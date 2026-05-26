"""Amortized Factor Inference Network (AFIN)."""
import copy
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from .tasks import (
    LIKELIHOOD_FAMILIES,
    LIKELIHOOD_FAMILY_TO_ID,
    PRIOR_FAMILIES,
    PRIOR_FAMILY_TO_ID,
    ExpType,
    gaussian_log_prob_from_precision_chol,
    gaussian_sample_from_precision_chol,
    standard_normal_dist_like,
    symlog,
)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def _zero_last(module):
    last = module.mlp[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


def _zero_last_bias(module):
    last = module.mlp[-1]
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def _init_softplus_unit(module):
    last = module.mlp[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        last.bias.data.fill_(math.log(math.expm1(1.0)))


def _symmetrize(x):
    return 0.5 * (x + x.transpose(-1, -2))


def _sym_pair(pair):
    return 0.5 * (pair + pair.transpose(-3, -2))


def _safe_log(x):
    return torch.log(x.clamp_min(1e-6))


def _eye(batch_shape, d, device, dtype):
    return torch.eye(d, device=device, dtype=dtype).expand(*batch_shape, d, d)


def _pair_expand(a, b):
    d = a.shape[-1]
    return a.unsqueeze(-1).expand(*a.shape, d), b.unsqueeze(-2).expand(*b.shape[:-1], d, d)


def _meta_tensor(task, key, shape, like):
    value = task.meta.get(key)
    if torch.is_tensor(value):
        value = value.to(device=like.device, dtype=like.dtype)
        if tuple(value.shape) == tuple(shape):
            return value
        try:
            return value.expand(shape)
        except RuntimeError:
            return value.reshape(shape)
    return torch.zeros(shape, device=like.device, dtype=like.dtype)


def _rich_node_features(pair_feat, node_feat):
    row = pair_feat.mean(dim=-2)
    col = pair_feat.mean(dim=-3)
    diag = torch.diagonal(pair_feat, dim1=-3, dim2=-2).movedim(-1, -2)
    global_node = node_feat.mean(dim=-2, keepdim=True).expand_as(node_feat)
    global_pair = pair_feat.mean(dim=(-3, -2)).unsqueeze(-2).expand_as(node_feat)
    return torch.cat([node_feat, 0.5 * (row + col), diag, global_node, global_pair], dim=-1)


# -----------------------------------------------------------------------------
# SPD safety helpers (Gaussian decoder)
# -----------------------------------------------------------------------------


def _safe_cholesky(matrix, rel_jitter=1e-12, abs_jitter=1e-10, max_tries=2):
    matrix = _symmetrize(matrix.double())
    B, d, _ = matrix.shape
    eye = torch.eye(d, device=matrix.device, dtype=matrix.dtype).unsqueeze(0).expand(B, -1, -1)
    diag_mean = torch.diagonal(matrix, dim1=-2, dim2=-1).abs().mean(dim=-1)
    base_jitter = torch.clamp(rel_jitter * diag_mean, min=abs_jitter)
    total_jitter = torch.zeros_like(base_jitter)
    chol, info = torch.linalg.cholesky_ex(matrix)
    bad = info > 0
    if not bad.any():
        return chol, total_jitter
    for attempt in range(max_tries):
        attempt_jitter = base_jitter * (10.0 ** attempt)
        total_jitter = torch.where(bad, torch.maximum(total_jitter, attempt_jitter), total_jitter)
        chol, info = torch.linalg.cholesky_ex(matrix + eye * total_jitter[:, None, None])
        bad = info > 0
        if not bad.any():
            return chol, total_jitter
    first_bad = int(torch.nonzero(bad, as_tuple=False)[0].item())
    raise torch._C._LinAlgError(
        f"SPD Cholesky failed (batch={first_bad}, max_jitter={float(total_jitter[first_bad].item()):.3e})"
    )


# -----------------------------------------------------------------------------
# BoxMLP layer (no context — only invariant coordinate summaries)
# -----------------------------------------------------------------------------


class BoxLayer(nn.Module):
    """Two coordinate-wise heads that update (pair, node) using 6C summaries.

    Each head receives 6 length-C features per coordinate (or coordinate pair).
    The pair update is symmetrized in the two coordinate indices.
    """

    def __init__(self, ch, hidden_dim=128, num_layers=3):
        super().__init__()
        self.pair_refine = MLP(6 * ch, ch, hidden_dim=hidden_dim, num_layers=num_layers)
        self.node_refine = MLP(6 * ch, ch, hidden_dim=hidden_dim, num_layers=num_layers)
        _zero_last(self.pair_refine)
        _zero_last(self.node_refine)

    def forward(self, pair, node):
        row = pair.mean(dim=-2)
        col = pair.mean(dim=-3)
        glob_pair = pair.mean(dim=(-2, -3))
        glob_node = node.mean(dim=-2)
        diag = torch.diagonal(pair, dim1=-3, dim2=-2).movedim(-1, -2)

        pair_input = torch.cat(
            [
                pair,
                row.unsqueeze(-2).expand(*row.shape[:-2], row.shape[-2], pair.shape[-2], row.shape[-1]),
                col.unsqueeze(-3).expand(*col.shape[:-2], pair.shape[-3], col.shape[-2], col.shape[-1]),
                node.unsqueeze(-2).expand(*node.shape[:-2], node.shape[-2], pair.shape[-2], node.shape[-1]),
                node.unsqueeze(-3).expand(*node.shape[:-2], pair.shape[-3], node.shape[-2], node.shape[-1]),
                glob_pair.unsqueeze(-2).unsqueeze(-2)
                .expand(*pair.shape[:-3], pair.shape[-3], pair.shape[-2], glob_pair.shape[-1]),
            ],
            dim=-1,
        )
        delta_pair = _sym_pair(self.pair_refine(pair_input))

        node_input = torch.cat(
            [
                node,
                diag,
                row,
                col,
                glob_pair.unsqueeze(-2).expand(*node.shape[:-2], node.shape[-2], glob_pair.shape[-1]),
                glob_node.unsqueeze(-2).expand(*node.shape[:-2], node.shape[-2], glob_node.shape[-1]),
            ],
            dim=-1,
        )
        delta_node = self.node_refine(node_input)
        return pair + delta_pair, node + delta_node


# -----------------------------------------------------------------------------
# Family-specific factor adapters
# -----------------------------------------------------------------------------


class FactorAdapter(nn.Module):
    PRIOR_NODE_DIMS = {"diag_gaussian": 3, "fullrank_gaussian": 3, "diag_student_t": 4, "diag_laplace": 3}
    PRIOR_PAIR_DIMS = {"diag_gaussian": 7, "fullrank_gaussian": 8, "diag_student_t": 8, "diag_laplace": 7}
    LIKE_NODE_DIMS = {"gaussian": 5, "gaussian_no_x": 5, "bernoulli_logit": 5, "binomial_logit": 7, "student_t": 6}
    LIKE_PAIR_DIMS = {"gaussian": 9, "gaussian_no_x": 6, "bernoulli_logit": 9, "binomial_logit": 11, "student_t": 10}

    def __init__(self, ch, hidden_dim=128, num_layers=3):
        super().__init__()
        self.ch = int(ch)

        def make_dict(dims):
            return nn.ModuleDict({
                name: MLP(in_dim, ch, hidden_dim=hidden_dim, num_layers=num_layers)
                for name, in_dim in dims.items()
            })

        self.prior_node_nets = make_dict(self.PRIOR_NODE_DIMS)
        self.prior_pair_nets = make_dict(self.PRIOR_PAIR_DIMS)
        self.like_node_nets = make_dict(self.LIKE_NODE_DIMS)
        self.like_pair_nets = make_dict(self.LIKE_PAIR_DIMS)
        for nets in (self.prior_node_nets, self.prior_pair_nets, self.like_node_nets, self.like_pair_nets):
            for net in nets.values():
                _zero_last_bias(net)

    def _prior_inputs(self, family, task, mask):
        loc = task.meta["prior_loc"][mask].to(dtype=task.z0.dtype)
        d = loc.shape[-1]
        diag_flag = _eye((loc.shape[0],), d, loc.device, loc.dtype)

        if family == "fullrank_gaussian":
            precision = task.meta["prior_precision"][mask].to(dtype=task.z0.dtype)
            diag = torch.diagonal(precision, dim1=-2, dim2=-1).clamp_min(1e-6)
            sqrt_diag = diag.sqrt()
            diag_i, diag_j = _pair_expand(diag, diag)
            loc_i, loc_j = _pair_expand(loc, loc)
            coupled_i = precision * loc_i
            coupled_j = precision * loc_j
            pair = torch.stack([loc_i, loc_j, diag_i, diag_j, precision, coupled_i, coupled_j, diag_flag], dim=-1)
            node = torch.stack([loc, diag, loc * sqrt_diag], dim=-1)
            return node, pair

        scale = task.meta["prior_scale"][mask].to(dtype=task.z0.dtype).clamp_min(1e-6)
        log_scale = _safe_log(scale)
        loc_over_scale = loc * scale.reciprocal()

        if family == "diag_student_t":
            df = task.meta["prior_df"][mask].to(dtype=task.z0.dtype).clamp_min(1e-6)
            log_df = _safe_log(df).expand(-1, d)
            loc_i, loc_j = _pair_expand(loc, loc)
            scale_i, scale_j = _pair_expand(log_scale, log_scale)
            ros_i, ros_j = _pair_expand(loc_over_scale, loc_over_scale)
            log_df_pair = log_df.unsqueeze(-1).expand(-1, d, d)
            node = torch.stack([log_df, loc, log_scale, loc_over_scale], dim=-1)
            pair = torch.stack([log_df_pair, loc_i, scale_i, ros_i, loc_j, scale_j, ros_j, diag_flag], dim=-1)
            return node, pair

        # diag_gaussian, diag_laplace
        loc_i, loc_j = _pair_expand(loc, loc)
        scale_i, scale_j = _pair_expand(log_scale, log_scale)
        ros_i, ros_j = _pair_expand(loc_over_scale, loc_over_scale)
        node = torch.stack([loc, log_scale, loc_over_scale], dim=-1)
        pair = torch.stack([loc_i, scale_i, ros_i, loc_j, scale_j, ros_j, diag_flag], dim=-1)
        return node, pair

    def _like_inputs(self, family, task, mask):
        x = task.X[mask]
        y = task.y[mask]
        M, d = x.shape
        norm = torch.log1p(x.norm(dim=-1))
        xi, xj = _pair_expand(x, x)
        xij = xi * xj
        diag_flag = _eye((M,), d, x.device, x.dtype)
        y_node = y.unsqueeze(-1).expand(-1, d)
        y_pair = y[:, None, None].expand(-1, d, d)
        norm_node = norm.unsqueeze(-1).expand(-1, d)
        norm_pair = norm[:, None, None].expand(-1, d, d)

        if family == "gaussian_no_x":
            y_vec = _meta_tensor(task, "likelihood_y_vector", task.X.shape, task.X)[mask]
            scale = _meta_tensor(task, "likelihood_scale", task.y.shape, task.y)[mask].clamp_min(1e-6)
            log_scale_node = _safe_log(scale).unsqueeze(-1).expand(-1, d)
            log_scale_pair = _safe_log(scale)[:, None, None].expand(-1, d, d)
            y_i, y_j = _pair_expand(y_vec, y_vec)
            y_norm = torch.log1p(y_vec.norm(dim=-1))
            y_mean = y_vec.mean(dim=-1)
            node = torch.stack(
                [y_vec, log_scale_node,
                 y_norm.unsqueeze(-1).expand(-1, d), y_mean.unsqueeze(-1).expand(-1, d), y_vec.square()],
                dim=-1,
            )
            pair = torch.stack(
                [y_i, y_j, log_scale_pair, y_i * y_j, diag_flag, y_norm[:, None, None].expand(-1, d, d)],
                dim=-1,
            )
            return node, pair

        if family == "gaussian":
            scale = _meta_tensor(task, "likelihood_scale", task.y.shape, task.y)[mask].clamp_min(1e-6)
            log_scale_node = _safe_log(scale).unsqueeze(-1).expand(-1, d)
            log_scale_pair = _safe_log(scale)[:, None, None].expand(-1, d, d)
            yx = y_node * x
            yx_i, yx_j = _pair_expand(yx, yx)
            node = torch.stack([y_node, log_scale_node, norm_node, x, yx], dim=-1)
            pair = torch.stack([y_pair, log_scale_pair, norm_pair, xi, xj, xij, yx_i, yx_j, diag_flag], dim=-1)
            return node, pair

        if family == "student_t":
            scale = _meta_tensor(task, "likelihood_scale", task.y.shape, task.y)[mask].clamp_min(1e-6)
            df = _meta_tensor(task, "likelihood_df", task.y.shape, task.y)[mask].clamp_min(1e-6)
            log_scale_node = _safe_log(scale).unsqueeze(-1).expand(-1, d)
            log_df_node = _safe_log(df).unsqueeze(-1).expand(-1, d)
            log_scale_pair = _safe_log(scale)[:, None, None].expand(-1, d, d)
            log_df_pair = _safe_log(df)[:, None, None].expand(-1, d, d)
            yx = y_node * x
            yx_i, yx_j = _pair_expand(yx, yx)
            node = torch.stack([y_node, log_scale_node, log_df_node, norm_node, x, yx], dim=-1)
            pair = torch.stack(
                [y_pair, log_scale_pair, log_df_pair, norm_pair, xi, xj, xij, yx_i, yx_j, diag_flag],
                dim=-1,
            )
            return node, pair

        if family == "bernoulli_logit":
            signed_y = 2.0 * y - 1.0
            signed_node = signed_y.unsqueeze(-1).expand(-1, d)
            signed_pair = signed_y[:, None, None].expand(-1, d, d)
            signed_x = signed_node * x
            sx_i, sx_j = _pair_expand(signed_x, signed_x)
            node = torch.stack([y_node, signed_node, norm_node, x, signed_x], dim=-1)
            pair = torch.stack([y_pair, signed_pair, norm_pair, xi, xj, xij, sx_i, sx_j, diag_flag], dim=-1)
            return node, pair

        if family == "binomial_logit":
            count = _meta_tensor(task, "likelihood_total_count", task.y.shape, task.y)[mask].clamp_min(1.0)
            log_count = torch.log1p(count)
            y_over_count = y / count
            signed_rate = 2.0 * y_over_count - 1.0
            signed_node = signed_rate.unsqueeze(-1).expand(-1, d)
            signed_x = signed_node * x
            sx_i, sx_j = _pair_expand(signed_x, signed_x)
            node = torch.stack(
                [y_node, log_count.unsqueeze(-1).expand(-1, d), y_over_count.unsqueeze(-1).expand(-1, d),
                 signed_node, norm_node, x, signed_x],
                dim=-1,
            )
            pair = torch.stack(
                [y_pair, log_count[:, None, None].expand(-1, d, d), y_over_count[:, None, None].expand(-1, d, d),
                 signed_rate[:, None, None].expand(-1, d, d), norm_pair, xi, xj, xij, sx_i, sx_j, diag_flag],
                dim=-1,
            )
            return node, pair

        raise ValueError(f"Unsupported likelihood family: {family}")

    def _adapt_prior(self, task):
        B, d = task.z0.shape
        node = torch.zeros(B, d, self.ch, device=task.z0.device, dtype=task.z0.dtype)
        pair = torch.zeros(B, d, d, self.ch, device=task.z0.device, dtype=task.z0.dtype)
        prior_family_ids = task.prior_family_ids
        if prior_family_ids is None:
            prior_family_ids = torch.full((B,), int(task.prior_family_id), device=task.z0.device, dtype=torch.long)
        else:
            prior_family_ids = prior_family_ids.to(device=task.z0.device, dtype=torch.long)
        for family in PRIOR_FAMILIES:
            if family not in self.prior_node_nets:
                continue
            mask = prior_family_ids == PRIOR_FAMILY_TO_ID[family]
            if not bool(mask.any().item()):
                continue
            node_in, pair_in = self._prior_inputs(family, task, mask)
            node[mask] = self.prior_node_nets[family](symlog(node_in)).to(dtype=node.dtype)
            pair[mask] = self.prior_pair_nets[family](symlog(pair_in)).to(dtype=pair.dtype)
        return _sym_pair(pair), node

    def _adapt_likelihood(self, task):
        B, N, d = task.X.shape
        node = torch.zeros(B, N, d, self.ch, device=task.X.device, dtype=task.X.dtype)
        pair = torch.zeros(B, N, d, d, self.ch, device=task.X.device, dtype=task.X.dtype)
        site_family_ids = task.site_family_ids
        if site_family_ids is None:
            site_family_ids = torch.full((B, N), int(task.likelihood_family_id), device=task.X.device, dtype=torch.long)
        else:
            site_family_ids = site_family_ids.to(device=task.X.device, dtype=torch.long)
        for family in LIKELIHOOD_FAMILIES:
            if family not in self.like_node_nets:
                continue
            mask = site_family_ids == LIKELIHOOD_FAMILY_TO_ID[family]
            if not bool(mask.any().item()):
                continue
            node_in, pair_in = self._like_inputs(family, task, mask)
            node[mask] = self.like_node_nets[family](symlog(node_in)).to(dtype=node.dtype)
            pair[mask] = self.like_pair_nets[family](symlog(pair_in)).to(dtype=pair.dtype)
        return _sym_pair(pair), node

    def forward(self, task):
        prior_pair, prior_node = self._adapt_prior(task)
        like_pair, like_node = self._adapt_likelihood(task)
        pair_tokens = torch.cat([prior_pair.unsqueeze(1), like_pair], dim=1)
        node_tokens = torch.cat([prior_node.unsqueeze(1), like_node], dim=1)
        return pair_tokens, node_tokens


# -----------------------------------------------------------------------------
# Box encoder (stack of BoxLayers, applied per factor)
# -----------------------------------------------------------------------------


class BoxEncoder(nn.Module):
    def __init__(self, ch, hidden_dim=128, num_layers=3, depth=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.layers = nn.ModuleList(
            [BoxLayer(ch, hidden_dim=hidden_dim, num_layers=num_layers) for _ in range(max(0, int(depth)))]
        )

    def forward(self, pair_tokens, node_tokens):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                pair_tokens, node_tokens = checkpoint_fn(layer, pair_tokens, node_tokens, use_reentrant=False)
            else:
                pair_tokens, node_tokens = layer(pair_tokens, node_tokens)
        return pair_tokens, node_tokens


# -----------------------------------------------------------------------------
# BoxTensorMap (Q/K/V/Out) — coordinate-wise MLP with invariant summaries
# -----------------------------------------------------------------------------


class BoxTensorMap(nn.Module):
    def __init__(self, ch, hidden_dim=128, num_layers=2, zero_init=False):
        super().__init__()
        self.pair_map = MLP(6 * ch, ch, hidden_dim=hidden_dim, num_layers=num_layers)
        self.node_map = MLP(6 * ch, ch, hidden_dim=hidden_dim, num_layers=num_layers)
        if zero_init:
            _zero_last(self.pair_map)
            _zero_last(self.node_map)

    def forward(self, pair, node):
        row = pair.mean(dim=-2)
        col = pair.mean(dim=-3)
        glob_pair = pair.mean(dim=(-2, -3))
        glob_node = node.mean(dim=-2)
        diag = torch.diagonal(pair, dim1=-3, dim2=-2).movedim(-1, -2)

        pair_input = torch.cat(
            [
                pair,
                row.unsqueeze(-2).expand(*row.shape[:-2], row.shape[-2], pair.shape[-2], row.shape[-1]),
                col.unsqueeze(-3).expand(*col.shape[:-2], pair.shape[-3], col.shape[-2], col.shape[-1]),
                node.unsqueeze(-2).expand(*node.shape[:-2], node.shape[-2], pair.shape[-2], node.shape[-1]),
                node.unsqueeze(-3).expand(*node.shape[:-2], pair.shape[-3], node.shape[-2], node.shape[-1]),
                glob_pair.unsqueeze(-2).unsqueeze(-2)
                .expand(*pair.shape[:-3], pair.shape[-3], pair.shape[-2], glob_pair.shape[-1]),
            ],
            dim=-1,
        )
        pair_out = _sym_pair(self.pair_map(pair_input))

        node_input = torch.cat(
            [
                node,
                diag,
                row,
                col,
                glob_pair.unsqueeze(-2).expand(*node.shape[:-2], node.shape[-2], glob_pair.shape[-1]),
                glob_node.unsqueeze(-2).expand(*node.shape[:-2], node.shape[-2], glob_node.shape[-1]),
            ],
            dim=-1,
        )
        node_out = self.node_map(node_input)
        return pair_out, node_out


def _common_num_heads(model_dim, value_dim):
    for heads in (8, 4, 2):
        if model_dim % heads == 0 and value_dim % heads == 0:
            return heads
    return 1


# -----------------------------------------------------------------------------
# BoxTransformer factor-attention block
# -----------------------------------------------------------------------------


class FactorAttentionBlock(nn.Module):
    def __init__(self, ch, model_dim=128, box_hidden_dim=128, box_num_layers=2):
        super().__init__()
        self.ch = int(ch)
        self.num_heads = _common_num_heads(int(model_dim), self.ch)
        self.head_dim = self.ch // self.num_heads
        self.score_mix_logits = nn.Parameter(torch.zeros(2))
        self.pair_attn_norm = nn.LayerNorm(ch)
        self.node_attn_norm = nn.LayerNorm(ch)
        self.pair_ffn_norm = nn.LayerNorm(ch)
        self.node_ffn_norm = nn.LayerNorm(ch)
        self.q_map = BoxTensorMap(ch, hidden_dim=box_hidden_dim, num_layers=box_num_layers)
        self.k_map = BoxTensorMap(ch, hidden_dim=box_hidden_dim, num_layers=box_num_layers)
        self.v_map = BoxTensorMap(ch, hidden_dim=box_hidden_dim, num_layers=box_num_layers)
        self.out_map = BoxTensorMap(ch, hidden_dim=box_hidden_dim, num_layers=box_num_layers, zero_init=True)
        self.ffn = BoxLayer(ch, hidden_dim=box_hidden_dim, num_layers=box_num_layers)

    def _split_heads_node(self, node):
        B, K, d, _ = node.shape
        return node.reshape(B, K, d, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

    def _split_heads_pair(self, pair):
        B, K, d, _, _ = pair.shape
        return pair.reshape(B, K, d, d, self.num_heads, self.head_dim).permute(0, 4, 1, 2, 3, 5)

    def _merge_heads_node(self, node):
        B, H, K, d, D = node.shape
        return node.permute(0, 2, 3, 1, 4).reshape(B, K, d, H * D)

    def _merge_heads_pair(self, pair):
        B, H, K, d, _, D = pair.shape
        return pair.permute(0, 2, 3, 4, 1, 5).reshape(B, K, d, d, H * D)

    def forward(self, pair_tokens, node_tokens):
        _, _, d, _, _ = pair_tokens.shape
        pair_attn = self.pair_attn_norm(pair_tokens)
        node_attn = self.node_attn_norm(node_tokens)
        q_pair, q_node = self.q_map(pair_attn, node_attn)
        k_pair, k_node = self.k_map(pair_attn, node_attn)
        v_pair, v_node = self.v_map(pair_attn, node_attn)

        q_node = self._split_heads_node(q_node)
        k_node = self._split_heads_node(k_node)
        v_node = self._split_heads_node(v_node)
        q_pair = self._split_heads_pair(q_pair)
        k_pair = self._split_heads_pair(k_pair)
        v_pair = self._split_heads_pair(v_pair)

        # Score normalization matches the paper appendix:
        #   node: 1 / (d * sqrt(head_dim));  pair: 1 / (d^2 * sqrt(head_dim))
        node_scores = torch.einsum("bhkdc,bhldc->bhkl", q_node, k_node) / float(max(1, d))
        pair_scores = torch.einsum("bhkijc,bhlijc->bhkl", q_pair, k_pair) / float(max(1, d * d))
        score_mix = torch.softmax(self.score_mix_logits, dim=0)
        scores = (score_mix[0] * node_scores + score_mix[1] * pair_scores) / math.sqrt(float(self.head_dim))
        attn = torch.softmax(scores, dim=-1)

        node_mix = self._merge_heads_node(torch.einsum("bhkl,bhldc->bhkdc", attn, v_node))
        pair_mix = self._merge_heads_pair(torch.einsum("bhkl,bhlijc->bhkijc", attn, v_pair))

        delta_pair, delta_node = self.out_map(pair_mix, node_mix)
        pair_tokens = _sym_pair(pair_tokens + delta_pair)
        node_tokens = node_tokens + delta_node

        pair_ffn = self.pair_ffn_norm(pair_tokens)
        node_ffn = self.node_ffn_norm(node_tokens)
        delta_pair, delta_node = self.ffn(pair_ffn, node_ffn)
        pair_tokens = _sym_pair(pair_tokens + (delta_pair - pair_ffn))
        node_tokens = node_tokens + (delta_node - node_ffn)
        return pair_tokens, node_tokens


# -----------------------------------------------------------------------------
# Set merge over factor tokens (sum pooling after attention blocks)
# -----------------------------------------------------------------------------


class FactorMerge(nn.Module):
    def __init__(self, ch, model_dim=128, num_layers=2, box_num_layers=2, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.blocks = nn.ModuleList(
            [
                FactorAttentionBlock(ch, model_dim=model_dim, box_hidden_dim=model_dim, box_num_layers=box_num_layers)
                for _ in range(max(0, int(num_layers)))
            ]
        )

    def forward(self, pair_tokens, node_tokens):
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                pair_tokens, node_tokens = checkpoint_fn(block, pair_tokens, node_tokens, use_reentrant=False)
            else:
                pair_tokens, node_tokens = block(pair_tokens, node_tokens)
        pair_feat = _sym_pair(pair_tokens.sum(dim=1))
        node_feat = node_tokens.sum(dim=1)
        return pair_feat, node_feat


# -----------------------------------------------------------------------------
# Gaussian decoder
# -----------------------------------------------------------------------------


class GaussianDecoder(nn.Module):
    def __init__(self, ch, hidden_dim=128, num_layers=3, residual_scale=0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.offdiag_head = MLP(ch, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        self.diag_head = MLP(ch, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        self.mean_head = MLP(5 * ch, 1, hidden_dim=hidden_dim, num_layers=num_layers)
        _zero_last(self.offdiag_head)
        _init_softplus_unit(self.diag_head)
        _zero_last(self.mean_head)

    def forward(self, pair_feat, node_feat):
        d = node_feat.shape[1]
        raw_offdiag = self.residual_scale * self.offdiag_head(pair_feat).squeeze(-1)
        A = _symmetrize(raw_offdiag.double())
        eye_bool = torch.eye(d, device=A.device, dtype=torch.bool).unsqueeze(0)
        A = A.masked_fill(eye_bool, 0.0)
        offdiag_abs_sum = A.abs().sum(dim=-1)
        diag_extra = F.softplus(self.diag_head(node_feat).squeeze(-1)).double() + 1e-4
        precision = _symmetrize(A + torch.diag_embed(offdiag_abs_sum + diag_extra))
        precision_chol, _ = _safe_cholesky(precision)
        precision = precision_chol @ precision_chol.transpose(-1, -2)
        mean = self.mean_head(_rich_node_features(pair_feat, node_feat)).squeeze(-1).double()
        return precision, mean, precision_chol


# -----------------------------------------------------------------------------
# Flow head: context projection + final affine + RealNVP coupling layers
# -----------------------------------------------------------------------------


class FlowContext(nn.Module):
    def __init__(self, ch, context_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        nl = max(2, int(num_layers))
        self.node_proj = MLP(3 * ch, context_dim, hidden_dim=max(hidden_dim, context_dim), num_layers=nl)
        self.pair_proj = MLP(ch, context_dim, hidden_dim=max(hidden_dim, context_dim), num_layers=nl)

    def forward(self, pair_feat, node_feat):
        row = pair_feat.mean(dim=-2)
        col = pair_feat.mean(dim=-3)
        diag = torch.diagonal(pair_feat, dim1=-3, dim2=-2).movedim(-1, -2)
        node_context = self.node_proj(torch.cat([node_feat, 0.5 * (row + col), diag], dim=-1))
        pair_context = _sym_pair(self.pair_proj(pair_feat))
        return node_context, pair_context


class FlowAffine(nn.Module):
    def __init__(self, ch, hidden_dim=128, max_log_scale=1.2, num_layers=2):
        super().__init__()
        self.max_log_scale = float(max_log_scale)
        self.head = MLP(5 * ch, 2, hidden_dim=max(hidden_dim, 5 * ch), num_layers=max(2, int(num_layers)))
        _zero_last(self.head)

    def forward(self, pair_feat, node_feat):
        raw_shift, raw_log_scale = self.head(_rich_node_features(pair_feat, node_feat)).unbind(dim=-1)
        log_scale = self.max_log_scale * torch.tanh(raw_log_scale / max(self.max_log_scale, 1e-6))
        return raw_shift, log_scale


class CouplingLayer(nn.Module):
    def __init__(self, ch, hidden_dim=128, num_layers=2, max_log_scale=1.2, layer_idx=0, mask_seed=1729):
        super().__init__()
        self.max_log_scale = float(max_log_scale)
        self.layer_idx = int(layer_idx)
        self.mask_seed = int(mask_seed)
        nl = max(2, int(num_layers))
        self.node_proj = MLP(ch + 2, ch, hidden_dim=hidden_dim, num_layers=nl)
        self.pair_proj = MLP(ch + 6, ch, hidden_dim=hidden_dim, num_layers=nl)
        self.box = BoxLayer(ch, hidden_dim=hidden_dim, num_layers=nl)
        self.param_head = MLP(ch, 2, hidden_dim=hidden_dim, num_layers=nl)
        _zero_last(self.param_head)

    def _mask(self, d, device, dtype):
        if d <= 1:
            return torch.zeros(d, device=device, dtype=dtype)
        idx = torch.arange(d, device=device, dtype=torch.float32)
        phase = float(self.mask_seed + 104729 * (self.layer_idx + 1))
        scores = torch.frac(torch.sin((idx + 1.0) * 12.9898 + phase) * 43758.5453)
        keep = max(1, min(d - 1, d // 2))
        top = torch.topk(scores, keep, largest=True).indices
        mask = torch.zeros(d, device=device, dtype=dtype)
        mask[top] = 1.0
        if self.layer_idx % 2 == 1:
            mask = 1.0 - mask
        return mask

    def _conditioner(self, x_keep, mask, node_context, pair_context):
        B, d = x_keep.shape
        mask_b = mask.unsqueeze(0).expand(B, -1)
        node_scalars = torch.stack([x_keep, mask_b], dim=-1)
        node = self.node_proj(torch.cat([node_context, symlog(node_scalars)], dim=-1))

        xi = x_keep.unsqueeze(-1).expand(-1, -1, d)
        xj = x_keep.unsqueeze(-2).expand(-1, d, -1)
        mi = mask_b.unsqueeze(-1).expand(-1, -1, d)
        mj = mask_b.unsqueeze(-2).expand(-1, d, -1)
        diag = torch.eye(d, device=x_keep.device, dtype=x_keep.dtype).expand(B, -1, -1)
        pair_scalars = torch.stack([xi, xj, mi, mj, mi * mj, diag], dim=-1)
        pair = self.pair_proj(torch.cat([pair_context, symlog(pair_scalars)], dim=-1))

        pair, node = self.box(pair, node)
        params = self.param_head(node)
        raw_shift, raw_log_scale = params.unbind(dim=-1)
        log_scale = self.max_log_scale * torch.tanh(raw_log_scale / max(self.max_log_scale, 1e-6))
        return raw_shift, log_scale

    def forward_transform(self, x, node_context, pair_context, strength=1.0):
        strength = float(strength)
        mask = self._mask(x.shape[-1], x.device, x.dtype).unsqueeze(0)
        transform_mask = 1.0 - mask
        x_keep = x * mask
        shift, log_scale = self._conditioner(x_keep, mask.squeeze(0), node_context, pair_context)
        shift, log_scale = strength * shift, strength * log_scale
        y = x_keep + transform_mask * (x * torch.exp(log_scale) + shift)
        return y, (transform_mask * log_scale).sum(dim=-1)

    def inverse_transform(self, y, node_context, pair_context, strength=1.0):
        strength = float(strength)
        mask = self._mask(y.shape[-1], y.device, y.dtype).unsqueeze(0)
        transform_mask = 1.0 - mask
        y_keep = y * mask
        shift, log_scale = self._conditioner(y_keep, mask.squeeze(0), node_context, pair_context)
        shift, log_scale = strength * shift, strength * log_scale
        x = y_keep + transform_mask * ((y - shift) * torch.exp(-log_scale))
        return x, -(transform_mask * log_scale).sum(dim=-1)


class RealNVPFlow(nn.Module):
    def __init__(self, ch, hidden_dim=128, num_layers=2, num_steps=4, max_log_scale=1.2):
        super().__init__()
        self.steps = nn.ModuleList(
            [
                CouplingLayer(ch, hidden_dim=hidden_dim, num_layers=num_layers,
                              max_log_scale=max_log_scale, layer_idx=idx)
                for idx in range(int(num_steps))
            ]
        )

    def sample(self, base_dist, context, strength=1.0):
        node_context, pair_context = context
        z = base_dist.rsample()
        for step in self.steps:
            z, _ = step.forward_transform(z, node_context, pair_context, strength=strength)
        return z

    def log_prob(self, base_dist, context, z, strength=1.0):
        node_context, pair_context = context
        x = z
        inv_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for step in reversed(self.steps):
            x, delta = step.inverse_transform(x, node_context, pair_context, strength=strength)
            inv_log_det = inv_log_det + delta
        return base_dist.log_prob(x) + inv_log_det


# -----------------------------------------------------------------------------
# Final affine z = exp(ℓ) v + a
# -----------------------------------------------------------------------------


def _apply_final_affine(v, shift, log_scale, strength):
    return v * torch.exp(strength * log_scale) + strength * shift


def _invert_final_affine(z, shift, log_scale, strength):
    return (z - strength * shift) * torch.exp(-strength * log_scale), (strength * log_scale).sum(dim=-1)


def _ensure_batch_z(z):
    return z.unsqueeze(0) if z.ndim == 1 else z


# -----------------------------------------------------------------------------
# AFIN = Amortized Factor Inference Network
# -----------------------------------------------------------------------------


class AFIN(nn.Module):
    def __init__(
        self,
        exp_type: ExpType,
        feat_dim=32,
        adapter_hidden_dim=64,
        adapter_num_layers=2,
        box_hidden_dim=128,
        box_num_layers=3,
        box_depth=1,
        merge_model_dim=128,
        merge_num_layers=2,
        decoder_hidden_dim=128,
        decoder_num_layers=3,
        decoder_residual_scale=0.1,
        flow_hidden_dim=128,
        flow_context_dim=128,
        flow_num_layers=2,
        flow_num_steps=2,
        flow_max_log_scale=1.2,
        use_encoder_checkpoint=False,
        use_merge_checkpoint=False,
        amp_dtype=None,
    ):
        super().__init__()
        self.exp_type = exp_type
        self.feat_dim = int(feat_dim)
        self.posterior_family = self.exp_type.posterior_family
        self.amp_dtype = amp_dtype
        self.flow_strength = 1.0

        self.adapter = FactorAdapter(
            ch=feat_dim, hidden_dim=adapter_hidden_dim, num_layers=adapter_num_layers,
        )
        self.encoder = BoxEncoder(
            ch=feat_dim, hidden_dim=box_hidden_dim, num_layers=box_num_layers,
            depth=box_depth, use_checkpoint=use_encoder_checkpoint,
        )
        self.merge = FactorMerge(
            ch=feat_dim, model_dim=merge_model_dim, num_layers=merge_num_layers,
            box_num_layers=max(2, box_num_layers), use_checkpoint=use_merge_checkpoint,
        )
        self.decoder = GaussianDecoder(
            ch=feat_dim, hidden_dim=decoder_hidden_dim, num_layers=decoder_num_layers,
            residual_scale=decoder_residual_scale,
        )
        self.flow_context_proj = FlowContext(
            ch=feat_dim, context_dim=flow_context_dim, hidden_dim=flow_hidden_dim, num_layers=flow_num_layers,
        )
        self.flow_affine = FlowAffine(
            ch=feat_dim, hidden_dim=flow_hidden_dim, max_log_scale=flow_max_log_scale, num_layers=flow_num_layers,
        )
        self.flow = RealNVPFlow(
            ch=flow_context_dim, hidden_dim=flow_hidden_dim, num_layers=flow_num_layers,
            num_steps=flow_num_steps, max_log_scale=flow_max_log_scale,
        )

    def set_flow_strength(self, value):
        self.flow_strength = float(max(0.0, min(1.0, value)))

    def _autocast_context(self, task):
        device_type = task.z0.device.type
        if self.amp_dtype is not None and device_type == "cuda":
            return torch.autocast(device_type=device_type, dtype=self.amp_dtype)
        return nullcontext()

    def _encode_features(self, task):
        with self._autocast_context(task):
            pair_tokens, node_tokens = self.adapter(task)
            pair_tokens, node_tokens = self.encoder(pair_tokens, node_tokens)
            pair_feat, node_feat = self.merge(pair_tokens, node_tokens)
        return pair_feat.float(), node_feat.float()

    def _gaussian_posterior(self, task):
        pair_feat, node_feat = self._encode_features(task)
        precision, mean, precision_chol = self.decoder(pair_feat, node_feat)
        return precision.float(), mean.float(), precision_chol.float()

    def _flow_posterior_parts(self, task):
        pair_feat, node_feat = self._encode_features(task)
        node_context, pair_context = self.flow_context_proj(pair_feat, node_feat)
        shift, log_scale = self.flow_affine(pair_feat, node_feat)
        return (node_context, pair_context), shift, log_scale, node_feat.shape[0], node_feat.shape[1]

    def log_prob(self, task, z):
        if self.posterior_family == "gaussian":
            _, mean, precision_chol = self._gaussian_posterior(task)
            return gaussian_log_prob_from_precision_chol(z, mean, precision_chol)
        # flow path
        flow_context, shift, log_scale, B, _ = self._flow_posterior_parts(task)
        z = _ensure_batch_z(z)
        flow_z, affine_log_det = _invert_final_affine(z, shift, log_scale, self.flow_strength)
        loc = torch.zeros_like(flow_z)
        base_dist = standard_normal_dist_like(loc)
        return self.flow.log_prob(base_dist, flow_context, flow_z, strength=self.flow_strength) - affine_log_det

    def forward(self, task):
        return -self.log_prob(task, task.z0).mean() / task.d

    @torch.no_grad()
    def sample(self, task):
        if self.posterior_family == "gaussian":
            _, mean, precision_chol = self._gaussian_posterior(task)
            return gaussian_sample_from_precision_chol(mean, precision_chol)
        flow_context, shift, log_scale, B, d = self._flow_posterior_parts(task)
        loc = torch.zeros(B, d, device=shift.device, dtype=shift.dtype)
        base_dist = standard_normal_dist_like(loc)
        flow_z = self.flow.sample(base_dist, flow_context, strength=self.flow_strength)
        return _apply_final_affine(flow_z, shift, log_scale, self.flow_strength)

    def posterior(self, task):
        if self.posterior_family == "gaussian":
            _, mean, precision_chol = self._gaussian_posterior(task)
            return Posterior(family="gaussian", mean=mean, precision_chol=precision_chol)
        flow_context, shift, log_scale, B, d = self._flow_posterior_parts(task)
        return Posterior(
            family="flow", flow=self.flow, flow_context=flow_context,
            flow_shift=shift, flow_log_scale=log_scale, flow_strength=self.flow_strength,
        )


class Posterior:
    """Cached posterior distribution returned by AFIN.posterior(task)."""

    def __init__(self, family, mean=None, precision_chol=None,
                 flow=None, flow_context=None, flow_shift=None, flow_log_scale=None, flow_strength=1.0):
        self.family = family
        if family == "gaussian":
            self.mean = mean
            self.precision_chol = precision_chol
            cov = torch.cholesky_inverse(precision_chol)
            scale_tril = torch.linalg.cholesky(cov)
            self._mvn = torch.distributions.MultivariateNormal(loc=mean, scale_tril=scale_tril, validate_args=False)
        else:
            self.flow = flow
            self.flow_context = flow_context
            self.flow_shift = flow_shift
            self.flow_log_scale = flow_log_scale
            self.flow_strength = flow_strength

    def sample(self, sample_shape=torch.Size()):
        if self.family == "gaussian":
            return self._mvn.sample(sample_shape)
        sample_shape = torch.Size(sample_shape)
        if len(tuple(sample_shape)) != 0:
            n = math.prod(sample_shape)
            B, d = self.flow_shift.shape

            def expand_batch(x):
                return x.unsqueeze(0).expand(n, *x.shape).reshape(n * B, *x.shape[1:])

            node_context, pair_context = self.flow_context
            flow_context = (expand_batch(node_context), expand_batch(pair_context))
            shift = expand_batch(self.flow_shift)
            log_scale = expand_batch(self.flow_log_scale)
            loc = torch.zeros(n * B, d, device=shift.device, dtype=shift.dtype)
            base_dist = standard_normal_dist_like(loc)
            flow_z = self.flow.sample(base_dist, flow_context, strength=self.flow_strength)
            z = _apply_final_affine(flow_z, shift, log_scale, self.flow_strength)
            return z.reshape(*sample_shape, B, d)
        loc = torch.zeros_like(self.flow_shift)
        base_dist = standard_normal_dist_like(loc)
        flow_z = self.flow.sample(base_dist, self.flow_context, strength=self.flow_strength)
        return _apply_final_affine(flow_z, self.flow_shift, self.flow_log_scale, self.flow_strength)

    def log_prob(self, z):
        if self.family == "gaussian":
            return self._mvn.log_prob(z)
        z = _ensure_batch_z(z)
        shift = self.flow_shift
        log_scale = self.flow_log_scale
        node_context, pair_context = self.flow_context
        if shift.shape[0] == 1 and z.shape[0] != 1:
            n = z.shape[0]
            shift = shift.expand(n, -1)
            log_scale = log_scale.expand(n, -1)
            node_context = node_context.expand(n, -1, -1)
            pair_context = pair_context.expand(n, -1, -1, -1)
        flow_z, affine_log_det = _invert_final_affine(z, shift, log_scale, self.flow_strength)
        loc = torch.zeros_like(flow_z)
        base_dist = standard_normal_dist_like(loc)
        return self.flow.log_prob(base_dist, (node_context, pair_context), flow_z, strength=self.flow_strength) - affine_log_det

    def sample_and_log_prob(self, sample_shape=torch.Size()):
        z = self.sample(sample_shape)
        return z, self.log_prob(z)


class EMA:
    def __init__(self, model, decay=0.999, update_after_step=0):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.update_after_step = update_after_step

    @torch.no_grad()
    def update(self, model, step):
        if step < self.update_after_step:
            self.module.load_state_dict(model.state_dict())
            return
        msd = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            value = msd[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value)
