"""Task sampling and family definitions for AFIN."""
import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


PRIOR_FAMILIES = (
    "diag_gaussian",
    "fullrank_gaussian",
    "diag_student_t",
    "diag_laplace",
)
LIKELIHOOD_FAMILIES = (
    "gaussian",
    "bernoulli_logit",
    "binomial_logit",
    "student_t",
    "gaussian_no_x",
)
PRIOR_FAMILY_TO_ID = {name: idx for idx, name in enumerate(PRIOR_FAMILIES)}
LIKELIHOOD_FAMILY_TO_ID = {name: idx for idx, name in enumerate(LIKELIHOOD_FAMILIES)}

X_FAMILIES = ("iid", "diag_scale", "correlated", "student_t")
X_FAMILY_TO_ID = {name: idx for idx, name in enumerate(X_FAMILIES)}
WHITENED_LIKE_X_FAMILY_PROBS = (0.70, 0.10, 0.10, 0.10)

HOMOGENEOUS_PROB = 0.5
HETERO_K_TAU = 1.0
HETERO_DIRICHLET_ALPHA = 0.5
FULLRANK_MATRIX_SCALE = 0.30
FULLRANK_PRIOR_EPS = 0.50


@dataclass
class Task:
    prior_family: str
    likelihood_family: str
    prior_family_id: int
    likelihood_family_id: int
    d: int
    N: int
    z0: torch.Tensor
    X: torch.Tensor
    y: torch.Tensor
    meta: dict
    prior_family_ids: torch.Tensor | None = None
    site_family_ids: torch.Tensor | None = None


@dataclass
class ExpType:
    d_min: int
    d_max: int
    N_min: int
    N_max: int
    posterior_family: str
    prior_families: list[str]
    likelihood_families: list[str]


def tree_map(obj, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: tree_map(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tree_map(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tree_map(v, fn) for v in obj)
    return obj


def canonicalize_posterior_family(value):
    key = value.lower().replace("_", "").replace("-", "")
    if key not in {"gaussian", "flow"}:
        raise argparse.ArgumentTypeError(f"Unknown posterior family: {value}")
    return key


def parse_family_list(value, allowed, name):
    families = []
    aliases = {
        "no_x_gaussian": "gaussian_no_x",
        "nox_gaussian": "gaussian_no_x",
        "gaussian_nox": "gaussian_no_x",
    }
    for item in value.split(","):
        key = item.strip().lower().replace("-", "_")
        if name == "likelihood":
            key = aliases.get(key, key)
        if not key:
            continue
        if key not in allowed:
            raise argparse.ArgumentTypeError(f"Unknown {name} family: {item}")
        families.append(key)
    if not families:
        raise argparse.ArgumentTypeError(f"Need at least one {name} family")
    return list(dict.fromkeys(families))


def make_exp_type(d_min, d_max, n_min, n_max, posterior_family="gaussian", prior_families=None, likelihood_families=None):
    return ExpType(
        d_min=d_min,
        d_max=d_max,
        N_min=n_min,
        N_max=n_max,
        posterior_family=canonicalize_posterior_family(posterior_family),
        prior_families=list(prior_families or PRIOR_FAMILIES),
        likelihood_families=list(likelihood_families or LIKELIHOOD_FAMILIES),
    )


# -----------------------------------------------------------------------------
# Numeric helpers (shared with model.py)
# -----------------------------------------------------------------------------


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def standard_normal_dist_like(loc):
    return torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros_like(loc), torch.ones_like(loc)), 1
    )


def gaussian_log_prob_from_precision_chol(z, mean, precision_chol):
    if z.ndim == 1:
        z = z.unsqueeze(0)
    delta = z - mean
    whitened = torch.einsum("bij,bj->bi", precision_chol.transpose(-1, -2), delta)
    quad = whitened.pow(2).sum(dim=-1)
    log_diag = torch.log(torch.diagonal(precision_chol, dim1=-2, dim2=-1)).sum(dim=-1)
    normalizer = mean.shape[-1] * math.log(2.0 * math.pi)
    return log_diag - 0.5 * (normalizer + quad)


def gaussian_sample_from_precision_chol(mean, precision_chol):
    eps = torch.randn_like(mean)
    delta = torch.linalg.solve_triangular(
        precision_chol.transpose(-1, -2), eps.unsqueeze(-1), upper=True
    ).squeeze(-1)
    return mean + delta


# -----------------------------------------------------------------------------
# Design matrix sampling
# -----------------------------------------------------------------------------


def _rand_uniform(shape, low, high, device):
    return low + (high - low) * torch.rand(*shape, device=device)


def _sample_log_scales(batch_size, d, low, high, device):
    return torch.exp(_rand_uniform((batch_size, d), low, high, device=device))


def _sample_spd(batch_size, d, eps, scale, device):
    M = scale * torch.randn(batch_size, d, d, device=device)
    return (M @ M.transpose(-1, -2)) / max(1, d) + eps * torch.eye(d, device=device)


def _random_orthogonal(batch_size, d, device, dtype):
    mats = torch.randn(batch_size, d, d, device=device, dtype=dtype)
    q, r = torch.linalg.qr(mats)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(-2)


def _sample_log_spectrum(batch_size, d, device, kappa_max, dtype):
    kappa_max = max(1.0, float(kappa_max))
    log_kappa = _rand_uniform((batch_size, 1), 0.0, math.log(kappa_max), device=device).to(dtype)
    log_eigs = -0.5 * log_kappa + torch.rand(batch_size, d, device=device, dtype=dtype) * log_kappa
    eigs = torch.exp(log_eigs)
    eigs = eigs * (float(d) / eigs.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    return eigs


def sample_x_family_ids(batch_size, device, mode):
    if mode == "whitened_like":
        probs = torch.tensor(WHITENED_LIKE_X_FAMILY_PROBS, device=device, dtype=torch.float32)
        return torch.multinomial(probs, batch_size, replacement=True)
    if mode not in X_FAMILY_TO_ID:
        raise ValueError(f"Unknown x sampler mode: {mode}")
    return torch.full((batch_size,), X_FAMILY_TO_ID[mode], device=device, dtype=torch.long)


def sample_design_matrix_batch(batch_size, N, d, device, mode="iid", kappa_max=100.0,
                                tail_df_min=5.0, tail_df_max=12.0):
    dtype = torch.float32
    base_scale = 0.9 / math.sqrt(max(1, d))
    family_ids = sample_x_family_ids(batch_size=batch_size, device=device, mode=mode)
    X = torch.empty(batch_size, N, d, device=device, dtype=dtype)
    cond = torch.ones(batch_size, device=device, dtype=dtype)
    erank = torch.full((batch_size,), float(d), device=device, dtype=dtype)

    for family_name in X_FAMILIES:
        family_id = X_FAMILY_TO_ID[family_name]
        mask = family_ids == family_id
        count = int(mask.sum().item())
        if count == 0:
            continue
        if family_name == "iid" or d == 1:
            X[mask] = base_scale * torch.randn(count, N, d, device=device, dtype=dtype)
            continue
        if family_name == "diag_scale":
            eigs = _sample_log_spectrum(count, d, device=device, kappa_max=kappa_max, dtype=dtype)
            X_family = torch.randn(count, N, d, device=device, dtype=dtype) * eigs.sqrt().unsqueeze(1)
        elif family_name == "correlated":
            eigs = _sample_log_spectrum(count, d, device=device, kappa_max=kappa_max, dtype=dtype)
            q = _random_orthogonal(count, d, device=device, dtype=dtype)
            chol = q @ torch.diag_embed(eigs.sqrt())
            X_family = torch.randn(count, N, d, device=device, dtype=dtype) @ chol.transpose(-1, -2)
        else:  # student_t
            eigs = _sample_log_spectrum(count, d, device=device, kappa_max=kappa_max, dtype=dtype)
            q = _random_orthogonal(count, d, device=device, dtype=dtype)
            chol = q @ torch.diag_embed(eigs.sqrt())
            gaussian_rows = torch.randn(count, N, d, device=device, dtype=dtype) @ chol.transpose(-1, -2)
            df = _rand_uniform((count, 1), tail_df_min, tail_df_max, device=device).to(dtype)
            chi2 = torch.distributions.Chi2(df.expand(count, N)).sample().to(device=device, dtype=dtype)
            row_scale = torch.sqrt((df.expand(count, N) - 2.0) / chi2.clamp_min(1e-6))
            X_family = gaussian_rows * row_scale.unsqueeze(-1)

        X[mask] = base_scale * X_family
        total = eigs.sum(dim=-1)
        cond[mask] = eigs.max(dim=-1).values / eigs.min(dim=-1).values.clamp_min(1e-12)
        erank[mask] = total.square() / eigs.square().sum(dim=-1).clamp_min(1e-12)

    return X, {"x_family_id": family_ids, "x_condition_number": cond, "x_effective_rank": erank}


# -----------------------------------------------------------------------------
# Family sampling
# -----------------------------------------------------------------------------


def _sample_prior_family_ids(batch_size, prior_families, device, fixed_family=None):
    if fixed_family is not None:
        return torch.full((batch_size,), PRIOR_FAMILY_TO_ID[fixed_family], device=device, dtype=torch.long)
    family_ids = torch.tensor([PRIOR_FAMILY_TO_ID[name] for name in prior_families], device=device, dtype=torch.long)
    return family_ids[torch.randint(family_ids.numel(), (batch_size,), device=device)]


def _sample_site_family_ids(batch_size, N, likelihood_families, device, fixed_family=None):
    if fixed_family is not None:
        return torch.full((batch_size, N), LIKELIHOOD_FAMILY_TO_ID[fixed_family], device=device, dtype=torch.long)
    family_ids = torch.tensor(
        [LIKELIHOOD_FAMILY_TO_ID[name] for name in likelihood_families], device=device, dtype=torch.long
    )
    site_family_ids = torch.empty(batch_size, N, device=device, dtype=torch.long)
    k_max = min(int(family_ids.numel()), int(N))
    if N <= 1 or k_max <= 1:
        homogeneous = torch.ones(batch_size, device=device, dtype=torch.bool)
    else:
        homogeneous = torch.rand(batch_size, device=device) < HOMOGENEOUS_PROB

    if bool(homogeneous.any().item()):
        chosen = family_ids[torch.randint(family_ids.numel(), (int(homogeneous.sum().item()),), device=device)]
        site_family_ids[homogeneous] = chosen.unsqueeze(-1).expand(-1, N)

    hetero_mask = ~homogeneous
    if bool(hetero_mask.any().item()):
        hetero_count = int(hetero_mask.sum().item())
        hetero_rows = hetero_mask.nonzero(as_tuple=True)[0]
        k_values = torch.arange(2, k_max + 1, device=device, dtype=torch.long)
        logits = -HETERO_K_TAU * (k_values.float() - 2.0)
        probs = torch.softmax(logits, dim=0)
        sampled_k = k_values[torch.multinomial(probs.expand(hetero_count, -1), 1, replacement=True).squeeze(-1)]
        random_order = torch.rand(hetero_count, N, device=device).argsort(dim=-1)

        for k in range(2, k_max + 1):
            k_mask = sampled_k == k
            if not bool(k_mask.any().item()):
                continue
            group_rows = hetero_rows[k_mask]
            group_size = int(group_rows.numel())
            scores = torch.rand(group_size, family_ids.numel(), device=device)
            chosen_pos = scores.topk(k, dim=-1).indices
            chosen = family_ids[chosen_pos]
            assignments = torch.empty(group_size, N, device=device, dtype=torch.long)
            assignments[:, :k] = chosen
            if N > k:
                pi = torch.distributions.Dirichlet(
                    torch.full((k,), HETERO_DIRICHLET_ALPHA, device=device)
                ).sample((group_size,))
                extra = torch.multinomial(pi, N - k, replacement=True)
                assignments[:, k:] = chosen.gather(1, extra)
            assignments = assignments.gather(1, random_order[k_mask])
            site_family_ids[group_rows] = assignments
    return site_family_ids


def _summarize_prior_batch(prior_family_ids):
    unique = torch.unique(prior_family_ids.detach().cpu())
    if unique.numel() == 1:
        family_id = int(unique.item())
        return PRIOR_FAMILIES[family_id], family_id
    return "mixed_batch", -1


def _summarize_likelihood_batch(site_family_ids):
    unique = torch.unique(site_family_ids.detach().cpu())
    if unique.numel() == 1:
        family_id = int(unique.item())
        return LIKELIHOOD_FAMILIES[family_id], family_id
    if bool((site_family_ids == site_family_ids[:, :1]).all().item()):
        return "mixed_batch", -1
    return "heterogeneous", -1


# -----------------------------------------------------------------------------
# Prior sampling
# -----------------------------------------------------------------------------


def _sample_prior_family(batch_size, d, prior_family, device):
    loc = 0.45 * torch.randn(batch_size, d, device=device)

    if prior_family == "diag_gaussian":
        scale = _sample_log_scales(batch_size, d, -0.8, 0.0, device)
        z0 = loc + scale * torch.randn(batch_size, d, device=device)
        meta = {"prior_loc": loc, "prior_scale": scale}
    elif prior_family == "fullrank_gaussian":
        precision = _sample_spd(batch_size, d, eps=FULLRANK_PRIOR_EPS, scale=FULLRANK_MATRIX_SCALE, device=device)
        cov = torch.linalg.inv(precision)
        chol = torch.linalg.cholesky(cov)
        z0 = loc + (chol @ torch.randn(batch_size, d, 1, device=device)).squeeze(-1)
        meta = {"prior_loc": loc, "prior_precision": precision}
    elif prior_family == "diag_student_t":
        scale = _sample_log_scales(batch_size, d, -0.7, 0.0, device)
        df = _rand_uniform((batch_size, 1), 3.0, 8.0, device=device)
        z0 = loc + scale * torch.distributions.StudentT(df.expand(batch_size, d)).sample()
        meta = {"prior_loc": loc, "prior_scale": scale, "prior_df": df}
    elif prior_family == "diag_laplace":
        scale = _sample_log_scales(batch_size, d, -1.0, -0.05, device)
        z0 = torch.distributions.Laplace(loc, scale).sample()
        meta = {"prior_loc": loc, "prior_scale": scale}
    else:
        raise ValueError(f"Unknown prior family: {prior_family}")

    return z0, meta


def _sample_prior_batch(prior_family_ids, d, device):
    batch_size = int(prior_family_ids.shape[0])
    z0 = torch.empty(batch_size, d, device=device)
    meta = {
        "prior_loc": torch.zeros(batch_size, d, device=device),
        "prior_scale": torch.zeros(batch_size, d, device=device),
        "prior_precision": torch.zeros(batch_size, d, d, device=device),
        "prior_df": torch.zeros(batch_size, 1, device=device),
    }
    for prior_family in PRIOR_FAMILIES:
        family_id = PRIOR_FAMILY_TO_ID[prior_family]
        mask = prior_family_ids == family_id
        if not bool(mask.any().item()):
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        z0_family, family_meta = _sample_prior_family(idx.numel(), d, prior_family, device)
        z0[idx] = z0_family
        for key, value in family_meta.items():
            meta[key][idx] = value
    return z0, meta


# -----------------------------------------------------------------------------
# Likelihood sampling
# -----------------------------------------------------------------------------


def _sample_likelihood_batch(batch_size, X, z0, site_family_ids, device):
    signal = torch.einsum("bnd,bd->bn", X, z0)
    N, d = X.shape[1], X.shape[2]
    y = torch.empty(batch_size, N, device=device)
    meta = {
        "likelihood_scale": torch.zeros(batch_size, N, device=device),
        "likelihood_y_vector": torch.zeros(batch_size, N, d, device=device),
        "likelihood_total_count": torch.zeros(batch_size, N, device=device),
        "likelihood_df": torch.zeros(batch_size, N, device=device),
        "likelihood_num_families": torch.tensor(
            [torch.unique(site_family_ids[b]).numel() for b in range(batch_size)],
            device=device, dtype=torch.float32,
        ),
        "likelihood_is_heterogeneous": (site_family_ids != site_family_ids[:, :1]).any(dim=1).float(),
    }

    for likelihood_family in LIKELIHOOD_FAMILIES:
        mask = site_family_ids == LIKELIHOOD_FAMILY_TO_ID[likelihood_family]
        if not bool(mask.any().item()):
            continue
        if likelihood_family == "gaussian":
            scale = torch.exp(_rand_uniform((batch_size, 1), -0.8, -0.05, device=device)).expand(batch_size, N)
            y_family = signal + scale * torch.randn(batch_size, N, device=device)
            meta["likelihood_scale"][mask] = scale[mask]
        elif likelihood_family == "gaussian_no_x":
            scale = torch.exp(_rand_uniform((batch_size, 1), -0.8, -0.05, device=device)).expand(batch_size, N)
            y_vector = z0.unsqueeze(1) + scale.unsqueeze(-1) * torch.randn(batch_size, N, d, device=device)
            y_family = y_vector.mean(dim=-1)
            meta["likelihood_scale"][mask] = scale[mask]
            meta["likelihood_y_vector"][mask] = y_vector[mask]
        elif likelihood_family == "bernoulli_logit":
            y_family = torch.bernoulli(torch.sigmoid(signal))
        elif likelihood_family == "binomial_logit":
            total_count = torch.randint(2, 9, (batch_size, N), device=device).float()
            y_family = torch.distributions.Binomial(total_count=total_count, probs=torch.sigmoid(signal)).sample()
            meta["likelihood_total_count"][mask] = total_count[mask]
        elif likelihood_family == "student_t":
            scale = torch.exp(_rand_uniform((batch_size, 1), -0.7, 0.0, device=device)).expand(batch_size, N)
            df = _rand_uniform((batch_size, 1), 3.5, 8.0, device=device).expand(batch_size, N)
            y_family = signal + scale * torch.distributions.StudentT(df).sample().to(device)
            meta["likelihood_scale"][mask] = scale[mask]
            meta["likelihood_df"][mask] = df[mask]
        else:
            raise ValueError(f"Unknown likelihood family: {likelihood_family}")
        y[mask] = y_family[mask]
    return y, meta


# -----------------------------------------------------------------------------
# Top-level task batch
# -----------------------------------------------------------------------------


def sample_task_batch(batch_size, exp_type, device="cpu", spec=None):
    spec = spec or {}
    d = int(spec.get("d", torch.randint(exp_type.d_min, exp_type.d_max + 1, (1,)).item()))
    N = int(spec.get("N", torch.randint(exp_type.N_min, exp_type.N_max + 1, (1,)).item()))

    prior_family_ids = _sample_prior_family_ids(
        batch_size, exp_type.prior_families, device, spec.get("prior_family")
    )
    z0, prior_meta = _sample_prior_batch(prior_family_ids, d, device)
    site_family_ids = _sample_site_family_ids(
        batch_size, N, exp_type.likelihood_families, device, spec.get("likelihood_family")
    )

    X, x_meta = sample_design_matrix_batch(
        batch_size=batch_size, N=N, d=d, device=device,
        mode=str(spec.get("x_mode", "iid")),
        kappa_max=float(spec.get("x_kappa_max", 100.0)),
    )
    y, like_meta = _sample_likelihood_batch(batch_size, X, z0, site_family_ids, device)

    prior_family, prior_family_id = _summarize_prior_batch(prior_family_ids)
    likelihood_family, likelihood_family_id = _summarize_likelihood_batch(site_family_ids)

    return Task(
        prior_family=prior_family, likelihood_family=likelihood_family,
        prior_family_id=prior_family_id, likelihood_family_id=likelihood_family_id,
        d=d, N=N, z0=z0, X=X, y=y,
        meta={**prior_meta, **like_meta, **x_meta},
        prior_family_ids=prior_family_ids, site_family_ids=site_family_ids,
    )


# -----------------------------------------------------------------------------
# Task moves and (de)serialization
# -----------------------------------------------------------------------------


def move_task(task, device):
    return Task(
        prior_family=task.prior_family, likelihood_family=task.likelihood_family,
        prior_family_id=task.prior_family_id, likelihood_family_id=task.likelihood_family_id,
        d=task.d, N=task.N,
        z0=task.z0.to(device), X=task.X.to(device), y=task.y.to(device),
        meta=tree_map(task.meta, lambda x: x.to(device)),
        prior_family_ids=None if task.prior_family_ids is None else task.prior_family_ids.to(device),
        site_family_ids=None if task.site_family_ids is None else task.site_family_ids.to(device),
    )


def repeat_task(task, repeats):
    assert task.z0.shape[0] == 1, "repeat_task is for cached eval tasks with batch_size=1"
    rep = lambda x: x.repeat(repeats, *([1] * (x.ndim - 1)))
    return Task(
        prior_family=task.prior_family, likelihood_family=task.likelihood_family,
        prior_family_id=task.prior_family_id, likelihood_family_id=task.likelihood_family_id,
        d=task.d, N=task.N,
        z0=rep(task.z0), X=rep(task.X), y=rep(task.y),
        meta=tree_map(task.meta, rep),
        prior_family_ids=None if task.prior_family_ids is None else rep(task.prior_family_ids),
        site_family_ids=None if task.site_family_ids is None else rep(task.site_family_ids),
    )


def task_to_dict(task):
    return {
        "prior_family": task.prior_family, "likelihood_family": task.likelihood_family,
        "prior_family_id": task.prior_family_id, "likelihood_family_id": task.likelihood_family_id,
        "d": task.d, "N": task.N,
        "z0": task.z0.cpu(), "X": task.X.cpu(), "y": task.y.cpu(),
        "meta": tree_map(task.meta, lambda x: x.cpu()),
        "prior_family_ids": None if task.prior_family_ids is None else task.prior_family_ids.cpu(),
        "site_family_ids": None if task.site_family_ids is None else task.site_family_ids.cpu(),
    }


def dict_to_task(payload):
    payload = dict(payload)
    meta = dict(payload.get("meta", {}))
    z0, y = payload["z0"], payload["y"]
    B, d, N = int(z0.shape[0]), int(z0.shape[-1]), int(y.shape[-1])
    device, dtype = z0.device, z0.dtype
    defaults = {
        "prior_loc": (B, d), "prior_scale": (B, d), "prior_precision": (B, d, d), "prior_df": (B, 1),
        "likelihood_scale": (B, N), "likelihood_y_vector": (B, N, d), "likelihood_total_count": (B, N),
        "likelihood_df": (B, N), "likelihood_num_families": (B,), "likelihood_is_heterogeneous": (B,),
        "x_family_id": (B,), "x_condition_number": (B,), "x_effective_rank": (B,),
    }
    for key, shape in defaults.items():
        if key not in meta:
            meta[key] = torch.zeros(*shape, device=device, dtype=dtype)
    payload["meta"] = meta
    payload.setdefault("prior_family_ids", None)
    payload.setdefault("site_family_ids", None)
    return Task(**payload)


# -----------------------------------------------------------------------------
# Reference posterior log probabilities
# -----------------------------------------------------------------------------


def prior_log_prob(task, z):
    if z.ndim == 1:
        z = z.unsqueeze(0)
    device = z.device
    prior_family_ids = task.prior_family_ids
    if prior_family_ids is None:
        prior_family_ids = torch.full((z.shape[0],), int(task.prior_family_id), device=device, dtype=torch.long)
    else:
        prior_family_ids = prior_family_ids.to(device)
    loc = task.meta["prior_loc"].to(device)
    scale = task.meta["prior_scale"].to(device)
    precision = task.meta["prior_precision"].to(device)
    df = task.meta["prior_df"].to(device)
    out = torch.empty(z.shape[0], device=device, dtype=z.dtype)
    for prior_family in PRIOR_FAMILIES:
        mask = prior_family_ids == PRIOR_FAMILY_TO_ID[prior_family]
        if not bool(mask.any().item()):
            continue
        z_m, loc_m = z[mask], loc[mask]
        if prior_family == "diag_gaussian":
            out[mask] = -0.5 * ((z_m - loc_m) / scale[mask]).pow(2).sum(dim=-1)
        elif prior_family == "fullrank_gaussian":
            delta = z_m - loc_m
            out[mask] = -0.5 * torch.einsum("bi,bij,bj->b", delta, precision[mask], delta)
        elif prior_family == "diag_student_t":
            residual = z_m - loc_m
            out[mask] = -0.5 * (
                (df[mask] + 1.0) * torch.log1p(residual.pow(2) / (df[mask] * scale[mask].pow(2)))
            ).sum(dim=-1)
        elif prior_family == "diag_laplace":
            out[mask] = -((z_m - loc_m).abs() / scale[mask]).sum(dim=-1)
    return out


def likelihood_log_prob(task, z):
    if z.ndim == 1:
        z = z.unsqueeze(0)
    device = z.device
    X = task.X.to(device)
    y = task.y.to(device)
    t = torch.einsum("bnd,bd->bn", X, z)
    site_family_ids = task.site_family_ids
    if site_family_ids is None:
        site_family_ids = torch.full(
            (z.shape[0], task.N), int(task.likelihood_family_id), device=device, dtype=torch.long
        )
    else:
        site_family_ids = site_family_ids.to(device)
    scale = task.meta["likelihood_scale"].to(device)
    y_vector = task.meta.get("likelihood_y_vector")
    if torch.is_tensor(y_vector):
        y_vector = y_vector.to(device=device, dtype=z.dtype)
    else:
        y_vector = torch.zeros(z.shape[0], task.N, task.d, device=device, dtype=z.dtype)
    total_count = task.meta["likelihood_total_count"].to(device)
    df = task.meta["likelihood_df"].to(device)
    site_log_prob = torch.zeros_like(t)
    for likelihood_family in LIKELIHOOD_FAMILIES:
        mask = site_family_ids == LIKELIHOOD_FAMILY_TO_ID[likelihood_family]
        if not bool(mask.any().item()):
            continue
        if likelihood_family == "gaussian":
            site_log_prob[mask] = (-0.5 * ((y - t) / scale.clamp_min(1e-8)).pow(2))[mask]
        elif likelihood_family == "gaussian_no_x":
            residual = (y_vector - z.unsqueeze(1)) / scale.clamp_min(1e-8).unsqueeze(-1)
            site_log_prob[mask] = (-0.5 * residual.pow(2).sum(dim=-1))[mask]
        elif likelihood_family == "bernoulli_logit":
            site_log_prob[mask] = (y * t - F.softplus(t))[mask]
        elif likelihood_family == "binomial_logit":
            site_log_prob[mask] = (y * t - total_count * F.softplus(t))[mask]
        elif likelihood_family == "student_t":
            residual = y - t
            safe_scale = scale.clamp_min(1e-8)
            safe_df = df.clamp_min(1e-8)
            site_log_prob[mask] = (
                -0.5 * (safe_df + 1.0) * torch.log1p(residual.pow(2) / (safe_df * safe_scale.pow(2)))
            )[mask]
    return site_log_prob.sum(dim=-1)


def unnormalized_log_posterior(task, z):
    return prior_log_prob(task, z) + likelihood_log_prob(task, z)


# -----------------------------------------------------------------------------
# Evaluation metrics over samples
# -----------------------------------------------------------------------------


def _empirical_cov(x):
    xc = x - x.mean(dim=0, keepdim=True)
    return xc.T @ xc / max(1, x.shape[0] - 1)


def m1_metric(x, y):
    return torch.norm(x.mean(dim=0) - y.mean(dim=0)).item()


def m2_metric(x, y):
    return torch.norm(_empirical_cov(x) - _empirical_cov(y), p="fro").item()


def sliced_w2_metric(x, y, num_projections=128, seed=0):
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n <= 0:
        return float("nan")
    x = x[:n].double()
    y = y[:n].double()
    d = int(x.shape[1])
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    directions = torch.randn(int(num_projections), d, generator=g, dtype=x.dtype, device=x.device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    proj_x = torch.sort(x @ directions.transpose(0, 1), dim=0).values
    proj_y = torch.sort(y @ directions.transpose(0, 1), dim=0).values
    return (proj_x - proj_y).pow(2).mean(dim=0).mean().sqrt().item()
