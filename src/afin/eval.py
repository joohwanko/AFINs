"""Evaluation suite, NUTS reference caching, and metrics."""
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .nuts import build_reference_payload, reference_worker
from .tasks import (
    X_FAMILIES,
    X_FAMILY_TO_ID,
    dict_to_task,
    m1_metric,
    m2_metric,
    move_task,
    repeat_task,
    sample_task_batch,
    sliced_w2_metric,
    task_to_dict,
    unnormalized_log_posterior,
)


PRIMARY_METRIC_KEYS = (
    "cross_entropy_p_to_q_per_dim",
    "psis_pareto_k",
    "sliced_w2",
    "ess_ratio_snis",
    "m1",
    "m2",
    "energy_gap",
    "var_over_Z_pbar_over_q",
)


def _progress(iterable, *, total=None, desc=None, leave=True, disable=False):
    if disable or _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


# -----------------------------------------------------------------------------
# Eval specs
# -----------------------------------------------------------------------------


def make_difficulty_tiers(d_min, d_max, n_min, n_max):
    d_easy = max(d_min, max(2, (d_max + d_min) // 4))
    d_med = max(d_min, (d_max + d_min) // 2)
    d_hard = d_max
    return [
        {"difficulty": "easy", "d": d_easy, "N": n_max},
        {"difficulty": "medium", "d": d_med, "N": max(n_min, (n_max + n_min) // 4)},
        {"difficulty": "hard", "d": d_hard, "N": n_min},
    ]


def make_eval_specs(exp_type):
    """One spec per (prior, likelihood, difficulty) cell."""
    tiers = make_difficulty_tiers(exp_type.d_min, exp_type.d_max, exp_type.N_min, exp_type.N_max)
    return [
        {"prior_family": p, "likelihood_family": l, "d": tier["d"], "N": tier["N"], "difficulty": tier["difficulty"]}
        for p in exp_type.prior_families
        for l in exp_type.likelihood_families
        for tier in tiers
    ]


def select_stratified_specs(specs, exp_type):
    """Pick one spec per (prior, difficulty) for fast online eval."""
    if not specs:
        return []
    prior_order = {name: idx for idx, name in enumerate(exp_type.prior_families)}
    like_order = {name: idx for idx, name in enumerate(exp_type.likelihood_families)}
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    selected = []
    for prior_family in exp_type.prior_families:
        prior_idx = prior_order[prior_family]
        for difficulty in ("easy", "medium", "hard"):
            difficulty_idx = difficulty_order[difficulty]
            candidates = sorted(
                [s for s in specs if s.get("prior_family") == prior_family and s.get("difficulty") == difficulty],
                key=lambda s: (like_order.get(s.get("likelihood_family"), 10**9), str(s.get("likelihood_family", ""))),
            )
            if not candidates:
                continue
            selected.append(dict(candidates[(prior_idx + difficulty_idx) % len(candidates)]))
    return selected


# -----------------------------------------------------------------------------
# Suite caching
# -----------------------------------------------------------------------------


def _suite_key(exp_type, seed, specs):
    signature = {"seed": int(seed), "exp_type": asdict(exp_type), "specs": specs}
    return hashlib.sha1(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]


def prepare_eval_suite(
    exp_type,
    ref_num_samples,
    ref_num_warmup,
    ref_source_num_samples,
    ref_source_num_warmup,
    cache_dir,
    seed,
    specs,
    workers=1,
):
    ref_num_samples = int(ref_num_samples)
    ref_num_warmup = int(ref_num_warmup)
    ref_source_num_samples = int(ref_source_num_samples or ref_num_samples)
    ref_source_num_warmup = int(ref_source_num_warmup or ref_num_warmup)
    if ref_source_num_samples < ref_num_samples:
        raise ValueError(
            f"ref_source_num_samples ({ref_source_num_samples}) must be >= ref_num_samples ({ref_num_samples})"
        )

    suite_dir = Path(cache_dir) / _suite_key(exp_type, seed, specs)
    suite_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = suite_dir / "suite.pt"
    refs_dir = suite_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    if tasks_path.exists():
        suite_payload = torch.load(tasks_path, map_location="cpu")
        task_dicts = suite_payload["tasks"]
        specs = suite_payload.get("specs", specs)
        print(f"Loaded eval suite from {tasks_path}")
    else:
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            task_dicts = [
                task_to_dict(sample_task_batch(batch_size=1, exp_type=exp_type, device="cpu", spec=spec))
                for spec in specs
            ]
        finally:
            torch.random.set_rng_state(rng_state)
        torch.save({"exp_type": asdict(exp_type), "seed": seed, "tasks": task_dicts, "specs": specs}, tasks_path)
        print(f"Built eval suite at {tasks_path}")

    suite = []
    missing_jobs = []
    refs = [None] * len(task_dicts)
    for i, task_dict in enumerate(task_dicts):
        ref_path = refs_dir / f"{i:03d}.pt"
        if ref_path.exists():
            payload = torch.load(ref_path, map_location="cpu")
            if int(payload.get("num_samples", 0)) >= ref_source_num_samples:
                refs[i] = payload
                continue
        missing_jobs.append(
            {
                "task_idx": i,
                "task_payload_path": str(refs_dir / f"{i:03d}_task.pt"),
                "output_path": str(ref_path),
                "num_samples": ref_source_num_samples,
                "num_warmup": ref_source_num_warmup,
                "seed": int(seed) + i + 1,
            }
        )
        torch.save({"task": task_dict, "task_idx": i}, refs_dir / f"{i:03d}_task.pt")

    if missing_jobs:
        worker_count = max(1, min(int(workers), len(missing_jobs)))
        print(f"[eval-cache] sampling {len(missing_jobs)} missing NUTS references with {worker_count} worker(s)")
        if worker_count > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count, mp_context=mp.get_context("spawn")
            ) as executor:
                future_to_job = {executor.submit(reference_worker, job): job for job in missing_jobs}
                for future in _progress(
                    concurrent.futures.as_completed(future_to_job),
                    total=len(future_to_job),
                    desc="[eval-cache] NUTS",
                ):
                    task_idx = future.result()
                    refs[task_idx] = torch.load(refs_dir / f"{task_idx:03d}.pt", map_location="cpu")
        else:
            for job in _progress(missing_jobs, total=len(missing_jobs), desc="[eval-cache] NUTS"):
                task_idx = reference_worker(job)
                refs[task_idx] = torch.load(refs_dir / f"{task_idx:03d}.pt", map_location="cpu")

    for i, (task_dict, spec) in enumerate(zip(task_dicts, specs)):
        payload = refs[i]
        samples = payload["samples"][:ref_num_samples].cpu()
        log_pbar = payload["log_pbar"][:ref_num_samples].cpu()
        suite.append({"task": dict_to_task(task_dict), "ref_samples": samples, "ref_log_pbar": log_pbar, "spec": spec})
    return suite


# -----------------------------------------------------------------------------
# Posterior log-prob (chunked) helpers
# -----------------------------------------------------------------------------


def chunked_log_posterior(task, z_samples, chunk_size=64):
    values = []
    total = z_samples.shape[0]
    chunk_size = max(1, min(int(chunk_size), total))
    for start in range(0, total, chunk_size):
        z_chunk = z_samples[start : start + chunk_size]
        task_chunk = repeat_task(task, z_chunk.shape[0])
        values.append(unnormalized_log_posterior(task_chunk, z_chunk).reshape(-1))
    return torch.cat(values, dim=0)


@torch.no_grad()
def chunked_model_log_prob(model, task, z_samples, chunk_size=64):
    values = []
    total = z_samples.shape[0]
    if total == 0:
        return torch.empty(0)
    chunk_size = max(1, min(int(chunk_size), total))
    for start in range(0, total, chunk_size):
        z_chunk = z_samples[start : start + chunk_size].to(next(model.parameters()).device)
        task_chunk = repeat_task(task, z_chunk.shape[0])
        values.append(model.log_prob(task_chunk, z_chunk).detach().cpu())
    return torch.cat(values, dim=0)


# -----------------------------------------------------------------------------
# Importance weight metrics + PSIS
# -----------------------------------------------------------------------------


def importance_weight_metrics(task, z_samples, log_q, ref_samples=None, ref_log_pbar=None, log_q_ref=None,
                               log_pbar=None):
    log_q = log_q.reshape(-1).double()
    k = min(z_samples.shape[0], log_q.shape[0])
    if log_pbar is None:
        log_pbar = chunked_log_posterior(task, z_samples[:k]).reshape(-1).double()
    else:
        log_pbar = log_pbar.reshape(-1).double()[:k]
    k = min(k, log_pbar.shape[0], log_q.shape[0])
    log_pbar, log_q = log_pbar[:k], log_q[:k]
    log_w = log_pbar - log_q
    log_k = torch.log(torch.tensor(float(k), dtype=log_w.dtype, device=log_w.device))
    log_mean_w = torch.logsumexp(log_w, dim=0) - log_k
    log_mean_w2 = torch.logsumexp(2.0 * log_w, dim=0) - log_k
    mean_w = torch.exp(log_mean_w)
    second_moment = torch.exp(log_mean_w2)
    cv2 = second_moment / torch.exp(2.0 * log_mean_w).clamp_min(1e-30) - 1.0
    norm_w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
    ess = 1.0 / norm_w.pow(2).sum().clamp_min(1e-30)
    out = {
        "cv2_pbar_over_q": float(cv2.clamp_min(0.0).item()),
        "ess_ratio_snis": float((ess / float(k)).clamp(0.0, 1.0).item()),
    }
    if ref_samples is not None and log_q_ref is not None:
        ref_samples = ref_samples[: log_q_ref.shape[0]]
        log_q_ref = log_q_ref.reshape(-1).double()[: ref_samples.shape[0]]
        if ref_log_pbar is None:
            ref_log_pbar = chunked_log_posterior(task, ref_samples).reshape(-1).double()
        else:
            ref_log_pbar = ref_log_pbar.reshape(-1).double()[: ref_samples.shape[0]].to(log_q_ref.device)
        log_mean_w_under_p = torch.logsumexp(ref_log_pbar - log_q_ref, dim=0) - torch.log(
            torch.tensor(float(ref_samples.shape[0]), dtype=log_q_ref.dtype, device=log_q_ref.device)
        )
        out["var_over_Z_pbar_over_q"] = float((torch.exp(log_mean_w_under_p) - mean_w).clamp_min(0.0).item())
    return out


def psis_pareto_k(log_w, tail_fraction=0.2):
    log_w = torch.as_tensor(log_w, dtype=torch.double).reshape(-1)
    log_w = log_w[torch.isfinite(log_w)]
    n = int(log_w.numel())
    if n < 10:
        return float("nan")
    log_w = log_w - log_w.max()
    w = np.asarray(torch.exp(log_w).cpu().numpy(), dtype=np.float64)
    w = w[np.isfinite(w)]
    n = int(w.size)
    if n < 10:
        return float("nan")
    tail_count = int(min(max(20, round(3.0 * math.sqrt(n))), max(5, round(float(tail_fraction) * n))))
    tail_count = max(5, min(tail_count, n - 1))
    w_sorted = np.sort(w)
    threshold = float(w_sorted[-tail_count - 1])
    excess = w_sorted[-tail_count:] - threshold
    excess = excess[np.isfinite(excess)]
    if excess.size < 5 or float(np.max(excess)) <= 0.0:
        return 0.0
    try:
        from scipy.stats import genpareto

        shape, _, _ = genpareto.fit(excess, floc=0.0)
        if np.isfinite(shape):
            return float(shape)
    except Exception:
        pass
    mean_excess = float(np.mean(excess))
    var_excess = float(np.var(excess, ddof=1)) if excess.size > 1 else 0.0
    if mean_excess <= 0.0 or var_excess <= 0.0 or not math.isfinite(mean_excess) or not math.isfinite(var_excess):
        return 0.0
    return float(0.5 * (1.0 - (mean_excess * mean_excess) / var_excess))


# -----------------------------------------------------------------------------
# Per-task metric summaries
# -----------------------------------------------------------------------------


def _sample_metrics(task, z_model, z_ref, item_idx, log_q=None, ref_log_pbar=None, log_q_ref=None):
    k = min(z_model.shape[0], z_ref.shape[0])
    z_model, z_ref = z_model[:k], z_ref[:k]
    log_pbar_model = chunked_log_posterior(task, z_model)
    metrics = {
        "m1": m1_metric(z_model, z_ref),
        "m2": m2_metric(z_model, z_ref),
        "sliced_w2": sliced_w2_metric(z_model, z_ref, num_projections=128, seed=24680 + item_idx * 1000 + k),
        "energy_gap": -log_pbar_model.mean().item() - (-ref_log_pbar[:k].mean().item() if ref_log_pbar is not None else 0.0),
    }
    if log_q_ref is not None:
        ce_value = -log_q_ref.reshape(-1).double().mean().item()
        metrics["cross_entropy_p_to_q"] = ce_value
        metrics["cross_entropy_p_to_q_per_dim"] = ce_value / max(1, int(task.d))
    if log_q is not None:
        metrics.update(
            importance_weight_metrics(
                task, z_model, log_q, ref_samples=z_ref, ref_log_pbar=ref_log_pbar,
                log_q_ref=log_q_ref, log_pbar=log_pbar_model,
            )
        )
        metrics["psis_pareto_k"] = psis_pareto_k(log_pbar_model - log_q[:k].reshape(-1).double())
    return metrics


def _make_record(task, metrics):
    x_family_id = int(task.meta["x_family_id"].reshape(-1)[0].item()) if "x_family_id" in task.meta else -1
    x_family = X_FAMILIES[x_family_id] if 0 <= x_family_id < len(X_FAMILIES) else "unknown"
    return {
        "prior_family": task.prior_family,
        "likelihood_family": task.likelihood_family,
        "d": task.d,
        "N": task.N,
        "x_family": x_family,
        "metrics": metrics,
    }


def _avg(metric_dicts):
    keys = metric_dicts[0].keys()
    return {k: sum(item[k] for item in metric_dicts) / len(metric_dicts) for k in keys}


def summarize_records(records):
    overall = _avg([record["metrics"] for record in records])
    by_x_family = {}
    for record in records:
        by_x_family.setdefault(record["x_family"], []).append(record["metrics"])
    by_x_family = {
        family: {**_avg(metric_dicts), "num_tasks": len(metric_dicts)}
        for family, metric_dicts in by_x_family.items()
    }
    return {"overall": overall, "by_x_family": by_x_family, "records": records}


# -----------------------------------------------------------------------------
# Model evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(model, eval_suite, num_samples, sample_chunk_size=64):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    records = []
    for item_idx, item in enumerate(eval_suite):
        task = move_task(item["task"], device)
        ref_samples = item["ref_samples"].cpu()
        ref_log_pbar = item.get("ref_log_pbar")
        if ref_log_pbar is not None:
            ref_log_pbar = ref_log_pbar.cpu()
        k = min(num_samples, ref_samples.shape[0])
        chunk = max(1, min(int(sample_chunk_size), k))

        if getattr(model, "posterior_family", "") == "gaussian" and hasattr(model, "posterior"):
            posterior = model.posterior(task)
            z_model, log_q = posterior.sample_and_log_prob((k,))
            z_model = z_model.reshape(k, -1, task.d).squeeze(1).detach().cpu()
            log_q = log_q.reshape(k, -1).squeeze(1).detach().cpu()
            log_q_ref = posterior.log_prob(ref_samples[:k].to(device)).detach().cpu()
        else:
            z_chunks, log_q_chunks = [], []
            for start in range(0, k, chunk):
                count = min(chunk, k - start)
                task_chunk = repeat_task(task, count)
                z_chunk = model.sample(task_chunk)
                log_q_chunk = model.log_prob(task_chunk, z_chunk)
                z_chunks.append(z_chunk.detach().cpu())
                log_q_chunks.append(log_q_chunk.detach().cpu())
            z_model = torch.cat(z_chunks, dim=0)
            log_q = torch.cat(log_q_chunks, dim=0)
            log_q_ref = chunked_model_log_prob(model, task, ref_samples, chunk_size=chunk)

        metrics = _sample_metrics(
            move_task(item["task"], "cpu"), z_model, ref_samples, item_idx,
            log_q=log_q, ref_log_pbar=ref_log_pbar, log_q_ref=log_q_ref,
        )
        records.append(_make_record(item["task"], metrics))
    if was_training:
        model.train()
    return records


# -----------------------------------------------------------------------------
# X-family training stats helpers (used in train log)
# -----------------------------------------------------------------------------


def x_family_train_stats(x_family_ids, raw_loss_values):
    stats = {}
    x_family_ids = x_family_ids.reshape(-1)
    raw_loss_values = raw_loss_values.reshape(-1)
    batch_size = max(1, int(x_family_ids.numel()))
    for family_name in X_FAMILIES:
        mask = x_family_ids == X_FAMILY_TO_ID[family_name]
        count = int(mask.sum().item())
        stats[f"x_family_count_{family_name}"] = count
        stats[f"x_family_frac_{family_name}"] = float(count) / float(batch_size)
        stats[f"x_family_raw_loss_{family_name}"] = (
            float(raw_loss_values[mask].mean().item()) if count > 0 else float("nan")
        )
    return stats


def aggregate_x_family_stats(micro_summaries):
    aggregated = {}
    total = max(1, sum(int(s.get(f"x_family_count_{f}", 0)) for s in micro_summaries for f in X_FAMILIES))
    for family_name in X_FAMILIES:
        family_count = sum(int(s.get(f"x_family_count_{family_name}", 0)) for s in micro_summaries)
        aggregated[f"x_family_frac_{family_name}"] = float(family_count) / float(total)
        weighted_loss, weighted_count = [], []
        for s in micro_summaries:
            count = int(s.get(f"x_family_count_{family_name}", 0))
            value = s.get(f"x_family_raw_loss_{family_name}", float("nan"))
            if count > 0 and math.isfinite(value):
                weighted_loss.append(float(value) * count)
                weighted_count.append(count)
        aggregated[f"x_family_raw_loss_{family_name}"] = (
            sum(weighted_loss) / max(1, sum(weighted_count)) if weighted_count else float("nan")
        )
    return aggregated


def x_family_eval_log(prefix, summary):
    log = {}
    for x_family, metrics in summary.get("by_x_family", {}).items():
        for key in PRIMARY_METRIC_KEYS:
            if key in metrics:
                log[f"{prefix}/x_family_{x_family}/{key}"] = metrics[key]
    return log


def x_family_eval_summary(summary):
    parts = []
    for x_family, metrics in summary.get("by_x_family", {}).items():
        bits = []
        for short, key in (("ce/d", "cross_entropy_p_to_q_per_dim"), ("k", "psis_pareto_k"),
                            ("sw2", "sliced_w2"), ("ess", "ess_ratio_snis")):
            if key in metrics:
                bits.append(f"{short}={metrics[key]:.3f}")
        if bits:
            parts.append(f"{x_family}:" + " ".join(bits))
    return " ".join(parts)


def filter_metrics(metrics):
    if metrics is None:
        return None
    return {k: metrics[k] for k in PRIMARY_METRIC_KEYS if k in metrics}
