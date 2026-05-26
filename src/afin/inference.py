"""Checkpoint loading and one-shot posterior helpers."""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from .model import AFIN
from .nuts import sample_nuts_reference
from .spec import Problem
from .tasks import ExpType, move_task
from .train import build_model_kwargs


PRETRAINED_REPO_ID = "joohwanko/AFINs"
PRETRAINED_CHECKPOINTS = {
    "gaussian": "v1-5m-gaussian-d1-16-n1-256",
    "flow": "v1-5m-flow-d1-16-n1-256",
}


def _sync_if_cuda(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _attach_runtime(posterior, *, forward_seconds, sampling_seconds=0.0, total_seconds=None):
    posterior.runtime_seconds = float(forward_seconds)
    posterior.forward_runtime_seconds = float(forward_seconds)
    posterior.sampling_runtime_seconds = float(sampling_seconds)
    if total_seconds is not None:
        posterior.total_runtime_seconds = float(total_seconds)
    return posterior


def _project_root():
    return Path(__file__).resolve().parents[2]


def find_checkpoint_dir(root=None, weights="final", posterior_family=None):
    """Find the newest local checkpoint run containing ``weights``."""
    root = Path(root) if root is not None else _project_root() / "checkpoints"
    candidates = []
    for config_path in root.glob("*/config.json"):
        run_dir = config_path.parent
        if (run_dir / f"{weights}.pt").exists():
            if posterior_family is not None:
                config = json.loads(config_path.read_text())
                family = config.get("exp_type", {}).get("posterior_family")
                family = family or config.get("args", {}).get("posterior_family")
                if family != posterior_family:
                    continue
            candidates.append(run_dir)
    if not candidates:
        details = f" with posterior_family={posterior_family!r}" if posterior_family is not None else ""
        raise FileNotFoundError(f"No checkpoint{details} and {weights}.pt found under {root}")
    return max(candidates, key=lambda p: (p / f"{weights}.pt").stat().st_mtime)


def _canonical_pretrained_family(family):
    key = str(family).lower().replace("_", "").replace("-", "")
    if key not in PRETRAINED_CHECKPOINTS:
        raise ValueError(f"Unknown pretrained posterior family: {family!r}")
    return key


def _pretrained_allow_patterns(families=None):
    if families is None:
        families = PRETRAINED_CHECKPOINTS
    if isinstance(families, str):
        families = [families]
    families = [_canonical_pretrained_family(family) for family in families]
    patterns = [f"{PRETRAINED_CHECKPOINTS[family]}/*" for family in families]
    patterns.append("README.md")
    return patterns


def download_checkpoint(repo_id, *, revision=None, cache_dir=None, allow_patterns=None):
    """Download an AFIN checkpoint snapshot from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
    ))


def download_pretrained_checkpoints(
    *,
    repo_id=PRETRAINED_REPO_ID,
    revision=None,
    cache_dir=None,
    families=None,
):
    """Download the public AFIN demo checkpoints from Hugging Face Hub."""
    return download_checkpoint(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=_pretrained_allow_patterns(families),
    )


def load_pretrained_afin(
    posterior_family,
    *,
    repo_id=PRETRAINED_REPO_ID,
    revision=None,
    cache_dir=None,
    weights="final",
    device=None,
    prefer_ema=True,
):
    """Load a public pretrained AFIN checkpoint by posterior family."""
    family = _canonical_pretrained_family(posterior_family)
    root = download_pretrained_checkpoints(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        families=[family],
    )
    return load_afin(
        root / PRETRAINED_CHECKPOINTS[family],
        weights=weights,
        device=device,
        prefer_ema=prefer_ema,
    )


def load_afin(
    checkpoint_dir=None,
    *,
    repo_id=None,
    revision=None,
    cache_dir=None,
    subfolder=None,
    allow_patterns=None,
    weights="final",
    device=None,
    prefer_ema=True,
    posterior_family=None,
):
    """Load an AFIN checkpoint.

    Pass ``checkpoint_dir`` for a local run directory, or pass ``repo_id`` and
    ``subfolder`` to download a run directory from Hugging Face. During
    development, omitting both falls back to the newest local ``checkpoints/*``
    run, optionally filtered by ``posterior_family``.
    """
    if checkpoint_dir is None:
        if repo_id is not None:
            if allow_patterns is None and posterior_family is not None:
                family = _canonical_pretrained_family(posterior_family)
                allow_patterns = _pretrained_allow_patterns([family])
                subfolder = subfolder or PRETRAINED_CHECKPOINTS[family]
            run_dir = download_checkpoint(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=allow_patterns,
            )
            if subfolder is not None:
                run_dir = run_dir / subfolder
        else:
            try:
                run_dir = find_checkpoint_dir(weights=weights, posterior_family=posterior_family)
            except FileNotFoundError:
                if weights == "best":
                    raise
                weights = "best"
                run_dir = find_checkpoint_dir(weights=weights, posterior_family=posterior_family)
    else:
        run_dir = Path(checkpoint_dir)

    config = json.loads((run_dir / "config.json").read_text())
    args = SimpleNamespace(**config["args"])
    exp_type = ExpType(**config["exp_type"])
    model_kwargs = build_model_kwargs(args)
    model_kwargs["amp_dtype"] = None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = run_dir / f"{weights}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = payload.get("ema_model_state_dict") if prefer_ema else None
    if state_dict is None:
        state_dict = payload["model_state_dict"]

    model = AFIN(exp_type=exp_type, **model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.checkpoint_dir = str(run_dir)
    return model


@dataclass
class GaussianPosterior:
    mean: torch.Tensor
    cov: torch.Tensor
    name: str = "AFIN"
    runtime_seconds: float | None = None
    forward_runtime_seconds: float | None = None
    sampling_runtime_seconds: float | None = None
    total_runtime_seconds: float | None = None
    samples: torch.Tensor | None = None
    posterior_log_prob: torch.Tensor | None = None
    target_log_prob: torch.Tensor | None = None

    @property
    def d(self):
        return int(self.mean.numel())

    @property
    def scale_tril(self):
        return torch.linalg.cholesky(self.cov)

    def sample(self, n=1, seed=None):
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        eps = torch.randn(int(n), self.d, generator=generator)
        return self.mean.reshape(1, -1) + eps @ self.scale_tril.T

    def log_prob(self, z):
        z = torch.as_tensor(z, dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        scale_tril = self.scale_tril
        delta = z - self.mean.reshape(1, -1)
        sol = torch.linalg.solve_triangular(scale_tril, delta.unsqueeze(-1), upper=False).squeeze(-1)
        quad = sol.square().sum(dim=-1)
        log_det = 2.0 * torch.log(torch.diagonal(scale_tril)).sum()
        return -0.5 * (self.d * math.log(2.0 * math.pi) + log_det + quad)


@dataclass
class SamplePosterior:
    samples: torch.Tensor
    name: str = "NUTS"
    runtime_seconds: float | None = None
    forward_runtime_seconds: float | None = None
    sampling_runtime_seconds: float | None = None
    total_runtime_seconds: float | None = None
    posterior_log_prob: torch.Tensor | None = None
    target_log_prob: torch.Tensor | None = None

    @property
    def d(self):
        return int(self.samples.shape[-1])

    @property
    def mean(self):
        return self.samples.mean(dim=0)

    @property
    def cov(self):
        centered = self.samples - self.mean.reshape(1, -1)
        return centered.T @ centered / max(1, int(self.samples.shape[0]) - 1)

    def sample(self, n=None, seed=None):
        if n is None or int(n) >= int(self.samples.shape[0]):
            return self.samples
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(int(self.samples.shape[0]), generator=generator)[: int(n)]
        return self.samples[idx]


@torch.no_grad()
def afin(
    model,
    problem: Problem,
    *,
    num_samples=None,
    seed=None,
    log_prob=False,
    flow_samples=None,
    flow_batch_size=512,
):
    """Run one-shot AFIN posterior inference for a built ``Problem``.

    Flow posteriors are sampled in chunks so demos can request thousands of
    samples without duplicating the full task batch on GPU.

    ``runtime_seconds`` on the returned posterior is the timed AFIN posterior
    forward pass. ``total_runtime_seconds`` includes post-processing and, for
    flow posteriors, sample generation.
    """
    total_start = time.perf_counter()
    sample_requested = num_samples is not None or flow_samples is not None
    if flow_samples is not None and num_samples is None:
        num_samples = flow_samples
    num_samples = 2000 if num_samples is None else int(num_samples)

    with _temporary_torch_seed(seed, model):
        device = next(model.parameters()).device
        task = move_task(problem.task, device)
        if getattr(model, "posterior_family", None) == "gaussian" and hasattr(model, "_gaussian_posterior"):
            _sync_if_cuda(device)
            forward_start = time.perf_counter()
            _, mean_raw, precision_chol = model._gaussian_posterior(task)
            _sync_if_cuda(device)
            forward_seconds = time.perf_counter() - forward_start
            sampling_start = time.perf_counter()
            mean = mean_raw.reshape(-1).detach().cpu().float()
            cov = torch.cholesky_inverse(precision_chol).reshape(problem.d, problem.d).detach().cpu().float()
            posterior = GaussianPosterior(mean=mean, cov=cov, name="AFIN")
            if sample_requested or log_prob:
                samples = posterior.sample(num_samples, seed=seed).detach().cpu().float()
                posterior.samples = samples
            if log_prob:
                posterior.posterior_log_prob = posterior.log_prob(samples).detach().cpu().float()
            _sync_if_cuda(device)
            sampling_seconds = time.perf_counter() - sampling_start
            return _attach_runtime(
                posterior,
                forward_seconds=forward_seconds,
                sampling_seconds=sampling_seconds,
                total_seconds=time.perf_counter() - total_start,
            )

        _sync_if_cuda(device)
        forward_start = time.perf_counter()
        post = model.posterior(task)
        _sync_if_cuda(device)
        forward_seconds = time.perf_counter() - forward_start
        sampling_start = time.perf_counter()
        samples = []
        posterior_log_prob = []
        remaining = int(num_samples)
        batch_size = max(1, int(flow_batch_size))
        while remaining > 0:
            n = min(batch_size, remaining)
            z = post.sample(torch.Size([n])).reshape(-1, problem.d)
            if log_prob:
                log_q = post.log_prob(z)
                posterior_log_prob.append(log_q.reshape(-1).detach().cpu().float())
            samples.append(z.detach().cpu().float())
            remaining -= n
        samples = torch.cat(samples, dim=0)
        log_q = torch.cat(posterior_log_prob, dim=0) if posterior_log_prob else None
        _sync_if_cuda(device)
        sampling_seconds = time.perf_counter() - sampling_start
        return _attach_runtime(
            SamplePosterior(samples=samples, name="AFIN", posterior_log_prob=log_q),
            forward_seconds=forward_seconds,
            sampling_seconds=sampling_seconds,
            total_seconds=time.perf_counter() - total_start,
        )


@contextmanager
def _temporary_torch_seed(seed, model):
    if seed is None:
        yield
        return

    device = next(model.parameters()).device
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


def nuts_reference(problem: Problem, *, num_samples=3000, num_warmup=800, seed=0, platform="auto", progress_bar=False):
    """Draw reference posterior samples using numpyro NUTS."""
    if platform is not None:
        os.environ["NUMPYRO_PLATFORM"] = str(platform)
    task = move_task(problem.task, "cpu")
    samples = sample_nuts_reference(
        task,
        num_samples=int(num_samples),
        num_warmup=int(num_warmup),
        seed=int(seed),
        progress_bar=bool(progress_bar),
    )
    return SamplePosterior(samples=samples.detach().cpu().float(), name="NUTS reference")


def exact_gaussian_regression_posterior(data_or_problem, *, name="Exact posterior"):
    """Closed-form posterior for the demo Gaussian regression model.

    Assumes z ~ N(0, I) and y | z ~ N(X z, sigma^2 I).
    """
    data = data_or_problem.data if isinstance(data_or_problem, Problem) else data_or_problem
    x = torch.as_tensor(data["X"], dtype=torch.double)
    y = torch.as_tensor(data["y"], dtype=torch.double).reshape(-1)
    sigma = float(torch.as_tensor(data["sigma"]).reshape(-1)[0].item())
    d = int(x.shape[1])
    precision = torch.eye(d, dtype=torch.double) + (x.T @ x) / (sigma * sigma)
    rhs = (x.T @ y) / (sigma * sigma)
    mean = torch.linalg.solve(precision, rhs)
    cov = torch.linalg.inv(precision)
    return GaussianPosterior(mean=mean.float(), cov=cov.float(), name=name)
