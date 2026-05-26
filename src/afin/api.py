"""Small public inference API."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .inference import afin as run_afin
from .inference import load_afin, load_pretrained_afin, nuts_reference
from .inference import SamplePosterior
from .spec import Problem, build_problem
from .tasks import move_task, repeat_task, unnormalized_log_posterior


class AFIN:
    """AFIN inference method loaded from a pretrained or local checkpoint."""

    def __init__(
        self,
        posterior_family="flow",
        *,
        family=None,
        model=None,
        checkpoint_dir=None,
        weights="final",
        device=None,
        prefer_ema=True,
        flow_batch_size=512,
    ):
        if family is not None:
            posterior_family = family
        if model is not None:
            self.model = model
        elif checkpoint_dir is not None:
            self.model = load_afin(
                checkpoint_dir,
                weights=weights,
                device=device,
                prefer_ema=prefer_ema,
            )
        else:
            self.model = load_pretrained_afin(
                posterior_family,
                weights=weights,
                device=device,
                prefer_ema=prefer_ema,
            )
        self.flow_batch_size = int(flow_batch_size)
        self.name = f"AFIN {self.model.posterior_family}"

    def infer(self, problem: Problem, **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        if num_samples is not None:
            num_samples = int(num_samples)
        seed = kwargs.pop("seed", None)
        log_prob = bool(kwargs.pop("log_prob", False))
        flow_batch_size = int(kwargs.pop("flow_batch_size", self.flow_batch_size))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown AFIN inference option(s): {unknown}")
        posterior = run_afin(
            self.model,
            problem,
            num_samples=num_samples,
            seed=seed,
            log_prob=log_prob,
            flow_batch_size=flow_batch_size,
        )
        posterior.name = self.name
        return posterior


@dataclass
class NumPyroNUTS:
    """NumPyro NUTS inference method."""

    num_warmup: int = 800
    platform: str = "auto"
    progress_bar: bool = True

    @property
    def name(self):
        return "NumPyro NUTS"

    def infer(self, problem: Problem, **kwargs):
        num_samples = int(kwargs.pop("num_samples", 3000))
        num_warmup = int(kwargs.pop("num_warmup", self.num_warmup))
        seed = int(kwargs.pop("seed", 0))
        platform = kwargs.pop("platform", self.platform)
        progress_bar = bool(kwargs.pop("progress_bar", self.progress_bar))
        kwargs.pop("log_prob", False)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown NumPyroNUTS option(s): {unknown}")
        posterior = nuts_reference(
            problem,
            num_samples=num_samples,
            num_warmup=num_warmup,
            seed=seed,
            platform=platform,
            progress_bar=progress_bar,
        )
        posterior.name = self.name
        return posterior


@dataclass
class PyroNUTS:
    """Pyro NUTS inference method."""

    num_warmup: int = 800
    progress_bar: bool = True

    @property
    def name(self):
        return "Pyro NUTS"

    def infer(self, problem: Problem, **kwargs):
        num_samples = int(kwargs.pop("num_samples", 3000))
        num_warmup = int(kwargs.pop("num_warmup", self.num_warmup))
        seed = int(kwargs.pop("seed", 0))
        progress_bar = bool(kwargs.pop("progress_bar", self.progress_bar))
        kwargs.pop("log_prob", False)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown PyroNUTS option(s): {unknown}")
        samples = _sample_pyro_nuts(
            problem,
            num_samples=num_samples,
            num_warmup=num_warmup,
            seed=seed,
            progress_bar=progress_bar,
        )
        return SamplePosterior(samples=samples, name=self.name)


def infer(
    method,
    prior=None,
    observed=None,
    *,
    num_samples=None,
    seed=None,
    log_prob=False,
    verbose=True,
    problem=None,
    likelihoods=None,
    observations=None,
    d=None,
    device="cpu",
    **method_options,
):
    """Run ``method`` on a built problem or on a prior plus observed factors."""
    _reject_public_metadata(method_options)
    if num_samples is not None:
        method_options["num_samples"] = num_samples
    if seed is not None:
        method_options["seed"] = seed
    if log_prob:
        method_options["log_prob"] = True
    label = _method_label(method)
    if verbose:
        print(f"Starting {label}...", flush=True)
    start = time.perf_counter()

    try:
        if problem is None:
            if isinstance(prior, Problem) and observed is None:
                problem = prior
            else:
                problem = build_problem(
                    prior,
                    observed,
                    likelihoods=likelihoods,
                    observations=observations,
                    d=d,
                    device=device,
                )
        if hasattr(method, "infer"):
            posterior = method.infer(problem, **method_options)
        elif getattr(method, "posterior_family", None) is not None:
            posterior = run_afin(method, problem, **method_options)
        else:
            raise TypeError("method must be an AFIN, NumPyroNUTS, PyroNUTS, or loaded AFIN model.")
    except Exception:
        if verbose:
            elapsed = time.perf_counter() - start
            print(f"Failed {label} after {_format_seconds(elapsed)}.", flush=True)
        raise

    elapsed = time.perf_counter() - start
    reported_elapsed = getattr(posterior, "runtime_seconds", None)
    if reported_elapsed is None:
        reported_elapsed = elapsed
    if verbose:
        print(f"Finished {label} in {_format_seconds(reported_elapsed)}.", flush=True)
    return _attach_problem(
        posterior,
        problem,
        reported_elapsed,
        total_runtime_seconds=elapsed,
        log_prob=bool(log_prob),
        num_samples=num_samples,
        seed=seed,
    )


def _attach_problem(
    posterior,
    problem,
    runtime_seconds=None,
    *,
    total_runtime_seconds=None,
    log_prob=False,
    num_samples=None,
    seed=None,
):
    posterior.problem = problem
    posterior.target = problem
    if runtime_seconds is not None:
        posterior.runtime_seconds = float(runtime_seconds)
    if total_runtime_seconds is not None:
        posterior.total_runtime_seconds = float(total_runtime_seconds)
    if log_prob:
        _attach_target_log_prob(posterior, problem, num_samples=num_samples, seed=seed)
    return posterior


def _reject_public_metadata(kwargs):
    blocked = sorted({"name", "true_z", "data"} & set(kwargs))
    if blocked:
        names = ", ".join(f"{key}=" for key in blocked)
        raise TypeError(f"infer(...) no longer accepts {names}; keep labels and raw data outside the inference call.")


def _attach_target_log_prob(posterior, problem, *, num_samples=None, seed=None):
    samples = getattr(posterior, "samples", None)
    if samples is None:
        n = 2000 if num_samples is None else int(num_samples)
        samples = posterior.sample(n, seed=seed).detach().cpu().float()
        posterior.samples = samples

    samples = torch.as_tensor(samples, dtype=torch.float32).reshape(-1, problem.d)
    values = []
    task = move_task(problem.task, "cpu")
    for start in range(0, samples.shape[0], 4096):
        z = samples[start : start + 4096]
        task_batch = repeat_task(task, z.shape[0])
        values.append(unnormalized_log_posterior(task_batch, z).detach().cpu().float())
    posterior.target_log_prob = torch.cat(values, dim=0)

    q_log_prob = getattr(posterior, "posterior_log_prob", None)
    if q_log_prob is None and callable(getattr(posterior, "log_prob", None)):
        posterior.posterior_log_prob = posterior.log_prob(samples).detach().cpu().float()


def _method_label(method):
    name = getattr(method, "name", None)
    if name:
        return str(name)
    family = getattr(method, "posterior_family", None)
    if family is not None:
        return f"AFIN {family}"
    return method.__class__.__name__


def _format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 0.01:
        return f"{seconds:.4f} seconds"
    if seconds < 10:
        return f"{seconds:.2f} seconds"
    return f"{seconds:.1f} seconds"


def _sample_pyro_nuts(problem, *, num_samples, num_warmup, seed, progress_bar):
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS

    pyro.set_rng_seed(int(seed))
    task = move_task(problem.task, "cpu")
    d = int(task.d)
    zeros = torch.zeros(d)
    ones = torch.ones(d)
    base = dist.Normal(zeros, ones).to_event(1)

    def model():
        z = pyro.sample("z", base)
        task_z = repeat_task(task, 1)
        log_p = unnormalized_log_posterior(task_z, z.reshape(1, -1))[0]
        pyro.factor("target", log_p - base.log_prob(z))

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_samples=int(num_samples),
        warmup_steps=int(num_warmup),
        initial_params={"z": task.meta["prior_loc"].reshape(-1).detach()},
        disable_progbar=not bool(progress_bar),
    )
    mcmc.run()
    return mcmc.get_samples()["z"].detach().cpu().float()
