"""Proposal-sampling helpers used by demos and notebooks."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .inference import SamplePosterior
from .plotting import sliced_w2
from .spec import Problem
from .tasks import move_task, repeat_task, unnormalized_log_posterior


@dataclass
class SNISResult:
    samples: torch.Tensor
    log_weights: torch.Tensor
    weights: torch.Tensor
    ess: float
    posterior: SamplePosterior
    runtime_seconds: float | None = None

    @property
    def ess_ratio(self):
        return float(self.ess / max(1, int(self.samples.shape[0])))


def log_target(problem: Problem, z):
    """Evaluate the unnormalized posterior log density at ``z``."""
    z = torch.as_tensor(z, dtype=torch.float32)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    task = repeat_task(move_task(problem.task, z.device), z.shape[0])
    return unnormalized_log_posterior(task, z).detach().cpu().reshape(-1)


@torch.no_grad()
def afin_proposal_samples(model, problem: Problem, n=4000, *, seed=None, chunk_size=512):
    """Draw samples from an AFIN posterior and compute log p - log q."""
    model = getattr(model, "model", model)
    if seed is not None:
        torch.manual_seed(int(seed))

    device = next(model.parameters()).device
    task = move_task(problem.task, device)
    post = model.posterior(task)
    samples = []
    log_p = []
    log_q = []
    for start in range(0, int(n), int(chunk_size)):
        batch_size = min(int(chunk_size), int(n) - start)
        z = post.sample(torch.Size([batch_size])).reshape(-1, problem.d)
        samples.append(z.detach().cpu().float())
        log_q.append(post.log_prob(z).detach().cpu().reshape(-1).double())
        task_batch = repeat_task(task, batch_size)
        log_p.append(unnormalized_log_posterior(task_batch, z).detach().cpu().reshape(-1).double())

    samples = torch.cat(samples, dim=0)
    log_p = torch.cat(log_p, dim=0)
    log_q = torch.cat(log_q, dim=0)
    return samples, log_p, log_q


def snis_from_afin(model, problem: Problem, n=4000, *, seed=0, chunk_size=512, name="AFIN + SNIS", verbose=True):
    """Self-normalized importance sampling using AFIN as the proposal."""
    if verbose:
        print("Starting AFIN + SNIS...", flush=True)
    start = time.perf_counter()
    samples, log_p, log_q = afin_proposal_samples(
        model,
        problem,
        n=n,
        seed=seed,
        chunk_size=chunk_size,
    )
    log_weights = log_p - log_q
    weights = torch.softmax(log_weights, dim=0).float()
    ess = float((1.0 / weights.square().sum()).item())

    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 10_000)
    idx = torch.multinomial(weights, int(n), replacement=True, generator=generator)
    posterior = SamplePosterior(samples=samples[idx], name=name)
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"Finished AFIN + SNIS in {_format_seconds(elapsed)}.", flush=True)
    return SNISResult(
        samples=samples,
        log_weights=log_weights.float(),
        weights=weights,
        ess=ess,
        posterior=posterior,
        runtime_seconds=elapsed,
    )


def _format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 0.01:
        return f"{seconds:.4f} seconds"
    if seconds < 10:
        return f"{seconds:.2f} seconds"
    return f"{seconds:.1f} seconds"


def run_random_walk_mh(problem: Problem, *, n_steps=1200, proposal_scale=0.115, seed=97):
    """A simple local random-walk Metropolis-Hastings baseline."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    current = torch.randn(problem.d, generator=generator)
    current_logp = log_target(problem, current)[0]

    states = []
    accepted = []
    for _ in range(int(n_steps)):
        proposal = current + float(proposal_scale) * torch.randn(problem.d, generator=generator)
        proposal_logp = log_target(problem, proposal)[0]
        log_alpha = proposal_logp - current_logp
        accept = torch.log(torch.rand((), generator=generator)) < torch.minimum(log_alpha, log_alpha.new_tensor(0.0))
        if bool(accept):
            current = proposal
            current_logp = proposal_logp
        states.append(current.clone())
        accepted.append(float(accept))
    return torch.stack(states), np.asarray(accepted, dtype=float)


def run_afin_imh(model, problem: Problem, *, n_steps=1200, seed=19, chunk_size=512):
    """Independence MH using the AFIN flow posterior as proposal."""
    proposals, log_p, log_q = afin_proposal_samples(
        model,
        problem,
        n=int(n_steps) + 1,
        seed=seed,
        chunk_size=chunk_size,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 123)

    current = proposals[0]
    current_score = log_p[0] - log_q[0]
    states = []
    accepted = []
    for step in range(1, int(n_steps) + 1):
        proposal = proposals[step]
        proposal_score = log_p[step] - log_q[step]
        log_alpha = proposal_score - current_score
        accept = torch.log(torch.rand((), generator=generator)) < torch.minimum(log_alpha, log_alpha.new_tensor(0.0))
        if bool(accept):
            current = proposal
            current_score = proposal_score
        states.append(current.clone())
        accepted.append(float(accept))
    return torch.stack(states), np.asarray(accepted, dtype=float)


def run_parallel_afin_imh(
    model,
    problem: Problem,
    *,
    num_chains=4000,
    num_steps=2,
    seed=19,
    chunk_size=512,
):
    """Many short independence-MH chains initialized from the AFIN proposal."""
    num_chains = int(num_chains)
    num_steps = int(num_steps)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 123)

    current, log_p, log_q = afin_proposal_samples(
        model,
        problem,
        n=num_chains,
        seed=seed,
        chunk_size=chunk_size,
    )
    current_score = (log_p - log_q).float()

    total_accepts = 0
    for step in range(num_steps):
        proposal, proposal_log_p, proposal_log_q = afin_proposal_samples(
            model,
            problem,
            n=num_chains,
            seed=int(seed) + 1000 + step,
            chunk_size=chunk_size,
        )
        proposal_score = (proposal_log_p - proposal_log_q).float()
        log_alpha = proposal_score - current_score
        accept = torch.log(torch.rand(num_chains, generator=generator)) < torch.minimum(
            log_alpha,
            log_alpha.new_tensor(0.0),
        )
        current[accept] = proposal[accept]
        current_score[accept] = proposal_score[accept]
        total_accepts += int(accept.sum().item())

    accept_rate = total_accepts / max(1, num_chains * num_steps)
    return current, accept_rate


def reference_region_fraction(samples, reference, *, level=5.991):
    """Fraction of samples inside the reference Gaussian 95% ellipse proxy."""
    mean = reference.mean.double()
    cov = reference.cov.double()
    precision = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.shape[0], dtype=torch.double))
    delta = samples.double() - mean.reshape(1, -1)
    mahal = (delta @ precision * delta).sum(dim=-1)
    return float((mahal <= float(level)).float().mean().item())


def chain_metric_trace(states, accepted, reference, steps, *, ref_samples=None, num_projections=96, seed=23):
    """Metrics for prefixes of a Markov chain against a reference posterior."""
    if ref_samples is None:
        ref_samples = reference.sample(max(int(max(steps)), 1), seed=123)
    records = []
    for step in steps:
        step = int(step)
        samples = states[:step]
        n = min(step, int(ref_samples.shape[0]))
        records.append({
            "step": step,
            "mean_error": float(torch.linalg.norm(samples.mean(dim=0) - reference.mean).item()),
            "sw2": sliced_w2(samples, ref_samples[:n], num_projections=num_projections, seed=seed),
            "in_region": reference_region_fraction(samples, reference),
            "accept_rate": float(accepted[:step].mean()) if step > 0 else 0.0,
        })
    return records
