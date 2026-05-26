"""Bayesian task specification for AFIN.

The preferred public API is a direct factor list:

    prior = GaussianPrior(0, 1)
    observed = [
        LinearGaussian(design_matrix=X, sigma=0.35).observe(y),
        BernoulliLogit(design_matrix=X_binary).observe(y_binary),
    ]
    problem = build_problem(prior, observed)

The older tiny-PPL ``Model`` builder remains available for compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .tasks import LIKELIHOOD_FAMILY_TO_ID, PRIOR_FAMILY_TO_ID, Task, move_task


@dataclass(frozen=True)
class Latent:
    name: str
    d: int

    def __rmatmul__(self, x):
        return Linear(self, _matrix(x, self.d, "X"))


@dataclass(frozen=True)
class Linear:
    latent: Latent
    x: torch.Tensor


class Normal:
    """Normal distribution.

    Prior: ``Normal(0, 1)``
    Likelihood: ``Normal(X @ z, sigma=0.35)`` or ``Normal(z, sigma=0.35)``
    """

    def __init__(self, mean, scale=None, *, sigma=None):
        if scale is None and sigma is None:
            raise ValueError("Normal needs scale= or sigma=")
        self.mean = mean
        self.scale = float(sigma) if sigma is not None else scale


class FullRankNormal:
    """Full-rank Gaussian prior, parameterized by a precision matrix."""

    def __init__(self, mean, precision):
        self.mean = mean
        self.precision = precision


class StudentT:
    """Student-t distribution for priors or linear likelihoods."""

    def __init__(self, mean, scale=None, df=4.0, *, sigma=None):
        if scale is None and sigma is None:
            raise ValueError("StudentT needs scale= or sigma=")
        self.mean = mean
        self.scale = float(sigma) if sigma is not None else scale
        self.df = float(df)


class Laplace:
    """Diagonal Laplace prior."""

    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale


@dataclass(frozen=True)
class Observed:
    """A likelihood factor paired with its observed values."""

    likelihood: Any
    value: Any


class _Likelihood:
    def observe(self, value):
        return Observed(self, value)


@dataclass(frozen=True)
class GaussianPrior:
    """Diagonal Gaussian prior over the latent vector."""

    loc: Any = 0.0
    scale: Any = 1.0


@dataclass(frozen=True)
class FullRankGaussianPrior:
    """Full-rank Gaussian prior, parameterized by a precision matrix."""

    loc: Any
    precision: Any


@dataclass(frozen=True)
class StudentTPrior:
    """Diagonal Student-t prior over the latent vector."""

    loc: Any = 0.0
    scale: Any = 1.0
    df: float = 4.0


@dataclass(frozen=True)
class LaplacePrior:
    """Diagonal Laplace prior over the latent vector."""

    loc: Any = 0.0
    scale: Any = 1.0


@dataclass(frozen=True)
class LinearGaussian(_Likelihood):
    """Linear Gaussian likelihood: ``y ~ Normal(X z, sigma)``."""

    design_matrix: Any
    sigma: Any

    def _to_observation(self, y, d):
        x = _matrix(self.design_matrix, d, "design_matrix")
        return _Observation(
            family="gaussian",
            y=_site_vector(y, x.shape[0], "observation"),
            x=x,
            scale=_site_vector(self.sigma, x.shape[0], "sigma"),
        )


@dataclass(frozen=True)
class DirectGaussian(_Likelihood):
    """Direct Gaussian likelihood: ``y ~ Normal(z, sigma)``."""

    sigma: Any

    def _to_observation(self, y, d):
        latent = Latent("z", d)
        return _distribution_to_observation(Normal(latent, sigma=self.sigma), y, latent)


@dataclass(frozen=True)
class LinearStudentT(_Likelihood):
    """Linear Student-t likelihood: ``y ~ StudentT(df, X z, sigma)``."""

    design_matrix: Any
    sigma: Any
    df: Any = 4.0

    def _to_observation(self, y, d):
        x = _matrix(self.design_matrix, d, "design_matrix")
        return _Observation(
            family="student_t",
            y=_site_vector(y, x.shape[0], "observation"),
            x=x,
            scale=_site_vector(self.sigma, x.shape[0], "sigma"),
            df=_site_vector(self.df, x.shape[0], "df"),
        )


class BernoulliLogit(_Likelihood):
    """Bernoulli likelihood with logits ``X @ z``."""

    def __init__(self, logits=None, *, design_matrix=None):
        if design_matrix is None and logits is not None and not _is_expression(logits):
            design_matrix, logits = logits, None
        self.logits = logits
        self.design_matrix = design_matrix

    def _to_observation(self, y, d):
        x = _matrix(self.design_matrix, d, "design_matrix")
        return _Observation(
            family="bernoulli_logit",
            y=_site_vector(y, x.shape[0], "observation"),
            x=x,
        )


class BinomialLogit(_Likelihood):
    """Binomial likelihood with logits ``X @ z``."""

    def __init__(self, logits=None, total_count=None, *, design_matrix=None):
        if design_matrix is None and logits is not None and not _is_expression(logits):
            design_matrix, logits = logits, None
        self.logits = logits
        self.design_matrix = design_matrix
        self.total_count = total_count

    def _to_observation(self, y, d):
        x = _matrix(self.design_matrix, d, "design_matrix")
        return _Observation(
            family="binomial_logit",
            y=_site_vector(y, x.shape[0], "observation"),
            x=x,
            total_count=_site_vector(self.total_count, x.shape[0], "total_count"),
        )


@dataclass
class _Observation:
    family: str
    y: torch.Tensor
    x: torch.Tensor
    scale: torch.Tensor | None = None
    df: torch.Tensor | None = None
    total_count: torch.Tensor | None = None
    y_vector: torch.Tensor | None = None

    @property
    def n(self):
        return int(self.y.shape[0])


@dataclass
class Problem:
    """A built Bayesian task ready for AFIN or a reference sampler."""

    task: Task
    name: str = ""
    true_z: torch.Tensor | None = None
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def d(self):
        return int(self.task.d)

    @property
    def N(self):
        return int(self.task.N)

    def to(self, device):
        return Problem(
            task=move_task(self.task, device),
            name=self.name,
            true_z=None if self.true_z is None else self.true_z.to(device),
            data={k: v.to(device) if torch.is_tensor(v) else v for k, v in self.data.items()},
        )

    def __repr__(self):
        inv = {v: k for k, v in LIKELIHOOD_FAMILY_TO_ID.items()}
        families = []
        ids = self.task.site_family_ids.reshape(-1).tolist()
        for family_id in ids:
            family = inv[int(family_id)]
            if not families or families[-1][0] != family:
                families.append([family, 0])
            families[-1][1] += 1
        pieces = ", ".join(f"{name} x{count}" for name, count in families) or "no observations"
        return f"Problem(name={self.name!r}, d={self.d}, N={self.N}, factors={pieces})"


class Model:
    """Tiny Bayesian model builder for one latent vector ``z``."""

    def __init__(self, d: int, name: str = ""):
        self.d = int(d)
        self.name = name
        self._latent: Latent | None = None
        self._prior: Any | None = None
        self._observations: list[_Observation] = []

    def latent(self, name: str, distribution):
        if self._latent is not None:
            raise ValueError("The public AFIN model API currently supports one latent vector.")
        if _is_expression(getattr(distribution, "mean", None)):
            raise ValueError("Prior mean must be numeric, not a latent expression.")
        self._latent = Latent(str(name), self.d)
        self._prior = distribution
        return self._latent

    def observe(self, distribution, value=None, *, y=None):
        if self._latent is None:
            raise ValueError("Call latent(...) before observe(...).")
        if y is None:
            y = value
        if y is None:
            raise ValueError("observe(...) needs an observed value.")
        self._observations.append(_distribution_to_observation(distribution, y, self._latent))
        return self

    def build(self, *, true_z=None, data=None, device="cpu"):
        if self._latent is None or self._prior is None:
            raise ValueError("Call latent(...) before build().")
        task = _build_task(
            d=self.d,
            prior=self._prior,
            observations=self._observations,
            device=torch.device(device),
        )
        return Problem(
            task=task,
            name=self.name,
            true_z=None if true_z is None else _vector(true_z, self.d, "true_z").cpu(),
            data={} if data is None else {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in data.items()
            },
        )


BayesianModel = Model


def build_problem(
    prior,
    observed=None,
    *,
    likelihoods=None,
    observations=None,
    d=None,
    name: str = "",
    true_z=None,
    data=None,
    device="cpu",
):
    """Build a :class:`Problem` from a prior and observed likelihood factors."""
    observed = _normalize_observed(observed, likelihoods, observations)
    d = _infer_d(d, prior, observed)
    task = _build_task(
        d=d,
        prior=_prior_distribution(prior),
        observations=[_likelihood_observation(item.likelihood, item.value, d) for item in observed],
        device=torch.device(device),
    )
    return Problem(
        task=task,
        name=name,
        true_z=None if true_z is None else _vector(true_z, d, "true_z").cpu(),
        data={} if data is None else {
            k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in data.items()
        },
    )


def _is_expression(x):
    return isinstance(x, (Latent, Linear))


def _vector(x, d, name):
    t = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    if t.numel() == 1 and int(d) > 1:
        t = t.expand(int(d)).clone()
    if t.numel() != int(d):
        raise ValueError(f"{name} has {t.numel()} elements, expected {d}")
    return t


def _site_vector(x, n, name):
    t = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    if t.numel() == 1 and int(n) > 1:
        t = t.expand(int(n)).clone()
    if t.numel() != int(n):
        raise ValueError(f"{name} has {t.numel()} elements, expected {n}")
    return t


def _matrix(x, d, name):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    if t.ndim != 2 or t.shape[1] != int(d):
        raise ValueError(f"{name} has shape {tuple(t.shape)}, expected (N, {d}) or ({d},)")
    return t


def _square_matrix(x, d, name):
    t = torch.as_tensor(x, dtype=torch.float32)
    if tuple(t.shape) != (int(d), int(d)):
        raise ValueError(f"{name} has shape {tuple(t.shape)}, expected ({d}, {d})")
    return t


def _normalize_observed(observed, likelihoods, observations):
    if observed is not None:
        if likelihoods is not None or observations is not None:
            raise ValueError("Pass either observed= or likelihoods= with observations=, not both.")
        if isinstance(observed, Observed):
            return [observed]
        return list(observed)
    if likelihoods is None and observations is None:
        return []
    if likelihoods is None or observations is None:
        raise ValueError("likelihoods= and observations= must be passed together.")
    likelihoods, observations = list(likelihoods), list(observations)
    if len(likelihoods) != len(observations):
        raise ValueError("likelihoods and observations must have the same length.")
    return [Observed(likelihood, value) for likelihood, value in zip(likelihoods, observations)]


def _prior_distribution(prior):
    if isinstance(prior, GaussianPrior):
        return Normal(prior.loc, prior.scale)
    if isinstance(prior, FullRankGaussianPrior):
        return FullRankNormal(prior.loc, prior.precision)
    if isinstance(prior, StudentTPrior):
        return StudentT(prior.loc, prior.scale, df=prior.df)
    if isinstance(prior, LaplacePrior):
        return Laplace(prior.loc, prior.scale)
    return prior


def _likelihood_observation(likelihood, value, d):
    to_observation = getattr(likelihood, "_to_observation", None)
    has_design = not hasattr(likelihood, "design_matrix") or likelihood.design_matrix is not None
    if to_observation is not None and has_design:
        return to_observation(value, d)
    return _distribution_to_observation(likelihood, value, Latent("z", d))


def _tensor_dim(x):
    if x is None:
        return None
    t = torch.as_tensor(x)
    if t.ndim == 0 or t.numel() == 1:
        return None
    if t.ndim == 1:
        return int(t.numel())
    if t.ndim == 2 and t.shape[0] == t.shape[1]:
        return int(t.shape[0])
    return None


def _design_dim(likelihood):
    design = getattr(likelihood, "design_matrix", None)
    if design is None:
        return None
    t = torch.as_tensor(design)
    if t.ndim == 1:
        return int(t.numel())
    if t.ndim == 2:
        return int(t.shape[1])
    raise ValueError(f"design_matrix has shape {tuple(t.shape)}, expected (N, d) or (d,)")


def _observed_direct_dim(observed):
    dims = []
    for item in observed:
        if isinstance(item.likelihood, DirectGaussian):
            y = torch.as_tensor(item.value)
            if y.ndim == 1:
                dims.append(int(y.numel()))
            elif y.ndim == 2:
                dims.append(int(y.shape[1]))
    return dims


def _infer_d(d, prior, observed):
    dims = []
    if d is not None:
        dims.append(int(d))

    base_prior = _prior_distribution(prior)
    if isinstance(base_prior, FullRankNormal):
        precision = torch.as_tensor(base_prior.precision)
        dims.append(_square_matrix(precision, int(precision.shape[0]), "prior.precision").shape[0])
    else:
        for value in (getattr(base_prior, "mean", None), getattr(base_prior, "scale", None)):
            prior_dim = _tensor_dim(value)
            if prior_dim is not None:
                dims.append(prior_dim)

    dims.extend(dim for dim in (_design_dim(item.likelihood) for item in observed) if dim is not None)
    dims.extend(_observed_direct_dim(observed))
    dims = [int(dim) for dim in dims if dim is not None]
    if not dims:
        raise ValueError("Could not infer latent dimension d; pass d= explicitly.")
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent latent dimensions: {dims}")
    return dims[0]


def _distribution_to_observation(dist, y, latent):
    if isinstance(dist, Normal):
        if isinstance(dist.mean, Linear):
            x = dist.mean.x
            return _Observation(
                family="gaussian",
                y=_site_vector(y, x.shape[0], "y"),
                x=x,
                scale=_site_vector(dist.scale, x.shape[0], "sigma"),
            )
        if isinstance(dist.mean, Latent):
            y_vector = torch.as_tensor(y, dtype=torch.float32)
            if y_vector.ndim == 1:
                y_vector = y_vector.reshape(1, -1)
            if y_vector.ndim != 2 or y_vector.shape[1] != latent.d:
                raise ValueError(f"Normal(z, sigma) observations must have shape (N, {latent.d}) or ({latent.d},)")
            n = y_vector.shape[0]
            return _Observation(
                family="gaussian_no_x",
                y=y_vector.mean(dim=-1),
                x=torch.zeros(n, latent.d),
                scale=_site_vector(dist.scale, n, "sigma"),
                y_vector=y_vector,
            )
        raise ValueError("Normal likelihood mean must be z or X @ z.")

    if isinstance(dist, StudentT):
        if not isinstance(dist.mean, Linear):
            raise ValueError("StudentT likelihood mean must be X @ z.")
        x = dist.mean.x
        return _Observation(
            family="student_t",
            y=_site_vector(y, x.shape[0], "y"),
            x=x,
            scale=_site_vector(dist.scale, x.shape[0], "sigma"),
            df=_site_vector(dist.df, x.shape[0], "df"),
        )

    if isinstance(dist, BernoulliLogit):
        if not isinstance(dist.logits, Linear):
            raise ValueError("BernoulliLogit logits must be X @ z.")
        x = dist.logits.x
        return _Observation(
            family="bernoulli_logit",
            y=_site_vector(y, x.shape[0], "y"),
            x=x,
        )

    if isinstance(dist, BinomialLogit):
        if not isinstance(dist.logits, Linear):
            raise ValueError("BinomialLogit logits must be X @ z.")
        x = dist.logits.x
        return _Observation(
            family="binomial_logit",
            y=_site_vector(y, x.shape[0], "y"),
            x=x,
            total_count=_site_vector(dist.total_count, x.shape[0], "total_count"),
        )

    raise TypeError(f"Unsupported likelihood distribution: {type(dist).__name__}")


def _prior_meta(prior, d):
    prior = _prior_distribution(prior)
    meta = {
        "prior_loc": torch.zeros(1, d),
        "prior_scale": torch.zeros(1, d),
        "prior_precision": torch.zeros(1, d, d),
        "prior_df": torch.zeros(1, 1),
    }
    if isinstance(prior, Normal):
        family = "diag_gaussian"
        meta["prior_loc"][0] = _vector(prior.mean, d, "prior.mean")
        meta["prior_scale"][0] = _vector(prior.scale, d, "prior.scale")
    elif isinstance(prior, FullRankNormal):
        family = "fullrank_gaussian"
        meta["prior_loc"][0] = _vector(prior.mean, d, "prior.mean")
        meta["prior_precision"][0] = _square_matrix(prior.precision, d, "prior.precision")
    elif isinstance(prior, StudentT):
        family = "diag_student_t"
        meta["prior_loc"][0] = _vector(prior.mean, d, "prior.mean")
        meta["prior_scale"][0] = _vector(prior.scale, d, "prior.scale")
        meta["prior_df"][0, 0] = float(prior.df)
    elif isinstance(prior, Laplace):
        family = "diag_laplace"
        meta["prior_loc"][0] = _vector(prior.mean, d, "prior.mean")
        meta["prior_scale"][0] = _vector(prior.scale, d, "prior.scale")
    else:
        raise TypeError(f"Unsupported prior distribution: {type(prior).__name__}")
    return family, meta


def _build_task(d, prior, observations, device):
    B = 1
    N = sum(obs.n for obs in observations)
    prior_family, meta = _prior_meta(prior, d)
    prior_family_id = PRIOR_FAMILY_TO_ID[prior_family]

    x = torch.zeros(B, N, d)
    y = torch.zeros(B, N)
    site_ids = torch.empty(B, N, dtype=torch.long)
    like_meta = {
        "likelihood_scale": torch.zeros(B, N),
        "likelihood_y_vector": torch.zeros(B, N, d),
        "likelihood_total_count": torch.zeros(B, N),
        "likelihood_df": torch.zeros(B, N),
    }

    offset = 0
    for obs in observations:
        sl = slice(offset, offset + obs.n)
        family_id = LIKELIHOOD_FAMILY_TO_ID[obs.family]
        x[0, sl] = obs.x
        y[0, sl] = obs.y
        site_ids[0, sl] = family_id
        if obs.scale is not None:
            like_meta["likelihood_scale"][0, sl] = obs.scale
        if obs.df is not None:
            like_meta["likelihood_df"][0, sl] = obs.df
        if obs.total_count is not None:
            like_meta["likelihood_total_count"][0, sl] = obs.total_count
        if obs.y_vector is not None:
            like_meta["likelihood_y_vector"][0, sl] = obs.y_vector
        offset += obs.n

    unique_families = sorted({obs.family for obs in observations})
    if len(unique_families) == 1:
        likelihood_family = unique_families[0]
        likelihood_family_id = LIKELIHOOD_FAMILY_TO_ID[likelihood_family]
    elif unique_families:
        likelihood_family = "heterogeneous"
        likelihood_family_id = -1
    else:
        likelihood_family = "none"
        likelihood_family_id = -1

    meta.update(like_meta)
    meta.update({
        "likelihood_num_families": torch.tensor([float(max(1, len(unique_families)))]),
        "likelihood_is_heterogeneous": torch.tensor([float(len(unique_families) > 1)]),
        "x_family_id": torch.zeros(B, dtype=torch.long),
        "x_condition_number": torch.ones(B),
        "x_effective_rank": torch.full((B,), float(d)),
    })

    task = Task(
        prior_family=prior_family,
        likelihood_family=likelihood_family,
        prior_family_id=prior_family_id,
        likelihood_family_id=likelihood_family_id,
        d=int(d),
        N=int(N),
        z0=torch.zeros(B, d),
        X=x,
        y=y,
        meta=meta,
        prior_family_ids=torch.tensor([prior_family_id], dtype=torch.long),
        site_family_ids=site_ids,
    )
    return move_task(task, device)
