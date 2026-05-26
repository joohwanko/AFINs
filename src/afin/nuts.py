"""NUTS reference posterior sampling via numpyro/jax."""
import math
import os

import torch

from .tasks import (
    LIKELIHOOD_FAMILIES,
    LIKELIHOOD_FAMILY_TO_ID,
    dict_to_task,
    task_to_dict,
    tree_map,
)


def _task_to_jax_payload(task):
    return tree_map(task_to_dict(task), lambda x: x.numpy())


def _log_posterior_jax(payload, z):
    import jax.nn as jnn
    import jax.numpy as jnp

    prior_family = payload["prior_family"]
    likelihood_family = payload["likelihood_family"]

    if prior_family == "diag_gaussian":
        loc = payload["meta"]["prior_loc"][0]
        scale = payload["meta"]["prior_scale"][0]
        residual = (z - loc) / scale
        prior = -0.5 * jnp.sum(residual**2)
    elif prior_family == "fullrank_gaussian":
        loc = payload["meta"]["prior_loc"][0]
        precision = payload["meta"]["prior_precision"][0]
        delta = z - loc
        prior = -0.5 * delta @ precision @ delta
    elif prior_family == "diag_student_t":
        loc = payload["meta"]["prior_loc"][0]
        scale = payload["meta"]["prior_scale"][0]
        df = payload["meta"]["prior_df"][0]
        residual = z - loc
        prior = -0.5 * jnp.sum((df + 1.0) * jnp.log1p((residual**2) / (df * scale**2)))
    elif prior_family == "diag_laplace":
        loc = payload["meta"]["prior_loc"][0]
        scale = payload["meta"]["prior_scale"][0]
        prior = -jnp.sum(jnp.abs(z - loc) / scale)
    else:
        raise ValueError(f"Unknown prior family: {prior_family}")

    X = payload["X"][0]
    y = payload["y"][0]
    y_vector = payload["meta"].get("likelihood_y_vector")
    y_vector = y_vector[0] if y_vector is not None else jnp.zeros_like(X)
    t = X @ z
    site_family_ids = payload.get("site_family_ids", None)

    if site_family_ids is not None:
        site_family_ids = site_family_ids[0]
        scale = payload["meta"]["likelihood_scale"][0]
        total_count = payload["meta"]["likelihood_total_count"][0]
        df = payload["meta"]["likelihood_df"][0]
        safe_scale = jnp.clip(scale, a_min=1e-8)
        safe_df = jnp.clip(df, a_min=1e-8)
        like_terms = jnp.zeros_like(t)
        for family_name in LIKELIHOOD_FAMILIES:
            mask = site_family_ids == LIKELIHOOD_FAMILY_TO_ID[family_name]
            if family_name == "gaussian":
                terms = -0.5 * ((y - t) / safe_scale) ** 2
            elif family_name == "gaussian_no_x":
                residual = (y_vector - z[None, :]) / safe_scale[:, None]
                terms = -0.5 * jnp.sum(residual**2, axis=-1)
            elif family_name == "bernoulli_logit":
                terms = y * t - jnn.softplus(t)
            elif family_name == "binomial_logit":
                terms = y * t - total_count * jnn.softplus(t)
            elif family_name == "student_t":
                residual = y - t
                terms = -0.5 * (safe_df + 1.0) * jnp.log1p((residual**2) / (safe_df * safe_scale**2))
            else:
                raise ValueError(f"Unknown likelihood family: {family_name}")
            like_terms = jnp.where(mask, terms, like_terms)
        like = jnp.sum(like_terms)
    elif likelihood_family == "gaussian":
        scale = payload["meta"]["likelihood_scale"][0]
        like = -0.5 * jnp.sum(((y - t) / scale) ** 2)
    elif likelihood_family == "gaussian_no_x":
        scale = payload["meta"]["likelihood_scale"][0]
        like = -0.5 * jnp.sum(((y_vector - z[None, :]) / scale[:, None]) ** 2)
    elif likelihood_family == "bernoulli_logit":
        like = jnp.sum(y * t - jnn.softplus(t))
    elif likelihood_family == "binomial_logit":
        total_count = payload["meta"]["likelihood_total_count"][0]
        like = jnp.sum(y * t - total_count * jnn.softplus(t))
    elif likelihood_family == "student_t":
        scale = payload["meta"]["likelihood_scale"][0]
        df = payload["meta"]["likelihood_df"][0]
        residual = y - t
        like = -0.5 * jnp.sum((df + 1.0) * jnp.log1p((residual**2) / (df * scale**2)))
    else:
        raise ValueError(f"Unknown likelihood family: {likelihood_family}")
    return prior + like


def sample_nuts_reference(task, num_samples, num_warmup, seed, progress_bar=False):
    """Run NUTS to draw `num_samples` posterior samples for `task`."""
    nuts_platform = os.environ.get("NUMPYRO_PLATFORM", "cuda")
    num_chains = max(1, int(os.environ.get("NUMPYRO_NUM_CHAINS", "1")))
    if nuts_platform != "auto":
        os.environ["JAX_PLATFORMS"] = nuts_platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    if nuts_platform != "auto":
        numpyro.set_platform(nuts_platform)
    payload = _task_to_jax_payload(task)
    d = task.d
    init_loc = jnp.asarray(task.meta["prior_loc"].squeeze(0).cpu().numpy())

    def model():
        base = dist.Normal(jnp.zeros((d,)), jnp.ones((d,))).to_event(1)
        z = numpyro.sample("z", base)
        numpyro.factor("joint", _log_posterior_jax(payload, z) - base.log_prob(z))

    kernel = NUTS(model)
    samples_per_chain = int(math.ceil(float(num_samples) / float(num_chains)))
    init_z = init_loc if num_chains == 1 else jnp.repeat(init_loc[None, :], num_chains, axis=0)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=samples_per_chain,
        num_chains=num_chains,
        chain_method="vectorized" if num_chains > 1 else "sequential",
        progress_bar=bool(progress_bar),
    )
    mcmc.run(jax.random.PRNGKey(seed), init_params={"z": init_z})
    z = np.asarray(mcmc.get_samples(group_by_chain=False)["z"])[:num_samples].copy()
    return torch.from_numpy(z).float()


def build_reference_payload(task_dict, num_samples, num_warmup, seed):
    from .eval import chunked_log_posterior

    task = dict_to_task(task_dict)
    samples = sample_nuts_reference(
        task,
        num_samples=num_samples,
        num_warmup=num_warmup,
        seed=seed,
        progress_bar=False,
    ).cpu()
    log_pbar = chunked_log_posterior(task, samples).cpu()
    return {
        "seed": int(seed),
        "num_samples": int(samples.shape[0]),
        "num_warmup": int(num_warmup),
        "samples": samples,
        "log_pbar": log_pbar,
    }


def reference_worker(job):
    task_payload = torch.load(job["task_payload_path"], map_location="cpu")
    payload = build_reference_payload(
        task_dict=task_payload["task"],
        num_samples=int(job["num_samples"]),
        num_warmup=int(job["num_warmup"]),
        seed=int(job["seed"]),
    )
    torch.save(payload, job["output_path"])
    return int(job["task_idx"])
