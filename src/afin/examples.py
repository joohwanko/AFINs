"""Small public examples used by notebooks and docs."""
import math

import torch

from .spec import BernoulliLogit, GaussianPrior, LinearGaussian, LinearStudentT, build_problem


def make_gaussian_2d_data(seed=0, n=36, sigma=0.35):
    """Generate a simple homogeneous 2D Gaussian regression problem."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    d = 2
    true_z = torch.tensor([1.15, -0.85])
    x_scale = 0.85 / math.sqrt(d)
    x = x_scale * torch.randn(int(n), d, generator=generator)
    y = x @ true_z + float(sigma) * torch.randn(int(n), generator=generator)
    return {"true_z": true_z, "X": x, "y": y, "sigma": torch.tensor(float(sigma))}


def make_gaussian_2d(seed=0, n=36, sigma=0.35):
    data = make_gaussian_2d_data(seed=seed, n=n, sigma=sigma)
    return build_problem(
        GaussianPrior(0, 1),
        [LinearGaussian(data["X"], sigma=float(data["sigma"])).observe(data["y"])],
        name="homogeneous Gaussian regression",
        true_z=data["true_z"],
        data=data,
    )


def make_mixed_2d_data(seed=0, n_gaussian=18, n_binary=18, n_student=8):
    """Generate a small heterogeneous 2D Bayesian regression problem."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    d = 2
    true_z = torch.tensor([1.15, -0.85])
    x_scale = 0.85 / math.sqrt(d)

    x_gauss = x_scale * torch.randn(int(n_gaussian), d, generator=generator)
    y_gauss = x_gauss @ true_z + 0.35 * torch.randn(int(n_gaussian), generator=generator)

    x_binary = x_scale * torch.randn(int(n_binary), d, generator=generator)
    probs = torch.sigmoid(x_binary @ true_z)
    y_binary = torch.bernoulli(probs, generator=generator)

    x_student = x_scale * torch.randn(int(n_student), d, generator=generator)
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed) + 12345)
        noise = torch.distributions.StudentT(torch.tensor(4.0)).sample((int(n_student),))
    finally:
        torch.random.set_rng_state(rng_state)
    y_student = x_student @ true_z + 0.60 * noise

    return {
        "true_z": true_z,
        "X_gauss": x_gauss,
        "y_gauss": y_gauss,
        "X_binary": x_binary,
        "y_binary": y_binary,
        "X_student": x_student,
        "y_student": y_student,
    }


def make_mixed_2d(seed=0, n_gaussian=18, n_binary=18, n_student=8):
    data = make_mixed_2d_data(seed, n_gaussian, n_binary, n_student)
    return build_problem(
        GaussianPrior(0, 1),
        [
            LinearGaussian(data["X_gauss"], sigma=0.35).observe(data["y_gauss"]),
            BernoulliLogit(data["X_binary"]).observe(data["y_binary"]),
            LinearStudentT(data["X_student"], sigma=0.60, df=4.0).observe(data["y_student"]),
        ],
        name="mixed Gaussian + Bernoulli + StudentT",
        true_z=data["true_z"],
        data=data,
    )
