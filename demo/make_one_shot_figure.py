"""Make a README/demo asset for one-shot posterior inference with AFINs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from figure_style import apply_figure_style


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afin import (  # noqa: E402
    AFIN,
    NumPyroNUTS,
    PRETRAINED_CHECKPOINTS,
    draw_one_shot_panel,
    download_pretrained_checkpoints,
    exact_gaussian_regression_posterior,
    infer,
    load_afin,
    make_gaussian_2d,
    make_mixed_2d,
)


apply_figure_style()

OUT = ROOT / "demo" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
PNG_PATH = OUT / "afin_one_shot_posteriors.png"


def load_models():
    checkpoint_root = download_pretrained_checkpoints()
    return (
        load_afin(checkpoint_root / PRETRAINED_CHECKPOINTS["gaussian"], weights="final"),
        load_afin(checkpoint_root / PRETRAINED_CHECKPOINTS["flow"], weights="final"),
    )


def main():
    torch.manual_seed(0)
    gaussian_model, flow_model = load_models()
    gaussian_method = AFIN("gaussian", model=gaussian_model)
    flow_method = AFIN("flow", model=flow_model)

    gaussian_problem = make_gaussian_2d(seed=4, n=36, sigma=0.35)
    gaussian_reference = exact_gaussian_regression_posterior(gaussian_problem)
    post_gaussian = infer(gaussian_method, gaussian_problem, seed=0)

    flow_problem = make_mixed_2d(seed=3, n_gaussian=18, n_binary=18, n_student=18)
    flow_reference = infer(
        NumPyroNUTS(
            num_warmup=1000,
            platform="cpu",
            progress_bar=False,
        ),
        flow_problem,
        num_samples=6000,
        seed=0,
    )
    post_flow = infer(flow_method, flow_problem, num_samples=3500, seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.55), dpi=180)
    fig.subplots_adjust(top=0.80, wspace=0.20)
    gaussian_metrics = draw_one_shot_panel(
        axes[0],
        post_gaussian,
        gaussian_reference,
        color="#5B7FDB",
        title="AFIN with Gaussian posterior",
        reference_label="Exact posterior",
        reference_mean_label="Exact mean",
    )
    flow_metrics = draw_one_shot_panel(
        axes[1],
        post_flow,
        flow_reference,
        color="#2E9F82",
        title="AFIN with flow posterior",
        reference_label="NUTS reference",
        reference_mean_label="NUTS mean",
    )

    fig.suptitle("One-shot posterior inference with AFINs", fontsize=18, y=0.985)
    fig.text(
        0.5,
        0.895,
        "Left: conjugate Gaussian regression.  Right: mixed Gaussian + Bernoulli-logit + Student-t likelihoods.",
        ha="center",
        va="center",
        fontsize=11.5,
    )
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"flow_reference_time_s={flow_reference.runtime_seconds:.2f}")
    print(f"saved_png={PNG_PATH}")
    print("gaussian_time_s", post_gaussian.runtime_seconds)
    print("flow_time_s", post_flow.runtime_seconds)
    print("gaussian_metrics", gaussian_metrics)
    print("flow_metrics", flow_metrics)


if __name__ == "__main__":
    main()
