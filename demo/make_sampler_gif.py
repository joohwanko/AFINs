"""Make README/demo assets comparing local MH with AFIN independence MH."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from figure_style import apply_figure_style


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afin import (  # noqa: E402
    NumPyroNUTS,
    PRETRAINED_CHECKPOINTS,
    chain_metric_trace,
    download_pretrained_checkpoints,
    infer,
    load_afin,
    make_mixed_2d,
    run_afin_imh,
    run_random_walk_mh,
)
from afin.plotting import _density_levels, _density_values  # noqa: E402


apply_figure_style()

OUT = ROOT / "demo" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

GIF_PATH = OUT / "afin_vs_random_walk_mh.gif"
PNG_PATH = OUT / "afin_vs_random_walk_mh.png"

N_STEPS = 1200
FRAME_STEPS = np.unique(np.round(np.geomspace(1, N_STEPS, 28)).astype(int))
X_LIM = (0.05, 1.35)
Y_LIM = (-1.15, 0.20)


def target_contours(reference):
    x = torch.linspace(*X_LIM, 180)
    y = torch.linspace(*Y_LIM, 180)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.numpy()
    yy = yy.numpy()
    zz = _density_values(reference, xx, yy, max_points=4500, seed=0)
    levels = _density_levels(zz, count=8)
    return xx, yy, zz, levels


def draw_panel(ax, xx, yy, zz, levels, states, metrics, step, color, title, mode):
    if levels is not None:
        ax.contourf(xx, yy, zz, levels=levels, cmap="Greys", alpha=0.38)
        ax.contour(xx, yy, zz, levels=levels[1::2], colors="#27272a", linestyles="--", linewidths=1.1)

    shown = states[:step]
    if mode == "walk":
        ax.plot(shown[:, 0], shown[:, 1], color=color, alpha=0.35, linewidth=1.2)
        ax.scatter(shown[:, 0], shown[:, 1], s=13, color=color, alpha=0.42, linewidths=0)
    else:
        ax.scatter(shown[:, 0], shown[:, 1], s=13, color=color, alpha=0.30, linewidths=0)
    ax.scatter(shown[-1, 0], shown[-1, 1], s=64, color=color, edgecolor="white", linewidth=1.1, zorder=10)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title(title, fontsize=13, pad=8)

    box = (
        f"step {step}/{N_STEPS}\n"
        f"m1 {metrics['mean_error']:.3f}\n"
        f"SW2 {metrics['sw2']:.3f}\n"
        f"in 95% {metrics['in_region']:.0%}\n"
        f"accept {metrics['accept_rate']:.0%}"
    )
    ax.text(
        0.03,
        0.04,
        box,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.84", alpha=0.94),
    )


def draw_metric_strip(ax, upto, rw_metrics, imh_metrics):
    steps = np.asarray([m["step"] for m in rw_metrics])
    rw = np.asarray([m["sw2"] for m in rw_metrics])
    imh = np.asarray([m["sw2"] for m in imh_metrics])
    current = int(FRAME_STEPS[upto])

    ax.plot(steps[: upto + 1], rw[: upto + 1], color="#C47A2C", linewidth=2.2, label="Random-walk MH")
    ax.plot(steps[: upto + 1], imh[: upto + 1], color="#5B7FDB", linewidth=2.2, label="AFIN-IMH")
    ax.scatter([current], [rw[upto]], color="#C47A2C", s=42, zorder=5)
    ax.scatter([current], [imh[upto]], color="#5B7FDB", s=42, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(1, N_STEPS)
    ymax = max(float(np.nanmax(rw[: upto + 1])), float(np.nanmax(imh[: upto + 1])), 0.04)
    ax.set_ylim(0.0, ymax * 1.18)
    ax.grid(alpha=0.20)
    ax.set_xlabel("posterior evaluations")
    ax.set_ylabel("sliced W2")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.set_title("Sliced W2 to the NUTS reference over the same sampling budget", fontsize=11, pad=4)


def render_frame(
    frame_idx,
    xx,
    yy,
    zz,
    levels,
    rw_states,
    imh_states,
    rw_metrics,
    imh_metrics,
):
    step = int(FRAME_STEPS[frame_idx])
    fig = plt.figure(figsize=(12.2, 6.95), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[4.7, 1.2], hspace=0.35, wspace=0.20)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    draw_panel(
        ax0,
        xx,
        yy,
        zz,
        levels,
        rw_states,
        rw_metrics[frame_idx],
        step,
        "#C47A2C",
        "Local MH from a prior draw",
        "walk",
    )
    draw_panel(
        ax1,
        xx,
        yy,
        zz,
        levels,
        imh_states,
        imh_metrics[frame_idx],
        step,
        "#5B7FDB",
        "AFIN proposal + IMH correction",
        "imh",
    )
    draw_metric_strip(ax2, frame_idx, rw_metrics, imh_metrics)
    fig.suptitle(
        "Same target, same step count: AFIN starts close and then corrects",
        fontsize=15,
        y=0.98,
    )
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    frame = Image.fromarray(image[:, :, :3].copy())
    plt.close(fig)
    return frame


def main():
    torch.manual_seed(0)
    problem = make_mixed_2d(seed=4, n_gaussian=18, n_binary=18, n_student=18)
    checkpoint_root = download_pretrained_checkpoints(families=["flow"])
    model = load_afin(checkpoint_root / PRETRAINED_CHECKPOINTS["flow"], weights="final")

    reference = infer(
        NumPyroNUTS(num_warmup=1000, platform="cpu", progress_bar=False),
        problem,
        num_samples=6000,
        seed=0,
    )
    ref_samples = reference.sample(N_STEPS, seed=123)
    rw_states, rw_accept = run_random_walk_mh(problem)
    imh_states, imh_accept = run_afin_imh(model, problem)
    rw_metrics = chain_metric_trace(rw_states, rw_accept, reference, FRAME_STEPS, ref_samples=ref_samples)
    imh_metrics = chain_metric_trace(imh_states, imh_accept, reference, FRAME_STEPS, ref_samples=ref_samples)
    xx, yy, zz, levels = target_contours(reference)

    frames = [
        render_frame(i, xx, yy, zz, levels, rw_states, imh_states, rw_metrics, imh_metrics)
        for i in range(len(FRAME_STEPS))
    ]
    frames[-1].save(PNG_PATH)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=[120] * (len(frames) - 1) + [1200],
        loop=0,
        optimize=True,
    )

    print(f"reference_time_s={reference.runtime_seconds:.2f}")
    print(f"saved_png={PNG_PATH}")
    print(f"saved_gif={GIF_PATH}")
    print("final_random_walk", rw_metrics[-1])
    print("final_afin_imh", imh_metrics[-1])
    print("first_random_walk", rw_metrics[0])
    print("first_afin_imh", imh_metrics[0])


if __name__ == "__main__":
    main()
