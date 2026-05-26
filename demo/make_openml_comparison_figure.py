"""Make compact OpenML comparison figures for the README."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import apply_figure_style


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "demo" / "assets"
RESULTS = ASSETS / "openml_standard_prior_benchmark.csv"
OUTPUT = ASSETS / "afin_openml_tabpfn_comparison.png"
AFIN_METHOD = "AFINs Flow HF N256 + SNIS"

BLUE = "#2563EB"
TEAL = "#0891B2"
CORAL = "#E76F51"
CHARCOAL = "#1F2937"
MUTED = "#94A3B8"
GRID = "#E5E7EB"
BG = "#FFFFFF"

NAME_OVERRIDES = {
    "banknote-authentication": "banknote",
    "blood-transfusion-service-center": "blood transfusion",
    "climate-model-simulation-crashes": "climate crashes",
    "heart-statlog": "heart statlog",
    "credit-approval": "credit approval",
    "qsar-biodeg": "qsar biodeg",
}


def short_name(name: str) -> str:
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]
    if len(name) <= 18:
        return name
    return name[:15] + "..."


def load_comparison(path: Path = RESULTS) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["error"].isna()].copy()

    tab = df[df["method"].eq("TabPFN public v2")].copy()
    tab = tab[["openml_id", "dataset", "d_features", "N_train", "N_test", "test_acc", "test_nll"]]

    afin = df[df["method"].eq(AFIN_METHOD)].copy()
    afin = afin.dropna(subset=["test_acc"])
    afin = afin[["dataset", "method", "prior", "test_acc", "test_nll", "time_s"]]

    out = afin.merge(tab, on="dataset", suffixes=("_afin", "_tabpfn"))
    out = out.rename(columns={
        "test_acc_afin": "afin_acc",
        "test_nll_afin": "afin_nll",
        "test_acc_tabpfn": "tabpfn_acc",
        "test_nll_tabpfn": "tabpfn_nll",
    })
    out["gap_pp"] = 100.0 * (out["afin_acc"] - out["tabpfn_acc"])
    out["win"] = out["gap_pp"] >= -1e-9
    out["label"] = out["dataset"].map(short_name)
    return out.sort_values("gap_pp").reset_index(drop=True)


def _common_axis_style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


def _summary(table: pd.DataFrame) -> tuple[int, int, float, float, float]:
    n = len(table)
    wins = int(table["win"].sum())
    afin_mean = float(table["afin_acc"].mean())
    tab_mean = float(table["tabpfn_acc"].mean())
    gap = 100.0 * (afin_mean - tab_mean)
    return wins, n, afin_mean, tab_mean, gap


def make_scatter(table: pd.DataFrame, path: Path):
    apply_figure_style()
    wins, n, afin_mean, tab_mean, gap = _summary(table)

    fig, ax = plt.subplots(figsize=(6.4, 5.7), dpi=220)
    colors = np.where(table["win"], BLUE, CORAL)
    ax.scatter(
        table["tabpfn_acc"],
        table["afin_acc"],
        s=76,
        c=colors,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.95,
        zorder=3,
    )

    lo = math.floor(min(table["tabpfn_acc"].min(), table["afin_acc"].min()) * 20 - 1) / 20
    hi = math.ceil(max(table["tabpfn_acc"].max(), table["afin_acc"].max()) * 20 + 1) / 20
    lo = max(0.45, lo)
    hi = min(1.02, hi)
    ax.plot([lo, hi], [lo, hi], color=CHARCOAL, linewidth=1.2, linestyle="--", alpha=0.55)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("TabPFN test accuracy")
    ax.set_ylabel("Flow + SNIS AFIN test accuracy")
    ax.set_title("Binary OpenML tasks")
    _common_axis_style(ax)

    text = f"{wins}/{n} tasks >= TabPFN\nmean acc: {afin_mean:.3f} vs {tab_mean:.3f}\nmean gap: {gap:+.1f} pp"
    ax.text(
        0.045,
        0.955,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        color=CHARCOAL,
        bbox=dict(boxstyle="round,pad=0.36,rounding_size=0.18", facecolor="white", edgecolor="#CBD5E1"),
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_gap_bars(table: pd.DataFrame, path: Path):
    apply_figure_style()
    fig, ax = plt.subplots(figsize=(8.8, 6.9), dpi=220)

    y = np.arange(len(table))
    colors = np.where(table["win"], BLUE, MUTED)
    ax.barh(y, table["gap_pp"], color=colors, height=0.66, alpha=0.95)
    ax.axvline(0.0, color=CHARCOAL, linewidth=1.1, alpha=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"])
    ax.set_xlabel("Flow + SNIS AFIN - TabPFN test accuracy (percentage points)")
    ax.set_title("Accuracy gap by dataset")
    _common_axis_style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="y")

    for yi, gap in zip(y, table["gap_pp"]):
        if gap >= 0:
            ax.text(gap + 0.4, yi, f"+{gap:.1f}", va="center", ha="left", fontsize=8.2, color=CHARCOAL)
    xpad = max(1.5, float(np.abs(table["gap_pp"]).max()) * 0.12)
    ax.set_xlim(table["gap_pp"].min() - xpad, table["gap_pp"].max() + xpad + 3.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_dumbbell(table: pd.DataFrame, path: Path):
    apply_figure_style()
    ordered = table.sort_values("tabpfn_acc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.8), dpi=220)

    y = np.arange(len(ordered))
    ax.hlines(y, ordered["afin_acc"], ordered["tabpfn_acc"], color="#CBD5E1", linewidth=2.2, zorder=1)
    ax.scatter(ordered["tabpfn_acc"], y, s=48, color=CHARCOAL, label="TabPFN", zorder=3)
    ax.scatter(ordered["afin_acc"], y, s=58, color=BLUE, edgecolor="white", linewidth=0.9, label="Flow + SNIS AFIN", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["label"])
    ax.set_xlabel("Test accuracy")
    ax.set_title("Per-dataset accuracy")
    _common_axis_style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(max(0.45, ordered[["afin_acc", "tabpfn_acc"]].min().min() - 0.04), 1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_readme_figure(table: pd.DataFrame, path: Path = OUTPUT):
    apply_figure_style()
    wins, n, afin_mean, tab_mean, gap = _summary(table)
    fig, ax = plt.subplots(figsize=(10.7, 6.5), dpi=220)
    ordered = table.sort_values("gap_pp").reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = np.where(ordered["gap_pp"] >= -1e-9, BLUE, MUTED)
    ax.barh(y, ordered["gap_pp"], color=colors, height=0.68, alpha=0.96)
    ax.axvline(0.0, color=CHARCOAL, linewidth=1.15, alpha=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["label"])
    ax.set_xlabel("Test accuracy gap: Flow + SNIS AFIN - TabPFN (percentage points)")
    ax.set_title("Flow + SNIS AFIN vs TabPFN on binary OpenML")
    _common_axis_style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.78)
    ax.grid(False, axis="y")

    x_min = min(-6.0, float(ordered["gap_pp"].min()) - 0.8)
    x_max = max(3.2, float(ordered["gap_pp"].max()) + 1.2)
    ax.set_xlim(x_min, x_max)
    for yi, gap_i in zip(y, ordered["gap_pp"]):
        label = f"{gap_i:+.1f}"
        if gap_i >= 0:
            ax.text(gap_i + 0.12, yi, label, va="center", ha="left", fontsize=8.6, color=CHARCOAL)
        else:
            ax.text(gap_i - 0.12, yi, label, va="center", ha="right", fontsize=8.6, color=CHARCOAL)

    summary = (
        f"{wins}/{n} datasets at or above TabPFN     "
        f"mean accuracy {afin_mean:.3f} vs {tab_mean:.3f}     "
        f"mean gap {gap:+.1f} pp"
    )
    fig.text(
        0.51,
        0.035,
        summary,
        ha="center",
        va="center",
        fontsize=10.6,
        color=CHARCOAL,
        bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.15", facecolor="white", edgecolor="#CBD5E1"),
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.88, bottom=0.16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--candidates-dir", type=Path, default=None)
    args = parser.parse_args()

    table = load_comparison(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_readme_figure(table, args.output)

    if args.candidates_dir is not None:
        args.candidates_dir.mkdir(parents=True, exist_ok=True)
        make_scatter(table, args.candidates_dir / "candidate_scatter.png")
        make_gap_bars(table, args.candidates_dir / "candidate_gap_bars.png")
        make_dumbbell(table, args.candidates_dir / "candidate_dumbbell.png")

    wins, n, afin_mean, tab_mean, gap = _summary(table)
    print(f"Wrote {args.output}")
    print(f"Flow + SNIS AFIN >= TabPFN on {wins}/{n} datasets")
    print(f"Mean accuracy: AFINs {afin_mean:.4f}, TabPFN {tab_mean:.4f}, gap {gap:+.2f} pp")


if __name__ == "__main__":
    main()
