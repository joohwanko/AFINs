"""Lightweight notebook plotting and reporting helpers."""
import math

import numpy as np
import torch

from .inference import GaussianPosterior
from .spec import Problem
from .tasks import move_task, repeat_task, unnormalized_log_posterior


def posterior_report(approx, reference=None, *, approx_time=None, reference_time=None, num_w2_samples=2048):
    rows = [{
        "method": approx.name,
        "time_s": approx_time,
        "mean": approx.mean.tolist(),
        "std": torch.sqrt(torch.diagonal(approx.cov)).tolist(),
        "mean_l2_to_ref": None,
        "cov_fro_to_ref": None,
        "sliced_w2_to_ref": None,
    }]
    if reference is not None:
        metrics = posterior_metrics(approx, reference, num_samples=num_w2_samples)
        rows.append({
            "method": reference.name,
            "time_s": reference_time,
            "mean": reference.mean.tolist(),
            "std": torch.sqrt(torch.diagonal(reference.cov)).tolist(),
            "mean_l2_to_ref": 0.0,
            "cov_fro_to_ref": 0.0,
            "sliced_w2_to_ref": 0.0,
        })
        rows[0].update(metrics)
    return rows


def posterior_metric_line(approx, reference, *, approx_time=None, num_w2_samples=2048, digits=4):
    metrics = posterior_metrics(approx, reference, num_samples=num_w2_samples)
    time_part = "" if approx_time is None else f" | time={_format_time(approx_time)}"
    return (
        f"{approx.name}{time_part} | "
        f"m1={_format_metric(metrics['mean_l2_to_ref'], digits)} | "
        f"m2={_format_metric(metrics['cov_fro_to_ref'], digits)} | "
        f"sliced W2={_format_metric(metrics['sliced_w2_to_ref'], digits)}"
    )


def print_posterior_metrics(approx, reference, *, approx_time=None, num_w2_samples=2048, digits=4):
    print(posterior_metric_line(
        approx,
        reference,
        approx_time=approx_time,
        num_w2_samples=num_w2_samples,
        digits=digits,
    ))


def posterior_metrics(approx, reference, *, num_samples=2048, num_projections=128, seed=0):
    """Small notebook metrics against a reference posterior."""
    return {
        "mean_l2_to_ref": float(torch.linalg.norm(approx.mean - reference.mean).item()),
        "cov_fro_to_ref": float(torch.linalg.norm(approx.cov - reference.cov).item()),
        "sliced_w2_to_ref": sliced_w2(
            _samples_for_metric(approx, num_samples, seed=seed),
            _samples_for_metric(reference, num_samples, seed=seed + 17),
            num_projections=num_projections,
            seed=seed + 101,
        ),
    }


def sliced_w2(x, y, *, num_projections=128, seed=0):
    """Sliced Wasserstein-2 distance between two sample clouds."""
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n <= 0:
        return float("nan")
    x = x[:n].double()
    y = y[:n].double()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn(int(num_projections), x.shape[1], generator=generator, dtype=torch.double)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    px = torch.sort(x @ directions.T, dim=0).values
    py = torch.sort(y @ directions.T, dim=0).values
    return float((px - py).square().mean(dim=0).mean().sqrt().item())


def _format_metric(value, digits):
    return f"{float(value):.{int(digits)}g}"


def _format_time(seconds):
    seconds = float(seconds)
    if seconds < 1.0:
        return f"{1e3 * seconds:.2f} ms"
    return f"{seconds:.2f} s"


def plot_2d(
    problem: Problem,
    approx,
    reference=None,
    *,
    ax=None,
    grid_size=180,
    max_points=2000,
    show_true=False,
    zoom=True,
    figsize=(6.2, 5.6)
):
    """Plot posterior PDFs/contours in 2D.

    Reference is shown as a grayscale density with dashed contours. AFIN is shown
    as blue contours. Sample-based posteriors are converted to a KDE.
    """
    if problem.d != 2:
        raise ValueError("plot_2d only supports d=2 problems.")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    xx, yy = _density_grid([p for p in (reference, approx) if p is not None], grid_size=grid_size, max_points=max_points)

    zoom_values = []
    if reference is not None:
        zz_ref = _density_values(reference, xx, yy, max_points=max_points, seed=0)
        zoom_values.append(zz_ref)
        levels_ref = _density_levels(zz_ref, count=8)
        if levels_ref is not None:
            ax.contourf(xx, yy, zz_ref, levels=levels_ref, cmap="Greys", alpha=0.45)
            ax.contour(xx, yy, zz_ref, levels=levels_ref[1::2], colors="black", linestyles="--", linewidths=1.3)
            ax.plot([], [], color="black", linestyle="--", label=f"{reference.name} PDF")
        ax.scatter(
            reference.mean[0], reference.mean[1], s=55,
            color="black", marker="x", label=f"{reference.name} mean",
        )

    zz_approx = _density_values(approx, xx, yy, max_points=max_points, seed=1)
    zoom_values.append(zz_approx)
    levels_approx = _density_levels(zz_approx, count=6)
    if levels_approx is not None:
        ax.contour(xx, yy, zz_approx, levels=levels_approx[1:], colors="tab:blue", linewidths=2.0)
        ax.plot([], [], color="tab:blue", linewidth=2.0, label=f"{approx.name} PDF")
    ax.scatter(approx.mean[0], approx.mean[1], s=70, color="tab:blue", marker="o", label=f"{approx.name} mean")

    if show_true and problem.true_z is not None:
        z = problem.true_z.reshape(-1)
        ax.scatter(z[0], z[1], s=80, color="tab:red", marker="*", label="true z")

    if zoom:
        _apply_density_zoom(ax, xx, yy, zoom_values)

    ax.set_title(problem.name or "AFIN posterior")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.grid(alpha=0.20)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_target_2d(
    problem: Problem,
    approx,
    *,
    ax=None,
    grid_size=180,
    max_points=2000,
    show_true=False,
    zoom=True,
    target_label="Target posterior",
    figsize=(6.2, 5.6)
):
    """Plot AFIN against the exact unnormalized target density on a 2D grid.

    This is useful for demos: in 2D we can evaluate the target density directly,
    so a NUTS chain is not needed just to draw the posterior shape.
    """
    if problem.d != 2:
        raise ValueError("plot_target_2d only supports d=2 problems.")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    xx, yy = _density_grid([approx], grid_size=grid_size, max_points=max_points)
    zz_target = _target_density_values(problem, xx, yy)
    zz_approx = _density_values(approx, xx, yy, max_points=max_points, seed=1)

    levels_target = _density_levels(zz_target, count=8)
    if levels_target is not None:
        ax.contourf(xx, yy, zz_target, levels=levels_target, cmap="Greys", alpha=0.45)
        ax.contour(xx, yy, zz_target, levels=levels_target[1::2], colors="black", linestyles="--", linewidths=1.3)
        ax.plot([], [], color="black", linestyle="--", label=f"{target_label} PDF")

    levels_approx = _density_levels(zz_approx, count=6)
    if levels_approx is not None:
        ax.contour(xx, yy, zz_approx, levels=levels_approx[1:], colors="tab:blue", linewidths=2.0)
        ax.plot([], [], color="tab:blue", linewidth=2.0, label=f"{approx.name} PDF")
    ax.scatter(approx.mean[0], approx.mean[1], s=70, color="tab:blue", marker="o", label=f"{approx.name} mean")

    if show_true and problem.true_z is not None:
        z = problem.true_z.reshape(-1)
        ax.scatter(z[0], z[1], s=80, color="tab:red", marker="*", label="true z")

    if zoom:
        _apply_density_zoom(ax, xx, yy, [zz_target, zz_approx])

    ax.set_title(problem.name or "AFIN posterior")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.grid(alpha=0.20)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_reference_2d(
    problem: Problem,
    approx,
    reference=None,
    *,
    reference_plots=("nuts",),
    title=None,
    figsize=(6.2, 5.6),
):
    """Draw one or more 2D demo reference plots.

    ``reference_plots`` can contain ``"target"`` for the unnormalized target
    grid and ``"nuts"`` for a supplied sample/analytic reference posterior.
    """
    import matplotlib.pyplot as plt

    axes = []
    unknown = set(reference_plots) - {"target", "nuts"}
    if unknown:
        raise ValueError(f"Unknown reference plot option(s): {sorted(unknown)}")

    if "target" in reference_plots:
        _, ax = plt.subplots(figsize=figsize)
        plot_target_2d(problem, approx, ax=ax)
        if title is not None:
            ax.set_title(f"{title}: target grid")
        plt.tight_layout()
        axes.append(ax)

    if "nuts" in reference_plots:
        if reference is None:
            raise ValueError("A reference posterior is required for the 'nuts' plot.")
        _, ax = plt.subplots(figsize=figsize)
        plot_2d(problem, approx, reference, ax=ax)
        if title is not None:
            ax.set_title(f"{title}: {reference.name}")
        plt.tight_layout()
        axes.append(ax)

    return axes


def format_metric(value):
    return f"{float(value):.3g}"


def draw_one_shot_panel(ax, approx, reference, *, color, title, reference_label, reference_mean_label):
    """Draw the polished one-shot posterior panel used by the README."""
    xx, yy = _density_grid([reference, approx], grid_size=190, max_points=3500)
    zz_ref = _density_values(reference, xx, yy, max_points=4500, seed=0)
    levels_ref = _density_levels(zz_ref, count=8)

    if levels_ref is not None:
        ax.contourf(xx, yy, zz_ref, levels=levels_ref, cmap="Greys", alpha=0.36)
        ax.contour(
            xx,
            yy,
            zz_ref,
            levels=levels_ref[1::2],
            colors="#27272a",
            linestyles="--",
            linewidths=1.1,
        )
        ax.plot([], [], color="#27272a", linestyle="--", label=reference_label)

    zz_approx = _density_values(approx, xx, yy, max_points=3500, seed=13)
    levels_approx = _density_levels(zz_approx, count=7)
    if levels_approx is not None:
        ax.contour(xx, yy, zz_approx, levels=levels_approx[1:], colors=color, linewidths=1.7, alpha=0.72)
        ax.plot([], [], color=color, linewidth=1.9, alpha=0.76, label="AFIN posterior")

    ax.scatter(reference.mean[0], reference.mean[1], s=60, color="black", marker="x", label=reference_mean_label)
    ax.scatter(approx.mean[0], approx.mean[1], s=68, color=color, edgecolor="white", linewidth=0.8, label="AFIN mean")

    metrics = posterior_metrics(approx, reference, num_samples=2200)
    note = (
        "one forward pass\n"
        f"m1 {format_metric(metrics['mean_l2_to_ref'])}\n"
        f"m2 {format_metric(metrics['cov_fro_to_ref'])}\n"
        f"SW2 {format_metric(metrics['sliced_w2_to_ref'])}"
    )
    ax.text(
        0.03,
        0.04,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.84", alpha=0.94),
    )

    _apply_density_zoom(ax, xx, yy, [zz_ref, zz_approx], min_frac=0.045)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.grid(alpha=0.18)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    return metrics


def _samples_for_metric(post, n, seed=0):
    return post.sample(int(n), seed=seed).detach().cpu().float()


def _density_grid(posts, *, grid_size, max_points):
    chunks = []
    for idx, post in enumerate(posts):
        if isinstance(post, GaussianPosterior):
            std = torch.sqrt(torch.diagonal(post.cov).clamp_min(1e-12))
            chunks.append(torch.stack([post.mean - 3.0 * std, post.mean + 3.0 * std]))
        else:
            samples = post.sample(max_points, seed=idx).detach().cpu().float()
            lo = torch.quantile(samples, 0.02, dim=0)
            hi = torch.quantile(samples, 0.98, dim=0)
            chunks.append(torch.stack([lo, hi]))
    bounds = torch.cat(chunks, dim=0)
    lo = bounds.min(dim=0).values
    hi = bounds.max(dim=0).values
    span = (hi - lo).clamp_min(1e-3)
    lo = lo - 0.06 * span
    hi = hi + 0.06 * span
    x = torch.linspace(float(lo[0]), float(hi[0]), int(grid_size))
    y = torch.linspace(float(lo[1]), float(hi[1]), int(grid_size))
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx.numpy(), yy.numpy()


def _apply_density_zoom(ax, xx, yy, density_values, *, min_frac=0.08):
    masks = []
    for values in density_values:
        vmax = float(np.nanmax(values))
        if math.isfinite(vmax) and vmax > 0:
            masks.append(values >= min_frac * vmax)
    if not masks:
        return
    mask = np.logical_or.reduce(masks)
    if not np.any(mask):
        return
    x_vals = xx[mask]
    y_vals = yy[mask]
    lo = np.array([float(np.min(x_vals)), float(np.min(y_vals))])
    hi = np.array([float(np.max(x_vals)), float(np.max(y_vals))])
    span = np.maximum(hi - lo, 1e-3)
    center = 0.5 * (lo + hi)
    half = 0.5 * span * 1.14
    max_half = float(np.max(half))
    ax.set_xlim(center[0] - max_half, center[0] + max_half)
    ax.set_ylim(center[1] - max_half, center[1] + max_half)


def _density_values(post, xx, yy, *, max_points, seed):
    points = torch.from_numpy(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)).float()
    if isinstance(post, GaussianPosterior):
        with torch.no_grad():
            values = post.log_prob(points).exp().reshape(xx.shape)
        return values.numpy()

    samples = post.sample(max_points, seed=seed).detach().cpu().float()
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(samples.T.numpy())
        values = kde(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=0)).reshape(xx.shape)
    except Exception:
        values = _histogram_density(samples, xx, yy)
    return np.asarray(values, dtype=float)


def _target_density_values(problem, xx, yy, *, chunk_size=8192):
    points = torch.from_numpy(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)).float()
    task = move_task(problem.task, "cpu")
    values = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        z = points[start : start + int(chunk_size)]
        task_chunk = repeat_task(task, z.shape[0])
        values.append(unnormalized_log_posterior(task_chunk, z).detach().cpu())
    logp = torch.cat(values, dim=0)
    finite = torch.isfinite(logp)
    if not bool(finite.any()):
        return np.zeros_like(xx, dtype=float)
    logp = logp - logp[finite].max()
    density = torch.exp(logp).reshape(xx.shape)
    return density.numpy()


def _histogram_density(samples, xx, yy):
    bins_x = np.linspace(float(xx.min()), float(xx.max()), xx.shape[1] + 1)
    bins_y = np.linspace(float(yy.min()), float(yy.max()), yy.shape[0] + 1)
    hist, _, _ = np.histogram2d(samples[:, 1].numpy(), samples[:, 0].numpy(), bins=(bins_y, bins_x), density=True)
    return hist


def _density_levels(values, *, count, min_frac=0.08):
    vmax = float(np.nanmax(values))
    if not math.isfinite(vmax) or vmax <= 0:
        return None
    return np.linspace(float(min_frac) * vmax, vmax, int(count))


def _draw_gaussian_ellipses(ax, mean, cov, color, label, linestyle="-", linewidth=2.0):
    from matplotlib.patches import Ellipse

    vals, vecs = torch.linalg.eigh(cov)
    vals = vals.clamp_min(1e-8)
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(float(vecs[1, 0]), float(vecs[0, 0])))
    first = True
    for nsig, alpha in ((1.0, 0.90), (2.0, 0.55), (3.0, 0.30)):
        width = 2.0 * nsig * math.sqrt(float(vals[0]))
        height = 2.0 * nsig * math.sqrt(float(vals[1]))
        patch = Ellipse(
            xy=(float(mean[0]), float(mean[1])),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            lw=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            color=color,
            label=label if first else None,
        )
        ax.add_patch(patch)
        first = False
