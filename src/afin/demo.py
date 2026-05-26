"""Small notebook helpers for the public demo suite.

These utilities keep the notebooks focused on the Bayesian model and the
result, while the repetitive data loading, timing, and table formatting lives
here.
"""
from __future__ import annotations

import time
import warnings
from statistics import median

import torch

from .inference import PRETRAINED_CHECKPOINTS, afin, download_pretrained_checkpoints, load_afin
from .spec import build_problem


def load_public_afins(families=("gaussian", "flow"), *, weights="final", labels=None):
    """Load public AFINs checkpoints from Hugging Face."""
    labels = labels or {
        "gaussian": "AFINs Gaussian",
        "flow": "AFINs Flow",
    }
    root = download_pretrained_checkpoints(families=list(families))
    return [
        (labels.get(family, f"AFINs {family}"), load_afin(root / PRETRAINED_CHECKPOINTS[family], weights=weights))
        for family in families
    ]


def binary_scores(probs, y):
    probs = probs.clamp(1e-6, 1 - 1e-6)
    pred = (probs >= 0.5).float()
    acc = (pred == y).float().mean().item()
    nll = torch.nn.functional.binary_cross_entropy(probs, y).item()
    return acc, nll


def posterior_predictive_metrics(data, samples):
    train_probs = torch.sigmoid(data["X_train"] @ samples.T).mean(dim=1)
    test_probs = torch.sigmoid(data["X_test"] @ samples.T).mean(dim=1)
    train_acc, train_nll = binary_scores(train_probs, data["y_train"])
    test_acc, test_nll = binary_scores(test_probs, data["y_test"])
    return train_acc, train_nll, test_acc, test_nll


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _problem_from_data(data):
    problem = data.get("problem")
    if problem is not None:
        return problem
    problem = build_problem(
        data["prior"],
        data["observed"],
        name=data.get("problem_name", data.get("name", "")),
        data=data,
    )
    data["problem"] = problem
    return problem


def run_one_shot_logistic(model, data, *, method, samples=4000, repeats=3, seed=0):
    """Run one-shot AFIN posterior inference and score posterior predictions."""
    kwargs = {"flow_samples": int(samples)} if model.posterior_family == "flow" else {}
    problem = _problem_from_data(data)

    torch.manual_seed(int(seed))
    _ = afin(model, problem, **kwargs)

    times = []
    posterior = None
    for repeat in range(int(repeats)):
        torch.manual_seed(int(seed) + repeat + 1)
        sync_cuda()
        start = time.perf_counter()
        posterior = afin(model, problem, **kwargs)
        sync_cuda()
        times.append(time.perf_counter() - start)

    z = posterior.sample(int(samples), seed=int(seed))  # type: ignore[union-attr]
    train_acc, train_nll, test_acc, test_nll = posterior_predictive_metrics(data, z)
    row = {
        "method": method,
        "dataset": data["name"],
        "prior": data.get("prior", "standard"),
        "d": data["d"],
        "time_s": median(times),
        "train_acc": train_acc,
        "train_nll": train_nll,
        "test_acc": test_acc,
        "test_nll": test_nll,
    }
    return row, {"posterior": posterior, "samples": z}


def run_afin_logistic_benchmark(models, datasets, *, samples=4000, repeats=3, seed=0):
    rows = []
    posterior_cache = {}
    for method, model in models:
        for data in datasets:
            row, cache = run_one_shot_logistic(
                model,
                data,
                method=method,
                samples=samples,
                repeats=repeats,
                seed=seed,
            )
            rows.append(row)
            posterior_cache[(method, data["name"])] = cache
    return rows, posterior_cache


def run_tabpfn_logistic_benchmark(
    datasets,
    *,
    device=None,
    seed=0,
    v26_estimators=8,
    public_v2_estimators=4,
):
    """Run TabPFN if installed; returns ``([], None)`` when unavailable."""
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError:
        print("TabPFN is not installed; install with `pip install tabpfn` to run this baseline.")
        return [], None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def quiet_fit(clf, x, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Running on CPU with more than 200 samples.*")
            clf.fit(x, y)
        return clf

    def warm_up(label, make_clf):
        data = datasets[0]
        try:
            quiet_fit(make_clf(), data["X_train_tab"][:64], data["y_train_tab"][:64])
            return True
        except Exception as exc:
            message = str(exc)
            if "TABPFN_TOKEN" in message or "license" in message.lower():
                print(f"{label} skipped: license/API token required.")
            else:
                print(f"{label} skipped: {type(exc).__name__}: {exc}")
            return False

    make_v26 = lambda: TabPFNClassifier(device=device, n_estimators=int(v26_estimators), random_state=int(seed))
    make_v2 = lambda: TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device=device,
        n_estimators=int(public_v2_estimators),
        random_state=int(seed),
    )

    if warm_up("TabPFN v2.6", make_v26):
        label, make_clf = "TabPFN v2.6", make_v26
    else:
        print("Falling back to the public TabPFN v2 checkpoint.")
        if not warm_up("TabPFN public v2", make_v2):
            return [], None
        label, make_clf = "TabPFN public v2", make_v2

    rows = []
    for data in datasets:
        clf = make_clf()
        start = time.perf_counter()
        quiet_fit(clf, data["X_train_tab"], data["y_train_tab"])
        train_proba = clf.predict_proba(data["X_train_tab"])
        test_proba = clf.predict_proba(data["X_test_tab"])
        elapsed = time.perf_counter() - start

        classes = list(getattr(clf, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else train_proba.shape[1] - 1
        train_probs = torch.as_tensor(train_proba[:, pos_idx], dtype=torch.float32)
        test_probs = torch.as_tensor(test_proba[:, pos_idx], dtype=torch.float32)
        train_acc, train_nll = binary_scores(train_probs, data["y_train"])
        test_acc, test_nll = binary_scores(test_probs, data["y_test"])
        rows.append({
            "method": label,
            "dataset": data["name"],
            "prior": "n/a",
            "d": data["d"],
            "time_s": elapsed,
            "train_acc": train_acc,
            "train_nll": train_nll,
            "test_acc": test_acc,
            "test_nll": test_nll,
        })
    print("TabPFN device:", device)
    return rows, label


def classification_result_markdown(rows):
    def method_order(method):
        if method == "AFINs Gaussian":
            return 0
        if method == "AFINs Flow":
            return 1
        if str(method).startswith("TabPFN"):
            return 2
        return 99

    rows = sorted(rows, key=lambda r: (r.get("dataset", r.get("features", "")), method_order(r["method"])))
    header = "| method | dataset | prior | d | time | train acc | train NLL | test acc | test NLL |\n"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {r['method']} | {r.get('dataset', r.get('features', ''))} | {r.get('prior', '')} | {r['d']} | "
        f"{r['time_s']:.3f}s | {r['train_acc']:.3f} | {r['train_nll']:.3f} | "
        f"{r['test_acc']:.3f} | {r['test_nll']:.3f} |\n"
        for r in rows
    )
    return header + sep + body


def plot_logistic_coefficients(datasets, posterior_cache, *, method="AFINs Flow", names=None):
    import matplotlib.pyplot as plt

    if names is None:
        names = tuple(data["name"] for data in datasets)
    datasets_by_name = {data["name"]: data for data in datasets}
    fig, axes = plt.subplots(1, len(names), figsize=(11.5, 4.8))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data = datasets_by_name[name]
        samples = posterior_cache[(method, name)]["samples"]
        coef_mean = samples.mean(dim=0)
        lo = torch.quantile(samples, 0.05, dim=0)
        hi = torch.quantile(samples, 0.95, dim=0)
        idx = torch.argsort(coef_mean.abs(), descending=True)[:12] if data["d"] > 12 else torch.arange(data["d"])

        ypos = torch.arange(len(idx))
        ax.errorbar(
            coef_mean[idx],
            ypos,
            xerr=torch.stack([coef_mean[idx] - lo[idx], hi[idx] - coef_mean[idx]]).numpy(),
            fmt="o",
            color="#5B7FDB",
            ecolor="#9bb1f0",
            capsize=3,
        )
        ax.axvline(0.0, color="0.25", lw=1.0, ls="--")
        ax.set_yticks(ypos)
        ax.set_yticklabels([data["feature_names"][int(i)] for i in idx])
        ax.invert_yaxis()
        ax.set_xlabel("posterior coefficient")
        ax.set_title(f"{method} {name} ({data['regime']})")
        ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    return fig
