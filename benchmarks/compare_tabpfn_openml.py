"""Compare public AFINs checkpoints against TabPFN on binary OpenML datasets.

This is intentionally a script rather than a notebook. It uses the same
Bayesian logistic-regression construction as ``demo/03_real_world_logistic``:

    prior = GaussianPrior(mu, sigma)
    BernoulliLogit(design_matrix=X).observe(y)

By default the Bayesian model includes an intercept latent, which is standard
for imbalanced real-world classification datasets.  The standard prior uses
mu=0 and sigma=1; the empirical prior uses a small ridge-logistic fit to set a
task-specific Normal prior. AFINs are evaluated from the public Hugging Face
checkpoints selected by the benchmark config. Every AFIN posterior can be used as an SNIS
proposal, with exact log p(z, y) / q(z) weights under the Bayesian
logistic-regression model.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from afin import (  # noqa: E402
    PRETRAINED_CHECKPOINTS,
    BernoulliLogit,
    GaussianPrior,
    afin,
    afin_proposal_samples,
    download_pretrained_checkpoints,
    load_afin,
)
from afin.spec import build_problem as build_afin_problem  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    data_id: int
    name: str


# Binary OpenML datasets used by the public comparison notebooks. We keep all
# rows and all features after one-hot encoding categorical columns; no d/N
# truncation is applied. The public paper/demo table excludes sonar, tecator,
# steel-plates-fault, and cylinder-bands from the earlier exploratory run.
DATASETS = [
    DatasetSpec(15, "breast-w"),
    DatasetSpec(29, "credit-approval"),
    DatasetSpec(31, "credit-g"),
    DatasetSpec(37, "diabetes"),
    DatasetSpec(43, "haberman"),
    DatasetSpec(53, "heart-statlog"),
    DatasetSpec(1049, "pc4"),
    DatasetSpec(1050, "pc3"),
    DatasetSpec(1063, "kc2"),
    DatasetSpec(1068, "pc1"),
    DatasetSpec(1462, "banknote-authentication"),
    DatasetSpec(1464, "blood-transfusion-service-center"),
    DatasetSpec(1480, "ilpd"),
    DatasetSpec(1494, "qsar-biodeg"),
    DatasetSpec(1510, "wdbc"),
    DatasetSpec(1467, "climate-model-simulation-crashes"),
]


@dataclass
class OpenMLBenchmarkConfig:
    dataset_ids: tuple[int, ...] = field(default_factory=lambda: tuple(spec.data_id for spec in DATASETS))
    samples: int = 4096
    snis_samples: int = 4096
    chunk_size: int = 256
    repeats: int = 1
    seed: int = 0
    test_size: float = 0.30
    tabpfn_estimators: int = 4
    prior_modes: tuple[str, ...] = ("standard",)
    posterior_families: tuple[str, ...] = ("flow",)
    afin_variants: tuple[str, ...] = ("snis",)
    empirical_prior_C: float = 1.0
    empirical_prior_shrink: float = 0.5
    empirical_prior_loc_clip: float = 3.0
    empirical_prior_scale_multiplier: float = 2.0
    empirical_prior_scale_min: float = 0.45
    empirical_prior_scale_max: float = 1.25
    skip_tabpfn: bool = False
    skip_snis: bool = False
    add_intercept: bool = True


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def binary_metrics(probs, y):
    probs = torch.as_tensor(probs, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    y = torch.as_tensor(y, dtype=torch.float32)
    pred = (probs >= 0.5).float()
    acc = float((pred == y).float().mean().item())
    nll = float(torch.nn.functional.binary_cross_entropy(probs, y).item())
    try:
        auc = float(roc_auc_score(y.cpu().numpy(), probs.cpu().numpy()))
    except ValueError:
        auc = float("nan")
    return acc, nll, auc


def load_openml_binary(spec: DatasetSpec, *, seed: int, test_size: float, add_intercept: bool):
    dataset = fetch_openml(data_id=spec.data_id, as_frame=True, parser="auto")
    X = dataset.data.copy()
    y = dataset.target.copy()

    frame = X.copy()
    frame["__target__"] = y
    frame = frame.dropna(axis=0)
    y = frame.pop("__target__")
    X = pd.get_dummies(frame, dummy_na=False, dtype=np.float32)
    if X.shape[1] == 0:
        raise ValueError(f"{spec.name} has no usable features after preprocessing.")

    labels = pd.Series(y).astype("category").cat.categories.tolist()
    if len(labels) != 2:
        raise ValueError(f"{spec.name} is not binary after preprocessing: {labels}")
    y01 = pd.Series(y).astype("category").cat.codes.to_numpy(dtype=np.int64)
    X_np = X.to_numpy(dtype=np.float32)

    X_train_raw, X_test_raw, y_train_np, y_test_np = train_test_split(
        X_np,
        y01,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y01,
    )

    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    X_train_std = (X_train_raw - mean) / std
    X_test_std = (X_test_raw - mean) / std

    d_features = int(X_train_std.shape[1])
    scale = 2.0 / np.sqrt(max(1, d_features))
    X_train_model = scale * X_train_std
    X_test_model = scale * X_test_std
    if add_intercept:
        X_train_model = np.concatenate([np.ones((X_train_model.shape[0], 1), dtype=np.float32), X_train_model], axis=1)
        X_test_model = np.concatenate([np.ones((X_test_model.shape[0], 1), dtype=np.float32), X_test_model], axis=1)
    d_model = int(X_train_model.shape[1])
    return {
        "openml_id": spec.data_id,
        "name": spec.name,
        "d": d_model,
        "d_features": d_features,
        "add_intercept": bool(add_intercept),
        "N_train": int(X_train_std.shape[0]),
        "N_test": int(X_test_std.shape[0]),
        "X_train": torch.tensor(X_train_model, dtype=torch.float32),
        "X_test": torch.tensor(X_test_model, dtype=torch.float32),
        "y_train": torch.tensor(y_train_np, dtype=torch.float32),
        "y_test": torch.tensor(y_test_np, dtype=torch.float32),
        "X_train_tabpfn": X_train_std.astype(np.float32),
        "X_test_tabpfn": X_test_std.astype(np.float32),
        "y_train_tabpfn": y_train_np,
        "y_test_tabpfn": y_test_np,
    }


def empirical_logistic_prior(
    data,
    *,
    C: float,
    shrink: float,
    loc_clip: float,
    scale_multiplier: float,
    scale_min: float,
    scale_max: float,
):
    """Build a simple data-informed Normal prior for the logistic coefficients.

    This is empirical-Bayes style: fit a ridge logistic model on the training
    split, shrink its coefficients, and use a clipped Hessian standard-error
    estimate as the prior scale.  The clipping keeps the prior within the range
    where the public AFINs checkpoints tend to behave sensibly.
    """
    x = data["X_train"].numpy().astype(np.float64)
    y = data["y_train_tabpfn"].astype(np.int64)
    d = x.shape[1]

    clf = LogisticRegression(
        C=float(C),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        clf.fit(x, y)

    beta = np.nan_to_num(clf.coef_.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    loc = float(shrink) * np.clip(beta, -float(loc_clip), float(loc_clip))

    logits = np.clip(x @ beta, -30.0, 30.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    weights = np.clip(probs * (1.0 - probs), 1e-6, None)
    ridge = 1.0 / max(float(C), 1e-6)
    hessian = x.T @ (weights[:, None] * x) + ridge * np.eye(d)
    try:
        cov = np.linalg.solve(hessian + 1e-5 * np.eye(d), np.eye(d))
        se = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
    except np.linalg.LinAlgError:
        se = np.ones(d, dtype=np.float64)
    scale = np.clip(float(scale_multiplier) * se, float(scale_min), float(scale_max))

    return torch.tensor(loc, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32)


def make_prior(data, mode, args):
    if mode == "standard":
        return 0.0, 1.0
    if mode == "empirical":
        return empirical_logistic_prior(
            data,
            C=args.empirical_prior_C,
            shrink=args.empirical_prior_shrink,
            loc_clip=args.empirical_prior_loc_clip,
            scale_multiplier=args.empirical_prior_scale_multiplier,
            scale_min=args.empirical_prior_scale_min,
            scale_max=args.empirical_prior_scale_max,
        )
    raise ValueError(f"Unknown prior mode: {mode}")


def build_problem(data, prior_loc=0.0, prior_scale=1.0):
    return build_afin_problem(
        GaussianPrior(prior_loc, prior_scale),
        [BernoulliLogit(data["X_train"]).observe(data["y_train"])],
        name=f"OpenML {data['openml_id']} {data['name']}",
        data=data,
    )


def posterior_predictive(data, samples, weights=None):
    train_probs = torch.sigmoid(data["X_train"] @ samples.T)
    test_probs = torch.sigmoid(data["X_test"] @ samples.T)
    if weights is None:
        return train_probs.mean(dim=1), test_probs.mean(dim=1)
    weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    return train_probs @ weights, test_probs @ weights


def run_afin_method(label, model, data, problem, *, prior: str, samples: int, chunk_size: int, repeats: int, seed: int):
    kwargs = {}
    if model.posterior_family == "flow":
        kwargs = {"flow_samples": int(samples), "flow_batch_size": int(chunk_size)}

    torch.manual_seed(int(seed))
    _ = afin(model, problem, **kwargs)

    timings = []
    posterior = None
    for repeat in range(int(repeats)):
        torch.manual_seed(int(seed) + repeat + 1)
        sync_cuda()
        start = time.perf_counter()
        posterior = afin(model, problem, **kwargs)
        sync_cuda()
        timings.append(time.perf_counter() - start)

    assert posterior is not None
    z = posterior.sample(int(samples), seed=int(seed) + 7)
    train_probs, test_probs = posterior_predictive(data, z)
    train_acc, train_nll, train_auc = binary_metrics(train_probs, data["y_train"])
    test_acc, test_nll, test_auc = binary_metrics(test_probs, data["y_test"])
    return {
        "method": label,
        "prior": prior,
        "time_s": median(timings),
        "train_acc": train_acc,
        "train_nll": train_nll,
        "train_auc": train_auc,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "test_auc": test_auc,
        "ess_ratio": "",
    }


def run_snis(label, model, data, problem, *, prior: str, samples: int, chunk_size: int, seed: int):
    torch.manual_seed(int(seed))
    sync_cuda()
    start = time.perf_counter()
    z, log_p, log_q = afin_proposal_samples(
        model,
        problem,
        n=int(samples),
        seed=int(seed),
        chunk_size=int(chunk_size),
    )
    sync_cuda()
    elapsed = time.perf_counter() - start

    log_w = log_p - log_q
    weights = torch.softmax(log_w, dim=0).float()
    ess = float((1.0 / weights.square().sum()).item())

    train_probs, test_probs = posterior_predictive(data, z, weights=weights)
    train_acc, train_nll, train_auc = binary_metrics(train_probs, data["y_train"])
    test_acc, test_nll, test_auc = binary_metrics(test_probs, data["y_test"])
    return {
        "method": f"{label} + SNIS",
        "prior": prior,
        "time_s": elapsed,
        "train_acc": train_acc,
        "train_nll": train_nll,
        "train_auc": train_auc,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "test_auc": test_auc,
        "ess_ratio": ess / max(1, int(samples)),
    }


def tabpfn_factory(device: str, estimators: int, seed: int, warmup_data):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    def quiet_fit(clf, X, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Running on CPU with more than 200 samples.*")
            clf.fit(X, y)
        return clf

    def make_v26():
        return TabPFNClassifier(device=device, n_estimators=int(estimators), random_state=int(seed))

    def make_v2():
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device=device,
            n_estimators=int(estimators),
            random_state=int(seed),
        )

    X_warm = warmup_data["X_train_tabpfn"][:64]
    y_warm = warmup_data["y_train_tabpfn"][:64]
    try:
        quiet_fit(make_v26(), X_warm, y_warm)
        return "TabPFN v2.6", make_v26, quiet_fit
    except Exception as exc:
        message = str(exc)
        if "TABPFN_TOKEN" in message or "license" in message.lower():
            print("TabPFN v2.6 skipped: license/API token required. Falling back to public v2.")
        else:
            print(f"TabPFN v2.6 skipped: {type(exc).__name__}: {exc}. Falling back to public v2.")
    quiet_fit(make_v2(), X_warm, y_warm)
    return "TabPFN public v2", make_v2, quiet_fit


def run_tabpfn(label, make_clf, quiet_fit, data):
    clf = make_clf()
    start = time.perf_counter()
    quiet_fit(clf, data["X_train_tabpfn"], data["y_train_tabpfn"])
    train_proba = clf.predict_proba(data["X_train_tabpfn"])
    test_proba = clf.predict_proba(data["X_test_tabpfn"])
    elapsed = time.perf_counter() - start

    classes = list(getattr(clf, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else train_proba.shape[1] - 1
    train_probs = torch.tensor(train_proba[:, pos_idx], dtype=torch.float32)
    test_probs = torch.tensor(test_proba[:, pos_idx], dtype=torch.float32)
    train_acc, train_nll, train_auc = binary_metrics(train_probs, data["y_train"])
    test_acc, test_nll, test_auc = binary_metrics(test_probs, data["y_test"])
    return {
        "method": label,
        "prior": "n/a",
        "time_s": elapsed,
        "train_acc": train_acc,
        "train_nll": train_nll,
        "train_auc": train_auc,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "test_auc": test_auc,
        "ess_ratio": "",
    }


def with_dataset_fields(data, result):
    return {
        "openml_id": data["openml_id"],
        "dataset": data["name"],
        "d_features": data["d_features"],
        "d_model": data["d"],
        "intercept": data["add_intercept"],
        "N_train": data["N_train"],
        "N_test": data["N_test"],
        **result,
    }


def run_safe(data, method_name, fn, *, prior=""):
    try:
        return with_dataset_fields(data, fn())
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return with_dataset_fields(data, {
            "method": method_name,
            "prior": prior,
            "time_s": "",
            "train_acc": "",
            "train_nll": "",
            "train_auc": "",
            "test_acc": "",
            "test_nll": "",
            "test_auc": "",
            "ess_ratio": "",
            "error": f"CUDA OOM: {exc}",
        })
    except Exception as exc:  # Keep the full benchmark moving.
        return with_dataset_fields(data, {
            "method": method_name,
            "prior": prior,
            "time_s": "",
            "train_acc": "",
            "train_nll": "",
            "train_auc": "",
            "test_acc": "",
            "test_nll": "",
            "test_auc": "",
            "ess_ratio": "",
            "error": f"{type(exc).__name__}: {exc}",
        })


def print_markdown_table(rows):
    columns = [
        "dataset", "d_features", "d_model", "N_train", "method", "prior", "time_s",
        "test_acc", "test_nll", "test_auc", "ess_ratio", "error",
    ]
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if col == "time_s":
                    value = f"{value:.3f}"
                else:
                    value = f"{value:.4f}"
            values.append(str(value))
        print("| " + " | ".join(values) + " |")


def write_csv(rows, path):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "openml_id", "dataset", "d_features", "d_model", "intercept", "N_train", "N_test", "method", "prior", "time_s",
        "train_acc", "train_nll", "train_auc", "test_acc", "test_nll",
        "test_auc", "ess_ratio", "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Wrote {path}")


def selected_dataset_specs(dataset_ids):
    selected = set(int(data_id) for data_id in dataset_ids)
    specs = [spec for spec in DATASETS if spec.data_id in selected]
    if not specs:
        raise ValueError("No matching dataset specs selected.")
    return specs


def load_openml_suite(config: OpenMLBenchmarkConfig, *, verbose=True):
    if verbose:
        print("Loading OpenML datasets...")
    datasets = [
        load_openml_binary(
            spec,
            seed=config.seed,
            test_size=config.test_size,
            add_intercept=config.add_intercept,
        )
        for spec in selected_dataset_specs(config.dataset_ids)
    ]
    if verbose:
        for data in datasets:
            print(
                f"{data['name']}: id={data['openml_id']} "
                f"d_features={data['d_features']} d_model={data['d']} "
                f"N_train={data['N_train']} N_test={data['N_test']}"
            )
    return datasets


def load_openml_afin_models(config: OpenMLBenchmarkConfig, *, verbose=True):
    if verbose:
        print("\nLoading AFINs checkpoints...")
    families = tuple(config.posterior_families)
    root = download_pretrained_checkpoints(families=families)
    labels = {
        "gaussian": "AFINs Gaussian HF N256",
        "flow": "AFINs Flow HF N256",
    }
    models = [
        (labels[family], load_afin(root / PRETRAINED_CHECKPOINTS[family], weights="final"))
        for family in families
    ]
    if verbose:
        print("AFINs device:", next(models[0][1].parameters()).device)
    return models


def _allowed_afin_methods(config: OpenMLBenchmarkConfig):
    labels = {
        "gaussian": "AFINs Gaussian HF N256",
        "flow": "AFINs Flow HF N256",
    }
    methods = []
    variants = set(config.afin_variants)
    for family in config.posterior_families:
        label = labels[family]
        if "direct" in variants:
            methods.append(label)
        if "snis" in variants and not config.skip_snis:
            methods.append(f"{label} + SNIS")
    return set(methods)


def run_openml_prior_benchmark(
    config: OpenMLBenchmarkConfig | None = None,
    *,
    cache_path=None,
    use_cache=False,
    verbose=True,
):
    """Run or load the binary OpenML prior benchmark used by demo 04."""
    config = config or OpenMLBenchmarkConfig()
    cache_path = Path(cache_path) if cache_path else None
    if use_cache and cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        selected = set(int(data_id) for data_id in config.dataset_ids)
        cached = cached[cached["openml_id"].astype(int).isin(selected)].copy()
        if "prior" in cached and config.prior_modes:
            prior = cached["prior"].fillna("n/a")
            keep_prior = prior.isin(set(config.prior_modes) | {"n/a"})
            cached = cached[keep_prior].copy()
        if config.skip_tabpfn:
            cached = cached[~cached["method"].str.startswith("TabPFN", na=False)].copy()
        if config.skip_snis:
            cached = cached[~cached["method"].str.contains("+ SNIS", regex=False, na=False)].copy()
        allowed_afin = _allowed_afin_methods(config)
        is_tabpfn = cached["method"].str.startswith("TabPFN", na=False)
        cached = cached[is_tabpfn | cached["method"].isin(allowed_afin)].copy()
        return cached.reset_index(drop=True)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    datasets = load_openml_suite(config, verbose=verbose)
    afin_models = load_openml_afin_models(config, verbose=verbose)

    tabpfn = None
    if not config.skip_tabpfn:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print("Preparing TabPFN on", device)
        try:
            tabpfn = tabpfn_factory(device, config.tabpfn_estimators, config.seed, datasets[0])
            if verbose:
                print("TabPFN method:", tabpfn[0])
        except Exception as exc:
            if verbose:
                print("TabPFN skipped:", type(exc).__name__, exc)

    rows = []
    for data in datasets:
        if verbose:
            print(f"\n=== {data['name']} (d_model={data['d']}, N_train={data['N_train']}) ===")
        for prior_mode in config.prior_modes:
            prior_loc, prior_scale = make_prior(data, prior_mode, config)
            problem = build_problem(data, prior_loc=prior_loc, prior_scale=prior_scale)
            if verbose:
                print(f"--- prior={prior_mode} ---")

            if "direct" in set(config.afin_variants):
                for label, afin_model in afin_models:
                    row = run_safe(
                        data,
                        label,
                        lambda label=label, afin_model=afin_model, data=data, problem=problem, prior_mode=prior_mode: run_afin_method(
                            label,
                            afin_model,
                            data,
                            problem,
                            prior=prior_mode,
                            samples=config.samples,
                            chunk_size=config.chunk_size,
                            repeats=config.repeats,
                            seed=config.seed,
                        ),
                        prior=prior_mode,
                    )
                    rows.append(row)
                    if verbose:
                        _print_progress_row(label, row, prior_mode)

            if not config.skip_snis and "snis" in set(config.afin_variants):
                for label, afin_model in afin_models:
                    method_name = f"{label} + SNIS"
                    row = run_safe(
                        data,
                        method_name,
                        lambda label=label, afin_model=afin_model, data=data, problem=problem, prior_mode=prior_mode: run_snis(
                            label,
                            afin_model,
                            data,
                            problem,
                            prior=prior_mode,
                            samples=config.snis_samples,
                            chunk_size=config.chunk_size,
                            seed=config.seed + 100,
                        ),
                        prior=prior_mode,
                    )
                    rows.append(row)
                    if verbose:
                        _print_progress_row(method_name, row, prior_mode)

        if tabpfn is not None:
            label, make_clf, quiet_fit = tabpfn
            row = run_safe(data, label, lambda data=data: run_tabpfn(label, make_clf, quiet_fit, data), prior="n/a")
            rows.append(row)
            if verbose:
                _print_progress_row(label, row, "n/a")

    result = pd.DataFrame(rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        if verbose:
            print(f"Wrote {cache_path}")
    return result


def _print_progress_row(method_name, row, prior_mode):
    if row.get("error"):
        print(f"{method_name} [{prior_mode}]: ERROR {row['error']}")
        return
    ess = row.get("ess_ratio", "")
    ess_text = f", ESS={ess:.1%}" if isinstance(ess, float) else ""
    print(
        f"{method_name} [{row.get('prior', '')}]: test_acc={row['test_acc']:.3f}, "
        f"test_nll={row['test_nll']:.3f}, test_auc={row['test_auc']:.3f}, "
        f"time={row['time_s']:.2f}s{ess_text}"
    )


def _numeric_results(results):
    results = results.copy()
    for col in ["test_acc", "test_nll", "test_auc", "time_s", "ess_ratio"]:
        if col in results:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    return results


def best_afin_vs_tabpfn(results):
    results = _numeric_results(results)
    is_tabpfn = results["method"].str.startswith("TabPFN", na=False)
    afin = results[~is_tabpfn & results["test_acc"].notna()]
    tabpfn = results[is_tabpfn & results["test_acc"].notna()]
    if afin.empty or tabpfn.empty:
        return pd.DataFrame()

    best_afin = afin.loc[afin.groupby("dataset")["test_acc"].idxmax()].set_index("dataset")
    tabpfn_best = tabpfn.loc[tabpfn.groupby("dataset")["test_acc"].idxmax()].set_index("dataset")
    comparison = pd.DataFrame({
        "best_afin_method": best_afin["method"],
        "best_afin_prior": best_afin["prior"],
        "best_afin_acc": best_afin["test_acc"],
        "best_afin_nll": best_afin["test_nll"],
        "tabpfn_acc": tabpfn_best["test_acc"],
        "tabpfn_nll": tabpfn_best["test_nll"],
    }).dropna()
    comparison["acc_delta_vs_tabpfn"] = comparison["best_afin_acc"] - comparison["tabpfn_acc"]
    comparison["nll_delta_vs_tabpfn"] = comparison["best_afin_nll"] - comparison["tabpfn_nll"]
    return comparison.sort_values("acc_delta_vs_tabpfn", ascending=False)


def prior_effect_table(results):
    results = _numeric_results(results)
    afin = results[~results["method"].str.startswith("TabPFN", na=False) & results["test_acc"].notna()]
    standard = afin[afin["prior"] == "standard"]
    empirical = afin[afin["prior"] == "empirical"]
    if standard.empty or empirical.empty:
        return pd.DataFrame()

    best_standard = standard.loc[standard.groupby("dataset")["test_acc"].idxmax()].set_index("dataset")
    best_empirical = empirical.loc[empirical.groupby("dataset")["test_acc"].idxmax()].set_index("dataset")
    effect = pd.DataFrame({
        "standard_method": best_standard["method"],
        "standard_acc": best_standard["test_acc"],
        "standard_nll": best_standard["test_nll"],
        "empirical_method": best_empirical["method"],
        "empirical_acc": best_empirical["test_acc"],
        "empirical_nll": best_empirical["test_nll"],
    }).dropna()
    effect["acc_delta"] = effect["empirical_acc"] - effect["standard_acc"]
    effect["nll_delta"] = effect["empirical_nll"] - effect["standard_nll"]
    return effect.sort_values("acc_delta", ascending=False)


def plot_prior_benchmark(results, prior_effect=None, *, figsize=(12.0, 4.6)):
    import matplotlib.pyplot as plt

    results = _numeric_results(results)
    prior_effect = prior_effect_table(results) if prior_effect is None else prior_effect
    comparison = best_afin_vs_tabpfn(results)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_df = prior_effect.sort_values("acc_delta")
    colors = np.where(plot_df["acc_delta"] >= 0, "#4374B3", "#C44E52")
    axes[0].barh(plot_df.index, 100.0 * plot_df["acc_delta"], color=colors)
    axes[0].axvline(0.0, color="0.25", lw=1)
    axes[0].set_xlabel("accuracy delta: empirical prior - standard prior (pp)")
    axes[0].set_title("Task-specific prior effect")
    axes[0].grid(axis="x", alpha=0.2)

    comparison_df = comparison.sort_values("acc_delta_vs_tabpfn")
    colors = np.where(comparison_df["acc_delta_vs_tabpfn"] >= 0, "#4374B3", "#C44E52")
    axes[1].barh(comparison_df.index, 100.0 * comparison_df["acc_delta_vs_tabpfn"], color=colors)
    axes[1].axvline(0.0, color="0.25", lw=1)
    axes[1].set_xlabel("accuracy delta: best AFINs - TabPFN (pp)")
    axes[1].set_title("Selected AFINs vs TabPFN")
    axes[1].grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig, axes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", type=int, default=[spec.data_id for spec in DATASETS])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--snis-samples", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--tabpfn-estimators", type=int, default=4)
    parser.add_argument("--prior-modes", nargs="*", default=["standard"], choices=["standard", "empirical"])
    parser.add_argument("--posterior-families", nargs="*", default=["flow"], choices=["gaussian", "flow"])
    parser.add_argument("--afin-variants", nargs="*", default=["snis"], choices=["direct", "snis"])
    parser.add_argument("--empirical-prior-C", type=float, default=1.0)
    parser.add_argument("--empirical-prior-shrink", type=float, default=0.5)
    parser.add_argument("--empirical-prior-loc-clip", type=float, default=3.0)
    parser.add_argument("--empirical-prior-scale-multiplier", type=float, default=2.0)
    parser.add_argument("--empirical-prior-scale-min", type=float, default=0.45)
    parser.add_argument("--empirical-prior-scale-max", type=float, default=1.25)
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--skip-snis", action="store_true")
    parser.add_argument("--no-intercept", action="store_true", help="Do not add a Bayesian logistic-regression intercept column for AFINs.")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OpenMLBenchmarkConfig(
        dataset_ids=tuple(args.datasets),
        samples=args.samples,
        snis_samples=args.snis_samples,
        chunk_size=args.chunk_size,
        repeats=args.repeats,
        seed=args.seed,
        test_size=args.test_size,
        tabpfn_estimators=args.tabpfn_estimators,
        prior_modes=tuple(args.prior_modes),
        posterior_families=tuple(args.posterior_families),
        afin_variants=tuple(args.afin_variants),
        empirical_prior_C=args.empirical_prior_C,
        empirical_prior_shrink=args.empirical_prior_shrink,
        empirical_prior_loc_clip=args.empirical_prior_loc_clip,
        empirical_prior_scale_multiplier=args.empirical_prior_scale_multiplier,
        empirical_prior_scale_min=args.empirical_prior_scale_min,
        empirical_prior_scale_max=args.empirical_prior_scale_max,
        skip_tabpfn=args.skip_tabpfn,
        skip_snis=args.skip_snis,
        add_intercept=not args.no_intercept,
    )
    result = run_openml_prior_benchmark(config, cache_path=args.output or None, verbose=True)
    rows = result.to_dict("records")

    print("\nSummary")
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
