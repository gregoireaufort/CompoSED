"""Analytic Gaussian conditional validation for experimental diffusion SBI.

This is the first validation rung for the conditional diffusion model.  The
joint data distribution is known exactly, so every conditional posterior can be
computed analytically.  That lets us test the diffusion machinery before any
SED physics, catalog cuts, or simulator mismatch enters the problem.

Feature vector:

    [phot_1, phot_2, phot_3, theta_1, theta_2]

The diffusion model trains on joint samples and is asked to sample arbitrary
unknown coordinates while clamping known coordinates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/composed_mplconfig")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inftools.diagnostics import marginal_coverage, rank_statistics
from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata


def make_joint_gaussian() -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed five-dimensional Gaussian with useful correlations."""

    mean = np.array([0.2, -0.3, 0.5, 0.1, -0.2], dtype=float)
    loadings = np.array(
        [
            [1.00, 0.25, -0.10],
            [0.80, 0.10, 0.20],
            [-0.25, 0.95, 0.35],
            [0.90, -0.45, 0.10],
            [-0.45, 0.75, 0.35],
        ],
        dtype=float,
    )
    unique_variance = np.array([0.18, 0.16, 0.22, 0.14, 0.17], dtype=float)
    cov = loadings @ loadings.T + np.diag(unique_variance)
    return mean, cov


def gaussian_conditional(
    mean: np.ndarray,
    cov: np.ndarray,
    known_index: np.ndarray,
    known_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Gaussian conditional for unknown coordinates.

    Returns ``(unknown_index, conditional_mean, conditional_covariance)`` for
    ``x_unknown | x_known = known_value``.
    """

    all_index = np.arange(mean.size)
    known_index = np.asarray(known_index, dtype=int)
    unknown_index = np.array([i for i in all_index if i not in set(known_index)], dtype=int)

    mu_known = mean[known_index]
    mu_unknown = mean[unknown_index]
    cov_kk = cov[np.ix_(known_index, known_index)]
    cov_uk = cov[np.ix_(unknown_index, known_index)]
    cov_ku = cov[np.ix_(known_index, unknown_index)]
    cov_uu = cov[np.ix_(unknown_index, unknown_index)]

    solve_known = np.linalg.solve(cov_kk, np.asarray(known_value, dtype=float) - mu_known)
    conditional_mean = mu_unknown + cov_uk @ solve_known
    conditional_cov = cov_uu - cov_uk @ np.linalg.solve(cov_kk, cov_ku)
    conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
    return unknown_index, conditional_mean, conditional_cov


def central_interval_coverage(samples: np.ndarray, truth: np.ndarray, level: float = 0.68) -> np.ndarray:
    """Marginal central-interval coverage for one posterior sample cube."""

    low_q = 0.5 * (1.0 - level)
    high_q = 1.0 - low_q
    lo = np.quantile(samples, low_q, axis=1)
    hi = np.quantile(samples, high_q, axis=1)
    return np.mean((truth >= lo) & (truth <= hi), axis=0)


def evaluate_case(
    estimator: ConditionalDiffusionEstimator,
    x_test: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    known_index: list[int],
    *,
    case_name: str,
    num_samples: int = 512,
    steps: int = 40,
    sampler: str = "edm_euler",
) -> dict:
    """Sample one conditional case and compare to analytic Gaussian truth."""

    known_index_arr = np.asarray(known_index, dtype=int)
    mask = np.zeros((x_test.shape[0], x_test.shape[1]), dtype=bool)
    mask[:, known_index_arr] = True
    known = np.full_like(x_test, np.nan)
    known[:, known_index_arr] = x_test[:, known_index_arr]

    t0 = time.perf_counter()
    samples_full = estimator.sample(
        known,
        mask,
        num_samples=num_samples,
        steps=steps,
        sampler=sampler,
        batch_size=min(64, x_test.shape[0]),
    )
    seconds = time.perf_counter() - t0

    unknown_index, _, _ = gaussian_conditional(mean, cov, known_index_arr, x_test[0, known_index_arr])
    samples_unknown = samples_full[:, :, unknown_index]
    truth_unknown = x_test[:, unknown_index]

    conditional_means = []
    conditional_covs = []
    for row in range(x_test.shape[0]):
        _, cond_mean, cond_cov = gaussian_conditional(mean, cov, known_index_arr, x_test[row, known_index_arr])
        conditional_means.append(cond_mean)
        conditional_covs.append(cond_cov)
    conditional_means = np.asarray(conditional_means)
    conditional_covs = np.asarray(conditional_covs)

    sample_means = np.mean(samples_unknown, axis=1)
    sample_covs = np.asarray([np.cov(samples_unknown[row].T) for row in range(samples_unknown.shape[0])])

    mean_rmse = float(np.sqrt(np.mean((sample_means - conditional_means) ** 2)))
    diag_true = np.asarray([np.diag(cov_row) for cov_row in conditional_covs])
    diag_sample = np.asarray([np.diag(cov_row) for cov_row in sample_covs])
    std_relative_error = float(
        np.mean(
            np.abs(np.sqrt(np.maximum(diag_sample, 0.0)) - np.sqrt(np.maximum(diag_true, 0.0)))
            / np.sqrt(np.maximum(diag_true, 1e-12))
        )
    )

    known_max_abs_error = float(np.max(np.abs(samples_full[:, :, known_index_arr] - x_test[:, None, known_index_arr])))
    coverage_68 = central_interval_coverage(samples_unknown, truth_unknown, level=0.68)
    ranks = rank_statistics(samples_unknown, truth_unknown)
    coverage_curve = marginal_coverage(samples_unknown, truth_unknown, levels=np.array([0.5, 0.68, 0.9]))

    return {
        "case_name": case_name,
        "known_index": known_index_arr.tolist(),
        "unknown_index": unknown_index.tolist(),
        "seconds": seconds,
        "samples_full": samples_full,
        "samples_unknown": samples_unknown,
        "truth_unknown": truth_unknown,
        "conditional_means": conditional_means,
        "conditional_covs": conditional_covs,
        "sample_means": sample_means,
        "sample_covs": sample_covs,
        "mean_rmse": mean_rmse,
        "std_relative_error": std_relative_error,
        "known_max_abs_error": known_max_abs_error,
        "coverage_68": coverage_68,
        "rank_percentiles": ranks["rank_percentiles"],
        "coverage_curve": coverage_curve,
    }


def plot_training_loss(history: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(history["train_loss"], color="C0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("diffusion score loss")
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close(fig)


def plot_case_summary(case: dict, feature_names: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    unknown_names = [feature_names[i] for i in case["unknown_index"]]
    truth_mean = case["conditional_means"].reshape(-1)
    sample_mean = case["sample_means"].reshape(-1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    ax.scatter(truth_mean, sample_mean, s=18, alpha=0.7)
    lo = min(float(np.min(truth_mean)), float(np.min(sample_mean)))
    hi = max(float(np.max(truth_mean)), float(np.max(sample_mean)))
    ax.plot([lo, hi], [lo, hi], color="0.4", ls="--")
    ax.set_xlabel("analytic conditional mean")
    ax.set_ylabel("diffusion sample mean")
    ax.set_title(f"{case['case_name']}: mean check")

    ax = axes[1]
    for j, name in enumerate(unknown_names):
        ax.hist(case["rank_percentiles"][:, j], bins=np.linspace(0.0, 1.0, 11), alpha=0.45, label=name)
    ax.set_xlabel("rank percentile of true value")
    ax.set_ylabel("count")
    ax.set_title("rank histogram")
    ax.legend(fontsize=8)

    ax = axes[2]
    curve = case["coverage_curve"]
    levels = curve["levels"]
    coverage = curve["coverage"]
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.4", ls="--", label="ideal")
    for j, name in enumerate(unknown_names):
        ax.plot(levels, coverage[:, j], marker="o", label=name)
    ax.set_xlabel("central credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("coverage")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"{case['case_name']}_summary.png", dpi=160)
    plt.close(fig)


def plot_inverse_posterior(case: dict, feature_names: list[str], output_dir: Path) -> None:
    """Plot one 2D inverse posterior against the analytic conditional ellipse."""

    import matplotlib.pyplot as plt

    if len(case["unknown_index"]) != 2:
        return

    samples = case["samples_unknown"][0]
    mean = case["conditional_means"][0]
    cov = case["conditional_covs"][0]
    truth = case["truth_unknown"][0]
    names = [feature_names[i] for i in case["unknown_index"]]

    evals, evecs = np.linalg.eigh(cov)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.22, label="diffusion samples")
    for nsig, label, color in [(1.0, "analytic 1 sigma", "C1"), (2.0, "analytic 2 sigma", "C3")]:
        ellipse = mean[:, None] + evecs @ (np.sqrt(np.maximum(evals, 0.0))[:, None] * nsig * circle)
        ax.plot(ellipse[0], ellipse[1], color=color, lw=1.5, label=label)
    ax.plot(truth[0], truth[1], "x", color="k", ms=8, label="held-out truth")
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Inverse conditional posterior for one held-out object")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "inverse_posterior_ellipse.png", dpi=160)
    plt.close(fig)


def serializable_metrics(cases: list[dict], feature_names: list[str], history: dict, device: str) -> dict:
    """Strip large sample arrays and return JSON-friendly metrics."""

    case_metrics = {}
    for case in cases:
        unknown_names = [feature_names[i] for i in case["unknown_index"]]
        case_metrics[case["case_name"]] = {
            "known_features": [feature_names[i] for i in case["known_index"]],
            "unknown_features": unknown_names,
            "seconds": float(case["seconds"]),
            "mean_rmse": float(case["mean_rmse"]),
            "std_relative_error": float(case["std_relative_error"]),
            "known_max_abs_error": float(case["known_max_abs_error"]),
            "coverage_68": {
                name: float(value) for name, value in zip(unknown_names, case["coverage_68"])
            },
        }
    return {
        "device": str(device),
        "final_train_loss": float(history["train_loss"][-1]),
        "n_epochs": int(len(history["train_loss"])),
        "cases": case_metrics,
    }


def run_validation(
    *,
    n_train: int = 4096,
    n_test: int = 48,
    epochs: int = 80,
    num_samples: int = 512,
    steps: int = 40,
    seed: int = 20260706,
    output_dir: str | Path = "outputs/diffusion_gaussian_conditionals",
) -> dict:
    """Train diffusion on a known Gaussian and evaluate analytic conditionals."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    mean, cov = make_joint_gaussian()
    x_train = rng.multivariate_normal(mean, cov, size=n_train)
    x_test = rng.multivariate_normal(mean, cov, size=n_test)

    feature_names = ["phot_1", "phot_2", "phot_3", "theta_1", "theta_2"]
    metadata = FeatureMetadata.from_groups(
        {
            "mags": feature_names[:3],
            "params": feature_names[3:],
        }
    )

    estimator = ConditionalDiffusionEstimator(
        metadata,
        model="mlp",
        hidden_features=96,
        model_config={"mlp_blocks": 3, "emb_dim": 64, "time_hidden": 96},
        sigma_min=0.02,
        sigma_max=2.0,
        learning_rate=2e-3,
        device="auto",
    )
    history = estimator.fit(
        x_train,
        mask_config={
            "curriculum": [
                {"weight": 1.0, "unknown_fraction": {"mags": 0.0, "params": 1.0}},
                {"weight": 1.0, "unknown_fraction": {"mags": 1.0, "params": 0.0}},
                {"weight": 1.0, "unknown_fraction": {"mags": [0.33, 0.67], "params": [0.5, 1.0]}},
            ]
        },
        epochs=epochs,
        batch_size=512,
        seed=seed,
        clamp_known_in_xt=True,
        loss_on_unknown_only=True,
        verbose=True,
    )

    cases = [
        evaluate_case(
            estimator,
            x_test,
            mean,
            cov,
            [0, 1, 2],
            case_name="inverse_theta_given_photometry",
            num_samples=num_samples,
            steps=steps,
        ),
        evaluate_case(
            estimator,
            x_test,
            mean,
            cov,
            [3, 4],
            case_name="forward_photometry_given_theta",
            num_samples=num_samples,
            steps=steps,
        ),
        evaluate_case(
            estimator,
            x_test,
            mean,
            cov,
            [0, 3],
            case_name="mixed_inpainting",
            num_samples=num_samples,
            steps=steps,
        ),
    ]

    plot_training_loss(history, output_path)
    for case in cases:
        plot_case_summary(case, feature_names, output_path)
    plot_inverse_posterior(cases[0], feature_names, output_path)

    metrics = serializable_metrics(cases, feature_names, history, estimator.device)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # These checks are deliberately broad.  They catch broken masking,
    # non-finite samples, or a wildly wrong conditional model without pretending
    # this short smoke run is a final calibration proof.
    if max(case["known_max_abs_error"] for case in cases) > 1e-5:
        raise RuntimeError("Known/clamped coordinates were not preserved exactly.")
    if not all(np.isfinite(case["samples_full"]).all() for case in cases):
        raise RuntimeError("Diffusion samples contain NaN or inf.")
    if cases[0]["mean_rmse"] > 0.75:
        raise RuntimeError("Inverse Gaussian conditional mean error is unexpectedly large.")

    return metrics


def main() -> None:
    metrics = run_validation()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
