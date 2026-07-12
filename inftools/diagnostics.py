"""Diagnostics for simulation-based inference posterior estimators.

The functions in this module are deliberately NumPy-facing.  They do not know
whether posterior samples came from a MAF, a diffusion model, a saved catalog,
or another sampler.  The only convention is that posterior samples are arranged
as ``(n_objects, n_samples, n_parameters)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PosteriorSampleSet:
    """Posterior samples and optional truth arrays used by SBI diagnostics.

    Parameters
    ----------
    samples:
        Posterior samples with shape ``(n_objects, n_samples, n_parameters)``.
    theta_true:
        Optional true parameters with shape ``(n_objects, n_parameters)``.
    x:
        Optional conditioning vectors with shape ``(n_objects, n_features)``.
    theta_names:
        Optional parameter names in the same order as the final sample axis.
    metadata:
        Free-form provenance.  Keep this small and JSON-like if the diagnostics
        will be written to disk.
    """

    samples: np.ndarray
    theta_true: np.ndarray | None = None
    x: np.ndarray | None = None
    theta_names: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = _as_samples_object_first(self.samples)
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("samples contain NaN or inf values.")

        n_objects, _, n_parameters = self.samples.shape
        if self.theta_true is not None:
            self.theta_true = _as_2d(self.theta_true, "theta_true")
            if self.theta_true.shape != (n_objects, n_parameters):
                raise ValueError(
                    "theta_true must have shape "
                    f"({n_objects}, {n_parameters}); got {self.theta_true.shape}."
                )
            if not np.all(np.isfinite(self.theta_true)):
                raise ValueError("theta_true contains NaN or inf values.")

        if self.x is not None:
            self.x = _as_2d(self.x, "x")
            if self.x.shape[0] != n_objects:
                raise ValueError("x must have the same number of rows as samples.")

        if self.theta_names is not None:
            self.theta_names = tuple(str(name) for name in self.theta_names)
            if len(self.theta_names) != n_parameters:
                raise ValueError("theta_names length must match samples.shape[2].")

    @property
    def n_objects(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[1])

    @property
    def n_parameters(self) -> int:
        return int(self.samples.shape[2])


def sample_posterior_dataset(
    estimator,
    x_test: np.ndarray,
    num_samples: int,
    batch_size: int | None = None,
) -> np.ndarray:
    """Draw posterior samples for a batch of conditioning vectors.

    ``estimator`` must expose ``sample(x_obs, num_samples=...)``.  Different
    estimators return single-object samples differently, so this function
    normalizes the result to ``(n_objects, n_samples, n_parameters)``.
    """

    x_arr = _as_2d(x_test, "x_test")
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    n_objects = x_arr.shape[0]
    if batch_size is None:
        raw = estimator.sample(x_arr, num_samples=num_samples)
        return _as_samples_object_first(raw, n_objects=n_objects)

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    chunks = []
    for start in range(0, n_objects, batch_size):
        end = min(start + batch_size, n_objects)
        raw = estimator.sample(x_arr[start:end], num_samples=num_samples)
        chunks.append(_as_samples_object_first(raw, n_objects=end - start))
    return np.concatenate(chunks, axis=0)


def rank_statistics(samples: np.ndarray, theta_true: np.ndarray) -> dict[str, np.ndarray]:
    """Return marginal posterior ranks of the true parameters.

    For each object and parameter, the rank is the number of posterior samples
    below the true value.  A calibrated posterior should give approximately
    uniform rank percentiles over many simulated test objects.
    """

    sample_set = PosteriorSampleSet(samples=samples, theta_true=theta_true)
    ranks = np.sum(sample_set.samples < sample_set.theta_true[:, None, :], axis=1)
    percentiles = ranks / float(sample_set.n_samples)
    return {
        "ranks": ranks.astype(int),
        "rank_percentiles": percentiles,
        "n_samples": np.full(sample_set.n_parameters, sample_set.n_samples, dtype=int),
    }


def marginal_coverage(
    samples: np.ndarray,
    theta_true: np.ndarray,
    levels: np.ndarray | list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute central-interval marginal coverage curves.

    ``coverage[level, parameter]`` is the fraction of objects whose true
    parameter lies inside the central posterior interval with probability
    ``level``.
    """

    sample_set = PosteriorSampleSet(samples=samples, theta_true=theta_true)
    if levels is None:
        levels_arr = np.linspace(0.05, 0.95, 19)
    else:
        levels_arr = np.asarray(levels, dtype=float)
    if levels_arr.ndim != 1 or np.any((levels_arr <= 0.0) | (levels_arr >= 1.0)):
        raise ValueError("levels must be one-dimensional values strictly between 0 and 1.")

    coverage = np.empty((levels_arr.size, sample_set.n_parameters), dtype=float)
    for i, level in enumerate(levels_arr):
        low_q = 0.5 * (1.0 - float(level))
        high_q = 1.0 - low_q
        lo = np.quantile(sample_set.samples, low_q, axis=1)
        hi = np.quantile(sample_set.samples, high_q, axis=1)
        inside = (sample_set.theta_true >= lo) & (sample_set.theta_true <= hi)
        coverage[i] = np.mean(inside, axis=0)

    return {
        "levels": levels_arr,
        "coverage": coverage,
        "mean_coverage": np.mean(coverage, axis=1),
    }


def prediction_summary(samples: np.ndarray, theta_true: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Summarize posterior location, scale, and residuals when truth is known."""

    sample_set = PosteriorSampleSet(samples=samples, theta_true=theta_true)
    median = np.median(sample_set.samples, axis=1)
    mean = np.mean(sample_set.samples, axis=1)
    std = np.std(sample_set.samples, axis=1)
    q16 = np.quantile(sample_set.samples, 0.16, axis=1)
    q84 = np.quantile(sample_set.samples, 0.84, axis=1)
    out = {"median": median, "mean": mean, "std": std, "q16": q16, "q84": q84}
    if sample_set.theta_true is not None:
        residual = median - sample_set.theta_true
        out["residual_median"] = residual
        out["pull_median"] = residual / np.where(std > 0.0, std, np.nan)
    return out


def run_sbi_diagnostics(
    estimator=None,
    x_test: np.ndarray | None = None,
    theta_true: np.ndarray | None = None,
    *,
    posterior_samples: np.ndarray | None = None,
    num_samples: int = 1000,
    batch_size: int | None = None,
    theta_names: list[str] | tuple[str, ...] | None = None,
    levels: np.ndarray | list[float] | None = None,
    output_dir: str | Path | None = None,
    make_plots: bool = True,
    use_tarp: bool = False,
) -> dict[str, Any]:
    """Run a compact SBI diagnostic suite.

    Either pass ``posterior_samples`` directly, or pass an ``estimator`` and
    ``x_test`` so samples can be drawn here.  The diagnostics returned are pure
    arrays; plots and `.npz`/`.json` files are optional side products.
    """

    if posterior_samples is None:
        if estimator is None or x_test is None:
            raise ValueError("Provide posterior_samples, or provide both estimator and x_test.")
        posterior_samples = sample_posterior_dataset(estimator, x_test, num_samples=num_samples, batch_size=batch_size)

    sample_set = PosteriorSampleSet(
        samples=posterior_samples,
        theta_true=theta_true,
        x=x_test,
        theta_names=None if theta_names is None else tuple(theta_names),
        metadata={"num_samples": int(num_samples)},
    )

    results: dict[str, Any] = {
        "sample_set": sample_set,
        "summary": prediction_summary(sample_set.samples, sample_set.theta_true),
    }
    if sample_set.theta_true is not None:
        results["ranks"] = rank_statistics(sample_set.samples, sample_set.theta_true)
        results["coverage"] = marginal_coverage(sample_set.samples, sample_set.theta_true, levels=levels)
    if use_tarp:
        if sample_set.theta_true is None:
            raise ValueError("TARP diagnostics require theta_true.")
        results["tarp"] = tarp_coverage(sample_set.samples, sample_set.theta_true)

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        _write_diagnostic_arrays(path, sample_set, results)
        if make_plots:
            _write_diagnostic_plots(path, sample_set, results)

    return results


def tarp_coverage(samples: np.ndarray, theta_true: np.ndarray, **kwargs) -> Any:
    """Run optional TARP coverage diagnostics when the package is installed."""

    try:
        import tarp
    except ImportError as exc:
        raise ImportError(
            "TARP diagnostics require the optional tarp package. "
            "Install the diagnostics extra or install tarp in this environment."
        ) from exc

    if not hasattr(tarp, "get_tarp_coverage"):
        raise ImportError("The installed tarp package does not expose get_tarp_coverage.")

    sample_set = PosteriorSampleSet(samples=samples, theta_true=theta_true)
    rng = np.random.default_rng(int(kwargs.pop("seed", 0)))
    lo = np.min(sample_set.samples, axis=(0, 1))
    hi = np.max(sample_set.samples, axis=(0, 1))
    references = rng.uniform(lo, hi, size=(sample_set.n_objects, sample_set.n_parameters))
    return tarp.get_tarp_coverage(sample_set.samples, sample_set.theta_true, references, **kwargs)


def plot_single_posterior(
    samples: np.ndarray,
    theta_true: np.ndarray | None = None,
    theta_names: list[str] | tuple[str, ...] | None = None,
    object_index: int = 0,
):
    """Plot one object's posterior as diagonal histograms and lower hex bins."""

    plt = _require_matplotlib()
    sample_set = PosteriorSampleSet(samples=samples, theta_true=theta_true, theta_names=theta_names)
    idx = int(object_index)
    if idx < 0 or idx >= sample_set.n_objects:
        raise IndexError("object_index is outside the sample set.")

    draws = sample_set.samples[idx]
    names = _parameter_names(sample_set)
    npar = sample_set.n_parameters
    fig, axes = plt.subplots(npar, npar, figsize=(2.2 * npar, 2.2 * npar), squeeze=False)

    for row in range(npar):
        for col in range(npar):
            ax = axes[row, col]
            if row == col:
                ax.hist(draws[:, col], bins=32, histtype="stepfilled", alpha=0.45, color="C0")
                if sample_set.theta_true is not None:
                    ax.axvline(sample_set.theta_true[idx, col], color="k", lw=1.5)
            elif row > col:
                ax.hexbin(draws[:, col], draws[:, row], gridsize=35, mincnt=1, cmap="Blues")
                if sample_set.theta_true is not None:
                    ax.plot(sample_set.theta_true[idx, col], sample_set.theta_true[idx, row], "x", color="k")
            else:
                ax.axis("off")
            if row == npar - 1:
                ax.set_xlabel(names[col])
            if col == 0 and row > 0:
                ax.set_ylabel(names[row])
    fig.tight_layout()
    return fig


def plot_rank_histograms(ranks: dict[str, np.ndarray] | np.ndarray, theta_names: list[str] | tuple[str, ...] | None = None):
    """Plot marginal rank histograms."""

    plt = _require_matplotlib()
    ranks_arr = ranks["rank_percentiles"] if isinstance(ranks, dict) else np.asarray(ranks, dtype=float)
    if ranks_arr.ndim != 2:
        raise ValueError("rank percentiles must have shape (n_objects, n_parameters).")
    names = _names_from_count(ranks_arr.shape[1], theta_names)
    fig, axes = plt.subplots(1, ranks_arr.shape[1], figsize=(3.0 * ranks_arr.shape[1], 2.8), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[0, i]
        ax.hist(ranks_arr[:, i], bins=np.linspace(0.0, 1.0, 11), histtype="stepfilled", alpha=0.5)
        ax.set_title(name)
        ax.set_xlabel("rank percentile")
        ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_coverage_curve(coverage: dict[str, np.ndarray], theta_names: list[str] | tuple[str, ...] | None = None):
    """Plot requested central credible level against empirical coverage."""

    plt = _require_matplotlib()
    levels = np.asarray(coverage["levels"], dtype=float)
    cov = np.asarray(coverage["coverage"], dtype=float)
    names = _names_from_count(cov.shape[1], theta_names)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.5", ls="--", label="ideal")
    for i, name in enumerate(names):
        ax.plot(levels, cov[:, i], marker="o", ms=3, label=name)
    ax.set_xlabel("central credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_prediction_scatter(
    samples: np.ndarray,
    theta_true: np.ndarray,
    theta_names: list[str] | tuple[str, ...] | None = None,
    statistic: str = "median",
):
    """Plot posterior point summaries against true parameters."""

    plt = _require_matplotlib()
    summary = prediction_summary(samples, theta_true)
    point = np.asarray(summary[statistic], dtype=float)
    truth = _as_2d(theta_true, "theta_true")
    names = _names_from_count(truth.shape[1], theta_names)
    fig, axes = plt.subplots(1, truth.shape[1], figsize=(3.2 * truth.shape[1], 3.0), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[0, i]
        ax.scatter(truth[:, i], point[:, i], s=8, alpha=0.6)
        lo = min(np.min(truth[:, i]), np.min(point[:, i]))
        hi = max(np.max(truth[:, i]), np.max(point[:, i]))
        ax.plot([lo, hi], [lo, hi], color="0.4", ls="--")
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"posterior {statistic}")
    fig.tight_layout()
    return fig


def _as_samples_object_first(samples: np.ndarray, n_objects: int | None = None) -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if n_objects is not None:
            if arr.shape[0] == n_objects:
                pass
            elif arr.shape[1] == n_objects:
                arr = np.transpose(arr, (1, 0, 2))
            else:
                raise ValueError(
                    "Could not align posterior samples with n_objects="
                    f"{n_objects}; got samples shape {arr.shape}."
                )
    else:
        raise ValueError("samples must have shape (n_samples, n_parameters) or (n_objects, n_samples, n_parameters).")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError("samples must have non-empty object, sample, and parameter axes.")
    return arr


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {arr.shape}.")
    return arr


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Diagnostic plotting requires matplotlib.") from exc
    return plt


def _parameter_names(sample_set: PosteriorSampleSet) -> tuple[str, ...]:
    return _names_from_count(sample_set.n_parameters, sample_set.theta_names)


def _names_from_count(n: int, names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if names is None:
        return tuple(f"theta_{i}" for i in range(int(n)))
    names_tuple = tuple(str(name) for name in names)
    if len(names_tuple) != int(n):
        raise ValueError("name count does not match parameter dimension.")
    return names_tuple


def _write_diagnostic_arrays(path: Path, sample_set: PosteriorSampleSet, results: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {"samples": sample_set.samples}
    if sample_set.theta_true is not None:
        arrays["theta_true"] = sample_set.theta_true
    if sample_set.x is not None:
        arrays["x"] = sample_set.x
    if "ranks" in results:
        arrays["rank_percentiles"] = results["ranks"]["rank_percentiles"]
    if "coverage" in results:
        arrays["coverage_levels"] = results["coverage"]["levels"]
        arrays["coverage"] = results["coverage"]["coverage"]
    np.savez(path / "posterior_diagnostics.npz", **arrays)

    metadata = {
        "n_objects": sample_set.n_objects,
        "n_samples": sample_set.n_samples,
        "n_parameters": sample_set.n_parameters,
        "theta_names": sample_set.theta_names,
        "metadata": sample_set.metadata,
        "has_truth": sample_set.theta_true is not None,
    }
    with (path / "posterior_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _write_diagnostic_plots(path: Path, sample_set: PosteriorSampleSet, results: dict[str, Any]) -> None:
    fig = plot_single_posterior(sample_set.samples, sample_set.theta_true, sample_set.theta_names)
    fig.savefig(path / "single_posterior.png", dpi=150)
    _close_figure(fig)
    if "ranks" in results:
        fig = plot_rank_histograms(results["ranks"], sample_set.theta_names)
        fig.savefig(path / "rank_histograms.png", dpi=150)
        _close_figure(fig)
    if "coverage" in results:
        fig = plot_coverage_curve(results["coverage"], sample_set.theta_names)
        fig.savefig(path / "coverage_curve.png", dpi=150)
        _close_figure(fig)
    if sample_set.theta_true is not None:
        fig = plot_prediction_scatter(sample_set.samples, sample_set.theta_true, sample_set.theta_names)
        fig.savefig(path / "prediction_scatter.png", dpi=150)
        _close_figure(fig)


def _close_figure(fig) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
