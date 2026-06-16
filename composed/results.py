from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np


@dataclass
class InferenceResult:
    """Normalized posterior samples and metadata from one inference run.

    ``weights`` are always stored as a normalized one-dimensional array. MCMC
    runs normally use uniform weights; grid and importance-sampling runs should
    pass their posterior weights explicitly.
    """

    samples: np.ndarray
    logp: np.ndarray
    weights: np.ndarray
    parameter_names: Sequence[str]
    sampler_name: str = "unknown"
    map_estimate: np.ndarray | None = None
    posterior_median: np.ndarray | None = None
    chain: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError("InferenceResult.samples must have shape (n_sample, n_parameter).")
        logp = np.asarray(self.logp, dtype=float)
        if logp.shape != (samples.shape[0],):
            raise ValueError(f"logp has shape {logp.shape}; expected {(samples.shape[0],)}.")
        weights = _normalize_weights(self.weights, samples.shape[0])
        names = tuple(str(name) for name in self.parameter_names)
        if len(names) != samples.shape[1]:
            raise ValueError("parameter_names length must match samples.shape[1].")

        if self.map_estimate is None:
            finite = np.isfinite(logp)
            map_estimate = samples[int(np.nanargmax(np.where(finite, logp, -np.inf)))]
        else:
            map_estimate = np.asarray(self.map_estimate, dtype=float)
            if map_estimate.shape != (samples.shape[1],):
                raise ValueError("map_estimate must have shape (n_parameter,).")

        if self.posterior_median is None:
            posterior_median = weighted_quantile(samples, weights, 0.5)
        else:
            posterior_median = np.asarray(self.posterior_median, dtype=float)
            if posterior_median.shape != (samples.shape[1],):
                raise ValueError("posterior_median must have shape (n_parameter,).")

        chain = None if self.chain is None else np.asarray(self.chain, dtype=float)

        self.samples = samples
        self.logp = logp
        self.weights = weights
        self.parameter_names = names
        self.sampler_name = str(self.sampler_name)
        self.map_estimate = map_estimate
        self.posterior_median = posterior_median
        self.chain = chain
        self.metadata = dict(self.metadata)


def normalize_sampling_result(
    sampling_result,
    parameter_space=None,
    *,
    parameter_names: Sequence[str] | None = None,
    sampler_name: str = "unknown",
    weights: Sequence[float] | None = None,
    chain: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InferenceResult:
    """Convert an inftools-style sampler output to ``InferenceResult``."""

    samples = np.asarray(sampling_result.samples, dtype=float)
    logp = np.asarray(sampling_result.logp, dtype=float)
    if parameter_names is None:
        if parameter_space is not None:
            parameter_names = tuple(parameter_space.names)
        else:
            parameter_names = tuple(f"theta_{i}" for i in range(samples.shape[1]))

    if weights is None:
        weights = getattr(sampling_result, "weights", None)
    if weights is None and hasattr(sampling_result, "meta"):
        weights = sampling_result.meta.get("weights_norm")
    if weights is None:
        weights = np.ones(samples.shape[0], dtype=float)

    meta = dict(metadata or {})
    if hasattr(sampling_result, "meta"):
        meta.setdefault("sampler_meta", _json_safe(sampling_result.meta))

    return InferenceResult(
        samples=samples,
        logp=logp,
        weights=np.asarray(weights, dtype=float),
        parameter_names=parameter_names,
        sampler_name=sampler_name,
        map_estimate=getattr(sampling_result, "map_estimate", None),
        chain=chain,
        metadata=meta,
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles) -> np.ndarray:
    """Column-wise weighted quantiles for posterior summaries."""

    values = np.asarray(values, dtype=float)
    weights = _normalize_weights(weights, values.shape[0])
    quantiles = np.asarray(quantiles, dtype=float)
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("quantiles must lie in [0, 1].")
    if values.ndim == 1:
        return _weighted_quantile_1d(values, weights, quantiles)
    if values.ndim != 2:
        raise ValueError("values must be one- or two-dimensional.")
    out = np.asarray([_weighted_quantile_1d(values[:, j], weights, quantiles) for j in range(values.shape[1])])
    if quantiles.ndim == 0:
        return out[:, 0]
    return out


def posterior_summary(result: InferenceResult, credible_interval: float = 0.68) -> dict[str, dict[str, float]]:
    """Return weighted median and central credible interval per parameter."""

    if not 0.0 < credible_interval < 1.0:
        raise ValueError("credible_interval must lie in (0, 1).")
    lo_q = 0.5 * (1.0 - credible_interval)
    hi_q = 1.0 - lo_q
    q = weighted_quantile(result.samples, result.weights, [lo_q, 0.5, hi_q])
    summary = {}
    for j, name in enumerate(result.parameter_names):
        summary[name] = {
            "q_lo": float(q[j, 0]),
            "median": float(q[j, 1]),
            "q_hi": float(q[j, 2]),
            "map": float(result.map_estimate[j]),
        }
    return summary


def save_inference_result(result: InferenceResult, path: str | Path) -> tuple[Path, Path]:
    """Save arrays to ``.npz`` and metadata to a JSON sidecar."""

    npz_path, json_path = _result_paths(path)
    arrays = {
        "samples": result.samples,
        "logp": result.logp,
        "weights": result.weights,
        "parameter_names": np.asarray(result.parameter_names, dtype=str),
        "map_estimate": result.map_estimate,
        "posterior_median": result.posterior_median,
    }
    if result.chain is not None:
        arrays["chain"] = result.chain
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    json_path.write_text(
        json.dumps(
            {
                "sampler_name": result.sampler_name,
                "metadata": _json_safe(result.metadata),
                "posterior_summary": posterior_summary(result),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return npz_path, json_path


def load_inference_result(path: str | Path) -> InferenceResult:
    """Load a result saved by ``save_inference_result``."""

    npz_path, json_path = _result_paths(path)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    payload = json.loads(json_path.read_text()) if json_path.exists() else {}
    return InferenceResult(
        samples=arrays["samples"],
        logp=arrays["logp"],
        weights=arrays["weights"],
        parameter_names=tuple(str(name) for name in arrays["parameter_names"]),
        sampler_name=payload.get("sampler_name", "unknown"),
        map_estimate=arrays.get("map_estimate"),
        posterior_median=arrays.get("posterior_median"),
        chain=arrays.get("chain"),
        metadata=payload.get("metadata", {}),
    )


def _normalize_weights(weights: Sequence[float], n_expected: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (int(n_expected),):
        raise ValueError(f"weights has shape {weights.shape}; expected {(int(n_expected),)}.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")
    total = np.sum(weights)
    if total <= 0.0:
        raise ValueError("weights must have positive total mass.")
    return weights / total


def _weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    return np.interp(np.atleast_1d(quantiles), cdf, values)


def _result_paths(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix == ".npz":
        npz_path = path
        json_path = path.with_suffix(".json")
    elif path.suffix:
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")
    else:
        npz_path = path / "inference_result.npz"
        json_path = path / "inference_result.json"
    return npz_path, json_path


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 256:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
