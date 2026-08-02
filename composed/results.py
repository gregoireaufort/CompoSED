from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
from composed.provenance import (
    artifact_provenance,
    collect_run_provenance,
    verify_artifact_provenance,
)


class InferenceFailure(RuntimeError):
    """Raised when a sampler produced no scientifically usable posterior state."""


@dataclass
class InferenceResult:
    """Normalized posterior samples and metadata from one inference run.

    ``weights`` are always stored as a normalized one-dimensional array. MCMC
    runs normally use uniform weights; grid and importance-sampling runs should
    pass their posterior weights explicitly. ``logp`` may be ``None`` for
    sample-only approximations such as conditional diffusion. In that case no
    MAP estimate is invented. ``inference_state`` can hold an in-memory trained
    guide or neural estimator and is intentionally not serialized by the
    generic result writer.

    ``chain`` uses shape ``(n_draw, n_chain, n_parameter)`` and should contain
    post-burn-in, thinned draws when available. ``diagnostics`` is an optional
    persisted :class:`composed.diagnostics.DiagnosticReport` dictionary; it is
    never inferred automatically while loading an old result.
    """

    samples: np.ndarray
    logp: np.ndarray | None
    weights: np.ndarray
    parameter_names: Sequence[str]
    sampler_name: str = "unknown"
    map_estimate: np.ndarray | None = None
    posterior_median: np.ndarray | None = None
    chain: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    inference_state: Any | None = field(default=None, repr=False)
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError("InferenceResult.samples must have shape (n_sample, n_parameter).")
        logp = None if self.logp is None else np.asarray(self.logp, dtype=float)
        if logp is not None and logp.shape != (samples.shape[0],):
            raise ValueError(f"logp has shape {logp.shape}; expected {(samples.shape[0],)}.")
        weights = _normalize_weights(self.weights, samples.shape[0])
        names = tuple(str(name) for name in self.parameter_names)
        if len(names) != samples.shape[1]:
            raise ValueError("parameter_names length must match samples.shape[1].")

        if logp is None:
            map_estimate = None
        else:
            finite = np.isfinite(logp)
            if not np.any(finite):
                raise InferenceFailure(
                    "Inference produced no finite log-probability sample; MAP and summaries are undefined."
                )
            if np.any(np.isnan(logp)) or np.any(np.isposinf(logp)):
                raise ValueError("logp may contain finite values or -inf, but not NaN or +inf.")
            if np.any((~finite) & (weights > 0.0)):
                raise ValueError("Samples with non-finite logp must have zero posterior weight.")
            map_estimate = samples[int(np.nanargmax(np.where(finite, logp, -np.inf)))]

        if self.map_estimate is not None:
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
        diagnostics = None if self.diagnostics is None else dict(self.diagnostics)

        self.samples = samples
        self.logp = logp
        self.weights = weights
        self.parameter_names = names
        self.sampler_name = str(self.sampler_name)
        self.map_estimate = map_estimate
        self.posterior_median = posterior_median
        self.chain = chain
        self.diagnostics = diagnostics
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
        meta.setdefault(
            "sampler_diagnostics",
            _diagnostic_sampler_meta(sampling_result.meta),
        )

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


def posterior_summary(
    result: InferenceResult,
    credible_interval: float = 0.68,
) -> dict[str, dict[str, float | None]]:
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
            "map": None if result.map_estimate is None else float(result.map_estimate[j]),
        }
    return summary


def save_inference_result(
    result: InferenceResult,
    path: str | Path,
    *,
    diagnostics=None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save arrays and cryptographically bind their scientific metadata.

    The numerical archive stores a SHA-256 digest of the canonical JSON
    metadata. The JSON sidecar stores the archive's own content hash. Together
    these checks detect modification of either the posterior arrays or the
    Problem/sampler/provenance metadata used to interpret them.

    Pass ``diagnostics=diagnose(result)`` to bind the diagnostic report into
    the same metadata digest. If omitted, ``result.diagnostics`` is preserved.
    Existing numerical or metadata files are never replaced unless
    ``overwrite=True`` is supplied explicitly.
    """

    npz_path, json_path = _result_paths(path)
    existing = tuple(candidate for candidate in (npz_path, json_path) if candidate.exists())
    if existing and not overwrite:
        paths = ", ".join(str(candidate) for candidate in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing inference result file(s): {paths}. "
            "Pass overwrite=True only when replacement is intentional."
        )
    metadata = dict(result.metadata)
    provenance = dict(
        metadata.get("provenance")
        or collect_run_provenance(
            extra={
                "artifact_type": "InferenceResult",
                "sampler_name": result.sampler_name,
                "parameter_names": tuple(result.parameter_names),
                "n_sample": int(result.samples.shape[0]),
            }
        )
    )
    metadata["provenance"] = provenance
    if diagnostics is None:
        diagnostic_payload = result.diagnostics
    elif hasattr(diagnostics, "to_dict"):
        diagnostic_payload = diagnostics.to_dict()
    elif isinstance(diagnostics, Mapping):
        diagnostic_payload = dict(diagnostics)
    else:
        raise TypeError("diagnostics must be a DiagnosticReport, mapping, or None.")

    payload = {
        "sampler_name": result.sampler_name,
        "metadata": _json_safe(metadata),
        "posterior_summary": posterior_summary(result),
        "diagnostics": _json_safe(diagnostic_payload),
    }
    metadata_sha256 = _result_metadata_sha256(payload)

    arrays = {
        "samples": result.samples,
        "weights": result.weights,
        "parameter_names": np.asarray(result.parameter_names, dtype=str),
        "posterior_median": result.posterior_median,
        "metadata_schema": np.asarray("composed.inference_result_metadata.v1"),
        "metadata_sha256": np.asarray(metadata_sha256),
    }
    if result.logp is not None:
        arrays["logp"] = result.logp
    if result.map_estimate is not None:
        arrays["map_estimate"] = result.map_estimate
    if result.chain is not None:
        arrays["chain"] = result.chain
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    provenance["output_artifact"] = artifact_provenance(npz_path)
    metadata["provenance"] = provenance
    payload["metadata"] = _json_safe(metadata)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    return npz_path, json_path


def load_inference_result(
    path: str | Path,
    *,
    verify_provenance: bool = True,
) -> InferenceResult:
    """Load a result saved by :func:`save_inference_result`.

    Verification is strict by default. Pass ``verify_provenance=False`` only
    to inspect a legacy result that predates content-hashed artifacts.
    """

    npz_path, json_path = _result_paths(path)
    payload = json.loads(json_path.read_text()) if json_path.exists() else {}
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if verify_provenance:
        metadata_schema = arrays.get("metadata_schema")
        metadata_sha256 = arrays.get("metadata_sha256")
        if metadata_schema is None or metadata_sha256 is None:
            raise ValueError(
                f"Saved inference result {npz_path} does not bind its scientific metadata. "
                "Rerun it, or pass verify_provenance=False only for legacy inspection."
            )
        if str(np.asarray(metadata_schema).item()) != "composed.inference_result_metadata.v1":
            raise ValueError(f"Unsupported inference-result metadata schema for {npz_path}.")
        expected_metadata_sha256 = str(np.asarray(metadata_sha256).item())
        actual_metadata_sha256 = _result_metadata_sha256(payload)
        if actual_metadata_sha256 != expected_metadata_sha256:
            raise ValueError(
                f"Saved inference result metadata hash mismatch for {json_path}. "
                "The scientific sidecar was changed after the posterior was saved."
            )
        provenance = payload.get("metadata", {}).get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(
                f"Saved inference result {npz_path} has no provenance record. "
                "Rerun it, or pass verify_provenance=False only for legacy inspection."
            )
        if provenance.get("schema") != "composed.provenance.v1":
            raise ValueError(f"Unsupported inference-result provenance schema for {npz_path}.")
        verify_artifact_provenance(npz_path, provenance)
    return InferenceResult(
        samples=arrays["samples"],
        logp=arrays.get("logp"),
        weights=arrays["weights"],
        parameter_names=tuple(str(name) for name in arrays["parameter_names"]),
        sampler_name=payload.get("sampler_name", "unknown"),
        map_estimate=arrays.get("map_estimate"),
        posterior_median=arrays.get("posterior_median"),
        chain=arrays.get("chain"),
        diagnostics=payload.get("diagnostics"),
        metadata=payload.get("metadata", {}),
    )


def problem_fingerprint(problem_or_specification: object) -> str:
    """Return a deterministic digest of a Problem's scientific specification.

    The fingerprint is based on the JSON-safe value returned by
    ``Problem.specification()``. It therefore records the backend
    configuration, ordered priors, filters, and observed arrays without
    embedding those arrays in a result sidecar.
    """

    if hasattr(problem_or_specification, "specification"):
        specification = problem_or_specification.specification()
    elif isinstance(problem_or_specification, Mapping):
        specification = problem_or_specification
    else:
        raise TypeError("problem_fingerprint expects a Problem or specification mapping.")
    payload = json.dumps(
        _json_safe(specification),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def require_result_matches_problem(result: InferenceResult, problem: object) -> InferenceResult:
    """Reject a cached result that was produced for a different Problem.

    Returning ``result`` makes the helper convenient in notebook load paths.
    A missing specification is treated as stale rather than silently trusted.
    """

    saved_specification = result.metadata.get("problem")
    if saved_specification is None:
        raise ValueError(
            "Saved inference result has no Problem specification and cannot be validated. "
            "Rerun the inference."
        )
    saved = problem_fingerprint(saved_specification)
    current = problem_fingerprint(problem)
    if saved != current:
        raise ValueError(
            "Saved inference result does not match the current backend, priors, filters, or data. "
            "Rerun the inference."
        )
    return result


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
    positive = weights > 0.0
    values = values[positive]
    weights = weights[positive]
    if values.size == 0:
        raise ValueError("Weighted quantile requires at least one positive weight.")
    if values.size == 1:
        return np.full(np.atleast_1d(quantiles).shape, values[0], dtype=float)
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(np.atleast_1d(quantiles), cdf, values, left=values[0], right=values[-1])


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


def _result_metadata_sha256(payload: Mapping[str, Any]) -> str:
    """Digest scientific result metadata without its self-referential NPZ hash."""

    canonical_payload = _json_safe(payload)
    metadata = canonical_payload.get("metadata")
    if isinstance(metadata, dict):
        provenance = metadata.get("provenance")
        if isinstance(provenance, dict):
            provenance.pop("output_artifact", None)
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


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


def _diagnostic_sampler_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Keep compact sampler telemetry even when large arrays are summarized."""

    keys = (
        "accept_rate",
        "acceptance_fraction_mean",
        "acceptance_fraction_min",
        "acceptance_fraction_max",
        "inner_acceptance_mean",
        "inner_acceptance_min",
        "inner_acceptance_max",
        "nwalkers",
        "nsteps",
        "burnin",
        "thin",
        "betas",
        "ESS",
        "tempered_ESS",
        "tmprd_ESS",
        "log_evidence",
        "log_evidence_err",
        "optimizer_success",
        "optimizer_status",
        "optimizer_message",
        "optimizer_nfev",
        "H",
        "continuous_names",
        "discrete_names",
        "fixed_names",
        "approximate_discrete_kernel",
    )
    kept = {}
    for key in keys:
        if key not in meta:
            continue
        value = meta[key]
        if isinstance(value, np.ndarray) and value.size > 10_000:
            continue
        if isinstance(value, np.ndarray):
            kept[key] = value.tolist()
        else:
            kept[key] = _json_safe(value)
    return kept
