"""Sampler-aware diagnostics for normalized CompoSED inference results.

The public :func:`diagnose` function accepts an
:class:`~composed.results.InferenceResult` and reports only diagnostics that
have a valid interpretation for its inference algorithm. MCMC chains use
rank-normalized ArviZ diagnostics when ArviZ is installed. Importance samplers
and finite grids use normalized-weight diagnostics computed with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping
import warnings as python_warnings

import numpy as np


_MCMC_NAMES = {"emcee", "random_walk", "rw_metropolis", "mixed_gibbs"}
_IMPORTANCE_NAMES = {"pocomc", "mixed_tamis", "tamis"}
_SBI_NAMES = {"maf", "mdn", "diffusion"}


@dataclass(frozen=True)
class DiagnosticReport:
    """Numerical diagnostics and interpretation notes for one inference run.

    ``global_metrics`` contains algorithm-level quantities such as weighted
    ESS, evidence uncertainty, or chain count. ``parameter_metrics`` follows
    the exact order of ``InferenceResult.parameter_names``. Warnings indicate
    a diagnostic threshold or missing information that should be investigated;
    they are not an automatic declaration that a scientific fit is invalid.
    """

    sampler_name: str
    family: str
    n_samples: int
    global_metrics: Mapping[str, Any] = field(default_factory=dict)
    parameter_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation suitable for persistence."""

        return {
            "schema": "composed.diagnostics.v1",
            "sampler_name": self.sampler_name,
            "family": self.family,
            "n_samples": int(self.n_samples),
            "global_metrics": _json_value(self.global_metrics),
            "parameter_metrics": _json_value(self.parameter_metrics),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """Return a compact scientist-readable text summary."""

        lines = [
            f"Sampler: {self.sampler_name}",
            f"Diagnostic family: {self.family}",
            f"Posterior rows: {self.n_samples}",
        ]
        for key, value in self.global_metrics.items():
            if isinstance(value, (list, tuple, dict)):
                continue
            lines.append(f"{key}: {_format_value(value)}")
        if self.parameter_metrics:
            lines.append("Per-parameter diagnostics:")
            for name, metrics in self.parameter_metrics.items():
                shown = ", ".join(
                    f"{key}={_format_value(value)}"
                    for key, value in metrics.items()
                    if not isinstance(value, (list, tuple, dict))
                )
                lines.append(f"  {name}: {shown}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {message}" for message in self.warnings)
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {message}" for message in self.notes)
        return "\n".join(lines)


def diagnose(
    result,
    *,
    rhat_threshold: float = 1.01,
    min_ess: float = 400.0,
    min_relative_weight_ess: float = 0.1,
    max_weight_threshold: float = 0.1,
    require_arviz: bool = False,
) -> DiagnosticReport:
    """Diagnose one :class:`~composed.results.InferenceResult`.

    Parameters
    ----------
    result
        Normalized CompoSED inference result.
    rhat_threshold
        Warning threshold for rank-normalized split R-hat.
    min_ess
        Warning threshold for bulk and tail MCMC ESS.
    min_relative_weight_ess
        Warning threshold for importance-sampling ESS divided by the number of
        weighted rows. This threshold is heuristic and is not applied to exact
        finite grids.
    max_weight_threshold
        Warning threshold for the largest normalized importance weight.
    require_arviz
        Raise a helpful :class:`ImportError` when MCMC diagnostics require
        ArviZ but it is unavailable. By default a partial report is returned.

    Notes
    -----
    R-hat requires multiple chains. Emcee walkers are interacting ensemble
    members, so their R-hat is a useful screening statistic rather than a
    replacement for comparing independently initialized ensembles.
    """

    from composed.results import InferenceResult

    if not isinstance(result, InferenceResult):
        raise TypeError("diagnose requires a composed.InferenceResult.")
    for name, value in (
        ("rhat_threshold", rhat_threshold),
        ("min_ess", min_ess),
        ("min_relative_weight_ess", min_relative_weight_ess),
        ("max_weight_threshold", max_weight_threshold),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    family = _diagnostic_family(result)
    if family == "mcmc":
        return _diagnose_mcmc(
            result,
            rhat_threshold=float(rhat_threshold),
            min_ess=float(min_ess),
            require_arviz=bool(require_arviz),
        )
    if family == "importance_sampling":
        return _diagnose_weighted(
            result,
            family=family,
            min_relative_ess=float(min_relative_weight_ess),
            max_weight_threshold=float(max_weight_threshold),
            warn_for_degeneracy=True,
        )
    if family == "grid":
        return _diagnose_weighted(
            result,
            family=family,
            min_relative_ess=float(min_relative_weight_ess),
            max_weight_threshold=float(max_weight_threshold),
            warn_for_degeneracy=False,
        )
    if family == "laplace":
        return _diagnose_laplace(result)
    if family == "sbi":
        return DiagnosticReport(
            sampler_name=result.sampler_name,
            family=family,
            n_samples=result.samples.shape[0],
            global_metrics={"posterior_draws": int(result.samples.shape[0])},
            notes=(
                "Neural posterior draws are not an MCMC chain; R-hat and chain ESS are not defined.",
                "Use held-out rank, coverage, prediction, OOD, and boundary-saturation diagnostics.",
            ),
        )
    return DiagnosticReport(
        sampler_name=result.sampler_name,
        family=family,
        n_samples=result.samples.shape[0],
        global_metrics={"posterior_draws": int(result.samples.shape[0])},
        warnings=(
            "No algorithm-specific diagnostic adapter is registered for this sampler.",
        ),
    )


def _diagnostic_family(result) -> str:
    name = str(result.sampler_name).lower()
    if name in _MCMC_NAMES or result.chain is not None:
        return "mcmc"
    if name in _IMPORTANCE_NAMES:
        return "importance_sampling"
    if name == "grid":
        return "grid"
    if name == "laplace":
        return "laplace"
    if name in _SBI_NAMES or result.logp is None:
        return "sbi"
    if not np.allclose(result.weights, np.full(result.weights.size, 1.0 / result.weights.size)):
        return "importance_sampling"
    return "unknown"


def _diagnose_mcmc(result, *, rhat_threshold: float, min_ess: float, require_arviz: bool) -> DiagnosticReport:
    chain = _chain_as_chain_draw_parameter(result)
    n_chains, n_draws, n_parameters = chain.shape
    sampler_meta = _sampler_meta(result)
    global_metrics: dict[str, Any] = {
        "n_chains": int(n_chains),
        "draws_per_chain": int(n_draws),
        "total_chain_draws": int(n_chains * n_draws),
        "rhat_warning_threshold": float(rhat_threshold),
        "ess_warning_threshold": float(min_ess),
    }
    _copy_scalar_metrics(
        sampler_meta,
        global_metrics,
        keys=(
            "accept_rate",
            "acceptance_fraction_mean",
            "acceptance_fraction_min",
            "acceptance_fraction_max",
            "inner_acceptance_mean",
            "inner_acceptance_min",
            "inner_acceptance_max",
        ),
    )

    diagnostic_warnings: list[str] = []
    notes: list[str] = []
    if n_chains < 2:
        notes.append("R-hat is unavailable because this result contains only one chain.")
    elif str(result.sampler_name).lower() == "emcee":
        notes.append(
            "Emcee walkers interact; walker R-hat is a screening diagnostic. "
            "Independent ensembles remain the stronger replication check."
        )

    arviz = _optional_arviz(require=require_arviz)
    if arviz is None:
        diagnostic_warnings.append(
            "ArviZ is unavailable, so rank-normalized R-hat, ESS, and MCSE were not computed. "
            "Install CompoSED with the 'mc-diagnostics' extra."
        )

    parameter_metrics: dict[str, dict[str, Any]] = {}
    bad_rhat: list[str] = []
    low_bulk: list[str] = []
    low_tail: list[str] = []
    for index, name in enumerate(result.parameter_names):
        values = chain[:, :, index]
        fixed = _is_fixed(values)
        metrics: dict[str, Any] = {"fixed": fixed}
        if fixed:
            metrics.update(
                {
                    "rhat": None,
                    "ess_bulk": None,
                    "ess_tail": None,
                    "mcse_mean": 0.0,
                    "mcse_over_sd": 0.0,
                    "autocorrelation_time_bulk": None,
                }
            )
        elif arviz is None:
            metrics.update(
                {
                    "rhat": None,
                    "ess_bulk": None,
                    "ess_tail": None,
                    "mcse_mean": None,
                    "mcse_over_sd": None,
                    "autocorrelation_time_bulk": None,
                }
            )
        else:
            with python_warnings.catch_warnings():
                python_warnings.simplefilter("ignore")
                rhat = (
                    _arviz_scalar(arviz.rhat(values, method="rank"))
                    if n_chains >= 2 and n_draws >= 4
                    else None
                )
                ess_bulk = _arviz_scalar(arviz.ess(values, method="bulk"))
                ess_tail = _arviz_scalar(arviz.ess(values, method="tail"))
                mcse_mean = _arviz_scalar(arviz.mcse(values, method="mean"))
            sample_sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            metrics.update(
                {
                    "rhat": rhat,
                    "ess_bulk": ess_bulk,
                    "ess_tail": ess_tail,
                    "mcse_mean": mcse_mean,
                    "mcse_over_sd": (
                        None
                        if sample_sd <= 0.0 or mcse_mean is None
                        else float(mcse_mean / sample_sd)
                    ),
                    "autocorrelation_time_bulk": (
                        None
                        if ess_bulk is None or ess_bulk <= 0.0
                        else float(n_chains * n_draws / ess_bulk)
                    ),
                }
            )
            if rhat is not None and rhat > rhat_threshold:
                bad_rhat.append(name)
            if ess_bulk is not None and ess_bulk < min_ess:
                low_bulk.append(name)
            if ess_tail is not None and ess_tail < min_ess:
                low_tail.append(name)
        parameter_metrics[name] = metrics

    if bad_rhat:
        diagnostic_warnings.append(
            f"Rank-normalized R-hat exceeds {rhat_threshold:g} for: {', '.join(bad_rhat)}."
        )
    if low_bulk:
        diagnostic_warnings.append(
            f"Bulk ESS is below {min_ess:g} for: {', '.join(low_bulk)}."
        )
    if low_tail:
        diagnostic_warnings.append(
            f"Tail ESS is below {min_ess:g} for: {', '.join(low_tail)}."
        )

    discrete_names = tuple(str(name) for name in sampler_meta.get("discrete_names", ()))
    for name in discrete_names:
        if name not in result.parameter_names:
            continue
        index = result.parameter_names.index(name)
        values = chain[:, :, index]
        transitions = values[:, 1:] != values[:, :-1]
        metric = parameter_metrics[name]
        metric["transition_rate"] = float(np.mean(transitions)) if transitions.size else 0.0
        states, counts = np.unique(values, return_counts=True)
        probabilities = counts / np.sum(counts)
        metric["visited_states"] = int(states.size)
        metric["state_entropy"] = float(-np.sum(probabilities * np.log(probabilities)))
        if states.size <= 1:
            diagnostic_warnings.append(
                f"Discrete parameter {name!r} visited only one state."
            )

    return DiagnosticReport(
        sampler_name=result.sampler_name,
        family="mcmc",
        n_samples=result.samples.shape[0],
        global_metrics=global_metrics,
        parameter_metrics=parameter_metrics,
        warnings=tuple(diagnostic_warnings),
        notes=tuple(notes),
    )


def _diagnose_weighted(
    result,
    *,
    family: str,
    min_relative_ess: float,
    max_weight_threshold: float,
    warn_for_degeneracy: bool,
) -> DiagnosticReport:
    weights = np.asarray(result.weights, dtype=float)
    n = weights.size
    positive = weights > 0.0
    entropy = float(-np.sum(weights[positive] * np.log(weights[positive])))
    weight_ess = float(1.0 / np.sum(weights**2))
    effective_support = float(np.exp(entropy))
    global_metrics: dict[str, Any] = {
        "weighted_rows": int(n),
        "positive_weight_rows": int(np.sum(positive)),
        "weight_ess": weight_ess,
        "relative_weight_ess": float(weight_ess / n),
        "weight_entropy": entropy,
        "entropy_effective_rows": effective_support,
        "normalized_perplexity": float(effective_support / n),
        "maximum_normalized_weight": float(np.max(weights)),
        "relative_weight_ess_warning_threshold": float(min_relative_ess),
        "maximum_weight_warning_threshold": float(max_weight_threshold),
    }
    sampler_meta = _sampler_meta(result)
    _copy_scalar_metrics(
        sampler_meta,
        global_metrics,
        keys=("log_evidence", "log_evidence_err"),
    )

    is_tamis = str(result.sampler_name).lower() in {"tamis", "mixed_tamis"}
    betas = _small_numeric_array(sampler_meta.get("betas")) if is_tamis else None
    ess_history = _small_numeric_array(sampler_meta.get("ESS")) if is_tamis else None
    tempered_ess = (
        _small_numeric_array(
            sampler_meta.get("tempered_ESS", sampler_meta.get("tmprd_ESS"))
        )
        if is_tamis
        else None
    )
    if betas is not None and betas.size:
        global_metrics["beta_history"] = betas.tolist()
        global_metrics["final_beta"] = float(betas[-1])
        global_metrics["beta_decrease_count"] = int(np.sum(np.diff(betas) < -1.0e-12))
    if ess_history is not None and ess_history.size:
        global_metrics["iteration_ess_history"] = ess_history.tolist()
        global_metrics["final_iteration_ess"] = float(ess_history[-1])
    if tempered_ess is not None and tempered_ess.size:
        global_metrics["tempered_ess_history"] = tempered_ess.tolist()

    parameter_metrics: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(result.parameter_names):
        values = np.asarray(result.samples[:, index], dtype=float)
        mean = float(np.sum(weights * values))
        variance = float(np.sum(weights * (values - mean) ** 2))
        metrics = {
            "weighted_mean": mean,
            "weighted_sd": float(np.sqrt(max(variance, 0.0))),
        }
        if family == "importance_sampling":
            metrics["approximate_mcse_mean"] = float(
                np.sqrt(max(variance, 0.0) / weight_ess)
            )
        parameter_metrics[name] = metrics

    diagnostic_warnings: list[str] = []
    notes: list[str] = []
    if warn_for_degeneracy:
        if weight_ess / n < min_relative_ess:
            diagnostic_warnings.append(
                "Relative importance-weight ESS is below "
                f"{min_relative_ess:g} ({weight_ess / n:.3g})."
            )
        if np.max(weights) > max_weight_threshold:
            diagnostic_warnings.append(
                "The largest normalized importance weight exceeds "
                f"{max_weight_threshold:g} ({np.max(weights):.3g})."
            )
        notes.append(
            "Importance-weight ESS measures weight degeneracy, not Markov-chain mixing. "
            "Independent repeated runs remain the strongest stability check."
        )
        if betas is not None:
            notes.append(
                "TAMIS beta controls proposal adaptation; final AMIS weights still target the "
                "untempered posterior, so beta alone is not a convergence certificate."
            )
    else:
        notes.append(
            "This is an exact finite-grid posterior. Effective support describes posterior "
            "concentration, not Monte Carlo convergence."
        )

    return DiagnosticReport(
        sampler_name=result.sampler_name,
        family=family,
        n_samples=result.samples.shape[0],
        global_metrics=global_metrics,
        parameter_metrics=parameter_metrics,
        warnings=tuple(diagnostic_warnings),
        notes=tuple(notes),
    )


def _diagnose_laplace(result) -> DiagnosticReport:
    sampler_meta = _sampler_meta(result)
    hessian = _small_numeric_array(sampler_meta.get("H"))
    global_metrics: dict[str, Any] = {}
    diagnostic_warnings: list[str] = []
    notes = [
        "Laplace is a local Gaussian approximation; MCMC convergence diagnostics are not defined."
    ]
    _copy_scalar_metrics(
        sampler_meta,
        global_metrics,
        keys=("optimizer_success", "optimizer_status", "optimizer_nfev"),
    )
    if "optimizer_success" in global_metrics and not bool(global_metrics["optimizer_success"]):
        diagnostic_warnings.append("The Laplace optimizer did not report success.")
    if hessian is not None and hessian.ndim == 2 and hessian.shape[0] == hessian.shape[1]:
        eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        positive = bool(np.all(eigenvalues > 0.0))
        global_metrics.update(
            {
                "hessian_positive_definite": positive,
                "hessian_min_eigenvalue": float(np.min(eigenvalues)),
                "hessian_max_eigenvalue": float(np.max(eigenvalues)),
                "hessian_condition_number": (
                    float(np.linalg.cond(hessian)) if positive else None
                ),
            }
        )
        if not positive:
            diagnostic_warnings.append("The negative-log-posterior Hessian is not positive definite.")
        elif float(np.linalg.cond(hessian)) > 1.0e12:
            diagnostic_warnings.append("The Laplace Hessian condition number exceeds 1e12.")
    else:
        diagnostic_warnings.append("The saved result does not contain a usable Laplace Hessian.")
    return DiagnosticReport(
        sampler_name=result.sampler_name,
        family="laplace",
        n_samples=result.samples.shape[0],
        global_metrics=global_metrics,
        warnings=tuple(diagnostic_warnings),
        notes=tuple(notes),
    )


def _chain_as_chain_draw_parameter(result) -> np.ndarray:
    if result.chain is None:
        return np.asarray(result.samples, dtype=float)[None, :, :]
    chain = np.asarray(result.chain, dtype=float)
    if chain.ndim != 3 or chain.shape[-1] != len(result.parameter_names):
        raise ValueError(
            "InferenceResult.chain must have shape (n_draw, n_chain, n_parameter)."
        )
    if not np.all(np.isfinite(chain)):
        raise ValueError("InferenceResult.chain contains NaN or inf values.")
    return np.transpose(chain, (1, 0, 2))


def _sampler_meta(result) -> Mapping[str, Any]:
    value = result.metadata.get("sampler_meta", {})
    compact = result.metadata.get("sampler_diagnostics", {})
    merged = dict(value) if isinstance(value, Mapping) else {}
    if isinstance(compact, Mapping):
        merged.update(compact)
    return merged


def _optional_arviz(*, require: bool):
    if importlib.util.find_spec("arviz") is None:
        if require:
            raise ImportError(
                "MCMC convergence diagnostics require ArviZ. "
                "Install CompoSED with the 'mc-diagnostics' extra."
            )
        return None
    try:
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("ignore", FutureWarning)
            import arviz
    except Exception as exc:
        if require:
            raise ImportError(
                "ArviZ is installed but could not be imported. Check its NumPy "
                "compatibility and that its cache directory is writable."
            ) from exc
        return None

    return arviz


def _arviz_scalar(value) -> float | None:
    scalar = float(np.asarray(value))
    return scalar if np.isfinite(scalar) else None


def _is_fixed(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(float(np.max(values) - np.min(values)) <= 1.0e-14 * scale)


def _copy_scalar_metrics(source: Mapping[str, Any], target: dict[str, Any], *, keys) -> None:
    for key in keys:
        if key not in source:
            continue
        value = source[key]
        if isinstance(value, (bool, str, int, float, np.generic)):
            target[key] = _json_value(value)


def _small_numeric_array(value) -> np.ndarray | None:
    if value is None or isinstance(value, Mapping):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.size <= 10_000 and np.all(np.isfinite(array)) else None


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
