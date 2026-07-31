from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from composed.diagnostics import DiagnosticReport, diagnose
from composed.results import (
    InferenceResult,
    load_inference_result,
    normalize_sampling_result,
    save_inference_result,
)
from inftools.core import SamplingResult


def _mcmc_result(chain_chain_draw, *, sampler_name="emcee", metadata=None):
    chain_chain_draw = np.asarray(chain_chain_draw, dtype=float)
    chain_draw_chain = np.transpose(chain_chain_draw, (1, 0, 2))
    samples = chain_draw_chain.reshape(-1, chain_draw_chain.shape[-1])
    return InferenceResult(
        samples=samples,
        logp=-0.5 * np.sum(samples**2, axis=1),
        weights=np.ones(samples.shape[0]),
        parameter_names=tuple(f"theta_{i}" for i in range(samples.shape[1])),
        sampler_name=sampler_name,
        chain=chain_draw_chain,
        metadata=metadata or {},
    )


@pytest.mark.skipif(importlib.util.find_spec("arviz") is None, reason="arviz is not installed")
def test_independent_gaussian_chains_have_good_rhat_and_high_ess():
    rng = np.random.default_rng(401)
    result = _mcmc_result(rng.normal(size=(4, 1200, 2)))

    report = diagnose(result)

    assert report.family == "mcmc"
    assert report.global_metrics["n_chains"] == 4
    for metrics in report.parameter_metrics.values():
        assert metrics["rhat"] < 1.01
        assert metrics["ess_bulk"] > 800
        assert np.isfinite(metrics["mcse_mean"])


@pytest.mark.skipif(importlib.util.find_spec("arviz") is None, reason="arviz is not installed")
def test_shifted_chain_is_flagged_by_rank_normalized_rhat():
    rng = np.random.default_rng(402)
    chain = rng.normal(size=(4, 900, 1))
    chain[-1, :, 0] += 1.5

    report = diagnose(_mcmc_result(chain))

    assert report.parameter_metrics["theta_0"]["rhat"] > 1.01
    assert any("R-hat" in message for message in report.warnings)


@pytest.mark.skipif(importlib.util.find_spec("arviz") is None, reason="arviz is not installed")
def test_ar1_chain_has_lower_ess_than_draw_count_and_no_rhat():
    rng = np.random.default_rng(403)
    values = np.zeros(3000, dtype=float)
    for index in range(1, values.size):
        values[index] = 0.95 * values[index - 1] + rng.normal()
    report = diagnose(
        _mcmc_result(values[None, :, None], sampler_name="random_walk"),
        min_ess=1.0,
    )

    metrics = report.parameter_metrics["theta_0"]
    assert metrics["rhat"] is None
    assert metrics["ess_bulk"] < 0.25 * values.size
    assert metrics["autocorrelation_time_bulk"] > 4.0
    assert any("only one chain" in note for note in report.notes)


@pytest.mark.skipif(importlib.util.find_spec("arviz") is None, reason="arviz is not installed")
def test_mixed_gibbs_reports_discrete_transitions_and_fixed_axes():
    continuous = np.linspace(-1.0, 1.0, 20)
    discrete = np.tile([0.0, 1.0], 10)
    fixed = np.full(20, 7.0)
    samples = np.column_stack([continuous, discrete, fixed])
    result = InferenceResult(
        samples=samples,
        logp=np.zeros(20),
        weights=np.ones(20),
        parameter_names=("x", "template", "fixed"),
        sampler_name="mixed_gibbs",
        chain=samples[:, None, :],
        metadata={"sampler_meta": {"discrete_names": ["template"]}},
    )

    report = diagnose(result, min_ess=1.0)

    assert report.parameter_metrics["template"]["transition_rate"] == pytest.approx(1.0)
    assert report.parameter_metrics["template"]["visited_states"] == 2
    assert report.parameter_metrics["fixed"]["fixed"] is True
    assert report.parameter_metrics["fixed"]["rhat"] is None


def test_importance_weight_diagnostics_distinguish_uniform_and_collapsed_weights():
    samples = np.arange(4.0)[:, None]
    common = dict(
        samples=samples,
        logp=np.zeros(4),
        parameter_names=("x",),
        sampler_name="pocomc",
    )
    uniform = diagnose(InferenceResult(weights=np.ones(4), **common))
    collapsed = diagnose(
        InferenceResult(weights=np.asarray([0.999, 0.001, 0.0, 0.0]), **common),
        min_relative_weight_ess=0.5,
    )

    assert uniform.global_metrics["weight_ess"] == pytest.approx(4.0)
    assert uniform.global_metrics["normalized_perplexity"] == pytest.approx(1.0)
    assert collapsed.global_metrics["weight_ess"] < 1.01
    assert collapsed.global_metrics["maximum_normalized_weight"] == pytest.approx(0.999)
    assert any("Relative importance-weight ESS" in message for message in collapsed.warnings)
    assert any("largest normalized" in message for message in collapsed.warnings)


def test_exact_grid_reports_concentration_without_false_convergence_warning():
    result = InferenceResult(
        samples=np.arange(3.0)[:, None],
        logp=np.asarray([0.0, -10.0, -20.0]),
        weights=np.asarray([1.0, np.exp(-10.0), np.exp(-20.0)]),
        parameter_names=("template",),
        sampler_name="grid",
    )

    report = diagnose(result)

    assert report.family == "grid"
    assert report.warnings == ()
    assert report.global_metrics["entropy_effective_rows"] < 1.01
    assert "approximate_mcse_mean" not in report.parameter_metrics["template"]
    assert any("exact finite-grid" in note for note in report.notes)


def test_pocomc_scalar_ess_is_not_mislabeled_as_tamis_iteration_history():
    result = InferenceResult(
        samples=np.arange(4.0)[:, None],
        logp=np.zeros(4),
        weights=np.ones(4),
        parameter_names=("x",),
        sampler_name="pocomc",
        metadata={"sampler_meta": {"ESS": 4.0, "log_evidence": -2.5}},
    )

    report = diagnose(result)

    assert report.global_metrics["weight_ess"] == pytest.approx(4.0)
    assert report.global_metrics["log_evidence"] == pytest.approx(-2.5)
    assert "iteration_ess_history" not in report.global_metrics


def test_mixed_tamis_report_preserves_adaptation_history():
    result = InferenceResult(
        samples=np.arange(8.0)[:, None],
        logp=np.zeros(8),
        weights=np.linspace(1.0, 2.0, 8),
        parameter_names=("x",),
        sampler_name="mixed_tamis",
        metadata={
            "sampler_meta": {
                "betas": [0.2, 0.5, 1.0],
                "ESS": [2.0, 4.0, 7.0],
                "tempered_ESS": [6.0, 6.0, 7.0],
            }
        },
    )

    report = diagnose(result)

    assert report.global_metrics["final_beta"] == pytest.approx(1.0)
    assert report.global_metrics["beta_decrease_count"] == 0
    assert report.global_metrics["iteration_ess_history"] == [2.0, 4.0, 7.0]
    assert any("beta controls proposal adaptation" in note for note in report.notes)


def test_long_tamis_history_survives_result_metadata_compaction():
    n_iterations = 300
    beta_history = np.linspace(0.1, 1.0, n_iterations)
    ess_history = np.linspace(20.0, 80.0, n_iterations)
    raw = SamplingResult(
        samples=np.arange(8.0)[:, None],
        logp=np.zeros(8),
        meta={
            "weights_norm": np.full(8, 1.0 / 8.0),
            "betas": beta_history,
            "ESS": ess_history,
        },
    )

    result = normalize_sampling_result(
        raw,
        parameter_names=("x",),
        sampler_name="mixed_tamis",
    )
    report = diagnose(result)

    # Generic metadata summarizes long arrays, but diagnostic telemetry keeps
    # the full adaptation history needed to audit one TAMIS run.
    assert result.metadata["sampler_meta"]["betas"]["shape"] == [n_iterations]
    assert len(result.metadata["sampler_diagnostics"]["betas"]) == n_iterations
    assert report.global_metrics["final_beta"] == pytest.approx(1.0)
    assert report.global_metrics["iteration_ess_history"] == pytest.approx(
        ess_history.tolist()
    )


def test_laplace_reports_hessian_failure_without_mcmc_claims():
    result = InferenceResult(
        samples=np.asarray([[0.0, 0.0]]),
        logp=np.asarray([0.0]),
        weights=np.ones(1),
        parameter_names=("x", "y"),
        sampler_name="laplace",
        metadata={
            "sampler_meta": {
                "optimizer_success": False,
                "optimizer_status": 2,
                "H": [[1.0, 0.0], [0.0, -1.0]],
            }
        },
    )

    report = diagnose(result)

    assert report.family == "laplace"
    assert report.global_metrics["hessian_positive_definite"] is False
    assert any("optimizer did not report success" in message for message in report.warnings)
    assert any("not positive definite" in message for message in report.warnings)


def test_neural_result_points_to_sbi_calibration_instead_of_mc_metrics():
    result = InferenceResult(
        samples=np.zeros((8, 1)),
        logp=None,
        weights=np.ones(8),
        parameter_names=("z",),
        sampler_name="maf",
    )

    report = diagnose(result)

    assert report.family == "sbi"
    assert report.parameter_metrics == {}
    assert any("coverage" in note for note in report.notes)


def test_missing_arviz_can_return_partial_report_or_raise(monkeypatch):
    result = _mcmc_result(np.arange(20.0)[None, :, None], sampler_name="random_walk")
    monkeypatch.setattr("composed.diagnostics.importlib.util.find_spec", lambda name: None)

    report = diagnose(result)
    assert report.parameter_metrics["theta_0"]["ess_bulk"] is None
    assert any("ArviZ is unavailable" in message for message in report.warnings)
    with pytest.raises(ImportError, match="ArviZ"):
        diagnose(result, require_arviz=True)


def test_broken_arviz_import_returns_partial_report_or_clear_error(monkeypatch):
    result = _mcmc_result(np.arange(20.0)[None, :, None], sampler_name="random_walk")
    monkeypatch.setattr("composed.diagnostics.importlib.util.find_spec", lambda name: object())

    real_import = __import__

    def fail_arviz_import(name, *args, **kwargs):
        if name == "arviz":
            raise RuntimeError("incompatible binary")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_arviz_import)

    report = diagnose(result)
    assert report.parameter_metrics["theta_0"]["ess_bulk"] is None
    with pytest.raises(ImportError, match="could not be imported"):
        diagnose(result, require_arviz=True)


def test_diagnostic_report_is_bound_into_saved_result(tmp_path):
    result = InferenceResult(
        samples=np.arange(4.0)[:, None],
        logp=np.zeros(4),
        weights=np.ones(4),
        parameter_names=("template",),
        sampler_name="grid",
    )
    report = diagnose(result)

    npz_path, _ = save_inference_result(result, tmp_path / "diagnosed", diagnostics=report)
    loaded = load_inference_result(npz_path)

    assert loaded.diagnostics["schema"] == "composed.diagnostics.v1"
    assert loaded.diagnostics["family"] == "grid"
    assert loaded.diagnostics["global_metrics"]["weight_ess"] == pytest.approx(4.0)


def test_report_summary_names_unavailable_quantities():
    report = DiagnosticReport(
        sampler_name="toy",
        family="mcmc",
        n_samples=2,
        parameter_metrics={"x": {"rhat": None}},
    )

    assert "rhat=unavailable" in report.summary()
