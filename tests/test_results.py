from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from composed import Gaussian, Problem, SEDDataset
import composed.problem as problem_module
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.filters import FilterSet
from inftools.core import SamplingResult
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.results import (
    InferenceFailure,
    InferenceResult,
    load_inference_result,
    normalize_sampling_result,
    posterior_summary,
    problem_fingerprint,
    require_result_matches_problem,
    save_inference_result,
)
from composed.units import MassNormalization


@dataclass
class ConfigurableBackend(SEDBackend):
    amplitude_scale: float = 1.0
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        flux = self.amplitude_scale * float(params["amplitude"])
        return ModelPhotometry(("g",), np.asarray([flux]))


GLOBAL_TRANSFORM_SCALE = 1.0


def transform_with_referenced_global(params):
    return {
        **params,
        "amplitude": GLOBAL_TRANSFORM_SCALE * float(params["amplitude"]),
    }


def make_problem(*, observed_flux=2.0, amplitude_scale=1.0):
    return Problem(
        backend=ConfigurableBackend(amplitude_scale=amplitude_scale),
        parameters=ParameterSpace(("amplitude",), {"amplitude": UniformPrior(1.0, 3.0)}),
        data=SEDDataset(("g",), np.asarray([observed_flux]), np.asarray([0.1])),
        likelihood=Gaussian(),
        filters=FilterSet(("g",), names=("g",)),
    )


def test_inference_result_normalizes_weights_and_summarizes():
    result = InferenceResult(
        samples=np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        logp=np.asarray([-2.0, -1.0, 0.0]),
        weights=np.asarray([0.0, 1.0, 3.0]),
        parameter_names=("x", "y"),
        sampler_name="toy",
    )

    assert np.isclose(np.sum(result.weights), 1.0)
    assert np.allclose(result.map_estimate, [4.0, 5.0])
    summary = posterior_summary(result)
    assert set(summary) == {"x", "y"}
    assert summary["x"]["map"] == 4.0


def test_normalize_sampling_result_reads_weights_from_sampler_meta():
    raw = SamplingResult(
        samples=np.asarray([[0.0], [1.0]]),
        logp=np.asarray([-1.0, 0.0]),
        map_estimate=np.asarray([1.0]),
        meta={"weights_norm": np.asarray([0.25, 0.75])},
    )
    space = ParameterSpace(names=("z",), priors={"z": UniformPrior(0.0, 2.0)})

    result = normalize_sampling_result(raw, space, sampler_name="grid")

    assert result.parameter_names == ("z",)
    assert np.allclose(result.weights, [0.25, 0.75])
    assert result.sampler_name == "grid"


def test_inference_result_save_load_roundtrip(tmp_path):
    result = InferenceResult(
        samples=np.asarray([[0.0], [1.0]]),
        logp=np.asarray([-1.0, 0.0]),
        weights=np.asarray([1.0, 1.0]),
        parameter_names=("z",),
        sampler_name="toy",
        metadata={"filters": ["u", "g"]},
    )

    npz_path, json_path = save_inference_result(result, tmp_path / "run_001")
    loaded = load_inference_result(npz_path)

    assert npz_path.exists()
    assert json_path.exists()
    assert loaded.parameter_names == ("z",)
    assert np.allclose(loaded.samples, result.samples)
    assert np.allclose(loaded.weights, result.weights)
    assert loaded.metadata["filters"] == ["u", "g"]


def test_problem_fingerprint_is_stable_across_json_roundtrip():
    specification = make_problem().specification()
    restored = json.loads(json.dumps(specification))

    assert problem_fingerprint(specification) == problem_fingerprint(restored)


def test_problem_fingerprint_tracks_referenced_global_transform_values(monkeypatch):
    problem = make_problem()
    problem.parameter_transform = transform_with_referenced_global
    problem.__post_init__()
    first_fingerprint = problem_fingerprint(problem)
    first_log_likelihood = problem.log_likelihood([2.0])

    monkeypatch.setitem(
        transform_with_referenced_global.__globals__,
        "GLOBAL_TRANSFORM_SCALE",
        2.0,
    )
    second_fingerprint = problem_fingerprint(problem)
    second_log_likelihood = problem.log_likelihood([2.0])

    assert second_fingerprint != first_fingerprint
    assert second_log_likelihood != first_log_likelihood


def test_engine_identity_hashes_source_without_recording_install_path(monkeypatch, tmp_path):
    source = tmp_path / "engine.py"
    source.write_text("ENGINE_VALUE = 1\n")
    specification = type("Specification", (), {"origin": str(source)})()
    monkeypatch.setattr(problem_module.importlib.util, "find_spec", lambda name: specification)

    first = problem_module._module_identity("example_engine")
    source.write_text("ENGINE_VALUE = 2\n")
    second = problem_module._module_identity("example_engine")

    assert first["source_name"] == "engine.py"
    assert first["source_sha256"] != second["source_sha256"]
    assert "origin" not in first
    assert str(tmp_path) not in json.dumps(first)


def test_external_engine_identity_does_not_record_machine_local_path(tmp_path):
    engine_tree = tmp_path / "FSPS"
    engine_tree.mkdir()

    identity = problem_module._scientific_directory_identity(str(engine_tree))

    assert identity == {"configured": True, "exists": True}
    assert str(tmp_path) not in json.dumps(identity)


def test_cached_result_must_match_backend_filters_and_observed_data():
    problem = make_problem()
    result = InferenceResult(
        samples=np.asarray([[2.0], [2.1]]),
        logp=np.asarray([0.0, -0.5]),
        weights=np.ones(2),
        parameter_names=("amplitude",),
        metadata={"problem": problem.specification()},
    )

    assert require_result_matches_problem(result, problem) is result
    with pytest.raises(ValueError, match="does not match"):
        require_result_matches_problem(result, make_problem(observed_flux=2.2))
    with pytest.raises(ValueError, match="does not match"):
        require_result_matches_problem(result, make_problem(amplitude_scale=1.1))


def test_cached_result_without_problem_specification_is_rejected():
    result = InferenceResult(
        samples=np.asarray([[2.0]]),
        logp=np.asarray([0.0]),
        weights=np.ones(1),
        parameter_names=("amplitude",),
    )

    with pytest.raises(ValueError, match="no Problem specification"):
        require_result_matches_problem(result, make_problem())


def test_inference_result_rejects_unnormalizable_weights():
    with pytest.raises(ValueError, match="positive total"):
        InferenceResult(
            samples=np.asarray([[0.0], [1.0]]),
            logp=np.asarray([-1.0, 0.0]),
            weights=np.asarray([0.0, 0.0]),
            parameter_names=("z",),
        )


def test_inference_result_rejects_total_log_probability_failure():
    with pytest.raises(InferenceFailure, match="no finite log-probability"):
        InferenceResult(
            samples=np.asarray([[0.0], [1.0]]),
            logp=np.asarray([-np.inf, -np.inf]),
            weights=np.asarray([0.5, 0.5]),
            parameter_names=("z",),
        )


def test_equal_weight_median_matches_ordinary_sample_median():
    result = InferenceResult(
        samples=np.asarray([[0.0], [1.0], [2.0]]),
        logp=np.asarray([-1.0, 0.0, -1.0]),
        weights=np.ones(3),
        parameter_names=("x",),
    )

    assert np.allclose(result.posterior_median, [1.0])


def test_sample_only_result_does_not_invent_log_density_or_map(tmp_path):
    result = InferenceResult(
        samples=np.asarray([[0.0], [1.0], [2.0]]),
        logp=None,
        weights=np.ones(3),
        parameter_names=("z",),
        sampler_name="diffusion",
    )

    assert result.logp is None
    assert result.map_estimate is None
    assert np.allclose(result.posterior_median, [1.0])

    npz_path, _ = save_inference_result(result, tmp_path / "sample_only")
    loaded = load_inference_result(npz_path)
    assert loaded.logp is None
    assert loaded.map_estimate is None
    assert posterior_summary(loaded)["z"]["map"] is None
