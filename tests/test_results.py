from __future__ import annotations

import numpy as np
import pytest

from inftools.core import SamplingResult
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.results import (
    InferenceResult,
    load_inference_result,
    normalize_sampling_result,
    posterior_summary,
    save_inference_result,
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


def test_inference_result_rejects_unnormalizable_weights():
    with pytest.raises(ValueError, match="positive total"):
        InferenceResult(
            samples=np.asarray([[0.0], [1.0]]),
            logp=np.asarray([-1.0, 0.0]),
            weights=np.asarray([0.0, 0.0]),
            parameter_names=("z",),
        )
