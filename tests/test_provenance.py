from __future__ import annotations

import json

import numpy as np
import pytest

from composed.provenance import (
    artifact_provenance,
    collect_run_provenance,
    provenance_path_for,
    read_provenance,
    require_provenance,
    save_npz_with_provenance,
    sha256_file,
    write_provenance,
)
from composed.results import InferenceResult, load_inference_result, save_inference_result


def test_file_hash_changes_when_contents_change(tmp_path):
    path = tmp_path / "ssp_like_table.h5"
    path.write_text("first version\n")
    first = sha256_file(path)

    path.write_text("second version\n")
    second = sha256_file(path)

    assert first != second


def test_collect_run_provenance_records_seed_args_and_artifact_hash(tmp_path):
    artifact = tmp_path / "model_table.npy"
    np.save(artifact, np.asarray([1.0, 2.0, 3.0]))

    provenance = collect_run_provenance(
        paths={"model_table": artifact},
        seed=123,
        command_args={"stage": "reference", "n_draws": 4},
        extra={"purpose": "unit-test"},
    )

    assert provenance["schema"] == "composed.provenance.v1"
    assert provenance["seed"] == 123
    assert provenance["command_args"]["stage"] == "reference"
    assert provenance["extra"]["purpose"] == "unit-test"
    assert provenance["artifacts"]["model_table"]["exists"]
    assert provenance["artifacts"]["model_table"]["sha256"] == sha256_file(artifact)
    assert "numpy" in provenance["packages"]
    assert "commit" in provenance["git"] or provenance["git"]["available"] is False


def test_save_npz_with_provenance_writes_sidecar_and_requires_it(tmp_path):
    data_path = tmp_path / "reference_spectra.npz"
    input_path = tmp_path / "input.txt"
    input_path.write_text("physical input\n")

    archive_path, provenance_path = save_npz_with_provenance(
        data_path,
        provenance_paths={"input": input_path},
        seed=7,
        command_args=["--stage", "references"],
        rest_wave_nm=np.asarray([100.0, 200.0]),
        spectrum_model=np.asarray([[1.0, 2.0]]),
    )
    loaded_provenance = require_provenance(archive_path)

    assert archive_path.exists()
    assert provenance_path == provenance_path_for(data_path)
    assert loaded_provenance["seed"] == 7
    assert loaded_provenance["artifacts"]["input"]["sha256"] == sha256_file(input_path)
    assert loaded_provenance["output_artifact"]["sha256"] == sha256_file(archive_path)
    with np.load(archive_path) as data:
        assert np.allclose(data["rest_wave_nm"], [100.0, 200.0])


def test_require_provenance_rejects_modified_artifact(tmp_path):
    archive_path, _ = save_npz_with_provenance(
        tmp_path / "reference.npz",
        x=np.asarray([1.0]),
    )
    with archive_path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        require_provenance(archive_path)


def test_require_provenance_fails_loudly_when_sidecar_is_missing(tmp_path):
    data_path = tmp_path / "stale_cache.npz"
    np.savez(data_path, x=np.asarray([1.0]))

    with pytest.raises(FileNotFoundError, match="Missing provenance sidecar"):
        require_provenance(data_path)


def test_write_and_read_provenance_roundtrip(tmp_path):
    path = tmp_path / "manual.provenance.json"
    payload = {"schema": "composed.provenance.v1", "array": np.asarray([1, 2])}

    write_provenance(payload, path)
    loaded = read_provenance(path)

    assert json.loads(path.read_text())["array"] == [1, 2]
    assert loaded["schema"] == "composed.provenance.v1"


def test_artifact_provenance_for_missing_input_is_explicit(tmp_path):
    missing = tmp_path / "does_not_exist.h5"

    info = artifact_provenance(missing)

    assert info["exists"] is False
    assert info["path"].endswith("does_not_exist.h5")


def test_save_inference_result_embeds_basic_provenance(tmp_path):
    result = InferenceResult(
        samples=np.asarray([[0.0], [1.0]]),
        logp=np.asarray([-1.0, 0.0]),
        weights=np.asarray([1.0, 1.0]),
        parameter_names=("z",),
        sampler_name="toy",
        metadata={"filters": ["g"]},
    )

    npz_path, _ = save_inference_result(result, tmp_path / "run_001")
    loaded = load_inference_result(npz_path)

    assert loaded.metadata["filters"] == ["g"]
    assert loaded.metadata["provenance"]["schema"] == "composed.provenance.v1"
    assert loaded.metadata["provenance"]["extra"]["sampler_name"] == "toy"
    assert loaded.metadata["provenance"]["output_artifact"]["sha256"] == sha256_file(npz_path)


def test_load_inference_result_rejects_modified_archive(tmp_path):
    result = InferenceResult(
        samples=np.asarray([[0.0], [1.0]]),
        logp=np.asarray([-1.0, 0.0]),
        weights=np.ones(2),
        parameter_names=("z",),
    )
    npz_path, _ = save_inference_result(result, tmp_path / "run")
    with npz_path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_inference_result(npz_path)
    legacy = load_inference_result(npz_path, verify_provenance=False)
    assert legacy.samples.shape == (2, 1)


@pytest.mark.parametrize("field", ["problem", "sampler_name", "posterior_summary"])
def test_load_inference_result_rejects_modified_scientific_sidecar(tmp_path, field):
    result = InferenceResult(
        samples=np.asarray([[0.0], [1.0]]),
        logp=np.asarray([-1.0, 0.0]),
        weights=np.ones(2),
        parameter_names=("z",),
        sampler_name="toy",
        metadata={"problem": {"backend": "A", "data": {"flux": [1.0]}}},
    )
    npz_path, json_path = save_inference_result(result, tmp_path / f"tampered_{field}")
    payload = json.loads(json_path.read_text())
    if field == "problem":
        payload["metadata"]["problem"]["backend"] = "B"
    elif field == "sampler_name":
        payload["sampler_name"] = "different-sampler"
    else:
        payload["posterior_summary"]["z"]["median"] = 999.0
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="metadata hash mismatch"):
        load_inference_result(npz_path)
    legacy = load_inference_result(npz_path, verify_provenance=False)
    assert legacy.samples.shape == (2, 1)
