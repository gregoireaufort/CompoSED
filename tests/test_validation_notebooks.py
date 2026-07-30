import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STABLE_NOTEBOOKS = tuple(
    sorted((ROOT / "notebooks" / "tutorials").glob("*.ipynb"))
    + sorted((ROOT / "notebooks" / "validation").glob("*.ipynb"))
)


def notebook_payload(path):
    return json.loads(path.read_text())


def notebook_source(name):
    payload = notebook_payload(ROOT / "notebooks" / "validation" / name)
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_stable_notebooks_have_no_developer_machine_paths_in_source_or_output():
    for path in STABLE_NOTEBOOKS:
        serialized = json.dumps(notebook_payload(path))
        assert "/Users/gregoire" not in serialized, f"{path.name} contains a developer home path"
        assert "/private/tmp" not in serialized, f"{path.name} contains a developer temporary path"


def test_stable_notebooks_have_unique_cell_ids_and_no_saved_errors():
    for path in STABLE_NOTEBOOKS:
        payload = notebook_payload(path)
        cell_ids = [cell["id"] for cell in payload["cells"] if "id" in cell]
        assert len(cell_ids) == len(set(cell_ids)), f"{path.name} has duplicate cell IDs"
        errors = [
            output
            for cell in payload["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, f"{path.name} contains a saved execution error"


def test_release_cigale_tutorials_ship_with_executed_results():
    for name in (
        "02_cigale_maf_cosmos2020_catalog.ipynb",
        "03_cigale_tamis_single_galaxy.ipynb",
    ):
        payload = notebook_payload(ROOT / "notebooks" / "tutorials" / name)
        code_cells = [
            cell for cell in payload["cells"] if cell.get("cell_type") == "code"
        ]
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        outputs = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
        ]
        assert any(output.get("output_type") == "stream" for output in outputs)
        assert any(
            "image/png" in output.get("data", {})
            for output in outputs
            if output.get("output_type") in {"display_data", "execute_result"}
        )


def test_fsps_and_cigale_tutorials_share_raw_sigma_and_model_discrepancy_convention():
    paths = [
        ROOT / "notebooks" / "tutorials" / name
        for name in (
            "01_fsps_maf_cosmos2020_catalog.ipynb",
            "02_cigale_maf_cosmos2020_catalog.ipynb",
            "03_cigale_tamis_single_galaxy.ipynb",
            "04_fsps_pocomc_single_galaxy.ipynb",
        )
    ]
    sources = [
        "\n".join("".join(cell.get("source", [])) for cell in notebook_payload(path)["cells"])
        for path in paths
    ]

    for source in sources:
        assert "MODEL_DISCREPANCY = 0.05" in source
        assert "Gaussian(photometric_model_discrepancy=MODEL_DISCREPANCY)" in source
        assert "sigma_effective" not in source
        assert "EmpiricalPhotometricNoise" not in source
    for source in sources[:2]:
        assert "ConditionalCatalogNoise" in source
        assert "noise_model=survey_noise" in source
        assert 'survey_noise.support_policy = "raise"' in source
        assert 'failure_policy="resample"' in source
        assert "sigma=sigma[TARGET_INDICES]" in source
    assert "require_result_matches_problem" in sources[1]
    assert 'result_label=f"MAF ({N_TRAIN:,} simulations, 256 x 3)"' in sources[1]
    assert '"metallicity": {"values": [0.008, 0.02]}' in sources[1]
    assert '"imf": {"values": [1], "dtype": "int"}' in sources[1]
    assert 'infer=parameters.names' in sources[1]
    assert '"redshift", "log10_mass", "metallicity"' in sources[1]
    assert "posterior.discrete_probabilities(data)" in sources[1]
    assert '"MAF exact P(BC03 metallicity | photometry):"' in sources[1]
    assert '"metallicity": {"values": [0.008, 0.02]}' in sources[2]
    assert '"imf": {"values": [1], "dtype": "int"}' in sources[2]
    assert 'result.parameter_names.index("metallicity")' in sources[2]


def test_validation_notebook_npz_outputs_use_provenance_helper():
    for name in ("09_cigale_mixed_prior_validation.ipynb",):
        source = notebook_source(name)
        assert "np.savez(" not in source
        assert "save_npz_with_provenance" in source
        assert "seed=RNG_SEED" in source


def test_validation_examples_do_not_bypass_provenance_or_use_developer_paths():
    for path in (ROOT / "examples").glob("*.py"):
        source = path.read_text()
        assert "np.savez(" not in source, f"{path.name} bypasses the provenance-aware save helper"
        assert "/Users/gregoire" not in source, f"{path.name} contains a developer-local path"


def test_cached_single_object_tutorials_validate_the_current_problem():
    for name in (
        "03_cigale_tamis_single_galaxy.ipynb",
        "04_fsps_pocomc_single_galaxy.ipynb",
    ):
        payload = notebook_payload(ROOT / "notebooks" / "tutorials" / name)
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        assert "require_result_matches_problem" in source
        assert "COMPOSED_TUTORIAL_FORCE" in source
