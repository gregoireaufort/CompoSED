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


def test_tutorial_notebooks_ship_without_machine_specific_saved_outputs():
    for path in sorted((ROOT / "notebooks" / "tutorials").glob("*.ipynb")):
        payload = notebook_payload(path)
        for cell in payload["cells"]:
            assert not cell.get("outputs", []), f"{path.name} contains stale saved output"
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None


def test_cigale_maf_and_tamis_tutorials_share_noise_and_problem_provenance():
    maf_path = ROOT / "notebooks" / "tutorials" / "02_cigale_maf_cosmos2020_catalog.ipynb"
    tamis_path = ROOT / "notebooks" / "tutorials" / "03_cigale_tamis_single_galaxy.ipynb"
    maf_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook_payload(maf_path)["cells"]
    )
    tamis_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook_payload(tamis_path)["cells"]
    )

    for source in (maf_source, tamis_source):
        assert "FRACTIONAL_MODEL_ERROR = 0.05" in source
        assert "sigma_effective" in source
    assert "fractional_error=0.0" not in maf_source
    assert "require_result_matches_problem" in maf_source
    assert 'result_label=f"MAF ({N_TRAIN:,} simulations, 256 x 3)"' in maf_source


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
