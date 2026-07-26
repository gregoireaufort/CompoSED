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
