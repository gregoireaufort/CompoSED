import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_source(name):
    payload = json.loads((ROOT / "notebooks" / "validation" / name).read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_public_validation_notebooks_have_no_developer_machine_paths():
    for name in (
        "04_backend_cross_validation_single_sed.ipynb",
        "05_cigale_mock_jaxcigale_photometric_validation.ipynb",
        "06_dsps_cue_spectrum_fitting_validation.ipynb",
        "07_jades_like_dsps_cue_prism_validation.ipynb",
        "08_dsps_cue_joint_spectrophotometry_validation.ipynb",
        "09_cigale_mixed_prior_validation.ipynb",
    ):
        source = notebook_source(name)
        assert "/Users/gregoire" not in source
        assert "/private/tmp/cue" not in source


def test_validation_notebook_npz_outputs_use_provenance_helper():
    for name in (
        "06_dsps_cue_spectrum_fitting_validation.ipynb",
        "07_jades_like_dsps_cue_prism_validation.ipynb",
        "08_dsps_cue_joint_spectrophotometry_validation.ipynb",
        "09_cigale_mixed_prior_validation.ipynb",
    ):
        source = notebook_source(name)
        assert "np.savez(" not in source
        assert "save_npz_with_provenance" in source
        assert "seed=RNG_SEED" in source


def test_validation_examples_do_not_bypass_provenance_or_use_local_cue_paths():
    for path in (ROOT / "examples").glob("*.py"):
        source = path.read_text()
        assert "np.savez(" not in source, f"{path.name} bypasses the provenance-aware save helper"
        assert "/private/tmp/cue" not in source, f"{path.name} contains a developer-local Cue path"
