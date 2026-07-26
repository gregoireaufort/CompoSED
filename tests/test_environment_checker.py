from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_checker_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"
    spec = importlib.util.spec_from_file_location("check_environment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_selection_checks_core_only():
    checker = load_checker_module()

    class Args:
        all = False
        core = False
        fsps = False
        cigale = False
        sbi = False

    assert checker.selected_components(Args()) == {"core"}


def test_all_selection_contains_only_release_stacks():
    checker = load_checker_module()

    class Args:
        all = True
        core = False
        fsps = False
        cigale = False
        sbi = False

    assert checker.selected_components(Args()) == {
        "core",
        "fsps",
        "cigale",
        "sbi",
        "mdn",
        "samplers",
        "diffusion",
    }


def test_mdn_check_requires_torch_only(monkeypatch):
    checker = load_checker_module()
    requested = []

    def fake_import_check(module, **kwargs):
        requested.append(module)
        return checker.Check(module, True, "fake")

    monkeypatch.setattr(checker, "import_check", fake_import_check)

    checks = checker.check_mdn()

    assert [check.name for check in checks] == ["numpy", "torch"]
    assert requested == ["numpy", "torch"]


def test_sampler_check_matches_sampler_extra(monkeypatch):
    checker = load_checker_module()
    requested = []

    def fake_import_check(module, **kwargs):
        requested.append(module)
        return checker.Check(module, True, "fake")

    monkeypatch.setattr(checker, "import_check", fake_import_check)

    checks = checker.check_samplers()

    assert [check.name for check in checks] == ["scipy", "emcee", "pocomc", "tqdm"]
    assert requested == ["scipy", "emcee", "pocomc", "tqdm"]


def test_diffusion_check_requires_torch_only(monkeypatch):
    checker = load_checker_module()
    requested = []

    def fake_import_check(module, **kwargs):
        requested.append(module)
        return checker.Check(module, True, "fake")

    monkeypatch.setattr(checker, "import_check", fake_import_check)

    checks = checker.check_diffusion()

    assert [check.name for check in checks] == ["numpy", "torch"]
    assert requested == ["numpy", "torch"]


def test_missing_required_path_is_failure():
    checker = load_checker_module()
    check = checker.path_check("MISSING_RESOURCE", None, required=True)

    assert not check.ok
    assert check.status == "FAIL"


def test_missing_optional_path_is_warning():
    checker = load_checker_module()
    check = checker.path_check("OPTIONAL_RESOURCE", None, required=False)

    assert not check.ok
    assert check.status == "WARN"


def test_cigale_check_reports_constant_sfh_numpy_compatibility(monkeypatch):
    checker = load_checker_module()
    monkeypatch.setattr(checker, "import_check", lambda *args, **kwargs: checker.Check("pcigale", True, "fake"))
    fake_pcigale = type("FakeCigale", (), {"__version__": "2022.0"})()
    monkeypatch.setattr(checker.importlib, "import_module", lambda name: fake_pcigale)

    checks = checker.check_cigale()
    constant_check = next(check for check in checks if check.name == "CIGALE constant SFH")
    target_check = next(check for check in checks if check.name == "CIGALE target")

    assert constant_check.required is False
    assert "NumPy" in constant_check.message
    assert target_check.ok


def test_cigale_check_rejects_unexpected_version(monkeypatch):
    checker = load_checker_module()
    monkeypatch.setattr(checker, "import_check", lambda *args, **kwargs: checker.Check("pcigale", True, "fake"))
    fake_pcigale = type("FakeCigale", (), {"__version__": "2025.0"})()
    monkeypatch.setattr(checker.importlib, "import_module", lambda name: fake_pcigale)

    checks = checker.check_cigale()
    target_check = next(check for check in checks if check.name == "CIGALE target")

    assert not target_check.ok
    assert target_check.required
    assert "expected 2022.0" in target_check.message
