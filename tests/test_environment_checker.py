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
        jaxcigale = False
        cue = False
        sbi = False

    assert checker.selected_components(Args()) == {"core"}


def test_cue_selection_implies_jaxcigale():
    checker = load_checker_module()

    class Args:
        all = False
        core = False
        fsps = False
        cigale = False
        jaxcigale = False
        cue = True
        sbi = False

    assert checker.selected_components(Args()) == {"cue", "jaxcigale"}


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
