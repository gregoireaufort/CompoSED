#!/usr/bin/env python
"""Check which CompoSED scientific backends are usable in this Python env.

This script deliberately does not install anything.  It answers the practical
question a scientist has before running a notebook:

    "Can the active Python interpreter see the packages and data files needed
    for the backend I want to use?"

Examples
--------
Core package only:

    python scripts/check_environment.py

FSPS after following the upstream python-fsps/FSPS install instructions:

    SPS_HOME=/path/to/fsps python scripts/check_environment.py --fsps

MAF/SBI dependencies:

    python scripts/check_environment.py --sbi

MDN-only SBI dependency:

    python scripts/check_environment.py --mdn

Traditional sampler adapters or experimental diffusion:

    python scripts/check_environment.py --samplers
    python scripts/check_environment.py --mc-diagnostics
    python scripts/check_environment.py --diffusion
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from dataclasses import dataclass


REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "composed").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON_FSPS_INSTALL_URL = "https://python-fsps.readthedocs.io/en/latest/installation/"
CIGALE_V2022_URL = "https://gitlab.lam.fr/cigale/cigale/-/tree/v2022.0"
EXPECTED_CIGALE_VERSION = "2022.0"


@dataclass
class Check:
    name: str
    ok: bool
    message: str
    required: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return "OK"
        return "FAIL" if self.required else "WARN"


def package_version(distribution: str) -> str | None:
    """Return installed distribution version when importlib metadata knows it."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_check(module: str, *, distribution: str | None = None, required: bool = True) -> Check:
    """Import a Python module and report the version when available."""

    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        return Check(module, False, f"not importable: {exc}", required=required)

    dist_name = distribution or module.split(".")[0]
    version = package_version(dist_name)
    if version is None:
        version = getattr(imported, "__version__", None)
    suffix = f" {version}" if version else ""
    location = getattr(imported, "__file__", None)
    if location:
        return Check(module, True, f"imported{suffix} from {location}", required=required)
    return Check(module, True, f"imported{suffix}", required=required)


def path_check(name: str, value: str | None, *, must_exist: bool = True, required: bool = True) -> Check:
    """Check an environment-variable path."""

    if not value:
        return Check(name, False, "not set", required=required)
    path = Path(value).expanduser()
    if must_exist and not path.exists():
        return Check(name, False, f"{path} does not exist", required=required)
    return Check(name, True, str(path), required=required)


def check_core() -> list[Check]:
    """Core CompoSED checks: no heavyweight scientific backend required."""

    checks = [
        Check("Python", True, f"{sys.version.split()[0]} on {platform.platform()}"),
        import_check("numpy"),
        import_check("astropy"),
        import_check("composed"),
    ]
    return checks


def check_fsps() -> list[Check]:
    """FSPS backend checks.

    python-fsps fails at import time when SPS_HOME is missing or invalid, so the
    path is checked before importing fsps.
    """

    checks = [
        path_check("SPS_HOME", os.environ.get("SPS_HOME"), required=True),
        import_check("sedpy", distribution="astro-sedpy"),
        import_check("astropy"),
    ]
    if checks[0].ok:
        checks.append(import_check("fsps"))
    else:
        checks.append(
            Check(
                "fsps",
                False,
                f"skipped import because SPS_HOME is not valid; see {PYTHON_FSPS_INSTALL_URL}",
                required=True,
            )
        )
    return checks


def check_cigale() -> list[Check]:
    """CIGALE backend checks."""

    pcigale_check = import_check("pcigale", distribution="pcigale")
    checks = [
        pcigale_check,
        import_check("pkg_resources", distribution="setuptools"),
    ]
    if pcigale_check.ok:
        try:
            imported = importlib.import_module("pcigale")
            version = getattr(imported, "__version__", None) or package_version("pcigale")
            version_matches = version is not None and str(version).split("+", maxsplit=1)[0] == EXPECTED_CIGALE_VERSION
            checks.append(
                Check(
                    "CIGALE target",
                    version_matches,
                    (
                        f"installed {version}; expected {EXPECTED_CIGALE_VERSION} from {CIGALE_V2022_URL}"
                        if version is not None
                        else f"installed version is unknown; expected {EXPECTED_CIGALE_VERSION}"
                    ),
                )
            )
        except Exception as exc:
            checks.append(Check("CIGALE target", False, f"could not inspect installed version: {exc}"))
    else:
        checks.append(
            Check(
                "CIGALE target",
                False,
                f"cannot verify {EXPECTED_CIGALE_VERSION} because pcigale is not importable",
            )
        )
    try:
        import numpy as np

        supports_periodic = np.lib.NumpyVersion(np.__version__) < "1.24.0"
        checks.append(
            Check(
                "CIGALE constant SFH",
                supports_periodic,
                (
                    f"NumPy {np.__version__} retains np.float for upstream sfhperiodic"
                    if supports_periodic
                    else f"NumPy {np.__version__} removed np.float; named constant SFH requires NumPy 1.23.5"
                ),
                required=False,
            )
        )
    except Exception as exc:
        checks.append(Check("CIGALE constant SFH", False, f"could not check NumPy: {exc}", required=False))
    return checks


def check_sbi() -> list[Check]:
    """MAF / neural posterior estimator checks."""

    return [
        # NumPy first avoids duplicate OpenMP initialization in some macOS
        # conda environments when torch is the first numerical import.
        import_check("numpy"),
        import_check("torch"),
        import_check("nflows"),
    ]


def check_mdn() -> list[Check]:
    """MDN-only neural posterior estimator check."""

    return [
        import_check("numpy"),
        import_check("torch"),
    ]


def check_samplers() -> list[Check]:
    """Traditional sampler-adapter dependency checks."""

    return [
        import_check("scipy"),
        import_check("emcee"),
        import_check("pocomc"),
        import_check("tqdm"),
    ]


def check_mc_diagnostics() -> list[Check]:
    """ArviZ-backed MCMC convergence diagnostics."""

    return [import_check("arviz")]


def check_diffusion() -> list[Check]:
    """Experimental conditional-diffusion dependency checks."""

    return [
        import_check("numpy"),
        import_check("torch"),
    ]


def selected_components(args: argparse.Namespace) -> set[str]:
    selected = set()
    if args.all:
        selected.update(
            {
                "core",
                "fsps",
                "cigale",
                "sbi",
                "mdn",
                "samplers",
                "mc_diagnostics",
                "diffusion",
            }
        )
    for name in (
        "core",
        "fsps",
        "cigale",
        "sbi",
        "mdn",
        "samplers",
        "mc_diagnostics",
        "diffusion",
    ):
        if getattr(args, name, False):
            selected.add(name)
    if not selected:
        selected.add("core")
    return selected


def run_checks(components: set[str]) -> list[tuple[str, list[Check]]]:
    grouped: list[tuple[str, list[Check]]] = []
    if "core" in components:
        grouped.append(("core", check_core()))
    if "fsps" in components:
        grouped.append(("fsps", check_fsps()))
    if "cigale" in components:
        grouped.append(("cigale", check_cigale()))
    if "sbi" in components:
        grouped.append(("sbi", check_sbi()))
    if "mdn" in components:
        grouped.append(("mdn", check_mdn()))
    if "samplers" in components:
        grouped.append(("samplers", check_samplers()))
    if "mc_diagnostics" in components:
        grouped.append(("mc-diagnostics", check_mc_diagnostics()))
    if "diffusion" in components:
        grouped.append(("diffusion", check_diffusion()))
    return grouped


def print_report(grouped: list[tuple[str, list[Check]]]) -> None:
    for group, checks in grouped:
        print(f"\n[{group}]")
        for check in checks:
            print(f"  {check.status:4s} {check.name:28s} {check.message}")


def has_required_failures(grouped: list[tuple[str, list[Check]]]) -> bool:
    return any((not check.ok) and check.required for _, checks in grouped for check in checks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all CompoSED v0.1 backend and inference stacks.",
    )
    parser.add_argument("--core", action="store_true", help="Check the lightweight core install.")
    parser.add_argument("--fsps", action="store_true", help="Check FSPS/python-fsps and SPS_HOME.")
    parser.add_argument("--cigale", action="store_true", help="Check CIGALE/pcigale.")
    parser.add_argument("--sbi", action="store_true", help="Check torch/nflows SBI dependencies.")
    parser.add_argument("--mdn", action="store_true", help="Check the torch-only MDN dependency.")
    parser.add_argument(
        "--samplers",
        action="store_true",
        help="Check traditional sampler-adapter dependencies.",
    )
    parser.add_argument(
        "--mc-diagnostics",
        dest="mc_diagnostics",
        action="store_true",
        help="Check ArviZ-backed MCMC convergence diagnostics.",
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        help="Check experimental conditional-diffusion dependencies.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    components = selected_components(args)
    grouped = run_checks(components)
    print_report(grouped)
    if has_required_failures(grouped):
        print("\nEnvironment check failed for one or more requested components.")
        return 1
    print("\nEnvironment check passed for requested components.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
