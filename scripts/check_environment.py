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

Full local science stack:

    SPS_HOME=/path/to/fsps \
    CUE_DATA_DIR=/path/to/cue/src/cue/data \
    DSPS_CONTINUUM_SSP_FILE=/path/to/fsps_continuum_ssp_data.h5 \
    python scripts/check_environment.py --all
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
CUE_URL = "https://github.com/yi-jia-li/cue"


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


def cue_data_checks(required: bool) -> list[Check]:
    """Check the public Cue data directory used by the JAX Cue port."""

    checks = [path_check("CUE_DATA_DIR", os.environ.get("CUE_DATA_DIR"), required=required)]
    if not checks[-1].ok:
        checks[-1].message += f" ; clone Cue from {CUE_URL} and point to src/cue/data"
        return checks

    data_dir = Path(os.environ["CUE_DATA_DIR"]).expanduser()
    required_files = [
        "FSPSlam.dat",
        "speculator_cont_new.pkl",
        "pca_cont_new.pkl",
        "speculator_line_new_H1.pkl",
        "pca_line_new_H1.pkl",
        "lineList_128lines.dat",
        "lineList_wav.npy",
    ]
    for filename in required_files:
        checks.append(path_check(f"Cue file {filename}", str(data_dir / filename), required=required))
    return checks


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

    checks = [import_check("pcigale", distribution="pcigale")]
    checks.append(
        Check(
            "CIGALE target",
            True,
            f"CompoSED validation targets upstream CIGALE v2022.0: {CIGALE_V2022_URL}",
        )
    )
    return checks


def check_sbi() -> list[Check]:
    """SBI / neural posterior estimator checks."""

    return [
        import_check("torch"),
        import_check("nflows"),
    ]


def check_jaxcigale(*, require_data: bool) -> list[Check]:
    """JAX-CIGALE checks."""

    checks = [
        import_check("jax"),
        import_check("jaxlib"),
        import_check("numpyro"),
        import_check("dsps"),
        import_check("h5py"),
        import_check("dill"),
        import_check("sklearn", distribution="scikit-learn"),
    ]

    # The analytic JAX-CIGALE smoke path does not need this file, but DSPS/Cue
    # validation does.  Treat it as required only when --all/--cue asks for the
    # complete science stack.
    checks.append(
        path_check(
            "DSPS_CONTINUUM_SSP_FILE",
            os.environ.get("DSPS_CONTINUUM_SSP_FILE"),
            required=require_data,
        )
    )

    try:
        import jax

        checks.append(Check("JAX backend", True, f"{jax.default_backend()} ; devices={jax.devices()}"))
    except Exception as exc:
        checks.append(Check("JAX backend", False, f"could not query devices: {exc}", required=False))
    return checks


def selected_components(args: argparse.Namespace) -> set[str]:
    selected = set()
    if args.all:
        selected.update({"core", "fsps", "cigale", "jaxcigale", "cue", "sbi"})
    for name in ("core", "fsps", "cigale", "jaxcigale", "cue", "sbi"):
        if getattr(args, name):
            selected.add(name)
    if not selected:
        selected.add("core")
    if "cue" in selected:
        selected.add("jaxcigale")
    return selected


def run_checks(components: set[str]) -> list[tuple[str, list[Check]]]:
    grouped: list[tuple[str, list[Check]]] = []
    if "core" in components:
        grouped.append(("core", check_core()))
    if "fsps" in components:
        grouped.append(("fsps", check_fsps()))
    if "cigale" in components:
        grouped.append(("cigale", check_cigale()))
    if "jaxcigale" in components:
        grouped.append(("jaxcigale", check_jaxcigale(require_data="cue" in components)))
    if "cue" in components:
        grouped.append(("cue-data", cue_data_checks(required=True)))
    if "sbi" in components:
        grouped.append(("sbi", check_sbi()))
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
    parser.add_argument("--all", action="store_true", help="Check all known CompoSED optional stacks.")
    parser.add_argument("--core", action="store_true", help="Check the lightweight core install.")
    parser.add_argument("--fsps", action="store_true", help="Check FSPS/python-fsps and SPS_HOME.")
    parser.add_argument("--cigale", action="store_true", help="Check CIGALE/pcigale.")
    parser.add_argument("--jaxcigale", action="store_true", help="Check JAX-CIGALE packages.")
    parser.add_argument("--cue", action="store_true", help="Check Cue public data plus JAX-CIGALE packages.")
    parser.add_argument("--sbi", action="store_true", help="Check torch/nflows SBI dependencies.")
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
