"""Validate named delayed-tau SFHs against explicit backend-native calls.

Run this script in the environment containing the requested engine:

    SPS_HOME=/path/to/fsps python examples/validate_named_sfh_backends.py fsps
    python examples/validate_named_sfh_backends.py cigale

The FSPS reference receives the exact tabular history explicitly. The CIGALE
reference calls ``SedWarehouse`` directly with the native ``sfhdelayed``
module. Neither reference calls the named backend path under test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from composed.backends.cigale import CIGALEBackend, MJY_PER_MAGGIE
from composed.backends.fsps import FSPSBackend
from composed.filters import FilterSet
from composed.sfh import DelayedTauSFH


def validate_fsps() -> None:
    """Compare the named FSPS path with the same explicit tabular history."""

    from sedpy.observate import load_filters

    filter_names = ["sdss_g0", "sdss_r0"]
    filters = FilterSet(load_filters(filter_names), names=filter_names)
    sfh = DelayedTauSFH(n_time=64)
    scalar_params = {
        "zred": 0.1,
        "tage_gyr": 5.0,
        "tau_gyr": 1.5,
        "logzsol": -0.3,
    }

    named = FSPSBackend(sfh=sfh).predict_photometry(scalar_params, filters)
    history = sfh.evaluate(scalar_params, redshift=scalar_params["zred"])
    explicit = FSPSBackend().predict_photometry(
        {
            "zred": scalar_params["zred"],
            "logzsol": scalar_params["logzsol"],
            "tabular_time_gyr": history.time_gyr,
            "tabular_sfr_msun_per_yr": history.sfr_msun_per_yr,
        },
        filters,
    )

    _report_and_check(named.flux, explicit.flux, tolerance=1.0e-12)
    print(f"formed mass: {named.metadata['formed_mass_msun']:.8f} Msun")
    print(f"surviving mass: {named.metadata['surviving_stellar_mass_msun']:.8f} Msun")
    print(f"SFH model: {named.metadata['sfh_model']['name']}")


def validate_cigale() -> None:
    """Compare the named CIGALE path with a direct native-module call."""

    from pcigale.warehouse import SedWarehouse

    filter_name = "sdss.gp"
    sfh = DelayedTauSFH()
    scalar_params = {"tage_gyr": 1.0, "tau_gyr": 3.0, "z": 0.1}
    modules = ["bc03", "redshifting"]
    module_parameters = {
        "bc03": {"imf": 1, "metallicity": 0.02, "separation_age": 10},
        "redshifting": {"redshift": {"name": "z", "range": [0.05, 0.2]}},
    }

    named = CIGALEBackend(
        modules=modules,
        module_parameters=module_parameters,
        sfh=sfh,
    ).predict_photometry(scalar_params, FilterSet([filter_name]))

    direct = SedWarehouse().get_sed(
        ["sfhdelayed", "bc03", "redshifting"],
        [
            sfh.cigale_parameters(scalar_params),
            module_parameters["bc03"],
            {"redshift": scalar_params["z"]},
        ],
    )
    direct_per_surviving_msun = np.array(
        [float(direct.compute_fnu(filter_name)) / MJY_PER_MAGGIE / float(direct.info["stellar.m_star"])]
    )

    _report_and_check(named.flux, direct_per_surviving_msun, tolerance=1.0e-12)
    print(f"formed mass: {named.metadata['formed_mass_msun']:.8f} Msun")
    print(f"surviving mass: {named.metadata['surviving_stellar_mass_msun']:.8f} Msun")
    print(f"native SFH module: {named.metadata['modules'][0]}")


def _report_and_check(actual: np.ndarray, reference: np.ndarray, *, tolerance: float) -> None:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    relative = np.abs(actual - reference) / np.maximum(np.abs(reference), np.finfo(float).tiny)
    print("named flux [maggies]:", actual)
    print("reference flux [maggies]:", reference)
    print("maximum relative difference:", float(np.max(relative)))
    if actual.shape != reference.shape or not np.all(np.isfinite(actual)):
        raise RuntimeError("Named SFH validation produced invalid or mismatched flux arrays.")
    if not np.allclose(actual, reference, rtol=tolerance, atol=0.0):
        raise RuntimeError(f"Named SFH validation exceeds relative tolerance {tolerance:.1e}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["fsps", "cigale"])
    args = parser.parse_args()
    if args.backend == "fsps":
        validate_fsps()
    else:
        validate_cigale()


if __name__ == "__main__":
    main()
