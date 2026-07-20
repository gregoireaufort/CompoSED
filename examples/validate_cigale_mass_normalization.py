"""Validate CIGALEBackend's surviving-stellar-mass normalization.

Run this in an environment containing the upstream CIGALE v2022.0 package and
database. The reference calculation calls ``SedWarehouse`` directly, obtains
the unit-formed-mass photometry and ``stellar.m_star``, and performs the
division explicitly without calling ``CIGALEBackend``.
"""

from __future__ import annotations

import numpy as np

from composed.backends.cigale import CIGALEBackend, MJY_PER_MAGGIE
from composed.filters import FilterSet


MODULES = ["sfhdelayed", "bc03", "redshifting"]
MODULE_PARAMETERS = {
    "sfhdelayed": {
        "age_main": 1000,
        "tau_main": 3000.0,
        "normalise": True,
    },
    "bc03": {
        "imf": 1,
        "metallicity": 0.02,
        "separation_age": 10,
    },
    "redshifting": {"redshift": 0.1},
}
FILTER_NAME = "sdss.gp"
FLUX_RTOL = 1.0e-12


def direct_cigale_reference() -> tuple[float, dict[str, float]]:
    """Return direct CIGALE maggies per surviving stellar solar mass."""

    from pcigale.warehouse import SedWarehouse

    parameter_list = [MODULE_PARAMETERS[module] for module in MODULES]
    sed = SedWarehouse().get_sed(MODULES, parameter_list)
    formed_mass = float(sed.info["sfh.integrated"])
    surviving_stellar_mass = float(sed.info["stellar.m_star"])
    if not np.isclose(formed_mass, 1.0, rtol=1e-8, atol=1e-8):
        raise RuntimeError(f"Direct CIGALE SED has sfh.integrated={formed_mass!r}, expected 1.")
    if not np.isfinite(surviving_stellar_mass) or surviving_stellar_mass <= 0.0:
        raise RuntimeError(f"Direct CIGALE SED has invalid stellar.m_star={surviving_stellar_mass!r}.")

    raw_maggies = float(sed.compute_fnu(FILTER_NAME)) / MJY_PER_MAGGIE
    reference_maggies = raw_maggies / surviving_stellar_mass
    return reference_maggies, {
        "formed_mass_msun": formed_mass,
        "surviving_stellar_mass_msun": surviving_stellar_mass,
        "surviving_stellar_mass_fraction": surviving_stellar_mass / formed_mass,
    }


def run_validation() -> dict[str, object]:
    reference_flux, reference_mass = direct_cigale_reference()
    backend = CIGALEBackend(modules=MODULES, module_parameters=MODULE_PARAMETERS)
    photometry = backend.predict_photometry({}, FilterSet([FILTER_NAME]))
    backend_flux = float(photometry.flux[0])

    if not np.isfinite(backend_flux) or backend_flux < 0.0:
        raise RuntimeError(f"CIGALEBackend returned invalid flux {backend_flux!r}.")
    relative_difference = abs(backend_flux - reference_flux) / max(
        abs(reference_flux), np.finfo(float).tiny
    )
    if relative_difference > FLUX_RTOL:
        raise RuntimeError(
            f"Relative flux difference {relative_difference:.3e} exceeds {FLUX_RTOL:.3e}."
        )
    for key, expected in reference_mass.items():
        if not np.isclose(float(photometry.metadata[key]), expected, rtol=1e-12, atol=1e-12):
            raise RuntimeError(
                f"CIGALE mass metadata disagree for {key}: "
                f"{photometry.metadata[key]!r} versus {expected!r}."
            )
    return {
        "photometry": photometry,
        "reference_flux": reference_flux,
        "reference_mass": reference_mass,
        "relative_difference": relative_difference,
    }


def main() -> None:
    result = run_validation()
    photometry = result["photometry"]
    print(f"CIGALEBackend mass normalization: per_solar_mass")
    print(f"CIGALEBackend mass reference: {photometry.metadata['mass_reference']}")
    print(
        "Unit-formed-mass surviving fraction: "
        f"{photometry.metadata['surviving_stellar_mass_fraction']:.8f}"
    )
    print(f"{FILTER_NAME} backend maggies:   {photometry.flux[0]:.12e}")
    print(f"{FILTER_NAME} reference maggies: {result['reference_flux']:.12e}")
    print(f"Relative flux difference: {result['relative_difference']:.3e}")
    print("CIGALEBackend mass-normalization validation passed.")


if __name__ == "__main__":
    main()
