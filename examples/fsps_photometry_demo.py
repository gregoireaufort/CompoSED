"""Compute FSPS broadband photometry for one named delayed-tau SFH.

This is intentionally written like a small analysis script:

1. choose filters;
2. define one physical parameter vector, with units in the variable names;
3. call the backend;
4. print the photometry and the mass-normalization convention.

Requirements: python-fsps, sedpy, and ``SPS_HOME`` pointing at the FSPS grids.
"""

from __future__ import annotations

from composed.backends.fsps import FSPSBackend
from composed.filters import FilterSet
from composed.sfh import DelayedTauSFH


FILTER_NAMES = ["sdss_g0", "sdss_r0", "sdss_i0"]

# Named SFH scalar parameters use Gyr. CompoSED constructs and validates the
# tabular history before calling FSPS.
GALAXY_PARAMETERS = {
    "zred": 0.1,
    "logzsol": -0.3,
    "dust2": 0.2,
    "tage_gyr": 5.0,
    "tau_gyr": 1.5,
}


def main() -> None:
    from sedpy.observate import load_filters

    filters = FilterSet(load_filters(FILTER_NAMES), names=FILTER_NAMES)
    backend = FSPSBackend(sfh=DelayedTauSFH())

    phot = backend.predict_photometry(GALAXY_PARAMETERS, filters)

    print("FSPSBackend photometry")
    print(f"mass normalization: {backend.mass_normalization.name}")
    print(f"mass reference: {phot.metadata['mass_reference']}")
    print(
        "surviving fraction for the unit-formed-mass SFH: "
        f"{phot.metadata['surviving_stellar_mass_fraction']:.6f}"
    )
    print("output units: maggies")
    for band, flux in zip(phot.band_names, phot.flux):
        print(f"{band}: {flux:.8e} maggies")


if __name__ == "__main__":
    main()
