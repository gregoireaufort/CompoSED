"""Compute FSPS broadband photometry for one simple tabular SFH.

This is intentionally written like a small analysis script:

1. choose filters;
2. define one physical parameter vector, with units in the variable names;
3. call the backend;
4. print the photometry and the mass-normalization convention.

Requirements: python-fsps, sedpy, and ``SPS_HOME`` pointing at the FSPS grids.
"""

from __future__ import annotations

import numpy as np

from composed.backends.fsps import FSPSBackend
from composed.filters import FilterSet


FILTER_NAMES = ["sdss_g0", "sdss_r0", "sdss_i0"]

# FSPS tabular SFH inputs are a monotonically increasing age/time grid in Gyr
# and SFR in Msun / yr. The backend validates monotonic time and non-negative
# SFR before calling FSPS.
GALAXY_PARAMETERS = {
    "zred": 0.1,
    "logzsol": -0.3,
    "dust2": 0.2,
    "tabular_time_gyr": np.array([0.01, 1.0, 5.0]),
    "tabular_sfr_msun_per_yr": np.array([1.0, 1.0, 0.2]),
}


def main() -> None:
    from sedpy.observate import load_filters

    filters = FilterSet(load_filters(FILTER_NAMES), names=FILTER_NAMES)
    backend = FSPSBackend()

    phot = backend.predict_photometry(GALAXY_PARAMETERS, filters)

    print("FSPSBackend photometry")
    print(f"mass normalization: {backend.mass_normalization.name}")
    print("output units: maggies")
    for band, flux in zip(phot.band_names, phot.flux):
        print(f"{band}: {flux:.8e} maggies")


if __name__ == "__main__":
    main()
