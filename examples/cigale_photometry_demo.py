"""Compute CIGALE photometry and evaluate one mock likelihood.

The point of this example is not to fit a real object. It shows the complete
data flow for one parameter vector:

1. define CIGALE modules and parameter ranges;
2. build a composed backend and ParameterSpace from those choices;
3. predict per-solar-mass photometry;
4. multiply by stellar mass explicitly;
5. build a Gaussian likelihood for the same flux vector.

Requirements: CIGALE/pcigale installed, with the CIGALE filter database
available. Filter names below are native CIGALE names.
"""

from __future__ import annotations

import numpy as np

from composed import GaussianPhotometricLikelihood, SEDDataset
from composed.backends.cigale import build_cigale_backend_and_parameter_space
from composed.filters import FilterSet
from composed.priors import UniformPrior


MODULES = ["sfhdelayed", "bc03", "redshifting"]

MODULE_PARAMETERS = {
    "sfhdelayed": {
        "tau_main": {"range": [500.0, 5000.0]},  # Myr
        "age_main": {"values": [1000, 3000, 5000], "dtype": "int"},  # Myr
    },
    "bc03": {
        "imf": 1,
        "metallicity": {"values": [0.008, 0.02]},
    },
    "redshifting": {
        "redshift": {"name": "z", "range": [0.0, 2.0]},
    },
}

FILTER_NAMES = ["sdss.u", "sdss.g", "sdss.r"]

GALAXY_PARAMETERS = {
    "log10_mass": 10.0,
    "tau_main": 2000.0,
    "age_main": 3000.0,
    "metallicity": 0.02,
    "z": 0.5,
}


def main() -> None:
    backend, parameter_space = build_cigale_backend_and_parameter_space(
        MODULES,
        MODULE_PARAMETERS,
        additional_priors={"log10_mass": UniformPrior(8.0, 12.0)},
    )
    filters = FilterSet(FILTER_NAMES)

    # CIGALE is configured to return luminosities per solar mass. The likelihood
    # applies this same factor internally when it sees log10_mass.
    params = GALAXY_PARAMETERS
    stellar_mass = 10.0 ** params["log10_mass"]
    backend_params = {name: value for name, value in params.items() if name != "log10_mass"}
    phot_per_msun = backend.predict_photometry(backend_params, filters)
    phot_absolute = stellar_mass * phot_per_msun.flux

    print("Parameter order:", parameter_space.names)
    print(f"mass normalization: {backend.mass_normalization.name}")
    print("output units: maggies")
    for name, flux in zip(phot_per_msun.band_names, phot_absolute):
        print(f"{name:12s} {flux:.6e} maggies")

    # Treat the predicted flux as a fake observation with 10% Gaussian errors.
    data = SEDDataset(
        band_names=phot_per_msun.band_names,
        flux=phot_absolute,
        sigma=0.1 * np.maximum(phot_absolute, 1e-30),
        metadata={"filters": filters},
    )
    theta = parameter_space.from_dict(params)
    likelihood = GaussianPhotometricLikelihood(backend, data, parameter_space)
    print("Self log posterior:", likelihood.log_prob(theta))


if __name__ == "__main__":
    main()
