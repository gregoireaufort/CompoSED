"""Fit mock photometry with the torch-only conditional MDN posterior.

The scientific path is the same as for MAF SBI:

1. define the backend, priors, filters, and observed uncertainties;
2. draw parameters from the prior and simulate noisy photometry;
3. train q(theta | measured flux, sigma);
4. sample physical parameters for the observed object.

Optional dependency: torch (``python -m pip install -e ".[mdn]"``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from composed import (
    Gaussian,
    MDN,
    ParameterSpace,
    PhotometricContext,
    Problem,
    SEDDataset,
    Simulate,
    UniformPrior,
    fit,
)
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.filters import FilterSet
from composed.units import MassNormalization


RNG_SEED = 123
BAND_NAMES = ["g", "r"]
FILTERS = FilterSet([object(), object()], names=BAND_NAMES)


class LinearColorBackend(SEDBackend):
    """Two-parameter flux model used only to expose the SBI calculation."""

    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        z = float(params["z"])
        dust = float(params["dust2"])
        flux = np.array([1.0 + z - 0.2 * dust, 0.8 + 0.5 * z + dust])
        return ModelPhotometry(band_names=filters.names, flux=flux)


def noise_sigma(noiseless_flux):
    """Gaussian sigma in the same flux units as the backend output."""

    return 0.02 + 0.05 * np.abs(noiseless_flux)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    parameters = ParameterSpace(
        names=["z", "dust2"],
        priors={
            "z": UniformPrior(0.0, 2.0),
            "dust2": UniformPrior(0.0, 1.0),
        },
    )
    backend = LinearColorBackend()

    truth = {"z": 0.8, "dust2": 0.3}
    noiseless_flux = backend.predict_photometry(truth, FILTERS).flux
    observed_sigma = noise_sigma(noiseless_flux)
    observed_flux = noiseless_flux + rng.normal(scale=observed_sigma)
    data = SEDDataset(BAND_NAMES, observed_flux, observed_sigma)

    problem = Problem(
        backend=backend,
        parameters=parameters,
        data=data,
        likelihood=Gaussian(),
        filters=FILTERS,
    )
    result = fit(
        problem,
        method=MDN(
            n_components=6,
            hidden_features=32,
            num_blocks=2,
            epochs=20,
            batch_size=128,
            validation_split=0.1,
            patience=5,
            num_samples=2_000,
            device="auto",
        ),
        training=Simulate(
            n=2_000,
            noise_fn=noise_sigma,
            infer=["z", "dust2"],
            context=PhotometricContext("snr_logsigma"),
        ),
        seed=RNG_SEED,
    )

    output = Path("outputs/sbi_mdn_mock_photometry_demo/mdn")
    result.inference_state.save(output, overwrite=True)

    print("device:", result.inference_state.estimator.device)
    print("true theta:", np.asarray([truth[name] for name in result.parameter_names]))
    print("observed flux:", observed_flux)
    print("observed sigma:", observed_sigma)
    print("posterior mean:", result.samples.mean(axis=0))
    print("posterior std:", result.samples.std(axis=0))
    print("checkpoint:", output)


if __name__ == "__main__":
    main()
