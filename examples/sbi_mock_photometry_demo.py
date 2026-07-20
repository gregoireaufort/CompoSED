"""Train and save a small uncertainty-conditioned MAF posterior.

This transparent mock follows the same stable CompoSED path as an FSPS or
CIGALE analysis: backend, priors, observed dataset, Problem, simulation noise,
MAF training, posterior sampling, and checkpointing.

Optional dependencies: torch and nflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from composed import (
    Gaussian,
    MAF,
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
    """Two-parameter linear flux model used only to expose the SBI plumbing."""

    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        z = float(params["z"])
        dust = float(params["dust2"])
        flux = np.array([1.0 + z - 0.2 * dust, 0.8 + 0.5 * z + dust])
        return ModelPhotometry(band_names=filters.names, flux=flux)


def noise_sigma(noiseless_flux):
    """Heteroscedastic Gaussian sigma in the same units as model flux."""

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
    noiseless = backend.predict_photometry(truth, FILTERS).flux
    observed_sigma = noise_sigma(noiseless)
    observed_flux = noiseless + rng.normal(scale=observed_sigma)
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
        method=MAF(
            hidden_features=32,
            num_transforms=2,
            num_blocks=1,
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

    output = Path("outputs/sbi_mock_photometry_demo/maf")
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
