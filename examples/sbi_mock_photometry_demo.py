"""Train a tiny MAF posterior on a transparent toy photometry problem.

The mock backend is deliberately simple: two physical parameters, two bands,
and a linear color response. That makes the example useful for checking the
SBI plumbing without hiding any astronomy in the simulator.

Optional dependencies:

    pip install torch nflows
"""

from __future__ import annotations

import numpy as np

from inftools.sbi import simulate_training_set, train_maf_posterior
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.data import SEDDataset
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.units import MassNormalization


RNG_SEED = 123

PARAMETER_PRIORS = {
    "z": UniformPrior(0.0, 2.0),
    "dust2": UniformPrior(0.0, 1.0),
}

N_TRAIN = 512
N_POSTERIOR_SAMPLES = 2000


class LinearColorBackend(SEDBackend):
    """Toy backend with an exactly readable flux model.

    This is not an astrophysical SED model. It just gives composed a backend
    with the same interface as FSPS or CIGALE.
    """

    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        z = float(params["z"])
        dust = float(params["dust2"])
        flux = np.array([1.0 + z - 0.2 * dust, 0.8 + 0.5 * z + dust], dtype=float)
        return ModelPhotometry(band_names=["g", "r"], flux=flux)


def toy_noise_sigma(flux):
    """Simple heteroscedastic Gaussian noise model in flux units."""

    sigma_floor = 0.02
    frac_error = 0.05
    return sigma_floor + frac_error * np.abs(flux)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    parameter_space = ParameterSpace(
        names=list(PARAMETER_PRIORS),
        priors=PARAMETER_PRIORS,
    )

    # The dataset only defines band names and active-band masking here. The
    # simulator below will generate the actual training fluxes.
    dataset = SEDDataset(["g", "r"], flux=np.zeros(2), sigma=np.ones(2))
    likelihood = GaussianPhotometricLikelihood(LinearColorBackend(), dataset, parameter_space)

    theta_train, x_train = simulate_training_set(
        parameter_space,
        likelihood,
        n=N_TRAIN,
        noise_fn=toy_noise_sigma,
        rng=rng,
    )

    estimator = train_maf_posterior(
        theta_train,
        x_train,
        hidden_features=32,
        num_transforms=2,
        num_blocks=1,
        epochs=20,
        batch_size=128,
        device="cpu",
        seed=RNG_SEED,
    )

    theta_true = np.array([0.8, 0.3])
    x_obs = likelihood.simulate(theta_true, noise_fn=toy_noise_sigma, rng=rng)
    samples = estimator.sample(x_obs, num_samples=N_POSTERIOR_SAMPLES)

    print("true theta:", theta_true)
    print("observed flux:", x_obs)
    print("posterior mean:", samples.mean(axis=0))
    print("posterior std:", samples.std(axis=0))


if __name__ == "__main__":
    main()
