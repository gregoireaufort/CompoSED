"""Minimal photometric diffusion-SBI pipeline.

This is the slide-friendly version of the workflow.  Replace ``ToySEDBackend``
with FSPS, CIGALE, or JAX-CIGALE for a real science run; the rest of the data
flow is the same.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/composed_mplconfig")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from composed import (
    Gaussian,
    ParameterSpace,
    Problem,
    SEDDataset,
    Simulate,
    UniformPrior,
    fit,
)
from composed.sbi import Diffusion
from composed.backends.base import ModelPhotometry
from composed.filters import FilterSet
from composed.units import MassNormalization


class ToySEDBackend:
    """Small deterministic SED model: theta -> three positive band fluxes."""

    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        z = float(params["z"])
        log10_mass = float(params["log10_mass"])
        dust = float(params["dust"])

        mass_scale = 10.0 ** (log10_mass - 10.0)
        color = np.array([1.0 + 0.8 * z, 0.9 + 0.2 * z, 0.7 - 0.1 * z])
        attenuation = np.exp(-dust * np.array([1.8, 1.0, 0.5]))
        flux = 1.0e-9 * mass_scale * color * attenuation
        return ModelPhotometry(band_names=filters.names, flux=flux)


def noise_model(flux):
    """Flux sigma for the simulated catalog: 5% relative noise plus a floor."""

    return 0.05 * np.abs(flux) + 1.0e-12


def main():
    # 1. Point to bands.  For sedpy filters use:
    # filters = load_filter_set(["sdss_u0", "sdss_g0", "sdss_r0"])
    filters = FilterSet([object(), object(), object()], names=["u", "g", "r"])

    # 2. Declare backend and priors.
    backend = ToySEDBackend()
    priors = ParameterSpace(
        names=["z", "log10_mass", "dust"],
        priors={
            "z": UniformPrior(0.0, 2.0),
            "log10_mass": UniformPrior(8.0, 11.5),
            "dust": UniformPrior(0.0, 1.0),
        },
    )

    # 3. Define one observed SED. This toy example knows the generating truth;
    # a real analysis would read flux and sigma from a catalog.
    truth = {"z": 0.7, "log10_mass": 9.8, "dust": 0.25}
    noiseless = backend.predict_photometry(truth, filters).flux
    sigma = noise_model(noiseless)
    observed = noiseless + np.random.default_rng(6).normal(scale=sigma)
    data = SEDDataset(filters.names, observed, sigma, flux_unit="maggies")

    # 4. Bind backend, priors, data, likelihood, and filters into one Problem.
    problem = Problem(
        backend=backend,
        parameters=priors,
        data=data,
        likelihood=Gaussian(),
        filters=filters,
    )

    # 5. Simulate from this exact Problem, train diffusion, and infer the
    # observed SED. For science runs use 1e5+ simulations and longer training.
    result = fit(
        problem,
        method=Diffusion(epochs=20, batch_size=256, num_samples=256, steps=32, device="auto"),
        training=Simulate(
            n=2_000,
            noise_fn=noise_model,
            infer=["z", "log10_mass", "dust"],
            context="flux",
            feature_transform="log10_flux",
        ),
        seed=7,
    )

    # 6. Diagnostics and plots. SBI returns samples without inventing a MAP or
    # log density when the estimator does not provide one.
    output_dir = Path("outputs/minimal_photometric_diffusion_sbi")
    truth_vector = np.asarray([[truth[name] for name in result.parameter_names]])
    diagnostics = result.inference_state.diagnostics(
        result.samples[None, :, :],
        truth_vector,
        output_dir=output_dir,
    )
    median = diagnostics["summary"]["median"][0]

    print("Posterior medians:")
    for name, value, true_value in zip(result.parameter_names, median, truth_vector[0]):
        print(f"  {name:>10s}: {value:.4g} (truth {true_value:.4g})")
    print(f"Diagnostic plots written to {output_dir}")


if __name__ == "__main__":
    main()
