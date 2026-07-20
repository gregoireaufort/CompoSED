import numpy as np
import pytest

import composed

from composed import (
    Gaussian,
    Emcee,
    Grid,
    Problem,
    RandomWalk,
    SEDDataset,
    SpectroPhotometricDataset,
    SpectrumDataset,
    fit,
)
from composed.backends.base import ModelPhotometry, ModelSpectrum, SEDBackend
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, UniformPrior
from composed.units import MassNormalization


def test_package_exposes_version():
    assert isinstance(composed.__version__, str)
    assert composed.__version__


class ParameterBackend(SEDBackend):
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        return ModelPhotometry(("g",), np.asarray([float(params["amplitude"])]))

    def predict_spectrum(self, params, wavelengths=None, wavelength_range=None, resolution=None):
        del wavelength_range, resolution
        wave = np.asarray(wavelengths, dtype=float)
        return ModelSpectrum(wave, np.full(wave.shape, float(params["amplitude"])))


def amplitude_transform(params):
    return {"amplitude": params["x"]}


def test_problem_exposes_prior_likelihood_posterior_and_transform():
    space = ParameterSpace(("x",), {"x": UniformPrior(1.0, 3.0)})
    problem = Problem(
        backend=ParameterBackend(),
        parameters=space,
        parameter_transform=amplitude_transform,
        data=SEDDataset(("g",), np.asarray([2.0]), np.asarray([0.1])),
        likelihood=Gaussian(),
    )

    expected_like = -0.5 * np.log(2.0 * np.pi * 0.1**2)
    expected_prior = -np.log(2.0)
    assert problem.log_likelihood([2.0]) == pytest.approx(expected_like)
    assert problem.log_prior([2.0]) == pytest.approx(expected_prior)
    assert problem.log_posterior([2.0]) == pytest.approx(expected_like + expected_prior)
    assert problem.specification()["parameter_transform"] == "amplitude_transform"
    assert problem.specification()["mass_normalization"] == "absolute"
    assert problem.specification()["mass_reference"] is None
    assert problem.specification()["mass_convention"] == "composed.mass.surviving_stellar.v1"


def test_joint_problem_adds_two_data_terms_but_prior_once():
    space = ParameterSpace(("x",), {"x": UniformPrior(1.0, 3.0)})
    photometry = SEDDataset(("g",), np.asarray([2.0]), np.asarray([0.1]))
    spectrum = SpectrumDataset(
        wavelength=np.asarray([5000.0, 6000.0]),
        flux=np.asarray([2.0, 2.0]),
        sigma=np.asarray([0.2, 0.2]),
    )
    problem = Problem(
        ParameterBackend(),
        space,
        SpectroPhotometricDataset(photometry=photometry, spectrum=spectrum),
        Gaussian(),
        parameter_transform=amplitude_transform,
    )

    expected_like = -0.5 * np.log(2.0 * np.pi * 0.1**2)
    expected_like += -0.5 * np.sum(np.log(2.0 * np.pi * np.asarray([0.2, 0.2]) ** 2))
    assert problem.log_likelihood([2.0]) == pytest.approx(expected_like)
    assert problem.log_posterior([2.0]) == pytest.approx(expected_like - np.log(2.0))


def test_fit_random_walk_returns_normalized_public_result():
    problem = Problem(
        ParameterBackend(),
        ParameterSpace(("x",), {"x": UniformPrior(1.0, 3.0)}),
        SEDDataset(("g",), np.asarray([2.0]), np.asarray([0.1])),
        Gaussian(),
        parameter_transform=amplitude_transform,
    )

    result = fit(
        problem,
        RandomWalk(nsteps=100, proposal_cov=np.asarray([[0.02]])),
        x0=np.asarray([2.0]),
        seed=3,
    )

    assert result.sampler_name == "random_walk"
    assert result.samples.shape == (100, 1)
    assert result.parameter_names == ("x",)
    assert np.isclose(np.sum(result.weights), 1.0)
    assert result.metadata["problem"]["backend"].endswith("ParameterBackend")


def test_fit_grid_handles_finite_choice_prior():
    problem = Problem(
        ParameterBackend(),
        ParameterSpace(("amplitude",), {"amplitude": ChoicePrior([1.0, 2.0])}),
        SEDDataset(("g",), np.asarray([2.0]), np.asarray([0.1])),
        Gaussian(),
    )

    result = fit(problem, Grid())

    assert np.allclose(result.map_estimate, [2.0])
    assert result.samples.shape == (2, 1)


def test_fit_rejects_sampler_without_discrete_capability_before_running():
    problem = Problem(
        ParameterBackend(),
        ParameterSpace(("amplitude",), {"amplitude": ChoicePrior([1.0, 2.0])}),
        SEDDataset(("g",), np.asarray([2.0]), np.asarray([0.1])),
        Gaussian(),
    )

    with pytest.raises(ValueError, match="does not support.*discrete"):
        fit(problem, Emcee(nwalkers=8, nsteps=2), x0=np.asarray([1.0]), seed=2)

    assert Grid().capabilities.discrete is True
    assert Emcee().capabilities.discrete is False
