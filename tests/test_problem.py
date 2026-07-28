import importlib.util

import numpy as np
import pytest

import composed

from composed import (
    DeltaPrior,
    Gaussian,
    Emcee,
    Grid,
    Laplace,
    MixedTAMIS,
    PocoMC,
    Problem,
    RandomWalk,
    SEDDataset,
    SpectroPhotometricDataset,
    SpectrumDataset,
    fit,
    TAMIS,
)
from composed.backends.base import ModelPhotometry, ModelSpectrum, SEDBackend
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, UniformPrior
from composed.units import MassNormalization


def test_package_exposes_version():
    assert isinstance(composed.__version__, str)
    assert composed.__version__


def test_package_exposes_discrete_priors():
    assert composed.ChoicePrior is ChoicePrior
    assert composed.IntegerUniformPrior.__name__ == "IntegerUniformPrior"


class ParameterBackend(SEDBackend):
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        return ModelPhotometry(("g",), np.asarray([float(params["amplitude"])]))

    def predict_spectrum(self, params, wavelengths=None, wavelength_range=None, resolution=None):
        del wavelength_range, resolution
        wave = np.asarray(wavelengths, dtype=float)
        return ModelSpectrum(wave, np.full(wave.shape, float(params["amplitude"])))


class ConditionedParameterBackend(SEDBackend):
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        flux = float(params["amplitude"]) + float(params["redshift"])
        return ModelPhotometry(("g",), np.asarray([flux]))


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


def test_fit_mixed_tamis_forwards_parallel_evaluation_options():
    problem = Problem(
        ParameterBackend(),
        ParameterSpace(("amplitude",), {"amplitude": UniformPrior(1.0, 3.0)}),
        SEDDataset(("g",), np.asarray([2.0]), np.asarray([0.1])),
        Gaussian(),
    )

    result = fit(
        problem,
        MixedTAMIS(
            n_comp=1,
            T_max=2,
            n_per_iter=16,
            continuous_transform="auto",
            n_workers=2,
            batch_size=4,
            mp_context="spawn",
        ),
        x0=np.asarray([2.0]),
        seed=4,
    )

    assert result.samples.shape == (32, 1)
    assert result.metadata["sampler_meta"]["parallel_evaluation"] == {
        "enabled": True,
        "n_workers": 2,
        "batch_size": 4,
        "mp_context": "spawn",
    }


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


def test_experimental_sampler_status_is_exposed_without_ambiguity():
    assert Laplace().capabilities.experimental is True
    assert TAMIS().capabilities.experimental is True
    assert MixedTAMIS().capabilities.experimental is False
    assert "external TAMIS" in TAMIS().capabilities.limitations


def test_fit_conditions_reduce_pocomc_space_and_restore_full_result(monkeypatch):
    from inftools.core import SamplingResult
    from inftools import pocomc_adapter

    problem = Problem(
        ConditionedParameterBackend(),
        ParameterSpace(
            ("redshift", "amplitude"),
            {
                "redshift": DeltaPrior(0.5),
                "amplitude": UniformPrior(1.0, 3.0),
            },
        ),
        SEDDataset(("g",), np.asarray([2.5]), np.asarray([0.1])),
        Gaussian(),
    )
    seen = {}

    def fake_prior(parameter_space):
        seen["prior_names"] = parameter_space.names
        return object()

    def fake_run(posterior, **options):
        seen["posterior_names"] = posterior.theta_names
        seen["options"] = options
        expected = -0.5 * np.log(2.0 * np.pi * 0.1**2)
        assert posterior.log_likelihood_fn(np.asarray([2.0])) == pytest.approx(expected)
        return SamplingResult(
            samples=np.asarray([[1.8], [2.0]]),
            logp=np.asarray([-1.0, 0.0]),
            map_estimate=np.asarray([2.0]),
            cov=np.asarray([[0.04]]),
            meta={
                "weights_norm": np.asarray([0.25, 0.75]),
                "raw_chain": np.asarray([[[1.8], [2.0]]]),
            },
        )

    monkeypatch.setattr(pocomc_adapter, "pocomc_prior_from_parameter_space", fake_prior)
    monkeypatch.setattr(pocomc_adapter, "run_pocomc", fake_run)

    result = fit(
        problem,
        PocoMC(),
        conditions={"redshift": 0.5},
        x0=np.asarray([0.5, 2.0]),
        seed=11,
    )

    assert seen["prior_names"] == ("amplitude",)
    assert seen["posterior_names"] == ("amplitude",)
    assert result.parameter_names == ("redshift", "amplitude")
    assert np.all(result.samples[:, 0] == 0.5)
    assert np.allclose(result.samples[:, 1], [1.8, 2.0])
    assert np.allclose(result.weights, [0.25, 0.75])
    assert np.allclose(result.map_estimate, [0.5, 2.0])
    assert result.chain.shape == (1, 2, 2)
    assert np.all(result.chain[..., 0] == 0.5)
    assert result.metadata["conditioned_parameter_names"] == ("redshift",)
    assert result.metadata["free_parameter_names"] == ("amplitude",)


@pytest.mark.parametrize("replacement", [{"prior": object()}, {"bounds": [(1.0, 3.0)]}])
def test_problem_driven_pocomc_rejects_sampler_specific_prior_replacement(replacement):
    problem = Problem(
        ParameterBackend(),
        ParameterSpace(("amplitude",), {"amplitude": UniformPrior(1.0, 3.0)}),
        SEDDataset(("g",), np.asarray([2.0]), np.asarray([0.1])),
        Gaussian(),
    )

    with pytest.raises(ValueError, match="derives its prior from Problem.parameters"):
        fit(problem, PocoMC(**replacement), seed=4)


def test_fit_conditions_run_through_real_pocomc_when_available():
    if importlib.util.find_spec("pocomc") is None:
        pytest.skip("pocomc is not installed.")

    problem = Problem(
        ConditionedParameterBackend(),
        ParameterSpace(
            ("redshift", "amplitude"),
            {
                "redshift": UniformPrior(0.0, 1.0),
                "amplitude": UniformPrior(1.0, 3.0),
            },
        ),
        SEDDataset(("g",), np.asarray([2.5]), np.asarray([0.1])),
        Gaussian(),
    )
    result = fit(
        problem,
        PocoMC(
            sampler_kwargs={"n_effective": 32, "n_active": 16},
            run_kwargs={
                "n_total": 64,
                "n_evidence": 64,
                "progress": False,
            },
        ),
        conditions={"redshift": 0.5},
        seed=12,
    )

    assert result.parameter_names == ("redshift", "amplitude")
    assert result.samples.shape[1] == 2
    assert np.all(result.samples[:, 0] == 0.5)
    assert np.isclose(np.sum(result.weights), 1.0)


def test_fit_conditions_validate_names_values_and_full_x0():
    problem = Problem(
        ConditionedParameterBackend(),
        ParameterSpace(
            ("redshift", "amplitude"),
            {
                "redshift": UniformPrior(0.0, 1.0),
                "amplitude": UniformPrior(1.0, 3.0),
            },
        ),
        SEDDataset(("g",), np.asarray([2.5]), np.asarray([0.1])),
        Gaussian(),
    )
    method = RandomWalk(nsteps=2, proposal_cov=np.asarray([[0.01]]))

    with pytest.raises(ValueError, match="Unknown conditioned parameter"):
        fit(problem, method, conditions={"zred": 0.5}, seed=1)
    with pytest.raises(ValueError, match="outside its declared prior support"):
        fit(problem, method, conditions={"redshift": 1.5}, seed=1)
    with pytest.raises(ValueError, match="must be finite"):
        fit(problem, method, conditions={"redshift": np.nan}, seed=1)
    with pytest.raises(ValueError, match="does not match"):
        fit(
            problem,
            method,
            conditions={"redshift": 0.5},
            x0=np.asarray([0.4, 2.0]),
            seed=1,
        )


def test_fit_all_parameters_conditioned_returns_one_model_evaluation():
    problem = Problem(
        ConditionedParameterBackend(),
        ParameterSpace(
            ("redshift", "amplitude"),
            {
                "redshift": UniformPrior(0.0, 1.0),
                "amplitude": UniformPrior(1.0, 3.0),
            },
        ),
        SEDDataset(("g",), np.asarray([2.5]), np.asarray([0.1])),
        Gaussian(),
    )

    result = fit(
        problem,
        PocoMC(),
        conditions={"redshift": 0.5, "amplitude": 2.0},
        seed=5,
    )

    assert result.samples.shape == (1, 2)
    assert np.allclose(result.samples[0], [0.5, 2.0])
    assert result.parameter_names == ("redshift", "amplitude")
    assert result.metadata["free_parameter_names"] == ()
    assert result.logp[0] == pytest.approx(
        -0.5 * np.log(2.0 * np.pi * 0.1**2)
    )
