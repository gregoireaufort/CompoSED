import importlib.util

import numpy as np
import pytest

from composed.backends.base import ModelPhotometry
from composed.data import SEDDataset
from composed.filters import FilterSet
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.problem import Gaussian, Problem, fit
from composed.sbi import (
    Diffusion,
    MAF,
    SBITrainingSet,
    Simulate,
    simulate_photometric_training_set,
    simulate_sbi_training_set,
    train_sbi,
    train_diffusion_photometric_sbi,
    train_maf_photometric_sbi,
    transform_photometry,
)
from composed.units import MassNormalization


class LinearColorBackend:
    """Small parameter-dependent backend for SBI pipeline tests."""

    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        z = float(params["z"])
        mass = float(params["log10_mass"])
        flux = np.array(
            [
                1.0 + 0.20 * z + 0.03 * (mass - 10.0),
                0.8 + 0.10 * z - 0.02 * (mass - 10.0),
                0.6 + 0.05 * z + 0.01 * (mass - 10.0),
            ],
            dtype=float,
        )
        return ModelPhotometry(band_names=filters.names, flux=flux)


def zero_noise(flux):
    return np.zeros_like(flux)


def small_noise(flux):
    return 0.02 * np.abs(flux)


def shifted_flux_features(flux):
    return np.asarray(flux, dtype=float) + 0.5


def toy_parameter_space():
    return ParameterSpace(
        names=["z", "log10_mass", "dust"],
        priors={
            "z": UniformPrior(0.0, 1.0),
            "log10_mass": UniformPrior(9.0, 11.0),
            "dust": UniformPrior(0.0, 0.5),
        },
    )


def toy_filters():
    return FilterSet([object(), object(), object()], names=["u", "g", "r"])


def toy_problem(mask=None):
    return Problem(
        backend=LinearColorBackend(),
        parameters=toy_parameter_space(),
        data=SEDDataset(
            band_names=["u", "g", "r"],
            flux=np.asarray([1.1, 0.9, 0.65]),
            sigma=np.asarray([0.05, 0.05, 0.05]),
            mask=mask,
        ),
        likelihood=Gaussian(),
        filters=toy_filters(),
    )


def test_transform_photometry_abmag_and_shape_errors():
    flux = np.array([[1.0, 0.1]])
    mag = transform_photometry(flux, "abmag")
    assert mag.shape == flux.shape
    assert np.allclose(mag, [[0.0, 2.5]])

    with pytest.raises(ValueError, match="strictly positive"):
        transform_photometry(np.array([[1.0, 0.0]]), "abmag")

    with pytest.raises(ValueError, match="changed shape"):
        transform_photometry(flux, lambda x: x[:, :1])


def test_simulate_photometric_training_set_uses_active_bands_and_inferred_subset():
    training = simulate_photometric_training_set(
        backend=LinearColorBackend(),
        filters=toy_filters(),
        parameter_space=toy_parameter_space(),
        noise_fn=zero_noise,
        n=12,
        infer=["z", "log10_mass"],
        mask=[True, False, True],
        feature_transform="flux",
        rng=123,
    )

    assert training.theta_full.shape == (12, 3)
    assert training.theta.shape == (12, 2)
    assert training.x_flux.shape == (12, 2)
    assert training.x.shape == (12, 2)
    assert training.band_names == ("u", "r")
    assert training.theta_names == ("z", "log10_mass")
    assert training.joint_features.shape == (12, 4)
    assert training.feature_metadata.groups["photometry"] == (0, 1)
    assert training.feature_metadata.groups["parameters"] == (2, 3)
    assert training.metadata["simulate_training_set"]["failures"] == []


def test_problem_simulation_is_the_declared_sbi_source():
    problem = toy_problem(mask=[True, False, True])
    training = simulate_sbi_training_set(
        problem,
        Simulate(n=12, noise_fn=zero_noise, infer=["z", "log10_mass"]),
        rng=123,
    )

    assert training.source == "composed.problem.simulate"
    assert training.x_names == ("u", "r")
    assert training.theta_names == ("z", "log10_mass")
    assert training.theta_full.shape == (12, 3)
    assert training.metadata["problem"] == problem.specification()


def test_preexisting_training_set_is_complete_without_a_problem():
    training = SBITrainingSet.from_arrays(
        theta=np.asarray([[0.1, 9.0], [0.5, 10.0], [0.9, 11.0]]),
        x=np.asarray([[1.0, 0.8], [1.1, 0.9], [1.2, 1.0]]),
        theta_names=["z", "log10_mass"],
        x_names=["g", "r"],
        source="empirical_catalog_labels",
        metadata={"catalog": "toy"},
    )

    assert training.source == "empirical_catalog_labels"
    assert training.observation_group == "observations"
    assert training.joint_features.shape == (3, 4)
    assert training.metadata["catalog"] == "toy"

    with_nonfinite = SBITrainingSet.from_arrays(
        theta=np.asarray([[0.1], [np.nan], [0.3]]),
        x=np.asarray([[1.0], [1.1], [1.2]]),
        theta_names=["z"],
        x_names=["g"],
        source="catalog_with_missing_label",
        finite="drop",
    )
    assert with_nonfinite.theta.shape == (2, 1)
    assert with_nonfinite.metadata["dropped_nonfinite_rows"] == 1

    grouped = SBITrainingSet.from_arrays(
        theta=np.asarray([[0.1], [0.2]]),
        x=np.asarray([[22.0, 0.1], [23.0, 0.2]]),
        theta_names=["z"],
        x_names=["g", "g_err"],
        source="empirical_catalog_with_errors",
        observation_groups={"mags": ["g"], "magerrs": ["g_err"]},
    )
    assert grouped.feature_metadata.groups["mags"] == (0,)
    assert grouped.feature_metadata.groups["magerrs"] == (1,)


def test_problem_fit_rejects_unrelated_preexisting_training_set():
    external = SBITrainingSet.from_arrays(
        theta=np.asarray([[0.1], [0.2]]),
        x=np.asarray([[1.0], [1.1]]),
        theta_names=["z"],
        x_names=["flux"],
        source="external",
    )
    with pytest.raises(TypeError, match="training=Simulate"):
        fit(toy_problem(), method=MAF(epochs=1), training=external, seed=1)


def test_problem_sbi_rejects_implicit_upper_limit_encoding():
    data = SEDDataset(
        band_names=["u", "g", "r"],
        flux=np.asarray([1.0, np.nan, 0.7]),
        sigma=np.asarray([0.1, 0.1, 0.1]),
        upper_limit=np.asarray([0.0, 0.3, 0.0]),
        upper_limit_mask=np.asarray([False, True, False]),
    )
    problem = Problem(
        LinearColorBackend(),
        toy_parameter_space(),
        data,
        Gaussian(),
        filters=toy_filters(),
    )
    with pytest.raises(NotImplementedError, match="upper-limit feature encoding"):
        simulate_sbi_training_set(problem, Simulate(n=4, noise_fn=zero_noise), rng=2)


@pytest.mark.diffusion
def test_train_diffusion_photometric_sbi_shape_and_clamping():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    result = train_diffusion_photometric_sbi(
        backend=LinearColorBackend(),
        filters=toy_filters(),
        parameter_space=toy_parameter_space(),
        noise_fn=small_noise,
        n_train=64,
        infer=["z", "log10_mass"],
        feature_transform="flux",
        rng=5,
        model="mlp",
        hidden_features=16,
        model_config={"mlp_blocks": 1, "emb_dim": 16, "time_hidden": 16},
        sigma_min=0.05,
        sigma_max=1.0,
        learning_rate=1e-3,
        device="cpu",
        epochs=1,
        batch_size=16,
        seed=6,
    )

    assert np.isfinite(result.history["train_loss"][-1])
    x_obs = result.training_set.x[:3]
    samples = result.sample(x_obs, num_samples=4, steps=3, sampler="edm_euler")
    assert samples.shape == (3, 4, 2)
    assert np.all(np.isfinite(samples))

    joint = result.sample_joint(x_obs, num_samples=4, steps=3, sampler="edm_euler")
    assert np.allclose(joint[:, :, :3], x_obs[:, None, :])


@pytest.mark.diffusion
def test_standalone_preexisting_diffusion_uses_generic_observation_group():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    rng = np.random.default_rng(9)
    theta = rng.normal(size=(48, 2))
    x = theta + 0.2 * rng.normal(size=(48, 2))
    training = SBITrainingSet.from_arrays(
        theta,
        x,
        theta_names=["a", "b"],
        x_names=["x0", "x1"],
        source="numerical_simulation",
    )
    trained = train_sbi(
        training,
        Diffusion(
            model="mlp",
            hidden_features=16,
            model_config={"mlp_blocks": 1, "emb_dim": 16, "time_hidden": 16},
            epochs=1,
            batch_size=16,
            validation_split=0.2,
        ),
        seed=10,
    )
    assert "observations" in trained.estimator.feature_metadata.groups
    assert len(trained.history["val_loss"]) == 1
    samples = trained.sample(x[:2], num_samples=3, steps=2, sampler="edm_euler")
    assert samples.shape == (2, 3, 2)


@pytest.mark.sbi
def test_train_maf_photometric_sbi_shape():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch/nflows are not installed.")

    training = simulate_photometric_training_set(
        backend=LinearColorBackend(),
        filters=toy_filters(),
        parameter_space=toy_parameter_space(),
        noise_fn=small_noise,
        n=48,
        infer=["z", "log10_mass"],
        feature_transform=shifted_flux_features,
        rng=15,
    )
    assert training.feature_transform_name == "shifted_flux_features"
    result = train_maf_photometric_sbi(
        training,
        hidden_features=16,
        num_transforms=2,
        num_blocks=1,
        learning_rate=1e-3,
        device="cpu",
        epochs=1,
        batch_size=16,
        seed=16,
    )
    samples = result.sample(training.x[:3], num_samples=5)
    assert samples.shape == (3, 5, 2)
    assert np.all(np.isfinite(samples))

    one_object = result.sample(training.x[0], num_samples=5)
    assert one_object.shape == (1, 5, 2)

    from_flux = result.sample(training.x_flux[:2], input_units="flux", num_samples=5)
    assert from_flux.shape == (2, 5, 2)


@pytest.mark.sbi
def test_standalone_preexisting_maf_and_problem_fit_share_training_api():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch/nflows are not installed.")

    rng = np.random.default_rng(33)
    theta = rng.normal(size=(64, 2))
    x = theta + 0.1 * rng.normal(size=(64, 2))
    external = SBITrainingSet.from_arrays(
        theta,
        x,
        theta_names=["a", "b"],
        x_names=["x0", "x1"],
        source="presampled_forward_model",
    )
    trained = train_sbi(
        external,
        MAF(hidden_features=16, num_transforms=2, num_blocks=1, epochs=1, batch_size=16, num_samples=5),
        seed=34,
    )
    assert trained.training_set.source == "presampled_forward_model"
    assert trained.sample(x[:2], num_samples=5).shape == (2, 5, 2)

    result = fit(
        toy_problem(),
        method=MAF(
            hidden_features=16,
            num_transforms=2,
            num_blocks=1,
            epochs=1,
            batch_size=16,
            num_samples=5,
        ),
        training=Simulate(n=64, noise_fn=small_noise, infer=["z", "log10_mass"]),
        seed=35,
    )
    assert result.samples.shape == (5, 2)
    assert result.logp is None
    assert result.map_estimate is None
    assert result.inference_state.training_set.source == "composed.problem.simulate"
