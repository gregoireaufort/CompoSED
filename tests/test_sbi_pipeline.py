import importlib.util
import json

import numpy as np
import pytest

import composed
from composed.backends.base import ModelPhotometry
from composed.data import SEDDataset, SpectrumDataset
from composed.filters import FilterSet
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.problem import Gaussian, Problem, fit
from composed.sbi import (
    Diffusion,
    MAF,
    MDN,
    PhotometricContext,
    PriorSupportTransform,
    SBITrainingSet,
    Simulate,
    TrainedDiffusionSBI,
    TrainedMAFSBI,
    TrainedMDNSBI,
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


def test_stable_top_level_exports_maf_and_mdn_but_not_experimental_diffusion():
    assert hasattr(composed, "MAF")
    assert hasattr(composed, "MDN")
    assert hasattr(composed, "PhotometricContext")
    assert hasattr(composed, "TrainedMAFSBI")
    assert hasattr(composed, "TrainedMDNSBI")
    assert not hasattr(composed, "Diffusion")
    assert not hasattr(composed, "TrainedDiffusionSBI")


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
        Simulate(n=12, noise_fn=zero_noise, infer=["z", "log10_mass"], context="flux"),
        rng=123,
    )

    assert training.source == "composed.problem.simulate"
    assert training.x_names == ("u", "r")
    assert training.theta_names == ("z", "log10_mass")
    assert training.theta_full.shape == (12, 3)
    assert training.metadata["problem"] == problem.specification()


def test_problem_simulation_separates_condition_columns_from_inferred_targets():
    problem = toy_problem()
    training = simulate_sbi_training_set(
        problem,
        Simulate(
            n=24,
            noise_fn=small_noise,
            infer=["log10_mass"],
            condition_on=["z"],
        ),
        rng=17,
    )

    z_index = problem.parameters.names.index("z")
    assert training.theta_names == ("log10_mass",)
    assert training.condition_names == ("z",)
    assert training.condition_values.shape == (24, 1)
    assert np.allclose(training.condition_values[:, 0], training.theta_full[:, z_index])
    assert training.x_names[-1] == "condition:z"
    assert np.allclose(training.x[:, -1], training.condition_values[:, 0])
    assert training.observation_groups["conditions"] == ("condition:z",)


def test_problem_fit_sbi_uses_conditions_as_training_context(monkeypatch):
    import composed.sbi as sbi_module

    problem = toy_problem()
    seen = {}

    class FakeTrained:
        def __init__(self, training_set):
            self.training_set = training_set
            self.theta_names = training_set.theta_names
            self.history = {"train_loss": [1.0]}
            self.estimator = type("Estimator", (), {"device": "cpu"})()

        def sample(
            self,
            photometry,
            *,
            conditions=None,
            num_samples,
            batch_size=None,
            seed=None,
        ):
            del photometry, batch_size, seed
            seen["inference_conditions"] = conditions
            values = np.linspace(9.8, 10.2, int(num_samples))
            return values[None, :, None]

    def fake_train_sbi(training_set, method, *, seed=None):
        del method, seed
        seen["training_set"] = training_set
        return FakeTrained(training_set)

    monkeypatch.setattr(sbi_module, "train_sbi", fake_train_sbi)

    result = fit(
        problem,
        method=MAF(num_samples=5),
        training=Simulate(
            n=32,
            noise_fn=small_noise,
            infer=["log10_mass"],
        ),
        conditions={"z": 0.35},
        seed=18,
    )

    training = seen["training_set"]
    assert training.condition_names == ("z",)
    assert seen["inference_conditions"] == {"z": 0.35}
    assert result.samples.shape == (5, 2)
    assert result.parameter_names == ("z", "log10_mass")
    assert np.all(result.samples[:, 0] == 0.35)
    assert np.allclose(result.samples[:, 1], np.linspace(9.8, 10.2, 5))
    assert result.metadata["conditions"] == {"z": 0.35}
    assert result.metadata["conditioned_parameter_names"] == ("z",)
    assert result.metadata["marginalized_parameter_names"] == ("dust",)


def test_problem_fit_sbi_rejects_target_condition_overlap():
    problem = toy_problem()

    with pytest.raises(ValueError, match="both inferred and conditioned"):
        fit(
            problem,
            method=MAF(num_samples=2),
            training=Simulate(
                n=8,
                noise_fn=small_noise,
                infer=["z", "log10_mass"],
            ),
            conditions={"z": 0.35},
            seed=19,
        )


@pytest.mark.sbi
def test_problem_fit_maf_conditions_survive_checkpoint_roundtrip(tmp_path):
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch or nflows is not installed.")

    problem = toy_problem()
    result = fit(
        problem,
        method=MAF(
            hidden_features=16,
            num_transforms=2,
            num_blocks=1,
            epochs=2,
            batch_size=16,
            validation_split=0.2,
            patience=None,
            num_samples=4,
            inference_batch_size=1,
            device="cpu",
        ),
        training=Simulate(
            n=96,
            noise_fn=small_noise,
            infer=["log10_mass"],
        ),
        conditions={"z": 0.35},
        seed=22,
    )

    assert result.samples.shape == (4, 2)
    assert result.parameter_names == ("z", "log10_mass")
    assert np.all(result.samples[:, 0] == 0.35)
    assert result.inference_state.condition_names == ("z",)
    checkpoint = result.inference_state.save(tmp_path / "conditioned_maf")
    loaded = TrainedMAFSBI.load(checkpoint, device="cpu")
    draws = loaded.sample(
        problem.data,
        conditions={"z": 0.35},
        num_samples=3,
        seed=23,
    )
    assert draws.shape == (1, 3, 1)
    assert np.all(np.isfinite(draws))


def test_trained_maf_appends_named_conditions_to_native_context():
    problem = toy_problem()
    training = simulate_sbi_training_set(
        problem,
        Simulate(
            n=16,
            noise_fn=small_noise,
            infer=["log10_mass"],
            condition_on=["z"],
        ),
        rng=20,
    )

    class RecordingEstimator:
        theta_dim = 1
        x_dim = training.x.shape[1]

        def __init__(self):
            self.context = None

        def sample(self, context, num_samples):
            self.context = np.asarray(context)
            return np.zeros((int(num_samples), 1))

    estimator = RecordingEstimator()
    trained = TrainedMAFSBI(
        estimator=estimator,
        training_set=training,
        history={},
    )
    draws = trained.sample(
        problem.data,
        conditions={"z": 0.35},
        num_samples=3,
    )

    assert draws.shape == (1, 3, 1)
    assert estimator.context.shape == (1, training.x.shape[1])
    assert estimator.context[0, -1] == pytest.approx(0.35)
    with pytest.raises(ValueError, match="requires condition values"):
        trained.sample(problem.data, num_samples=2)


def test_trained_diffusion_appends_named_conditions_to_native_context():
    problem = toy_problem()
    training = simulate_sbi_training_set(
        problem,
        Simulate(
            n=16,
            noise_fn=small_noise,
            infer=["log10_mass"],
            condition_on=["z"],
        ),
        rng=21,
    )

    class RecordingDiffusionEstimator:
        def __init__(self):
            self.known = None

        def sample(self, known, mask, *, num_samples, **kwargs):
            del kwargs
            self.known = np.asarray(known)
            cube = np.zeros(
                (known.shape[0], int(num_samples), known.shape[1]),
                dtype=float,
            )
            cube[:] = np.where(mask, known, 0.0)[:, None, :]
            return cube

    estimator = RecordingDiffusionEstimator()
    trained = TrainedDiffusionSBI(
        estimator=estimator,
        training_set=training,
        history={},
        mask_config={},
    )
    draws = trained.sample(
        problem.data,
        conditions={"z": 0.35},
        num_samples=3,
        steps=2,
    )

    condition_index = training.diffusion_observation_size - 1
    assert draws.shape == (1, 3, 1)
    assert estimator.known[0, condition_index] == pytest.approx(0.35)


def test_default_problem_sbi_context_retains_exact_sigma_and_accepts_negative_flux():
    context = PhotometricContext("snr_logsigma", flux_unit="maggies")
    encoded = context.encode(
        np.asarray([[-0.2, 0.4]]),
        np.asarray([[0.1, 0.2]]),
    )
    assert np.allclose(encoded, [[-2.0, 2.0, -1.0, np.log10(0.2)]])
    assert context.feature_names(["g", "r"]) == (
        "snr:g",
        "snr:r",
        "log10_sigma:g",
        "log10_sigma:r",
    )

    training = simulate_sbi_training_set(
        toy_problem(mask=[True, False, True]),
        Simulate(n=12, noise_fn=small_noise, infer=["z", "log10_mass"]),
        rng=123,
    )
    assert training.x.shape == (12, 4)
    assert training.x_native.shape == training.sigma_native.shape == (12, 2)
    assert np.all(training.sigma_native > 0.0)
    assert np.allclose(
        training.x,
        np.column_stack(
            [
                training.x_native / training.sigma_native,
                np.log10(training.sigma_native),
            ]
        ),
    )
    assert training.x_names == (
        "snr:u",
        "snr:r",
        "log10_sigma:u",
        "log10_sigma:r",
    )
    assert training.observation_groups == {
        "photometry": ("snr:u", "snr:r"),
        "uncertainty": ("log10_sigma:u", "log10_sigma:r"),
    }


def test_preexisting_photometry_constructor_uses_same_context_contract():
    parameter_space = ParameterSpace(["z"], {"z": UniformPrior(0.0, 2.0)})
    training = SBITrainingSet.from_photometry(
        theta=np.asarray([[0.2], [1.0]]),
        flux=np.asarray([[1.0, -0.1], [2.0, 0.2]]),
        sigma=np.asarray([[0.1, 0.2], [0.4, 0.1]]),
        theta_names=["z"],
        band_names=["g", "r"],
        source="presampled_photometry",
        parameter_space=parameter_space,
    )
    assert training.x.shape == (2, 4)
    assert training.band_names == ("g", "r")
    assert training.context.mode == "snr_logsigma"
    unconstrained = training.theta_transform.transform(training.theta)
    assert np.allclose(training.theta_transform.inverse(unconstrained), training.theta)


def test_prior_support_transform_roundtrip_support_and_jacobian():
    from composed.priors import LogUniformPrior, NormalPrior, StudentTPrior

    space = ParameterSpace(
        ["u", "logu", "normal", "student_t"],
        {
            "u": UniformPrior(-2.0, 3.0),
            "logu": LogUniformPrior(0.1, 100.0),
            "normal": NormalPrior(0.0, 2.0),
            "student_t": StudentTPrior(2.0, 0.0, 0.3),
        },
    )
    transform = PriorSupportTransform.from_parameter_space(space, space.names)
    theta = np.asarray([[-1.0, 0.2, -3.0, -0.2], [2.5, 50.0, 4.0, 0.6]])
    unconstrained = transform.transform(theta)
    recovered = transform.inverse(unconstrained)

    assert np.allclose(recovered, theta)
    assert np.all(np.isfinite(transform.log_abs_det_forward(theta)))
    extreme = transform.inverse(np.asarray([[-1.0e6, 1.0e6, 0.0, -3.0]]))
    assert -2.0 <= extreme[0, 0] <= 3.0
    assert 0.1 <= extreme[0, 1] <= 100.0
    assert extreme[0, 2] == 0.0
    assert extreme[0, 3] == -3.0


def test_maf_catalog_sampling_chunks_context_rows_without_object_loops():
    class FakeEstimator:
        theta_dim = 1
        x_dim = 2

        def __init__(self):
            self.calls = []

        def sample(self, context, num_samples):
            context = np.asarray(context)
            self.calls.append(context.shape[0])
            return np.repeat(context[:, None, :1], int(num_samples), axis=1)

    training = SBITrainingSet.from_arrays(
        theta=np.asarray([[0.0], [1.0]]),
        x=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        theta_names=["theta"],
        x_names=["x0", "x1"],
        source="batching_test",
    )
    estimator = FakeEstimator()
    trained = TrainedMAFSBI(estimator, training, history={"train_loss": [0.0]})
    samples = trained.sample(np.ones((5, 2)), num_samples=3, batch_size=2)

    assert estimator.calls == [2, 2, 1]
    assert samples.shape == (5, 3, 1)

    estimator.calls.clear()
    summary = trained.summarize_catalog(
        np.arange(10, dtype=float).reshape(5, 2),
        num_samples=4,
        batch_size=2,
    )
    assert estimator.calls == [2, 2, 1]
    assert summary.quantile_values.shape == (5, 3, 1)
    assert np.allclose(summary.median[:, 0], [0.0, 2.0, 4.0, 6.0, 8.0])
    assert summary.mean.dtype == np.float32


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


def test_problem_fit_maf_rejects_spectral_data_with_controlled_error():
    problem = Problem(
        backend=LinearColorBackend(),
        parameters=toy_parameter_space(),
        data=SpectrumDataset(
            wavelength=np.asarray([5000.0, 5001.0]),
            flux=np.asarray([1.0, 1.0]),
            sigma=np.asarray([0.1, 0.1]),
        ),
        likelihood=Gaussian(),
    )

    with pytest.raises(
        NotImplementedError,
        match="supports photometric SEDDataset observations",
    ):
        fit(
            problem,
            method=MAF(epochs=1),
            training=Simulate(n=8, noise_fn=small_noise),
            seed=1,
        )


def test_simulate_configuration_rejects_negative_retry_budget():
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        Simulate(n=8, noise_fn=small_noise, max_retries=-1)


def test_preexisting_photometry_encodes_availability_and_upper_limits_explicitly():
    training = SBITrainingSet.from_photometry(
        theta=np.asarray([[0.2], [0.7]]),
        flux=np.asarray([[1.0, np.nan, np.nan], [2.0, 0.5, np.nan]]),
        sigma=np.asarray([[0.1, 0.2, np.nan], [0.2, 0.1, 0.05]]),
        measurement_mask=np.asarray(
            [[True, True, False], [True, True, True]]
        ),
        upper_limit=np.asarray(
            [[np.nan, 0.3, np.nan], [np.nan, np.nan, 0.2]]
        ),
        upper_limit_mask=np.asarray(
            [[False, True, False], [False, False, True]]
        ),
        theta_names=["z"],
        band_names=["u", "g", "r"],
        source="censored_catalog",
        parameter_space=ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)}),
    )

    observation = training.diffusion_observation_features
    groups = training.diffusion_feature_metadata.groups
    assert observation.shape == (2, 18)
    assert np.allclose(observation[:, groups["photometry"]], [[10.0, 0.0, 0.0], [10.0, 5.0, 0.0]])
    assert np.array_equal(
        observation[:, groups["availability"]],
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
    )
    assert np.array_equal(
        observation[:, groups["censoring"]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert np.allclose(
        observation[:, groups["upper_limit"]],
        [[0.0, 1.5, 0.0], [0.0, 0.0, 4.0]],
    )
    assert np.allclose(
        training.diffusion_joint_features[:, -1:],
        training.theta_transform.transform(training.theta),
    )


def test_problem_sbi_encodes_simulated_censoring_without_latent_flux_leakage():
    data = SEDDataset(
        band_names=["u", "g", "r"],
        flux=np.asarray([1.0, np.nan, 0.7]),
        sigma=np.asarray([0.1, 0.1, 0.1]),
        upper_limit=np.asarray([0.0, 0.84, 0.0]),
        upper_limit_mask=np.asarray([False, True, False]),
    )
    problem = Problem(
        LinearColorBackend(),
        toy_parameter_space(),
        data,
        Gaussian(),
        filters=toy_filters(),
    )
    training = simulate_sbi_training_set(
        problem,
        Simulate(n=128, noise_fn=small_noise),
        rng=2,
    )

    assert training.has_photometric_state
    assert np.all(training.measurement_mask_native)
    assert np.all(~training.upper_limit_mask_native[:, [0, 2]])
    censored = training.upper_limit_mask_native[:, 1]
    assert np.any(censored)
    assert np.any(~censored)
    assert np.all(np.isnan(training.x_native[censored, 1]))
    assert np.all(training.x_native[~censored, 1] > 0.84)
    assert np.all(training.upper_limit_native[:, 1] == 0.84)
    assert np.all(np.isnan(training.upper_limit_native[:, [0, 2]]))
    assert training.metadata["observation_state"]["censoring_rule"] == (
        "measured_flux <= upper_limit"
    )

    with pytest.raises(NotImplementedError, match="MAF does not yet encode"):
        fit(
            problem,
            method=MAF(epochs=1),
            training=Simulate(n=8, noise_fn=small_noise),
            seed=3,
        )


def test_diffusion_data_state_and_conditioning_mask_are_distinct():
    parameter_space = ParameterSpace(
        ["z", "mass"],
        {
            "z": UniformPrior(0.0, 1.0),
            "mass": UniformPrior(9.0, 11.0),
        },
    )
    training = SBITrainingSet.from_photometry(
        theta=np.asarray([[0.2, 9.5], [0.8, 10.5]]),
        flux=np.asarray([[1.0, np.nan], [1.2, 0.8]]),
        sigma=np.asarray([[0.1, np.nan], [0.1, 0.2]]),
        measurement_mask=np.asarray([[True, False], [True, True]]),
        theta_names=["z", "mass"],
        band_names=["g", "r"],
        source="state_mask_test",
        parameter_space=parameter_space,
    )

    class FakeDiffusionEstimator:
        def sample(self, known, mask, *, num_samples, **kwargs):
            self.known = np.asarray(known)
            self.mask = np.asarray(mask)
            samples = np.repeat(
                np.nan_to_num(self.known, nan=0.0)[:, None, :],
                int(num_samples),
                axis=1,
            )
            samples[:, :, -2:] = np.asarray([-1.0e6, 1.0e6])
            return samples

    estimator = FakeDiffusionEstimator()
    trained = TrainedDiffusionSBI(
        estimator=estimator,
        training_set=training,
        history={"train_loss": [0.0]},
        mask_config={},
    )
    data = SEDDataset(
        band_names=["g", "r"],
        flux=np.asarray([1.0, np.nan]),
        sigma=np.asarray([0.1, np.nan]),
        mask=np.asarray([True, False]),
    )
    samples = trained.sample(data, num_samples=3)

    n_observation = training.diffusion_observation_size
    availability_cols = training.diffusion_feature_metadata.groups["availability"]
    assert np.array_equal(estimator.known[0, availability_cols], [1.0, 0.0])
    assert np.all(estimator.mask[:, :n_observation])
    assert np.all(~estimator.mask[:, n_observation:])
    assert samples.shape == (1, 3, 2)
    assert np.all((samples[:, :, 0] >= 0.0) & (samples[:, :, 0] <= 1.0))
    assert np.all((samples[:, :, 1] >= 9.0) & (samples[:, :, 1] <= 11.0))


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
    x_obs = result.training_set.diffusion_observation_features[:3]
    samples = result.sample(x_obs, num_samples=4, steps=3, sampler="edm_euler")
    assert samples.shape == (3, 4, 2)
    assert np.all(np.isfinite(samples))
    assert np.all((samples[:, :, 0] >= 0.0) & (samples[:, :, 0] <= 1.0))
    assert np.all((samples[:, :, 1] >= 9.0) & (samples[:, :, 1] <= 11.0))

    joint = result.sample_joint(x_obs, num_samples=4, steps=3, sampler="edm_euler")
    n_observation = result.training_set.diffusion_observation_size
    assert np.allclose(joint[:, :, :n_observation], x_obs[:, None, :])


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


@pytest.mark.diffusion
def test_problem_fit_diffusion_uses_sigma_upper_limit_and_physical_priors():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    problem = Problem(
        LinearColorBackend(),
        toy_parameter_space(),
        SEDDataset(
            band_names=["u", "g", "r"],
            flux=np.asarray([1.0, np.nan, 0.7]),
            sigma=np.asarray([0.1, 0.1, 0.1]),
            upper_limit=np.asarray([0.0, 0.84, 0.0]),
            upper_limit_mask=np.asarray([False, True, False]),
        ),
        Gaussian(),
        filters=toy_filters(),
    )
    result = fit(
        problem,
        method=Diffusion(
            model="mlp",
            hidden_features=16,
            model_config={"mlp_blocks": 1, "emb_dim": 16, "time_hidden": 16},
            epochs=1,
            batch_size=16,
            num_samples=4,
            steps=2,
            sampler="edm_euler",
        ),
        training=Simulate(
            n=64,
            noise_fn=small_noise,
            infer=["z", "log10_mass"],
        ),
        seed=19,
    )

    assert result.samples.shape == (4, 2)
    assert np.all((result.samples[:, 0] >= 0.0) & (result.samples[:, 0] <= 1.0))
    assert np.all((result.samples[:, 1] >= 9.0) & (result.samples[:, 1] <= 11.0))
    training = result.inference_state.training_set
    assert training.context.mode == "snr_logsigma"
    assert "censoring" in training.diffusion_observation_groups
    assert tuple(result.inference_state.mask_config["tie_groups"]) == tuple(
        training.diffusion_observation_groups
    )


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
    assert result.inference_state.training_set.context.mode == "snr_logsigma"
    assert np.all((result.samples[:, 0] >= 0.0) & (result.samples[:, 0] <= 1.0))
    assert np.all((result.samples[:, 1] >= 9.0) & (result.samples[:, 1] <= 11.0))


@pytest.mark.sbi
def test_problem_fit_mdn_returns_normalized_bounded_physical_posterior():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    problem = toy_problem()
    result = fit(
        problem,
        method=MDN(
            n_components=3,
            hidden_features=16,
            num_blocks=2,
            epochs=3,
            batch_size=16,
            num_samples=7,
            device="cpu",
        ),
        training=Simulate(
            n=96,
            noise_fn=small_noise,
            infer=["z", "log10_mass"],
        ),
        seed=46,
    )

    assert result.sampler_name == "mdn"
    assert result.samples.shape == (7, 2)
    assert np.all((result.samples[:, 0] >= 0.0) & (result.samples[:, 0] <= 1.0))
    assert np.all((result.samples[:, 1] >= 9.0) & (result.samples[:, 1] <= 11.0))
    logp = result.inference_state.log_prob(result.samples, problem.data)
    assert logp.shape == (7,)
    assert np.all(np.isfinite(logp))


@pytest.mark.sbi
def test_trained_mdn_save_load_roundtrip_preserves_density_and_samples(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    training = simulate_sbi_training_set(
        toy_problem(),
        Simulate(n=96, noise_fn=small_noise, infer=["z", "log10_mass"]),
        rng=61,
    )
    trained = train_sbi(
        training,
        MDN(
            n_components=3,
            hidden_features=16,
            num_blocks=2,
            epochs=3,
            batch_size=16,
            device="cpu",
        ),
        seed=62,
    )
    checkpoint = trained.save(tmp_path / "mdn")
    loaded = TrainedMDNSBI.load(checkpoint, device="cpu")

    theta = np.asarray([[0.2, 9.5], [0.8, 10.5]])
    context = training.x[:2]
    assert np.allclose(
        loaded.log_prob(theta, context),
        trained.log_prob(theta, context),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    original_samples = trained.sample(context, num_samples=5, seed=63)
    loaded_samples = loaded.sample(context, num_samples=5, seed=63)
    assert np.allclose(loaded_samples, original_samples)


@pytest.mark.sbi
def test_trained_maf_save_load_roundtrip_preserves_schema_and_weights(tmp_path):
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch/nflows are not installed.")

    problem = toy_problem()
    training = simulate_sbi_training_set(
        problem,
        Simulate(n=64, noise_fn=small_noise, infer=["z", "log10_mass"]),
        rng=51,
    )
    trained = train_sbi(
        training,
        MAF(
            hidden_features=16,
            num_transforms=2,
            num_blocks=1,
            epochs=2,
            batch_size=16,
            validation_split=0.2,
            patience=2,
            device="cpu",
        ),
        seed=52,
    )
    checkpoint = trained.save(tmp_path / "toy_maf")
    loaded = type(trained).load(checkpoint, device="cpu")
    manifest = json.loads((checkpoint / "manifest.json").read_text())

    before = trained.sample(problem.data, num_samples=7, batch_size=1, seed=53)
    after = loaded.sample(problem.data, num_samples=7, batch_size=1, seed=53)

    assert loaded.training_set is None
    assert loaded.theta_names == trained.theta_names
    assert loaded.x_names == trained.x_names
    assert loaded.band_names == trained.band_names
    assert loaded.context.specification() == trained.context.specification()
    saved_problem = manifest["metadata"]["training_set_metadata"]["problem"]
    assert saved_problem["backend"] == problem.specification()["backend"]
    assert saved_problem["parameters"] == list(problem.parameters.names)
    assert manifest["metadata"]["training_set_metadata"]["noise_model"] == "small_noise"
    assert np.allclose(after, before)
    assert np.all(np.isfinite(loaded.log_prob(after[0, :3], problem.data)))
    assert {path.name for path in checkpoint.iterdir()} == {
        "manifest.json",
        "standardizers.npz",
        "weights.pt",
    }
