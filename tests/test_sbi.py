import importlib.util

import numpy as np
import pytest

from composed._numerics import trapezoid
from inftools.sbi import (
    HybridMAFPosteriorEstimator,
    MAFPosteriorEstimator,
    MDNPosteriorEstimator,
    simulate_training_set,
    train_maf_posterior_from_dataset,
)
from composed.backends.mock import MockBackend
from composed.data import SEDDataset
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior


def identity_simulator(theta, noise_fn=None, rng=None):
    """Pickleable simulator used by process-executor regression tests."""

    del noise_fn, rng
    return np.asarray(theta, dtype=float)


def threshold_simulator(theta, noise_fn=None, rng=None):
    """Reject part of the prior so process workers exercise replacement waves."""

    del noise_fn, rng
    if float(theta[0]) < 0.25:
        raise ValueError("outside toy simulator domain")
    return np.asarray(theta, dtype=float)


def zero_noise(flux):
    """Pickleable deterministic noise function for process workers."""

    return np.zeros_like(flux)


def test_importing_inftools_works_without_constructing_sbi_estimator():
    import inftools

    assert hasattr(inftools, "Posterior")
    assert hasattr(inftools, "MAFPosteriorEstimator")
    assert hasattr(inftools, "HybridMAFPosteriorEstimator")
    assert hasattr(inftools, "MDNPosteriorEstimator")
    assert hasattr(inftools, "SBISimulationFailureWarning")


def test_importing_inftools_sbi_works_without_dependencies():
    import inftools.sbi as sbi

    assert hasattr(sbi, "simulate_training_set")
    assert hasattr(sbi, "SBISimulationFailureWarning")
    assert hasattr(sbi, "train_maf_posterior_from_dataset")
    assert hasattr(sbi, "train_mdn_posterior_from_dataset")


def test_train_maf_posterior_from_dataset_prepares_paired_arrays(monkeypatch):
    import inftools.sbi as sbi

    calls = {}

    def fake_train(theta_train, x_train, **kwargs):
        calls["theta"] = theta_train
        calls["x"] = x_train
        calls["kwargs"] = kwargs
        return {"estimator": "fake"}

    monkeypatch.setattr(sbi, "train_maf_posterior", fake_train)

    theta = np.array([[0.1, 9.0], [0.2, 9.5], [0.3, 10.0]])
    x = np.array([[21.0, 22.0], [20.0, 21.0], [19.0, 20.0]])
    estimator, meta = train_maf_posterior_from_dataset(
        theta,
        x,
        theta_names=["z", "log10_mass"],
        x_names=["g", "r"],
        source="empirical_catalog",
        epochs=3,
        batch_size=2,
        return_metadata=True,
    )

    assert estimator == {"estimator": "fake"}
    assert np.allclose(calls["theta"], theta)
    assert np.allclose(calls["x"], x)
    assert calls["kwargs"]["epochs"] == 3
    assert calls["kwargs"]["batch_size"] == 2
    assert meta["source"] == "empirical_catalog"
    assert meta["theta_names"] == ("z", "log10_mass")
    assert meta["x_names"] == ("g", "r")
    assert meta["n_train"] == 3


def test_train_maf_posterior_from_dataset_can_drop_nonfinite_rows(monkeypatch):
    import inftools.sbi as sbi

    calls = {}

    def fake_train(theta_train, x_train, **kwargs):
        calls["theta"] = theta_train
        calls["x"] = x_train
        return "trained"

    monkeypatch.setattr(sbi, "train_maf_posterior", fake_train)

    theta = np.array([[0.1], [np.nan], [0.3]])
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    estimator, meta = train_maf_posterior_from_dataset(theta, x, finite="drop", return_metadata=True)

    assert estimator == "trained"
    assert np.allclose(calls["theta"], [[0.1], [0.3]])
    assert np.allclose(calls["x"], [[1.0, 2.0], [5.0, 6.0]])
    assert meta["dropped_nonfinite"] == 1
    assert meta["n_train"] == 2


def test_train_maf_posterior_from_dataset_rejects_bad_pairing(monkeypatch):
    import inftools.sbi as sbi

    monkeypatch.setattr(sbi, "train_maf_posterior", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="same number of rows"):
        train_maf_posterior_from_dataset(np.ones((3, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="NaN or inf"):
        train_maf_posterior_from_dataset(np.array([[1.0], [np.inf]]), np.ones((2, 1)))
    with pytest.raises(ValueError, match="theta_names"):
        train_maf_posterior_from_dataset(np.ones((3, 2)), np.ones((3, 1)), theta_names=["z"])


def test_constructing_maf_without_nflows_gives_helpful_import_error(monkeypatch):
    import inftools.sbi as sbi

    def fake_import_module(name):
        if name == "torch":
            class FakeCuda:
                @staticmethod
                def is_available():
                    return False

            class FakeTorch:
                cuda = FakeCuda()

            return FakeTorch()
        if name == "nflows":
            raise ImportError("no nflows")
        return importlib.import_module(name)

    monkeypatch.setattr(sbi.importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="torch and nflows"):
        sbi.MAFPosteriorEstimator(theta_dim=1, x_dim=1)


def test_constructing_mdn_without_torch_gives_helpful_import_error(monkeypatch):
    import inftools.sbi as sbi

    original_import = importlib.import_module

    def fake_import_module(name):
        if name == "torch":
            raise ImportError("no torch")
        return original_import(name)

    monkeypatch.setattr(sbi.importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="MDN posterior estimator requires"):
        sbi.MDNPosteriorEstimator(theta_dim=1, x_dim=1)


def test_maf_constructor_forces_float32_before_device_move(monkeypatch):
    import inftools.sbi as sbi

    calls = []

    class FakeDevice:
        def __init__(self, value):
            self.value = value

    class FakeTorch:
        float32 = "float32"

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def device(value):
            return FakeDevice(value)

    class FakeFlow:
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    monkeypatch.setattr(sbi, "_require_sbi_dependencies", lambda: (FakeTorch, object()))
    monkeypatch.setattr(sbi, "build_maf", lambda **kwargs: FakeFlow())

    est = sbi.MAFPosteriorEstimator(theta_dim=1, x_dim=1, device="mps", validate_device=False)
    assert est.flow is not None
    assert calls[0] == ((), {"dtype": "float32"})
    assert isinstance(calls[1][1]["device"], FakeDevice)
    assert calls[1][1]["device"].value == "mps"


@pytest.mark.sbi
def test_maf_resolve_torch_device_cpu_and_bad_explicit_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    import torch
    import inftools.sbi as sbi

    device = sbi.resolve_torch_device(torch, "cpu", validate=True)
    assert device.type == "cpu"

    with pytest.raises(RuntimeError, match="not usable"):
        sbi.resolve_torch_device(torch, "not_a_real_device", validate=True, allow_fallback=False)


def test_simulate_training_set_with_toy_likelihood():
    data = SEDDataset(["g", "r"], flux=np.zeros(2), sigma=np.ones(2))
    backend = MockBackend([1.0, 2.0], band_names=["g", "r"])
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    like = GaussianPhotometricLikelihood(backend, data, ps)
    theta, x = simulate_training_set(ps, like, n=5, noise_fn=lambda flux: np.zeros_like(flux), rng=np.random.default_rng(3))

    assert theta.shape == (5, 1)
    assert x.shape == (5, 2)
    assert np.all(np.isfinite(theta))
    assert np.allclose(x, np.array([[1.0, 2.0]] * 5))


def test_simulate_training_set_can_return_exact_sigma():
    data = SEDDataset(["g", "r"], flux=np.zeros(2), sigma=np.ones(2))
    backend = MockBackend([1.0, 2.0], band_names=["g", "r"])
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    like = GaussianPhotometricLikelihood(backend, data, ps)

    def noise(flux, theta=None, rng=None):
        del rng
        return 0.1 * flux + 0.01 * theta[0]

    theta, x, sigma, metadata = simulate_training_set(
        ps,
        like,
        n=5,
        noise_fn=noise,
        rng=np.random.default_rng(3),
        return_sigma=True,
        return_metadata=True,
    )
    expected = 0.1 * np.asarray([1.0, 2.0])[None, :] + 0.01 * theta
    assert sigma.shape == x.shape == (5, 2)
    assert np.allclose(sigma, expected)
    assert metadata["returned_sigma"] is True


@pytest.mark.sbi
def test_mdn_density_is_normalized_and_sampling_is_finite():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    rng = np.random.default_rng(90)
    x = rng.normal(size=(256, 1))
    theta = 0.7 * x + 0.4 * rng.normal(size=(256, 1))
    estimator = MDNPosteriorEstimator(
        theta_dim=1,
        x_dim=1,
        n_components=3,
        hidden_features=24,
        num_blocks=2,
        device="cpu",
        initialization_seed=91,
    )
    history = estimator.fit(
        theta,
        x,
        epochs=3,
        batch_size=64,
        validation_split=0.2,
        seed=92,
    )

    assert np.all(np.isfinite(history["train_loss"]))
    samples = estimator.sample(np.asarray([[0.0], [0.5]]), 7, seed=93)
    assert samples.shape == (2, 7, 1)
    assert np.all(np.isfinite(samples))

    mixture = estimator.mixture_parameters(np.asarray([[0.0]]))
    assert np.allclose(np.sum(mixture["weights"], axis=1), 1.0)
    assert np.all(mixture["scales"] > 0.0)

    grid = np.linspace(-20.0, 20.0, 10_001)
    log_density = estimator.log_prob(grid[:, None], np.asarray([[0.0]]))
    integral = trapezoid(np.exp(log_density), grid)
    assert integral == pytest.approx(1.0, abs=2.0e-3)


def test_simulate_training_set_parallel_thread_chunks():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    def simulator(theta, noise_fn=None, rng=None):
        del noise_fn, rng
        return np.array([theta[0], theta[0] + 1.0])

    theta, x, meta = simulate_training_set(
        ps,
        simulator,
        n=17,
        noise_fn=lambda flux: np.zeros_like(flux),
        rng=np.random.default_rng(11),
        batch_size=4,
        n_workers=2,
        executor="thread",
        return_metadata=True,
    )

    assert theta.shape == (17, 1)
    assert x.shape == (17, 2)
    assert np.allclose(x[:, 0], theta[:, 0])
    assert np.allclose(x[:, 1], theta[:, 0] + 1.0)
    assert meta["batch_size"] == 4
    assert meta["n_workers"] == 2
    assert meta["executor"] == "thread"


def test_parallel_training_simulation_does_not_run_unused_retry_reserve():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    def simulator(theta, noise_fn=None, rng=None):
        del noise_fn, rng
        return np.array([theta[0]])

    theta, x, meta = simulate_training_set(
        ps,
        simulator,
        n=17,
        noise_fn=lambda flux: np.zeros_like(flux),
        rng=np.random.default_rng(13),
        batch_size=4,
        n_workers=2,
        executor="thread",
        return_metadata=True,
    )

    assert theta.shape == (17, 1)
    assert x.shape == (17, 1)
    assert meta["attempts"] == 17
    assert meta["failures"] == []


def test_process_training_simulation_does_not_run_unused_retry_reserve():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    theta, x, meta = simulate_training_set(
        ps,
        identity_simulator,
        n=17,
        noise_fn=zero_noise,
        rng=np.random.default_rng(14),
        batch_size=4,
        n_workers=2,
        executor="process",
        mp_context="spawn",
        return_metadata=True,
    )

    assert theta.shape == (17, 1)
    assert x.shape == (17, 1)
    assert meta["attempts"] == 17
    assert meta["failures"] == []


def test_process_training_simulation_replaces_only_failed_rows():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    with pytest.warns(RuntimeWarning, match="conditioned on simulator success"):
        theta, x, meta = simulate_training_set(
            ps,
            threshold_simulator,
            n=17,
            noise_fn=zero_noise,
            rng=np.random.default_rng(15),
            warn_retry_fraction=0.05,
            failure_policy="resample",
            batch_size=4,
            n_workers=2,
            executor="process",
            mp_context="spawn",
            return_metadata=True,
        )

    assert theta.shape == x.shape == (17, 1)
    assert np.all(theta[:, 0] >= 0.25)
    assert len(meta["failures"]) > 0
    assert meta["attempts"] == 17 + len(meta["failures"])
    assert meta["failure_policy"] == "resample"
    assert meta["returned_prior"] == "simulator_success_conditioned"
    assert meta["acceptance_fraction"] == pytest.approx(17 / meta["attempts"])
    assert meta["failure_fraction"] == pytest.approx(
        len(meta["failures"]) / meta["attempts"]
    )


def test_parallel_training_simulation_returns_matching_sigma_rows():
    class Simulator:
        def simulate_with_uncertainty(self, theta, noise_fn=None, rng=None):
            flux = np.array([theta[0], theta[0] + 1.0])
            sigma = np.asarray(noise_fn(flux), dtype=float)
            return flux + rng.normal(scale=sigma), sigma

    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    theta, x, sigma = simulate_training_set(
        ps,
        Simulator(),
        n=17,
        noise_fn=lambda flux: 0.1 + 0.05 * flux,
        rng=np.random.default_rng(12),
        batch_size=4,
        n_workers=2,
        executor="thread",
        return_sigma=True,
    )

    expected = np.column_stack([0.1 + 0.05 * theta[:, 0], 0.15 + 0.05 * theta[:, 0]])
    assert x.shape == sigma.shape == (17, 2)
    assert np.allclose(sigma, expected)


def test_simulate_training_set_retries_failures():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    calls = {"n": 0}

    def simulator(theta, noise_fn=None, rng=None):
        del theta, noise_fn, rng
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("first one fails")
        return np.array([42.0])

    with pytest.warns(RuntimeWarning, match="conditioned on simulator success"):
        theta, x, meta = simulate_training_set(
            ps,
            simulator,
            n=2,
            noise_fn=lambda flux: np.zeros_like(flux),
            rng=np.random.default_rng(4),
            warn_retry_fraction=0.05,
            failure_policy="resample",
            return_metadata=True,
        )
    assert theta.shape == (2, 1)
    assert x.shape == (2, 1)
    assert len(meta["failures"]) == 1
    assert meta["returned_prior"] == "simulator_success_conditioned"


def test_resampling_warns_on_failure_fraction_but_keeps_running():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    calls = {"n": 0}

    def simulator(theta, noise_fn=None, rng=None):
        del noise_fn, rng
        calls["n"] += 1
        if calls["n"] <= 5:
            raise ValueError("temporary invalid domain")
        return np.asarray(theta, dtype=float)

    with pytest.warns(RuntimeWarning, match="failure fraction"):
        theta, x, meta = simulate_training_set(
            ps,
            simulator,
            n=3,
            noise_fn=zero_noise,
            rng=np.random.default_rng(45),
            warn_retry_fraction=0.1,
            failure_policy="resample",
            return_metadata=True,
        )

    assert theta.shape == x.shape == (3, 1)
    assert meta["attempts"] == 8
    assert meta["n_failures"] == 5
    assert meta["failure_fraction"] == pytest.approx(5 / 8)
    assert meta["failure_fraction_warning_count"] >= 1


def test_simulate_training_set_fails_loudly_by_default():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    def simulator(theta, noise_fn=None, rng=None):
        del theta, noise_fn, rng
        raise ValueError("invalid toy model")

    with pytest.raises(RuntimeError, match="failure_policy='resample'") as error:
        simulate_training_set(
            ps,
            simulator,
            n=2,
            noise_fn=zero_noise,
            rng=np.random.default_rng(41),
        )
    assert isinstance(error.value.__cause__, ValueError)


def test_successful_training_set_retains_exact_declared_prior_draws():
    ps = ParameterSpace(["z"], {"z": UniformPrior(-2.0, 3.0)})
    seed = 84

    theta, x, meta = simulate_training_set(
        ps,
        identity_simulator,
        n=12,
        noise_fn=zero_noise,
        rng=np.random.default_rng(seed),
        return_metadata=True,
    )
    reference_rng = np.random.default_rng(seed)
    expected = np.vstack([ps.sample_prior(1, reference_rng)[0] for _ in range(12)])

    assert np.array_equal(theta, expected)
    assert np.array_equal(x, expected)
    assert meta["returned_prior"] == "declared_prior"
    assert meta["n_failures"] == 0
    assert meta["acceptance_fraction"] == 1.0


def test_simulate_training_set_stops_only_at_explicit_attempt_budget():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    def simulator(theta, noise_fn=None, rng=None):
        raise ValueError("always fails")

    with pytest.warns(RuntimeWarning, match="failure fraction"):
        with pytest.raises(RuntimeError, match="max_attempts=2"):
            simulate_training_set(
                ps,
                simulator,
                n=1,
                noise_fn=lambda flux: flux,
                max_attempts=2,
                failure_policy="resample",
            )


def test_simulate_training_set_rejects_attempt_budget_below_requested_rows():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    with pytest.raises(ValueError, match="max_attempts must be at least n"):
        simulate_training_set(
            ps,
            identity_simulator,
            n=2,
            noise_fn=lambda flux: flux,
            max_attempts=1,
        )


def test_simulate_training_set_rejects_invalid_retry_warning_fraction():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    with pytest.raises(ValueError, match="warn_retry_fraction"):
        simulate_training_set(
            ps,
            lambda theta, noise_fn=None, rng=None: np.asarray(theta),
            n=1,
            noise_fn=lambda flux: flux,
            warn_retry_fraction=1.1,
        )


def test_simulate_training_set_rejects_unknown_failure_policy():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    with pytest.raises(ValueError, match="failure_policy"):
        simulate_training_set(
            ps,
            identity_simulator,
            n=1,
            noise_fn=zero_noise,
            failure_policy="ignore",
        )


@pytest.mark.sbi
def test_tiny_maf_training_if_dependencies_available():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch/nflows are not installed.")

    rng = np.random.default_rng(5)
    theta = rng.normal(size=(64, 1))
    x = theta + 0.1 * rng.normal(size=(64, 1))
    estimator = MAFPosteriorEstimator(
        theta_dim=1,
        x_dim=1,
        hidden_features=8,
        num_transforms=1,
        num_blocks=1,
        learning_rate=5e-3,
        device="cpu",
    )
    history = estimator.fit(theta, x, epochs=2, batch_size=32, validation_split=0.25, seed=6)
    samples = estimator.sample(np.array([0.0]), num_samples=16)
    logp = estimator.log_prob(samples[:4], np.array([0.0]))
    assert samples.shape == (16, 1)
    assert logp.shape == (4,)
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(logp))
    assert len(history["val_loss"]) == 2
    assert np.all(np.isfinite(history["val_loss"]))


@pytest.mark.sbi
def test_hybrid_maf_learns_categories_and_category_conditioned_continuous_draws():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch/nflows are not installed.")

    rng = np.random.default_rng(105)
    category = rng.integers(0, 2, size=512)
    x = (4.0 * category - 2.0 + 0.25 * rng.normal(size=category.size))[:, None]
    theta = (
        0.4 * x[:, 0]
        + 0.8 * category
        + 0.15 * rng.normal(size=category.size)
    )[:, None]
    estimator = HybridMAFPosteriorEstimator(
        continuous_dim=1,
        x_dim=1,
        n_categories=2,
        hidden_features=16,
        num_transforms=1,
        num_blocks=1,
        classifier_hidden_features=16,
        classifier_num_blocks=1,
        learning_rate=5.0e-3,
        device="cpu",
        initialization_seed=106,
    )
    history = estimator.fit(
        theta,
        category,
        x,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        seed=107,
    )

    assert np.all(np.isfinite(history["train_loss"]))
    probabilities = estimator.category_probabilities([[-2.0], [2.0]])
    assert probabilities[0, 0] > 0.9
    assert probabilities[1, 1] > 0.9
    continuous, categories = estimator.sample(
        np.asarray([[-2.0], [2.0]]),
        num_samples=32,
        seed=108,
    )
    assert continuous.shape == (2, 32, 1)
    assert categories.shape == (2, 32)
    assert np.all(np.isfinite(continuous))
    logp = estimator.log_prob(
        np.asarray([[-0.8], [1.6]]),
        np.asarray([0, 1]),
        np.asarray([[-2.0], [2.0]]),
    )
    assert logp.shape == (2,)
    assert np.all(np.isfinite(logp))


@pytest.mark.sbi
def test_maf_training_seed_controls_weight_initialization():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch/nflows are not installed.")

    rng = np.random.default_rng(81)
    theta = rng.normal(size=(48, 1))
    x = theta + 0.1 * rng.normal(size=(48, 1))
    settings = {
        "hidden_features": 8,
        "num_transforms": 2,
        "num_blocks": 1,
        "device": "cpu",
        "epochs": 2,
        "batch_size": 16,
        "seed": 82,
    }
    first = train_maf_posterior_from_dataset(theta, x, **settings)
    second = train_maf_posterior_from_dataset(theta, x, **settings)

    for name, value in first.flow.state_dict().items():
        assert first.torch.equal(value, second.flow.state_dict()[name])


@pytest.mark.sbi
def test_maf_stays_float32_when_global_default_dtype_is_float64():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("torch/nflows are not installed.")

    import torch

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        rng = np.random.default_rng(8)
        theta = rng.normal(size=(32, 1))
        x = theta + 0.1 * rng.normal(size=(32, 1))
        estimator = MAFPosteriorEstimator(
            theta_dim=1,
            x_dim=1,
            hidden_features=8,
            num_transforms=1,
            num_blocks=1,
            learning_rate=5e-3,
            device="cpu",
        )
        assert next(estimator.flow.parameters()).dtype == torch.float32
        estimator.fit(theta, x, epochs=1, batch_size=16, seed=9)
        samples = estimator.sample(np.array([0.0]), num_samples=4)
        assert samples.shape == (4, 1)
        assert np.all(np.isfinite(samples))
    finally:
        torch.set_default_dtype(old_dtype)
