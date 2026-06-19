import importlib.util

import numpy as np
import pytest

from inftools.sbi import MAFPosteriorEstimator, simulate_training_set, train_maf_posterior_from_dataset
from composed.backends.mock import MockBackend
from composed.data import SEDDataset
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior


def test_importing_inftools_works_without_constructing_sbi_estimator():
    import inftools

    assert hasattr(inftools, "Posterior")
    assert hasattr(inftools, "MAFPosteriorEstimator")


def test_importing_inftools_sbi_works_without_dependencies():
    import inftools.sbi as sbi

    assert hasattr(sbi, "simulate_training_set")
    assert hasattr(sbi, "train_maf_posterior_from_dataset")


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

    est = sbi.MAFPosteriorEstimator(theta_dim=1, x_dim=1, device="mps")
    assert est.flow is not None
    assert calls[0] == ((), {"dtype": "float32"})
    assert isinstance(calls[1][1]["device"], FakeDevice)
    assert calls[1][1]["device"].value == "mps"


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


def test_simulate_training_set_retries_failures():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    calls = {"n": 0}

    def simulator(theta, noise_fn=None, rng=None):
        del theta, noise_fn, rng
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("first one fails")
        return np.array([42.0])

    theta, x, meta = simulate_training_set(
        ps,
        simulator,
        n=2,
        noise_fn=lambda flux: np.zeros_like(flux),
        rng=np.random.default_rng(4),
        max_retries=3,
        return_metadata=True,
    )
    assert theta.shape == (2, 1)
    assert x.shape == (2, 1)
    assert len(meta["failures"]) == 1


def test_simulate_training_set_raises_after_too_many_failures():
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    def simulator(theta, noise_fn=None, rng=None):
        raise ValueError("always fails")

    with pytest.raises(RuntimeError, match="Too many failed simulations"):
        simulate_training_set(ps, simulator, n=1, noise_fn=lambda flux: flux, max_retries=1)


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
    estimator.fit(theta, x, epochs=2, batch_size=32, seed=6)
    samples = estimator.sample(np.array([0.0]), num_samples=16)
    logp = estimator.log_prob(samples[:4], np.array([0.0]))
    assert samples.shape == (16, 1)
    assert logp.shape == (4,)
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(logp))
