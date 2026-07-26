import numpy as np
import pytest

from inftools.core import Posterior
from inftools.mixed_tamis import run_mixed_tamis
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, UniformPrior


def _parallel_test_log_prob(theta):
    x, template = np.asarray(theta, dtype=float)
    if not (-5.0 <= x <= 5.0) or template not in {-1.0, 1.0}:
        return -np.inf
    template_bonus = 0.0 if template == 1.0 else -4.0
    log_prior = -np.log(10.0) - np.log(2.0)
    return log_prior + template_bonus - 0.5 * ((x - template) / 0.35) ** 2


def _parallel_test_space():
    return ParameterSpace(
        names=("x", "template"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([-1.0, 1.0]),
        },
    )


def test_mixed_tamis_returns_weighted_samples_on_discrete_support():
    space = ParameterSpace(
        names=("x", "template"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([-1.0, 1.0]),
        },
    )

    def log_prob(theta):
        theta = np.asarray(theta, dtype=float)
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        x, template = theta
        template_bonus = 0.0 if template == 1.0 else -4.0
        return prior + template_bonus - 0.5 * ((x - template) / 0.35) ** 2

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_mixed_tamis(
        posterior,
        space,
        x0=np.asarray([0.0, -1.0]),
        n_comp=2,
        T_max=4,
        n_per_iter=80,
        alpha=30,
        seed=10,
    )

    assert result.samples.shape == (320, 2)
    assert set(np.unique(result.samples[:, 1])).issubset({-1.0, 1.0})
    assert np.all(np.isfinite(result.meta["weights_norm"]))
    assert np.isclose(np.sum(result.meta["weights_norm"]), 1.0)
    assert result.map_estimate[1] == 1.0
    assert np.all(np.isfinite(result.meta["betas"]))
    assert result.meta["parallel_evaluation"] == {
        "enabled": False,
        "n_workers": 1,
        "batch_size": 32,
        "mp_context": None,
    }


def test_mixed_tamis_delegates_all_discrete_case_to_grid_sampler():
    space = ParameterSpace(
        names=("template",),
        priors={"template": ChoicePrior([0.0, 1.0])},
    )

    def log_prob(theta):
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        return prior if theta[0] == 1.0 else prior - 3.0

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_mixed_tamis(posterior, space, seed=1)

    assert result.samples.shape == (2, 1)
    assert result.map_estimate[0] == 1.0
    assert np.isclose(np.sum(result.meta["weights_norm"]), 1.0)


def test_mixed_tamis_can_sample_continuous_block_in_box_logit_coordinates():
    space = ParameterSpace(
        names=("x", "scale", "template"),
        priors={
            "x": UniformPrior(0.0, 1.0),
            "scale": UniformPrior(100.0, 1000.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )

    def log_prob(theta):
        theta = np.asarray(theta, dtype=float)
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        x, scale, template = theta
        return prior - 0.5 * ((x - 0.3) / 0.08) ** 2 - 0.5 * ((scale - 700.0) / 60.0) ** 2 - 5.0 * abs(
            template - 1.0
        )

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_mixed_tamis(
        posterior,
        space,
        x0=np.asarray([0.5, 500.0, 0.0]),
        n_comp=2,
        T_max=3,
        n_per_iter=80,
        init_span=1.0,
        var0=np.asarray([1.0, 1.0]),
        continuous_transform="auto",
        alpha=30,
        seed=11,
    )

    assert result.samples.shape == (240, 3)
    assert result.meta["continuous_transform"] == "BoxLogitTransform"
    assert result.meta["continuous_transform_names"] == ("x", "scale")
    assert np.all((result.samples[:, 0] > 0.0) & (result.samples[:, 0] < 1.0))
    assert np.all((result.samples[:, 1] > 100.0) & (result.samples[:, 1] < 1000.0))
    assert np.isclose(np.sum(result.meta["weights_norm"]), 1.0)


def test_mixed_tamis_parallel_spawn_matches_serial_exactly():
    space = _parallel_test_space()
    posterior = Posterior(_parallel_test_log_prob, dim=space.ndim, theta_names=space.names)
    options = {
        "x0": np.asarray([0.0, -1.0]),
        "n_comp": 2,
        "T_max": 3,
        "n_per_iter": 48,
        "alpha": 20,
        "seed": 14,
    }

    serial = run_mixed_tamis(posterior, space, **options)
    parallel = run_mixed_tamis(
        posterior,
        space,
        **options,
        n_workers=2,
        batch_size=7,
        mp_context="spawn",
    )

    np.testing.assert_array_equal(parallel.samples, serial.samples)
    np.testing.assert_array_equal(parallel.logp, serial.logp)
    np.testing.assert_array_equal(parallel.map_estimate, serial.map_estimate)
    np.testing.assert_array_equal(parallel.cov, serial.cov)
    for key in ("weights_norm", "final_log_weights", "betas", "ESS", "tempered_ESS"):
        np.testing.assert_array_equal(parallel.meta[key], serial.meta[key])
    assert parallel.meta["parallel_evaluation"] == {
        "enabled": True,
        "n_workers": 2,
        "batch_size": 7,
        "mp_context": "spawn",
    }


def test_mixed_tamis_default_serial_path_does_not_construct_pool(monkeypatch):
    import inftools.mixed_tamis as mixed_tamis_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Serial MixedTAMIS must not construct a process pool.")

    monkeypatch.setattr(mixed_tamis_module, "ProcessPoolExecutor", fail_if_called)
    space = _parallel_test_space()
    posterior = Posterior(_parallel_test_log_prob, dim=space.ndim, theta_names=space.names)

    result = run_mixed_tamis(
        posterior,
        space,
        x0=np.asarray([0.0, -1.0]),
        n_comp=1,
        T_max=1,
        n_per_iter=8,
        seed=15,
    )

    assert result.samples.shape == (8, 2)


def test_mixed_tamis_parallel_reports_unpickleable_log_prob_before_starting_workers():
    space = _parallel_test_space()
    offset = 0.25

    def local_log_prob(theta):
        return _parallel_test_log_prob(theta) + offset

    posterior = Posterior(local_log_prob, dim=space.ndim, theta_names=space.names)
    with pytest.raises(TypeError, match="pickleable posterior log-probability"):
        run_mixed_tamis(
            posterior,
            space,
            x0=np.asarray([0.0, -1.0]),
            n_comp=1,
            T_max=1,
            n_per_iter=8,
            seed=16,
            n_workers=2,
        )


def test_mixed_tamis_rejects_invalid_parallel_configuration():
    space = _parallel_test_space()
    posterior = Posterior(_parallel_test_log_prob, dim=space.ndim, theta_names=space.names)

    with pytest.raises(ValueError, match="n_workers must be positive"):
        run_mixed_tamis(posterior, space, n_workers=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        run_mixed_tamis(posterior, space, batch_size=0)


@pytest.mark.cigale
def test_mixed_tamis_parallel_spawn_matches_serial_for_real_cigale():
    pytest.importorskip("pcigale")

    from composed import Gaussian, Problem, SEDDataset
    from composed.backends.cigale import CIGALEBackend
    from composed.filters import FilterSet

    filters = FilterSet(["sdss.gp"])
    backend = CIGALEBackend(
        modules=["sfhdelayed", "bc03", "redshifting"],
        module_parameters={
            "sfhdelayed": {
                "tau_main": 3000.0,
                "age_main": 1000,
                "tau_burst": 50.0,
                "age_burst": 20,
                "f_burst": 0.0,
                "normalise": True,
            },
            "bc03": {
                "imf": 1,
                "metallicity": 0.02,
                "separation_age": 10,
            },
            "redshifting": {
                "redshift": {"range": [0.05, 0.2]},
            },
        },
    )
    space = ParameterSpace(
        names=("log10_mass", "redshift"),
        priors={
            "log10_mass": UniformPrior(9.5, 10.5),
            "redshift": UniformPrior(0.05, 0.2),
        },
    )
    truth = np.asarray([10.0, 0.1])
    try:
        unit_flux = backend.predict_photometry({"redshift": truth[1]}, filters).flux
    except Exception as exc:
        pytest.skip(f"CIGALE v2022.0 database or sdss.gp filter is unavailable: {exc}")
    observed_flux = unit_flux * 10.0 ** truth[0]
    data = SEDDataset(
        filters.names,
        observed_flux,
        np.maximum(0.05 * observed_flux, 1.0e-20),
        flux_unit="maggies",
    )
    problem = Problem(backend, space, data, Gaussian(), filters=filters)
    posterior = problem.to_inftools_posterior()
    options = {
        "x0": truth,
        "n_comp": 2,
        "T_max": 2,
        "n_per_iter": 12,
        "alpha": 8,
        "continuous_transform": "auto",
        "seed": 17,
    }

    serial = run_mixed_tamis(posterior, space, **options)
    parallel = run_mixed_tamis(
        posterior,
        space,
        **options,
        n_workers=2,
        batch_size=3,
        mp_context="spawn",
    )

    np.testing.assert_array_equal(parallel.samples, serial.samples)
    np.testing.assert_allclose(parallel.logp, serial.logp, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        parallel.meta["weights_norm"],
        serial.meta["weights_norm"],
        rtol=0.0,
        atol=0.0,
    )
