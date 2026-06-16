import numpy as np
import pytest

from composed.backends.mock import MockBackend
from composed.data import SEDDataset
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import DeltaPrior, UniformPrior
from composed.units import MassNormalization


def loglike_no_residual(sigma):
    sigma = np.asarray(sigma, dtype=float)
    return -0.5 * np.sum(np.log(2.0 * np.pi * sigma**2))


def test_gaussian_likelihood_value_for_mock_backend():
    data = SEDDataset(["g", "r"], flux=np.array([1.0, 2.0]), sigma=np.array([0.1, 0.2]))
    backend = MockBackend([1.1, 1.8], band_names=["g", "r"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.5)})
    like = GaussianPhotometricLikelihood(backend, data, ps)
    theta = np.array([0.5])

    resid = np.array([-1.0, 1.0])
    expected = -0.5 * (np.sum(resid**2) + np.sum(np.log(2.0 * np.pi * data.sigma**2)))
    assert np.isclose(like.log_prob(theta), expected)


def test_mass_scaling_only_for_per_solar_mass_backend():
    data = SEDDataset(["g"], flux=np.array([20.0]), sigma=np.array([1.0]))
    ps = ParameterSpace(["log10_mass"], {"log10_mass": DeltaPrior(1.0)})

    per_mass = MockBackend([2.0], band_names=["g"], mass_normalization=MassNormalization.PER_SOLAR_MASS)
    absolute = MockBackend([2.0], band_names=["g"], mass_normalization=MassNormalization.ABSOLUTE)

    assert GaussianPhotometricLikelihood(per_mass, data, ps).log_prob([1.0]) > -1.0
    assert GaussianPhotometricLikelihood(absolute, data, ps).log_prob([1.0]) < -100.0


def test_missing_log10_mass_allowed_for_absolute_backend():
    data = SEDDataset(["g"], flux=np.array([1.0]), sigma=np.array([0.1]))
    backend = MockBackend([1.0], band_names=["g"], mass_normalization=MassNormalization.ABSOLUTE)
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    assert np.isfinite(GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.5]))


def test_log10_mass_is_not_double_applied_for_absolute_backend():
    data = SEDDataset(["g"], flux=np.array([20.0]), sigma=np.array([1.0]))
    backend = MockBackend([20.0], band_names=["g"], mass_normalization=MassNormalization.ABSOLUTE)
    ps = ParameterSpace(["log10_mass"], {"log10_mass": DeltaPrior(1.0)})
    assert np.isclose(GaussianPhotometricLikelihood(backend, data, ps).log_prob([1.0]), loglike_no_residual([1.0]))


def test_missing_log10_mass_for_per_solar_mass_raises_clear_error():
    data = SEDDataset(["g"], flux=np.array([1.0]), sigma=np.array([0.1]))
    backend = MockBackend([1.0], band_names=["g"], mass_normalization=MassNormalization.PER_SOLAR_MASS)
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    with pytest.raises(ValueError, match="log10_mass"):
        GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.5])


def test_masked_bands_are_ignored():
    data = SEDDataset(
        ["g", "r"],
        flux=np.array([1.0, 1000.0]),
        sigma=np.array([0.1, 0.1]),
        mask=np.array([True, False]),
    )
    backend = MockBackend([1.0, -999.0], band_names=["g", "r"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    logp = GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.0])
    expected = -0.5 * np.log(2.0 * np.pi * 0.1**2)
    assert np.isclose(logp, expected)


def test_masked_bands_excluded_from_flux_and_sigma():
    data = SEDDataset(
        ["u", "g", "r"],
        flux=np.array([10.0, 20.0, 30.0]),
        sigma=np.array([1.0, 2.0, 3.0]),
        mask=np.array([False, True, False]),
    )
    flux, sigma, idx, bands = data.active_arrays()
    assert np.allclose(flux, [20.0])
    assert np.allclose(sigma, [2.0])
    assert np.allclose(idx, [1])
    assert bands == ("g",)


def test_all_masked_photometry_raises_clear_error():
    with pytest.raises(ValueError, match="at least one active band"):
        SEDDataset(
            ["g", "r"],
            flux=np.array([1.0, 2.0]),
            sigma=np.array([0.1, 0.2]),
            mask=np.array([False, False]),
        )


def test_all_bad_sigma_photometry_raises_clear_error():
    with pytest.raises(ValueError, match="at least one active band"):
        SEDDataset(["g", "r"], flux=np.array([1.0, 2.0]), sigma=np.array([0.0, np.nan]))


def test_nonfinite_or_bad_sigma_bands_are_masked_automatically():
    data = SEDDataset(
        ["u", "g", "r", "i"],
        flux=np.array([np.nan, 2.0, 3.0, 4.0]),
        sigma=np.array([1.0, np.inf, 0.0, 0.4]),
    )
    assert data.active_band_names == ("i",)
    assert np.allclose(data.active_flux, [4.0])
    assert np.allclose(data.active_sigma, [0.4])


def test_upper_limit_band_can_have_nan_flux_and_stays_active():
    data = SEDDataset(
        ["fuv"],
        flux=np.array([np.nan]),
        sigma=np.array([0.2]),
        upper_limit=np.array([1.0]),
        upper_limit_mask=np.array([True]),
    )

    assert data.active_band_names == ("fuv",)
    assert np.allclose(data.active_upper_limit, [1.0])
    assert np.all(data.active_upper_limit_mask)


def test_upper_limit_likelihood_uses_gaussian_cdf():
    data = SEDDataset(
        ["fuv"],
        flux=np.array([np.nan]),
        sigma=np.array([0.2]),
        upper_limit=np.array([1.0]),
        upper_limit_mask=np.array([True]),
    )
    backend = MockBackend([1.0], band_names=["fuv"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})

    logp = GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.0])
    assert np.isclose(logp, np.log(0.5))


def test_upper_limit_and_detection_terms_can_mix():
    data = SEDDataset(
        ["g", "fuv"],
        flux=np.array([2.0, np.nan]),
        sigma=np.array([0.1, 0.5]),
        upper_limit=np.array([0.0, 1.0]),
        upper_limit_mask=np.array([False, True]),
    )
    backend = MockBackend([2.0, 1.0], band_names=["g", "fuv"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})

    expected_detection = loglike_no_residual([0.1])
    expected_upper = np.log(0.5)
    logp = GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.0])
    assert np.isclose(logp, expected_detection + expected_upper)


def test_invalid_upper_limit_raises_clear_error():
    with pytest.raises(ValueError, match="upper_limit"):
        SEDDataset(
            ["fuv"],
            flux=np.array([np.nan]),
            sigma=np.array([0.2]),
            upper_limit=np.array([np.nan]),
            upper_limit_mask=np.array([True]),
        )


def test_upper_limit_band_must_respect_mask():
    with pytest.raises(ValueError, match="Upper-limit bands"):
        SEDDataset(
            ["fuv"],
            flux=np.array([np.nan]),
            sigma=np.array([0.2]),
            mask=np.array([False]),
            upper_limit=np.array([1.0]),
            upper_limit_mask=np.array([True]),
        )


def test_model_data_shape_mismatch_raises_clear_error():
    data = SEDDataset(["g", "r"], flux=np.array([1.0, 2.0]), sigma=np.array([0.1, 0.2]))
    backend = MockBackend([1.0], band_names=["g"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    with pytest.raises(ValueError, match="missing active band"):
        GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.0])


def test_model_photometry_with_duplicate_band_names_raises_clear_error():
    data = SEDDataset(["g"], flux=np.array([1.0]), sigma=np.array([0.1]))
    backend = MockBackend([1.0, 2.0], band_names=["g", "g"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    with pytest.raises(ValueError, match="unique"):
        GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.0])


def test_sigma_floor_is_added_in_quadrature():
    data = SEDDataset(["g"], flux=np.array([2.0]), sigma=np.array([1.0]))
    backend = MockBackend([0.0], band_names=["g"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    logp = GaussianPhotometricLikelihood(backend, data, ps, sigma_floor=3.0).log_prob([0.0])
    sigma = np.sqrt(10.0)
    expected = -0.5 * ((2.0 / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    assert np.isclose(logp, expected)


def test_negative_sigma_floor_raises_clear_error():
    data = SEDDataset(["g"], flux=np.array([2.0]), sigma=np.array([1.0]))
    backend = MockBackend([0.0], band_names=["g"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    with pytest.raises(ValueError, match="sigma_floor"):
        GaussianPhotometricLikelihood(backend, data, ps, sigma_floor=-1.0).log_prob([0.0])


def test_backend_numerical_error_returns_minus_inf():
    data = SEDDataset(["g"], flux=np.array([1.0]), sigma=np.array([0.1]))
    backend = MockBackend([1.0], band_names=["g"], fail_on_call=True)
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    assert GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.0]) == -np.inf


@pytest.mark.parametrize("bad_flux", [[np.nan], [np.inf]])
def test_bad_backend_outputs_return_minus_inf(bad_flux):
    data = SEDDataset(["g"], flux=np.array([1.0]), sigma=np.array([0.1]))
    backend = MockBackend(bad_flux, band_names=["g"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    assert GaussianPhotometricLikelihood(backend, data, ps).log_prob([0.0]) == -np.inf


def test_simulate_supports_single_and_batched_theta_shapes():
    data = SEDDataset(["g", "r"], flux=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0]))
    backend = MockBackend([1.0, 2.0], band_names=["g", "r"])
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})
    like = GaussianPhotometricLikelihood(backend, data, ps)
    noise_fn = lambda flux: np.zeros_like(flux)

    one = like.simulate(np.array([0.5]), noise_fn=noise_fn, rng=np.random.default_rng(1))
    batch = like.simulate(np.array([[0.5], [0.7]]), noise_fn=noise_fn, rng=np.random.default_rng(1))
    assert one.shape == (2,)
    assert batch.shape == (2, 2)
    assert np.allclose(one, [1.0, 2.0])
    assert np.allclose(batch, [[1.0, 2.0], [1.0, 2.0]])


def test_simulate_uses_noise_fn_and_active_mask():
    data = SEDDataset(
        ["g", "r", "i"],
        flux=np.zeros(3),
        sigma=np.ones(3),
        mask=np.array([True, False, True]),
    )
    backend = MockBackend([10.0, 99.0, 30.0], band_names=["g", "r", "i"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    like = GaussianPhotometricLikelihood(backend, data, ps)

    called = {}

    def noise_fn(flux, theta=None, rng=None):
        called["flux"] = flux.copy()
        called["theta"] = theta.copy()
        called["rng_seen"] = rng is not None
        return np.zeros_like(flux)

    out = like.rvs([0.0], noise_fn=noise_fn, rng=np.random.default_rng(2))
    assert np.allclose(out, [10.0, 30.0])
    assert np.allclose(called["flux"], [10.0, 30.0])
    assert np.allclose(called["theta"], [0.0])
    assert called["rng_seen"]


def test_simulate_mass_scaling_matches_likelihood_convention():
    data = SEDDataset(["g"], flux=np.array([0.0]), sigma=np.array([1.0]))
    per_mass = MockBackend([2.0], band_names=["g"], mass_normalization=MassNormalization.PER_SOLAR_MASS)
    absolute = MockBackend([2.0], band_names=["g"], mass_normalization=MassNormalization.ABSOLUTE)
    ps = ParameterSpace(["log10_mass"], {"log10_mass": DeltaPrior(1.0)})
    noise_fn = lambda flux: np.zeros_like(flux)

    assert np.allclose(GaussianPhotometricLikelihood(per_mass, data, ps).simulate([1.0], noise_fn), [20.0])
    assert np.allclose(GaussianPhotometricLikelihood(absolute, data, ps).simulate([1.0], noise_fn), [2.0])


def test_simulate_invalid_backend_output_raises_controlled_error():
    from composed.likelihood import PhotometricSimulationError

    data = SEDDataset(["g"], flux=np.array([0.0]), sigma=np.array([1.0]))
    backend = MockBackend([np.nan], band_names=["g"])
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    with pytest.raises(PhotometricSimulationError, match="non-finite"):
        GaussianPhotometricLikelihood(backend, data, ps).simulate([0.0], lambda flux: np.zeros_like(flux))
