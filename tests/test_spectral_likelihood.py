import numpy as np
import pytest

from composed.backends.base import ModelSpectrum
from composed.backends.mock import MockBackend
from composed.data import SpectrumDataset
from composed.likelihood import GaussianSpectralLikelihood, SpectralSimulationError
from composed.parameters import ParameterSpace
from composed.priors import DeltaPrior, UniformPrior
from composed.units import MassNormalization


def loglike_no_residual(sigma):
    sigma = np.asarray(sigma, dtype=float)
    return -0.5 * np.sum(np.log(2.0 * np.pi * sigma**2))


def test_spectrum_dataset_active_arrays_exclude_mask_flux_and_sigma():
    data = SpectrumDataset(
        wavelength=np.array([4000.0, 5000.0, 6000.0, 7000.0]),
        flux=np.array([1.0, np.nan, 3.0, 4.0]),
        sigma=np.array([0.1, 0.2, 0.0, 0.4]),
        mask=np.array([True, True, True, False]),
    )

    wave, flux, sigma, idx = data.active_arrays()
    assert np.allclose(wave, [4000.0])
    assert np.allclose(flux, [1.0])
    assert np.allclose(sigma, [0.1])
    assert np.allclose(idx, [0])


def test_all_masked_spectrum_raises_clear_error():
    with pytest.raises(ValueError, match="at least one active spectral pixel"):
        SpectrumDataset(
            wavelength=np.array([4000.0, 5000.0]),
            flux=np.array([1.0, 2.0]),
            sigma=np.array([0.1, 0.2]),
            mask=np.array([False, False]),
        )


def test_all_bad_sigma_spectrum_raises_clear_error():
    with pytest.raises(ValueError, match="at least one active spectral pixel"):
        SpectrumDataset(
            wavelength=np.array([4000.0, 5000.0]),
            flux=np.array([1.0, 2.0]),
            sigma=np.array([0.0, np.nan]),
        )


def test_gaussian_spectral_likelihood_value_for_mock_backend():
    data = SpectrumDataset(
        wavelength=np.array([4000.0, 5000.0]),
        flux=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.2]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[4000.0, 5000.0],
        spectrum_flux=[1.1, 1.8],
    )
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.5)})
    like = GaussianSpectralLikelihood(backend, data, ps)

    resid = np.array([-1.0, 1.0])
    expected = -0.5 * (np.sum(resid**2) + np.sum(np.log(2.0 * np.pi * data.sigma**2)))
    assert np.isclose(like.log_prob([0.5]), expected)


def test_spectral_mass_scaling_only_for_per_solar_mass_backend():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([20.0, 40.0]),
        sigma=np.array([1.0, 1.0]),
    )
    ps = ParameterSpace(["log10_mass"], {"log10_mass": DeltaPrior(1.0)})

    per_mass = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[2.0, 4.0],
        mass_normalization=MassNormalization.PER_SOLAR_MASS,
    )
    absolute = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[2.0, 4.0],
        mass_normalization=MassNormalization.ABSOLUTE,
    )

    assert GaussianSpectralLikelihood(per_mass, data, ps).log_prob([1.0]) > -3.0
    assert GaussianSpectralLikelihood(absolute, data, ps).log_prob([1.0]) < -400.0


def test_missing_log10_mass_for_per_solar_mass_spectrum_raises_clear_error():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.1]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[1.0, 2.0],
        mass_normalization=MassNormalization.PER_SOLAR_MASS,
    )
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    with pytest.raises(ValueError, match="log10_mass"):
        GaussianSpectralLikelihood(backend, data, ps).log_prob([0.5])


def test_missing_log10_mass_allowed_for_absolute_spectrum():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.1]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[1.0, 2.0],
        mass_normalization=MassNormalization.ABSOLUTE,
    )
    ps = ParameterSpace(["z"], {"z": UniformPrior(0.0, 1.0)})

    assert np.isfinite(GaussianSpectralLikelihood(backend, data, ps).log_prob([0.5]))


def test_log10_mass_is_not_double_applied_for_absolute_spectrum():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([20.0, 40.0]),
        sigma=np.array([1.0, 2.0]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[20.0, 40.0],
        mass_normalization=MassNormalization.ABSOLUTE,
    )
    ps = ParameterSpace(["log10_mass"], {"log10_mass": DeltaPrior(1.0)})

    assert np.isclose(GaussianSpectralLikelihood(backend, data, ps).log_prob([1.0]), loglike_no_residual([1.0, 2.0]))


def test_spectral_sigma_floor_is_added_in_quadrature():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([2.0, 0.0]),
        sigma=np.array([1.0, 1.0]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[0.0, 0.0],
    )
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})

    logp = GaussianSpectralLikelihood(backend, data, ps, sigma_floor=3.0).log_prob([0.0])
    sigma = np.sqrt(10.0)
    expected = -0.5 * ((2.0 / sigma) ** 2 + np.sum(np.log(2.0 * np.pi * np.array([sigma, sigma]) ** 2)))
    assert np.isclose(logp, expected)


def test_spectral_backend_numerical_error_returns_minus_inf():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.1]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[1.0, 2.0],
        fail_on_call=True,
    )
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    assert GaussianSpectralLikelihood(backend, data, ps).log_prob([0.0]) == -np.inf


@pytest.mark.parametrize("bad_flux", [[np.nan, 1.0], [np.inf, 1.0]])
def test_bad_spectral_backend_outputs_return_minus_inf(bad_flux):
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.1]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=bad_flux,
    )
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    assert GaussianSpectralLikelihood(backend, data, ps).log_prob([0.0]) == -np.inf


def test_spectral_model_shape_mismatch_raises_clear_error():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.1]),
    )
    backend = WrongShapeSpectrumBackend()
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    with pytest.raises(ValueError, match="Model spectrum shape"):
        GaussianSpectralLikelihood(backend, data, ps).log_prob([0.0])


def test_spectral_model_wavelength_mismatch_raises_clear_error():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.1]),
    )
    backend = WrongWavelengthSpectrumBackend()
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    with pytest.raises(ValueError, match="wavelength grid"):
        GaussianSpectralLikelihood(backend, data, ps).log_prob([0.0])


def test_spectral_simulate_supports_batch_and_mask():
    data = SpectrumDataset(
        wavelength=np.array([4000.0, 5000.0, 6000.0]),
        flux=np.zeros(3),
        sigma=np.ones(3),
        mask=np.array([True, False, True]),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[4000.0, 5000.0, 6000.0],
        spectrum_flux=[1.0, 99.0, 3.0],
    )
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    like = GaussianSpectralLikelihood(backend, data, ps)
    noise_fn = lambda flux: np.zeros_like(flux)

    one = like.simulate([0.0], noise_fn=noise_fn, rng=np.random.default_rng(1))
    batch = like.simulate([[0.0], [0.0]], noise_fn=noise_fn, rng=np.random.default_rng(1))
    assert np.allclose(one, [1.0, 3.0])
    assert np.allclose(batch, [[1.0, 3.0], [1.0, 3.0]])


def test_spectral_simulate_invalid_backend_output_raises_controlled_error():
    data = SpectrumDataset(
        wavelength=np.array([5000.0, 6000.0]),
        flux=np.zeros(2),
        sigma=np.ones(2),
    )
    backend = MockBackend(
        flux=[],
        spectrum_wavelength=[5000.0, 6000.0],
        spectrum_flux=[np.nan, 1.0],
    )
    ps = ParameterSpace(["z"], {"z": DeltaPrior(0.0)})
    with pytest.raises(SpectralSimulationError, match="non-finite"):
        GaussianSpectralLikelihood(backend, data, ps).simulate([0.0], lambda flux: np.zeros_like(flux))


class WrongShapeSpectrumBackend:
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_spectrum(self, params, wavelengths=None, wavelength_range=None, resolution=None):
        del params, wavelengths, wavelength_range, resolution
        return ModelSpectrum(wavelength=np.array([5000.0]), flux=np.array([1.0]))


class WrongWavelengthSpectrumBackend:
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_spectrum(self, params, wavelengths=None, wavelength_range=None, resolution=None):
        del params, wavelength_range, resolution
        return ModelSpectrum(wavelength=np.asarray(wavelengths, dtype=float) + 1.0, flux=np.ones_like(wavelengths))
