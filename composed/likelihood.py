from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Sequence

import numpy as np

from composed.data import SEDDataset, SpectrumDataset
from composed.parameters import ParameterSpace
from composed.units import MassNormalization


class PhotometricSimulationError(RuntimeError):
    """Controlled error raised when photometric simulation cannot produce finite fluxes."""


class SpectralSimulationError(RuntimeError):
    """Controlled error raised when spectral simulation cannot produce finite fluxes."""


@dataclass
class GaussianPhotometricLikelihood:
    """Backend-agnostic Gaussian photometric log posterior for one SED.

    The returned value is ``log_prior(theta) + log_likelihood(data | theta)``.
    The likelihood aligns model and observed photometry by band name, applies an
    optional uncertainty floor in quadrature, and applies ``10**log10_mass`` only
    when the backend explicitly declares ``MassNormalization.PER_SOLAR_MASS``.
    Backend numerical failures and non-finite model fluxes return ``-inf``;
    configuration errors such as missing mass parameters or shape mismatches
    raise clear exceptions.
    """

    backend: object
    dataset: SEDDataset
    parameter_space: ParameterSpace
    filters: object | None = None
    sigma_floor: float | None = None

    def log_prob(self, theta: Sequence[float]) -> float:
        return self._log_prob_checked(theta)

    def _log_prob_checked(self, theta: Sequence[float]) -> float:
        theta = np.asarray(theta, dtype=float)
        log_prior = self.parameter_space.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf

        f_obs, sigma, idx, active_bands = self.dataset.active_arrays()
        if f_obs.size == 0:
            raise ValueError("SEDDataset has no active photometric bands.")
        if self.sigma_floor is not None:
            floor = float(self.sigma_floor)
            if floor < 0.0:
                raise ValueError("sigma_floor must be non-negative.")
            sigma = np.sqrt(sigma**2 + floor**2)

        try:
            model_flux = self._predict_active_model_flux(theta, idx=idx, active_bands=active_bands)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return -np.inf

        if model_flux.shape != f_obs.shape:
            raise ValueError(f"Model flux shape {model_flux.shape} does not match data shape {f_obs.shape}.")
        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        upper_mask = self.dataset.active_upper_limit_mask
        detection_mask = ~upper_mask

        log_like = 0.0
        if np.any(detection_mask):
            residual = (f_obs[detection_mask] - model_flux[detection_mask]) / sigma[detection_mask]
            logdet = np.sum(np.log(2.0 * np.pi * sigma[detection_mask] ** 2))
            log_like += -0.5 * (np.sum(residual**2) + logdet)
        if np.any(upper_mask):
            upper_limit = self.dataset.active_upper_limit
            z = (upper_limit[upper_mask] - model_flux[upper_mask]) / sigma[upper_mask]
            log_like += np.sum(_normal_logcdf(z))

        return float(log_prior + log_like)

    def simulate(self, theta: Sequence[float], noise_fn, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate flux-like observations for active/masked bands.

        ``theta`` may be a single vector with shape ``(dim,)`` or a batch with
        shape ``(n, dim)``. The simulator returns the same active-band vector
        convention consumed by ``log_prob``. ``noise_fn`` is called with the
        noiseless active flux and must return Gaussian sigma values with the
        same shape. Extended signatures accepting ``theta=`` and/or ``rng=`` are
        also supported.
        """

        if rng is None:
            rng = np.random.default_rng()
        theta_arr = np.asarray(theta, dtype=float)
        single = theta_arr.ndim == 1
        if single:
            theta_batch = theta_arr[None, :]
        elif theta_arr.ndim == 2:
            theta_batch = theta_arr
        else:
            raise ValueError("theta must have shape (dim,) or (n, dim).")

        _, _, idx, active_bands = self.dataset.active_arrays()
        if idx.size == 0:
            raise ValueError("SEDDataset has no active photometric bands.")
        draws = []
        for row in theta_batch:
            try:
                flux = self._predict_active_model_flux(row, idx=idx, active_bands=active_bands)
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise PhotometricSimulationError(f"Backend numerical failure during simulation: {exc}") from exc
            if not np.all(np.isfinite(flux)):
                raise PhotometricSimulationError("Backend produced non-finite noiseless flux.")
            sigma = np.asarray(_call_noise_fn(noise_fn, flux, theta=row, rng=rng), dtype=float)
            if sigma.shape != flux.shape:
                raise ValueError(f"noise_fn returned sigma shape {sigma.shape}; expected {flux.shape}.")
            if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
                raise ValueError("noise_fn must return finite non-negative sigma values.")
            draws.append(flux + rng.normal(loc=0.0, scale=sigma, size=flux.shape))

        out = np.stack(draws, axis=0)
        return out[0] if single else out

    rvs = simulate

    def _predict_active_model_flux(
        self, theta: Sequence[float], idx: np.ndarray | None = None, active_bands: Sequence[str] | None = None
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        params = self.parameter_space.to_dict(theta)
        backend_params, mass_scale = _backend_params_and_mass_scale(
            params,
            self.backend,
            quantity_name="photometry",
        )

        if idx is None or active_bands is None:
            _, _, idx, active_bands = self.dataset.active_arrays()

        filters = self.filters
        if filters is None:
            filters = self.dataset.metadata.get("filters")

        model = self.backend.predict_photometry(backend_params, filters)
        model_flux = self._align_model_flux(model, idx, active_bands)
        return mass_scale * np.asarray(model_flux, dtype=float)

    @staticmethod
    def _align_model_flux(model, active_indices: np.ndarray, active_bands: Sequence[str]) -> np.ndarray:
        del active_indices
        flux = np.asarray(model.flux, dtype=float)
        names = tuple(str(name) for name in getattr(model, "band_names", ()))
        if len(names) != flux.size:
            raise ValueError("ModelPhotometry band_names length must match flux length.")
        if len(set(names)) != len(names):
            raise ValueError("ModelPhotometry band_names must be unique.")
        active_bands = tuple(str(name) for name in active_bands)
        missing = [name for name in active_bands if name not in names]
        if missing:
            raise ValueError(f"Model photometry is missing active band(s): {', '.join(missing)}")
        lookup = {name: i for i, name in enumerate(names)}
        return np.asarray([flux[lookup[name]] for name in active_bands], dtype=float)


@dataclass
class GaussianSpectralLikelihood:
    """Backend-agnostic Gaussian spectral log posterior for one spectrum.

    The dataset is interpreted as observed-frame ``f_lambda`` sampled on a
    strictly increasing wavelength grid. The likelihood asks the backend for a
    model spectrum on the active data wavelengths, applies the same explicit
    mass-normalization rule as photometry, and evaluates a diagonal Gaussian
    log likelihood plus the parameter-space log prior.
    """

    backend: object
    dataset: SpectrumDataset
    parameter_space: ParameterSpace
    sigma_floor: float | None = None

    def log_prob(self, theta: Sequence[float]) -> float:
        theta = np.asarray(theta, dtype=float)
        log_prior = self.parameter_space.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf

        wavelength, f_obs, sigma, _ = self.dataset.active_arrays()
        if f_obs.size == 0:
            raise ValueError("SpectrumDataset has no active spectral pixels.")
        if self.sigma_floor is not None:
            floor = float(self.sigma_floor)
            if floor < 0.0:
                raise ValueError("sigma_floor must be non-negative.")
            sigma = np.sqrt(sigma**2 + floor**2)

        try:
            model_flux = self._predict_active_model_flux(theta, wavelength)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return -np.inf

        if model_flux.shape != f_obs.shape:
            raise ValueError(f"Model spectrum shape {model_flux.shape} does not match data shape {f_obs.shape}.")
        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        residual = (f_obs - model_flux) / sigma
        logdet = np.sum(np.log(2.0 * np.pi * sigma**2))
        return float(log_prior - 0.5 * (np.sum(residual**2) + logdet))

    def simulate(self, theta: Sequence[float], noise_fn, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate flux-like spectral observations on active data pixels."""

        if rng is None:
            rng = np.random.default_rng()
        theta_arr = np.asarray(theta, dtype=float)
        single = theta_arr.ndim == 1
        if single:
            theta_batch = theta_arr[None, :]
        elif theta_arr.ndim == 2:
            theta_batch = theta_arr
        else:
            raise ValueError("theta must have shape (dim,) or (n, dim).")

        wavelength = self.dataset.active_wavelength
        if wavelength.size == 0:
            raise ValueError("SpectrumDataset has no active spectral pixels.")
        draws = []
        for row in theta_batch:
            try:
                flux = self._predict_active_model_flux(row, wavelength)
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise SpectralSimulationError(f"Backend numerical failure during simulation: {exc}") from exc
            if not np.all(np.isfinite(flux)):
                raise SpectralSimulationError("Backend produced non-finite noiseless spectrum.")
            sigma = np.asarray(_call_noise_fn(noise_fn, flux, theta=row, rng=rng), dtype=float)
            if sigma.shape != flux.shape:
                raise ValueError(f"noise_fn returned sigma shape {sigma.shape}; expected {flux.shape}.")
            if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
                raise ValueError("noise_fn must return finite non-negative sigma values.")
            draws.append(flux + rng.normal(loc=0.0, scale=sigma, size=flux.shape))

        out = np.stack(draws, axis=0)
        return out[0] if single else out

    rvs = simulate

    def _predict_active_model_flux(self, theta: Sequence[float], wavelength: np.ndarray) -> np.ndarray:
        params = self.parameter_space.to_dict(theta)
        backend_params, mass_scale = _backend_params_and_mass_scale(
            params,
            self.backend,
            quantity_name="spectrum",
        )
        model = self.backend.predict_spectrum(backend_params, wavelengths=wavelength)
        model_wavelength = np.asarray(model.wavelength, dtype=float)
        model_flux = np.asarray(model.flux, dtype=float)
        if model_flux.shape != wavelength.shape:
            raise ValueError(f"Model spectrum shape {model_flux.shape} does not match data shape {wavelength.shape}.")
        if model_wavelength.shape != wavelength.shape:
            raise ValueError(
                f"Model wavelength shape {model_wavelength.shape} does not match data shape {wavelength.shape}."
            )
        if not np.allclose(model_wavelength, wavelength, rtol=1e-10, atol=1e-8):
            raise ValueError("Model spectrum wavelength grid does not match the active data wavelength grid.")
        return mass_scale * model_flux


def _backend_params_and_mass_scale(
    params: dict[str, float],
    backend: object,
    *,
    quantity_name: str,
) -> tuple[dict[str, float], float]:
    backend_params = dict(params)
    mass_norm = getattr(backend, "mass_normalization", None)
    if mass_norm is None:
        raise ValueError("Backend must declare mass_normalization.")
    mass_norm = MassNormalization(mass_norm)

    if mass_norm == MassNormalization.PER_SOLAR_MASS:
        if "log10_mass" not in params:
            raise ValueError(
                f"Backend returns PER_SOLAR_MASS {quantity_name}, but ParameterSpace is missing 'log10_mass'."
            )
        log10_mass = float(params["log10_mass"])
        backend_params.pop("log10_mass", None)
        return backend_params, 10.0**log10_mass
    return backend_params, 1.0


def _call_noise_fn(noise_fn, flux: np.ndarray, *, theta: np.ndarray, rng: np.random.Generator):
    try:
        params = inspect.signature(noise_fn).parameters
    except (TypeError, ValueError):
        return noise_fn(flux)
    kwargs = {}
    if "theta" in params:
        kwargs["theta"] = theta
    if "rng" in params:
        kwargs["rng"] = rng
    return noise_fn(flux, **kwargs)


def _normal_logcdf(z: np.ndarray) -> np.ndarray:
    """Stable enough log CDF for a standard normal without requiring SciPy."""

    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    very_negative = z < -10.0
    if np.any(very_negative):
        zv = z[very_negative]
        out[very_negative] = -0.5 * zv**2 - np.log(-zv) - 0.5 * np.log(2.0 * np.pi)
    if np.any(~very_negative):
        zv = z[~very_negative]
        cdf = np.array([0.5 * math.erfc(-float(value) / math.sqrt(2.0)) for value in zv], dtype=float)
        out[~very_negative] = np.log(np.maximum(cdf, np.finfo(float).tiny))
    return out
