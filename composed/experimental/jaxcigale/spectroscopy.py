from __future__ import annotations

from typing import Sequence

import numpy as np

from composed.experimental.jaxcigale.dependencies import require_jax

C_KM_PER_S = 299792.458
GAUSSIAN_FWHM_OVER_SIGMA = 2.3548200450309493


def pixel_edges_from_centers_numpy(wavelength_obs_a: Sequence[float]) -> np.ndarray:
    """Return wavelength-bin edges from strictly increasing pixel centers.

    The input and output are observed-frame Angstrom.  Interior edges are the
    midpoints between neighboring pixels; the first and last edges extrapolate
    by half of the nearest pixel spacing.  This is the simple convention used
    by many reduced spectra when the native pixel edges are not available.
    """

    wavelength = np.asarray(wavelength_obs_a, dtype=float)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength_obs_a must be a one-dimensional grid with at least two pixels.")
    if not np.all(np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_obs_a must be finite and strictly increasing.")
    interior = 0.5 * (wavelength[1:] + wavelength[:-1])
    first = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    last = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return np.concatenate([[first], interior, [last]])


def validate_pixel_edges_numpy(pixel_edges_obs_a: Sequence[float], wavelength_obs_a: Sequence[float]) -> np.ndarray:
    """Validate observed-frame spectral pixel edges."""

    edges = np.asarray(pixel_edges_obs_a, dtype=float)
    centers = np.asarray(wavelength_obs_a, dtype=float)
    if edges.ndim != 1 or edges.shape != (centers.size + 1,):
        raise ValueError("pixel_edges_obs_a must have length len(wavelength_obs_a) + 1.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("pixel_edges_obs_a must be finite and strictly increasing.")
    if np.any(centers <= edges[:-1]) or np.any(centers >= edges[1:]):
        raise ValueError("Each wavelength center must lie inside its pixel edges.")
    return edges


def model_spectrum_on_observed_pixels(
    model_wavelength_obs_a,
    model_flux_lambda_cgs,
    data_wavelength_obs_a,
    pixel_edges_obs_a,
    *,
    resample_mode: str = "bin",
    lsf_fwhm_a: float | None = None,
    resolving_power: float | Sequence[float] | None = None,
    velocity_sigma_kms: float | None = None,
):
    """Broaden and resample a model spectrum onto observed spectral pixels.

    All wavelengths are observed-frame Angstrom.  Fluxes are
    ``f_lambda`` in ``erg s^-1 cm^-2 Angstrom^-1``.

    The intended sequence is:

    1. Start from the model's high-resolution observed-frame spectrum.
    2. Apply one Gaussian line-spread function on that observed grid.
    3. Compare to data either by integrating over the data pixels
       (``resample_mode='bin'``) or, for debugging, by point interpolation
       (``resample_mode='interp'``).

    The Gaussian smoothing is a direct O(N^2) quadrature.  That is not the
    final high-throughput algorithm, but it is transparent and works on
    irregular wavelength grids, which is exactly what we want for the first
    science-audit implementation.
    """

    _, jnp = require_jax()
    wave_model = jnp.asarray(model_wavelength_obs_a)
    flux_model = jnp.asarray(model_flux_lambda_cgs)
    wave_data = jnp.asarray(data_wavelength_obs_a)
    pixel_edges = jnp.asarray(pixel_edges_obs_a)

    broadened = gaussian_lsf_smooth_observed(
        wave_model,
        flux_model,
        lsf_fwhm_a=lsf_fwhm_a,
        resolving_power=resolving_power,
        resolving_power_wavelength_obs_a=wave_data,
        velocity_sigma_kms=velocity_sigma_kms,
    )

    if resample_mode == "interp":
        return jnp.interp(wave_data, wave_model, broadened, left=0.0, right=0.0)
    if resample_mode == "bin":
        return bin_average_spectrum(wave_model, broadened, pixel_edges)
    raise ValueError("resample_mode must be 'bin' or 'interp'.")


def gaussian_lsf_smooth_observed(
    wavelength_obs_a,
    flux_lambda_cgs,
    *,
    lsf_fwhm_a: float | None = None,
    resolving_power: float | Sequence[float] | None = None,
    resolving_power_wavelength_obs_a=None,
    velocity_sigma_kms: float | None = None,
):
    """Apply a Gaussian LSF to an observed-frame spectrum.

    Exactly zero or one broadening description may be supplied:

    - ``lsf_fwhm_a``: constant observed-frame FWHM in Angstrom.
    - ``resolving_power``: scalar or wavelength-dependent R = lambda / FWHM.
    - ``velocity_sigma_kms``: constant Gaussian velocity sigma.

    The returned spectrum remains in ``f_lambda`` cgs per Angstrom.
    """

    _, jnp = require_jax()
    wave = jnp.asarray(wavelength_obs_a)
    flux = jnp.asarray(flux_lambda_cgs)
    n_modes = sum(value is not None for value in (lsf_fwhm_a, resolving_power, velocity_sigma_kms))
    if n_modes == 0:
        return flux
    if n_modes > 1:
        raise ValueError("Specify only one of lsf_fwhm_a, resolving_power, or velocity_sigma_kms.")

    if lsf_fwhm_a is not None:
        sigma_a = jnp.ones_like(wave) * (float(lsf_fwhm_a) / GAUSSIAN_FWHM_OVER_SIGMA)
    elif resolving_power is not None:
        resolving_power_array = jnp.asarray(resolving_power)
        if resolving_power_array.ndim == 0:
            r_on_wave = jnp.ones_like(wave) * resolving_power_array
        else:
            if resolving_power_wavelength_obs_a is None:
                r_wave = wave
            else:
                r_wave = jnp.asarray(resolving_power_wavelength_obs_a)
            r_on_wave = jnp.interp(wave, r_wave, resolving_power_array, left=resolving_power_array[0], right=resolving_power_array[-1])
        sigma_a = wave / (r_on_wave * GAUSSIAN_FWHM_OVER_SIGMA)
    else:
        sigma_a = wave * (float(velocity_sigma_kms) / C_KM_PER_S)
    sigma_a = jnp.maximum(sigma_a, 1.0e-12)

    source_width = trapezoid_widths(wave)
    delta = (wave[:, None] - wave[None, :]) / sigma_a[:, None]
    weights = jnp.exp(-0.5 * delta * delta) * source_width[None, :]
    norm = jnp.maximum(jnp.sum(weights, axis=1), 1.0e-300)
    return jnp.sum(weights * flux[None, :], axis=1) / norm


def bin_average_spectrum(wavelength_obs_a, flux_lambda_cgs, pixel_edges_obs_a):
    """Average ``f_lambda`` over observed spectral pixels.

    The output value for each pixel is

    ``integral f_lambda d_lambda / pixel_width``.

    This preserves integrated flux through resampling.  It is the right default
    when emission lines are narrower than the data pixels.
    """

    _validate_bin_coverage_if_concrete(wavelength_obs_a, pixel_edges_obs_a)
    _, jnp = require_jax()
    wave = jnp.asarray(wavelength_obs_a)
    flux = jnp.asarray(flux_lambda_cgs)
    edges = jnp.asarray(pixel_edges_obs_a)

    cumulative = cumulative_trapezoid(wave, flux)
    cumulative_at_edges = jnp.interp(edges, wave, cumulative, left=0.0, right=cumulative[-1])
    integrated = cumulative_at_edges[1:] - cumulative_at_edges[:-1]
    widths = jnp.maximum(edges[1:] - edges[:-1], 1.0e-300)
    averaged = integrated / widths
    coverage_ok = (edges[0] >= wave[0]) & (edges[-1] <= wave[-1])
    return jnp.where(coverage_ok, averaged, jnp.nan)


def _validate_bin_coverage_if_concrete(wavelength_obs_a, pixel_edges_obs_a) -> None:
    """Reject bin averaging when concrete pixel edges extend beyond model support.

    ``jnp.interp`` is intentionally allowed to extrapolate for point-sampling
    debug plots, but bin-averaging a spectrum outside the model wavelength
    coverage loses half of the first/last pixels for even a constant spectrum.
    During JIT tracing the inputs may be tracers, so this check is best-effort;
    concrete calls, tests, and notebook setup get the explicit error.
    """

    try:
        wave = np.asarray(wavelength_obs_a, dtype=float)
        edges = np.asarray(pixel_edges_obs_a, dtype=float)
    except Exception:
        return
    if wave.ndim != 1 or edges.ndim != 1 or wave.size == 0 or edges.size == 0:
        return
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(edges)):
        return
    scale = max(1.0, float(np.max(np.abs(np.concatenate([wave[[0, -1]], edges[[0, -1]]])))))
    tol = 32.0 * np.finfo(float).eps * scale
    if edges[0] < wave[0] - tol or edges[-1] > wave[-1] + tol:
        raise ValueError(
            "Model wavelength grid must cover all spectral pixel edges for bin averaging. "
            "Extend the model wavelength grid, provide narrower pixel_edges_obs_a, or use resample_mode='interp'."
        )


def cumulative_trapezoid(x, y):
    """Cumulative trapezoidal integral with the first value equal to zero."""

    _, jnp = require_jax()
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    increments = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
    return jnp.concatenate([jnp.zeros((1,), dtype=y.dtype), jnp.cumsum(increments)])


def trapezoid_widths(x):
    """Quadrature widths associated with sample centers."""

    _, jnp = require_jax()
    x = jnp.asarray(x)
    first = 0.5 * (x[1:2] - x[0:1])
    middle = 0.5 * (x[2:] - x[:-2])
    last = 0.5 * (x[-1:] - x[-2:-1])
    return jnp.concatenate([first, middle, last])
