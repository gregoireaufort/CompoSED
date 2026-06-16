from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple, Sequence

import numpy as np

from composed.experimental.jaxcigale.dependencies import require_jax
from composed.experimental.jaxcigale.parameters import JaxParameterSpace
from composed.experimental.jaxcigale.photometry import JaxFilterSet, integrate_maggies
from composed.experimental.jaxcigale.spectroscopy import (
    model_spectrum_on_observed_pixels,
    pixel_edges_from_centers_numpy,
    validate_pixel_edges_numpy,
)
from composed.units import LSUN_CGS

MPC_CM = 3.0856775814913673e24
C_KM_PER_S = 299792.458

_ATTENUATION_MODULE_NAMES = frozenset(
    {
        "calzetti_attenuation",
        "modified_starburst_attenuation",
        "gordon16_rvfa_extinction",
        "smc_screen_attenuation",
    }
)
_DUST_EMISSION_MODULE_NAMES = frozenset({"modified_blackbody_dust", "two_temperature_dust"})
_REST_FRAME_TERMINAL_MODULE_NAMES = frozenset({"madau_igm", "redshift"})


class SEDState(NamedTuple):
    """Fixed-shape arrays carried through the differentiable SED graph."""

    wave_rest_a: object
    sfh_time_gyr: object
    sfr_msun_per_yr: object
    formed_mass_msun: object
    intrinsic_lum_lsun_per_a: object
    stellar_lum_lsun_per_a: object
    stellar_young_lum_lsun_per_a: object
    stellar_old_lum_lsun_per_a: object
    nebular_lum_lsun_per_a: object
    nebular_continuum_lum_lsun_per_a: object
    nebular_line_lum_lsun_per_a: object
    attenuated_lum_lsun_per_a: object
    dust_lum_lsun_per_a: object
    total_lum_lsun_per_a: object
    absorbed_lum_lsun: object
    wave_obs_a: object
    flux_lambda_cgs: object


@dataclass(frozen=True)
class CompiledModule:
    """One fixed module in the JAX-CIGALE graph."""

    name: str
    state: object
    apply_fn: Callable[[Mapping[str, object], SEDState, object], SEDState]

    def apply(self, params: Mapping[str, object], sed_state: SEDState) -> SEDState:
        return self.apply_fn(params, sed_state, self.state)


@dataclass(frozen=True)
class ModuleSpec:
    """Python setup plus pure JAX apply function."""

    name: str
    setup_fn: Callable[[Mapping[str, object]], object]
    apply_fn: Callable[[Mapping[str, object], SEDState, object], SEDState]
    config: Mapping[str, object] | None = None

    def setup(self, grids: Mapping[str, object]) -> CompiledModule:
        merged = dict(grids)
        if self.config is not None:
            merged.update(dict(self.config))
        return CompiledModule(name=self.name, state=self.setup_fn(merged), apply_fn=self.apply_fn)


@dataclass(frozen=True)
class GaussianPhotometricData:
    """Observed photometry in maggies for the JAX likelihood.

    Detections use the usual diagonal Gaussian log likelihood.  Non-detections
    are represented explicitly with ``upper_limit_mask=True`` and
    ``upper_limit_maggies`` set to the limiting flux.  For those bands the
    likelihood contribution is the Gaussian cumulative probability

    ``P(observed flux < upper_limit | model flux, sigma)``.

    ``sigma_maggies`` is always the one-sigma noise scale in maggies.  If a
    catalog reports a 5-sigma depth, convert it to ``sigma=depth/5`` and put
    ``upper_limit_maggies=depth``.
    """

    flux_maggies: np.ndarray
    sigma_maggies: np.ndarray
    mask: np.ndarray | None = None
    upper_limit_maggies: np.ndarray | None = None
    upper_limit_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        flux = np.asarray(self.flux_maggies, dtype=float)
        sigma = np.asarray(self.sigma_maggies, dtype=float)
        if flux.ndim != 1 or sigma.shape != flux.shape:
            raise ValueError("flux_maggies and sigma_maggies must be matching one-dimensional arrays.")
        if self.mask is None:
            mask = np.ones(flux.shape, dtype=bool)
        else:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != flux.shape:
                raise ValueError("mask shape must match flux_maggies.")
        if self.upper_limit_mask is None:
            upper_mask = np.zeros(flux.shape, dtype=bool)
        else:
            upper_mask = np.asarray(self.upper_limit_mask, dtype=bool)
            if upper_mask.shape != flux.shape:
                raise ValueError("upper_limit_mask shape must match flux_maggies.")
        if np.any(upper_mask & ~mask):
            raise ValueError("upper-limit bands must also be active in mask.")
        if not np.any(mask):
            raise ValueError("GaussianPhotometricData requires at least one active band.")
        if self.upper_limit_maggies is None:
            upper_limit = np.zeros(flux.shape, dtype=float)
        else:
            upper_limit = np.asarray(self.upper_limit_maggies, dtype=float)
            if upper_limit.shape != flux.shape:
                raise ValueError("upper_limit_maggies shape must match flux_maggies.")

        detection_mask = mask & ~upper_mask
        if not np.all(np.isfinite(flux[detection_mask])):
            raise ValueError("Detected flux_maggies must be finite.")
        if not np.all(np.isfinite(sigma[mask])) or np.any(sigma[mask] <= 0.0):
            raise ValueError("Active sigma_maggies must be finite and strictly positive.")
        if not np.all(np.isfinite(upper_limit[upper_mask])):
            raise ValueError("Active upper_limit_maggies values must be finite.")

        safe_flux = np.where(detection_mask, flux, 0.0)
        safe_sigma = np.where(mask, sigma, 1.0)
        safe_upper_limit = np.where(upper_mask, upper_limit, 0.0)
        object.__setattr__(self, "flux_maggies", flux)
        object.__setattr__(self, "flux_maggies", safe_flux)
        object.__setattr__(self, "sigma_maggies", safe_sigma)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "upper_limit_maggies", safe_upper_limit)
        object.__setattr__(self, "upper_limit_mask", upper_mask)

    def as_jax(self):
        _, jnp = require_jax()
        return (
            jnp.asarray(self.flux_maggies),
            jnp.asarray(self.sigma_maggies),
            jnp.asarray(self.mask),
            jnp.asarray(self.upper_limit_maggies),
            jnp.asarray(self.upper_limit_mask),
        )


@dataclass(frozen=True)
class GaussianSpectralData:
    """Observed-frame spectrum for a diagonal Gaussian JAX likelihood.

    Wavelengths are observed-frame Angstrom.  Flux and sigma are
    ``f_lambda`` in cgs units, ``erg s^-1 cm^-2 Angstrom^-1``.

    The model is always compared after an explicit spectral response step:

    - optionally broaden on the model's observed wavelength grid with one
      Gaussian LSF description;
    - then either integrate over spectral pixel edges (``resample_mode='bin'``)
      or point-interpolate at pixel centers (``resample_mode='interp'``).

    ``bin`` is the scientifically safer default when emission lines matter.
    """

    wavelength_obs_a: np.ndarray
    flux_lambda_cgs: np.ndarray
    sigma_lambda_cgs: np.ndarray
    mask: np.ndarray | None = None
    pixel_edges_obs_a: np.ndarray | None = None
    resample_mode: str = "bin"
    lsf_fwhm_a: float | None = None
    resolving_power: object | None = None
    velocity_sigma_kms: float | None = None

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_obs_a, dtype=float)
        flux = np.asarray(self.flux_lambda_cgs, dtype=float)
        sigma = np.asarray(self.sigma_lambda_cgs, dtype=float)
        if wavelength.ndim != 1 or flux.shape != wavelength.shape or sigma.shape != wavelength.shape:
            raise ValueError("wavelength_obs_a, flux_lambda_cgs, and sigma_lambda_cgs must be matching 1D arrays.")
        if wavelength.size < 2 or np.any(np.diff(wavelength) <= 0.0) or not np.all(np.isfinite(wavelength)):
            raise ValueError("wavelength_obs_a must be a finite strictly increasing one-dimensional grid.")
        if self.mask is None:
            mask = np.ones(wavelength.shape, dtype=bool)
        else:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != wavelength.shape:
                raise ValueError("mask shape must match wavelength_obs_a.")
        if not np.any(mask):
            raise ValueError("GaussianSpectralData requires at least one active spectral pixel.")
        if not np.all(np.isfinite(flux[mask])):
            raise ValueError("Active spectral flux values must be finite.")
        if not np.all(np.isfinite(sigma[mask])) or np.any(sigma[mask] <= 0.0):
            raise ValueError("Active spectral sigma values must be finite and strictly positive.")
        if self.resample_mode not in {"bin", "interp"}:
            raise ValueError("resample_mode must be 'bin' or 'interp'.")
        n_lsf = sum(value is not None for value in (self.lsf_fwhm_a, self.resolving_power, self.velocity_sigma_kms))
        if n_lsf > 1:
            raise ValueError("Specify only one of lsf_fwhm_a, resolving_power, or velocity_sigma_kms.")
        if self.lsf_fwhm_a is not None and self.lsf_fwhm_a <= 0.0:
            raise ValueError("lsf_fwhm_a must be positive.")
        if self.resolving_power is not None:
            resolving_power = np.asarray(self.resolving_power, dtype=float)
            if resolving_power.ndim == 0:
                if not np.isfinite(float(resolving_power)) or float(resolving_power) <= 0.0:
                    raise ValueError("resolving_power must be finite and positive.")
                resolving_power_out = float(resolving_power)
            else:
                if resolving_power.shape != wavelength.shape:
                    raise ValueError("Array-valued resolving_power must match wavelength_obs_a shape.")
                if not np.all(np.isfinite(resolving_power)) or np.any(resolving_power <= 0.0):
                    raise ValueError("Array-valued resolving_power must be finite and positive.")
                resolving_power_out = resolving_power
        else:
            resolving_power_out = None
        if self.velocity_sigma_kms is not None and self.velocity_sigma_kms <= 0.0:
            raise ValueError("velocity_sigma_kms must be positive.")
        if self.pixel_edges_obs_a is None:
            pixel_edges = pixel_edges_from_centers_numpy(wavelength)
        else:
            pixel_edges = validate_pixel_edges_numpy(self.pixel_edges_obs_a, wavelength)

        object.__setattr__(self, "wavelength_obs_a", wavelength)
        object.__setattr__(self, "flux_lambda_cgs", np.where(mask, flux, 0.0))
        object.__setattr__(self, "sigma_lambda_cgs", np.where(mask, sigma, 1.0))
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "pixel_edges_obs_a", pixel_edges)
        object.__setattr__(self, "resample_mode", str(self.resample_mode))
        object.__setattr__(self, "lsf_fwhm_a", None if self.lsf_fwhm_a is None else float(self.lsf_fwhm_a))
        object.__setattr__(self, "resolving_power", resolving_power_out)
        object.__setattr__(
            self,
            "velocity_sigma_kms",
            None if self.velocity_sigma_kms is None else float(self.velocity_sigma_kms),
        )

    def as_jax(self):
        _, jnp = require_jax()
        return (
            jnp.asarray(self.wavelength_obs_a),
            jnp.asarray(self.flux_lambda_cgs),
            jnp.asarray(self.sigma_lambda_cgs),
            jnp.asarray(self.mask),
        )


@dataclass(frozen=True)
class GaussianSpectroPhotometricData:
    """Joint photometry plus spectroscopy data for one object."""

    photometry: GaussianPhotometricData | None = None
    spectrum: GaussianSpectralData | None = None

    def __post_init__(self) -> None:
        if self.photometry is None and self.spectrum is None:
            raise ValueError("GaussianSpectroPhotometricData requires photometry, spectrum, or both.")


@dataclass(frozen=True)
class JaxSedModel:
    """Compiled differentiable SED model with fixed modules and filters."""

    modules: tuple[CompiledModule, ...]
    wavelength_grid_a: np.ndarray
    filters: JaxFilterSet
    parameter_space: JaxParameterSpace
    mass_parameter: str = "log10_mass"
    fixed_parameters: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        wave = np.asarray(self.wavelength_grid_a, dtype=float)
        if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0.0):
            raise ValueError("wavelength_grid_a must be a strictly increasing one-dimensional grid.")
        fixed = {} if self.fixed_parameters is None else {str(k): float(v) for k, v in self.fixed_parameters.items()}
        overlap = set(fixed).intersection(self.parameter_space.names)
        if overlap:
            raise ValueError(f"fixed_parameters overlap fitted parameters: {sorted(overlap)}")
        object.__setattr__(self, "wavelength_grid_a", wave)
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "fixed_parameters", fixed)

    def initial_state(self) -> SEDState:
        _, jnp = require_jax()
        wave = jnp.asarray(self.wavelength_grid_a)
        zeros = jnp.zeros_like(wave)
        return SEDState(
            wave_rest_a=wave,
            sfh_time_gyr=jnp.zeros(2),
            sfr_msun_per_yr=jnp.zeros(2),
            formed_mass_msun=jnp.asarray(0.0),
            intrinsic_lum_lsun_per_a=zeros,
            stellar_lum_lsun_per_a=zeros,
            stellar_young_lum_lsun_per_a=zeros,
            stellar_old_lum_lsun_per_a=zeros,
            nebular_lum_lsun_per_a=zeros,
            nebular_continuum_lum_lsun_per_a=zeros,
            nebular_line_lum_lsun_per_a=zeros,
            attenuated_lum_lsun_per_a=zeros,
            dust_lum_lsun_per_a=zeros,
            total_lum_lsun_per_a=zeros,
            absorbed_lum_lsun=jnp.asarray(0.0),
            wave_obs_a=wave,
            flux_lambda_cgs=zeros,
        )

    def run_modules(self, theta) -> SEDState:
        params = self.params_from_theta(theta)
        state = self.initial_state()
        for module in self.modules:
            state = module.apply(params, state)
        return state

    def params_from_theta(self, theta) -> dict[str, object]:
        """Merge fitted vector parameters with fixed run-level parameters."""

        params = dict(self.fixed_parameters)
        params.update(self.parameter_space.params_from_theta(theta))
        return params

    def predict_photometry_per_msun(self, theta):
        state = self.run_modules(theta)
        return integrate_maggies(state.wave_obs_a, state.flux_lambda_cgs, self.filters)

    def predict_photometry(self, theta):
        state = self.run_modules_mass_scaled(theta)
        return integrate_maggies(state.wave_obs_a, state.flux_lambda_cgs, self.filters)

    def predict_spectrum(self, theta, wavelength_obs_a):
        """Return observed-frame ``f_lambda`` on requested wavelengths.

        Output units are cgs ``erg s^-1 cm^-2 Angstrom^-1`` after applying
        ``10**log10_mass`` once.
        """

        _, jnp = require_jax()
        state = self.run_modules_mass_scaled(theta)
        wavelength = jnp.asarray(wavelength_obs_a)
        return jnp.interp(wavelength, state.wave_obs_a, state.flux_lambda_cgs, left=0.0, right=0.0)

    def run_modules_mass_scaled(self, theta) -> SEDState:
        _, jnp = require_jax()
        params = self.params_from_theta(theta)
        if self.mass_parameter not in params:
            raise ValueError(f"JaxSedModel requires mass parameter {self.mass_parameter!r}.")
        mass = 10.0 ** params[self.mass_parameter]
        state = self.run_modules(theta)
        # Apply mass before filter integration. Per-solar-mass broadband f_nu
        # can be subnormal in float32 and get flushed to zero on GPU/MPS; the
        # mass-scaled observed spectrum is still comfortably finite. This keeps
        # the same scientific convention: mass is applied exactly once here.
        return state._replace(flux_lambda_cgs=state.flux_lambda_cgs * mass)

    def log_prob(self, theta, data):
        _, jnp = require_jax()
        log_prior = self.parameter_space.log_prior(theta)
        state = self.run_modules_mass_scaled(theta)

        if isinstance(data, GaussianSpectroPhotometricData):
            log_like = jnp.asarray(0.0)
            if data.photometry is not None:
                log_like = log_like + self._photometric_log_likelihood_from_state(state, data.photometry)
            if data.spectrum is not None:
                log_like = log_like + self._spectral_log_likelihood_from_state(state, data.spectrum)
            return log_prior + log_like
        if isinstance(data, GaussianSpectralData):
            return log_prior + self._spectral_log_likelihood_from_state(state, data)
        return log_prior + self._photometric_log_likelihood_from_state(state, data)

    def _photometric_log_likelihood_from_state(self, state: SEDState, data: GaussianPhotometricData):
        _, jnp = require_jax()
        from jax.scipy.special import log_ndtr

        model_flux = integrate_maggies(state.wave_obs_a, state.flux_lambda_cgs, self.filters)
        obs, sigma, mask, upper_limit, upper_mask = data.as_jax()
        valid_model = jnp.all(jnp.isfinite(model_flux)) & jnp.all(model_flux >= 0.0)
        detection_mask = mask & ~upper_mask
        sigma_safe = jnp.where(mask, sigma, 1.0)

        residual = jnp.where(detection_mask, (obs - model_flux) / sigma_safe, 0.0)
        logdet = jnp.where(detection_mask, jnp.log(2.0 * jnp.pi * sigma_safe**2), 0.0)
        detection_log_like = -0.5 * (jnp.sum(residual**2) + jnp.sum(logdet))

        limit_z = (upper_limit - model_flux) / sigma_safe
        upper_log_like = jnp.sum(jnp.where(upper_mask, log_ndtr(limit_z), 0.0))
        return jnp.where(valid_model, detection_log_like + upper_log_like, -jnp.inf)

    def _spectral_log_likelihood_from_state(self, state: SEDState, data: GaussianSpectralData):
        _, jnp = require_jax()
        wavelength, obs, sigma, mask = data.as_jax()
        model_flux = model_spectrum_on_observed_pixels(
            state.wave_obs_a,
            state.flux_lambda_cgs,
            wavelength,
            data.pixel_edges_obs_a,
            resample_mode=data.resample_mode,
            lsf_fwhm_a=data.lsf_fwhm_a,
            resolving_power=data.resolving_power,
            velocity_sigma_kms=data.velocity_sigma_kms,
        )
        valid_model = jnp.all(jnp.isfinite(model_flux))
        sigma_safe = jnp.where(mask, sigma, 1.0)
        residual = jnp.where(mask, (obs - model_flux) / sigma_safe, 0.0)
        logdet = jnp.where(mask, jnp.log(2.0 * jnp.pi * sigma_safe**2), 0.0)
        log_like = -0.5 * (jnp.sum(residual**2) + jnp.sum(logdet))
        return jnp.where(valid_model, log_like, -jnp.inf)


def build_jax_sed_model(
    modules: Sequence[ModuleSpec],
    wavelength_grid_a: Sequence[float],
    filters: JaxFilterSet,
    parameter_space: JaxParameterSpace,
    grids: Mapping[str, object] | None = None,
    mass_parameter: str = "log10_mass",
    fixed_parameters: Mapping[str, float] | None = None,
    dust_mode: str = "auto",
    default_dust_emission_module: ModuleSpec | None = None,
) -> JaxSedModel:
    """Setup modules once, then return a pure-JAX model object.

    Dust bookkeeping is deliberately explicit:

    - ``dust_mode='auto'`` (default) inserts a simple modified-blackbody
      re-emission module whenever an attenuation module is present and no dust
      emission module was supplied by the caller.
    - ``dust_mode='attenuation_only'`` preserves the historical diagnostic
      behavior where attenuated energy is recorded in ``absorbed_lum_lsun`` but
      is not reradiated.

    The automatic dust module is inserted after the last attenuation module and
    before IGM/redshift modules, so absorbed stellar/nebular light is reradiated
    in the rest frame.
    """

    grids = {} if grids is None else dict(grids)
    modules = _apply_default_dust_mode(
        modules,
        dust_mode=dust_mode,
        default_dust_emission_module=default_dust_emission_module,
    )
    compiled = tuple(module.setup(grids) for module in modules)
    return JaxSedModel(
        modules=compiled,
        wavelength_grid_a=np.asarray(wavelength_grid_a, dtype=float),
        filters=filters,
        parameter_space=parameter_space,
        mass_parameter=mass_parameter,
        fixed_parameters=fixed_parameters,
    )


def _apply_default_dust_mode(
    modules: Sequence[ModuleSpec],
    *,
    dust_mode: str,
    default_dust_emission_module: ModuleSpec | None,
) -> tuple[ModuleSpec, ...]:
    """Return the module chain after applying the JAX-CIGALE dust convention."""

    mode = str(dust_mode).lower()
    if mode not in {"auto", "attenuation_only"}:
        raise ValueError("dust_mode must be 'auto' or 'attenuation_only'.")

    module_list = list(modules)
    if mode == "attenuation_only":
        return tuple(module_list)

    names = [module.name for module in module_list]
    has_attenuation = any(name in _ATTENUATION_MODULE_NAMES for name in names)
    has_emission = any(name in _DUST_EMISSION_MODULE_NAMES for name in names)
    if not has_attenuation or has_emission:
        return tuple(module_list)

    if default_dust_emission_module is None:
        from composed.experimental.jaxcigale.modules import modified_blackbody_dust_module

        default_dust_emission_module = modified_blackbody_dust_module(
            normalization_wavelength_grid_a=np.geomspace(1.0e4, 3.0e6, 512)
        )

    last_attenuation = max(index for index, name in enumerate(names) if name in _ATTENUATION_MODULE_NAMES)
    insert_at = last_attenuation + 1
    for index in range(last_attenuation + 1, len(module_list)):
        if module_list[index].name in _REST_FRAME_TERMINAL_MODULE_NAMES:
            insert_at = index
            break
    module_list.insert(insert_at, default_dust_emission_module)
    return tuple(module_list)


def flat_lcdm_luminosity_distance_mpc(z, omega_m=0.3075, h=0.6774, n_grid=256):
    """JAX flat-LCDM luminosity distance in Mpc."""

    _, jnp = require_jax()
    z = jnp.asarray(z, dtype=jnp.result_type(z, 1.0))
    zz = jnp.linspace(0.0, z, int(n_grid))
    e_z = jnp.sqrt(omega_m * (1.0 + zz) ** 3 + (1.0 - omega_m))
    integral = jnp.sum(0.5 * (1.0 / e_z[1:] + 1.0 / e_z[:-1]) * (zz[1:] - zz[:-1]))
    d_comoving_mpc = (C_KM_PER_S / (100.0 * h)) * integral
    return (1.0 + z) * d_comoving_mpc


def flat_lcdm_luminosity_distance_cm(z, omega_m=0.3075, h=0.6774, n_grid=256):
    """JAX flat-LCDM luminosity distance in cm."""

    _, jnp = require_jax()
    return flat_lcdm_luminosity_distance_mpc(z, omega_m=omega_m, h=h, n_grid=n_grid) * jnp.asarray(MPC_CM)


def flat_lcdm_age_gyr(z, omega_m=0.3075, h=0.6774):
    """Analytic age of a flat matter+Lambda universe in Gyr."""

    _, jnp = require_jax()
    z = jnp.asarray(z, dtype=jnp.result_type(z, 1.0))
    omega_l = 1.0 - omega_m
    hubble_time_gyr = 9.778 / h
    arg = jnp.sqrt(omega_l / omega_m) / (1.0 + z) ** 1.5
    # Avoid jnp.arcsinh here.  Some accelerator backends used for exploratory
    # NUTS runs, notably Metal/MPS, do not lower arcsinh yet.  This identity is
    # mathematically equivalent for the positive cosmological argument used here.
    asinh_arg = jnp.log(arg + jnp.sqrt(1.0 + arg * arg))
    return (2.0 / (3.0 * jnp.sqrt(omega_l))) * asinh_arg * hubble_time_gyr


def flat_lcdm_age_gyr_numpy(z, omega_m=0.3075, h=0.6774):
    """NumPy version of the analytic flat-LCDM age used for setup tables."""

    z = np.asarray(z, dtype=float)
    omega_l = 1.0 - omega_m
    hubble_time_gyr = 9.778 / h
    arg = np.sqrt(omega_l / omega_m) / (1.0 + z) ** 1.5
    return (2.0 / (3.0 * np.sqrt(omega_l))) * np.arcsinh(arg) * hubble_time_gyr


def observed_flux_from_luminosity(wave_rest_a, lum_lsun_per_a, z):
    """Convert rest-frame L_lambda in Lsun/A to observed f_lambda cgs/A."""

    _, jnp = require_jax()
    z = jnp.asarray(z, dtype=jnp.result_type(z, 1.0))
    valid_z = z > 0.0
    safe_z = jnp.where(valid_z, z, jnp.asarray(1.0, dtype=z.dtype))
    d_l_mpc = flat_lcdm_luminosity_distance_mpc(safe_z)
    wave_obs_a = wave_rest_a * (1.0 + safe_z)
    # Compute the inverse-square dilution in Mpc rather than cm. Squaring a
    # luminosity distance in cm overflows float32 at cosmological distances,
    # which makes GPU/MPS runs unusable. This is algebraically identical to
    # using d_L in cm, but the intermediate numbers stay finite in float32.
    cgs_per_lsun_per_mpc2 = jnp.asarray(LSUN_CGS / (4.0 * np.pi * MPC_CM**2))
    flux_lambda_cgs = lum_lsun_per_a * cgs_per_lsun_per_mpc2 / (d_l_mpc**2 * (1.0 + safe_z))
    flux_lambda_cgs = jnp.where(valid_z, flux_lambda_cgs, jnp.nan)
    return wave_obs_a, flux_lambda_cgs
