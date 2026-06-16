from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from composed.backends.base import ModelPhotometry, ModelSpectrum, SEDBackend
from composed.filters import FilterSet
from composed.transforms.sfh import normalize_sfh_to_formed_mass
from composed.units import LSUN_CGS, MassNormalization, PARSEC_CM

REDSHIFT_KEYS = ("z", "zred", "redshift")
FSPS_PARAMETER_KEYS = (
    "logzsol",
    "dust2",
    "dust1",
    "dust_index",
    "gas_logz",
    "gas_logu",
    "fagn",
    "agn_tau",
)


@dataclass
class FSPSBackend(SEDBackend):
    """Production FSPS forward-model backend.

    Dependencies are lazy-imported. Construction checks that the ``fsps`` module
    is discoverable without importing it, so import errors are raised early with
    a targeted message while avoiding FSPS initialization until first use.

    FSPS conventions used here:

    - ``sp.get_spectrum(tage=..., peraa=True)`` returns rest-frame wavelength in
      Angstrom and luminosity density ``L_lambda`` in ``Lsun / Angstrom`` for the
      currently configured stellar population and tabular SFH normalization.
    - The backend redshifts wavelengths via ``lambda_obs = lambda_rest * (1+z)``.
    - It converts rest luminosity density to observed ``f_lambda`` in
      ``erg / s / cm^2 / Angstrom`` using
      ``L_lambda * Lsun_cgs / (4*pi*d_L^2*(1+z))``.
    - ``sedpy.observate.getSED`` integrates that observed spectrum through the
      supplied filters and returns AB magnitudes, which are converted to maggies.

    If ``mass_normalization`` is ``PER_SOLAR_MASS``, the tabular SFH is
    normalized to one solar mass formed before evaluating FSPS. If it is
    ``ABSOLUTE``, the SFH normalization is passed through unchanged.

    The default cosmology is Astropy ``Planck18``. It is used for luminosity
    distances and for rejecting tabular SFH ages that exceed the age of the
    Universe at the requested redshift.
    """

    sp_kwargs: Mapping[str, Any] | None = None
    cosmology: Any | None = None
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    default_z_key: str = "zred"
    age_tolerance_gyr: float = 1e-6
    _sp: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mass_normalization = MassNormalization(self.mass_normalization)
        if self.default_z_key not in REDSHIFT_KEYS:
            raise ValueError(f"default_z_key must be one of {REDSHIFT_KEYS}.")
        if float(self.age_tolerance_gyr) < 0.0:
            raise ValueError("age_tolerance_gyr must be non-negative.")
        if not _module_available("fsps"):
            raise ImportError(
                "FSPSBackend requires python-fsps and FSPS stellar population grids. "
                "Install python-fsps and configure SPS_HOME before constructing FSPSBackend."
            )

    def predict_photometry(self, params: Mapping[str, Any], filters: FilterSet | Sequence[object]) -> ModelPhotometry:
        """Predict observed-frame filter photometry in maggies."""

        filter_set = coerce_filter_set(filters)
        wave_obs_a, flam_obs, metadata = self._observed_spectrum_from_params(params)
        get_sed = _load_getsed()

        mags = get_sed(wave_obs_a, flam_obs, list(filter_set.filters), linear_flux=False)
        flux_maggies = 10.0 ** (-0.4 * np.asarray(mags, dtype=float))
        if flux_maggies.shape != (len(filter_set),):
            raise ValueError(
                f"sedpy returned photometry shape {flux_maggies.shape}; expected {(len(filter_set),)}."
            )
        if not np.all(np.isfinite(flux_maggies)):
            raise FloatingPointError("FSPS photometry contains non-finite values.")
        return ModelPhotometry(band_names=filter_set.names, flux=flux_maggies, metadata=metadata)

    def predict_spectrum(
        self,
        params: Mapping[str, Any],
        wavelengths: Sequence[float] | None = None,
        wavelength_range: tuple[float, float] | None = None,
        resolution: float | None = None,
    ) -> ModelSpectrum:
        """Predict observed-frame ``f_lambda`` in cgs per Angstrom.

        ``wavelengths`` and ``wavelength_range`` are observed-frame Angstrom.
        Instrumental convolution is not implemented in this first pass, so
        ``resolution`` must be omitted.
        """

        if resolution is not None:
            raise NotImplementedError("FSPSBackend spectral resolution convolution is not implemented yet.")
        wave_obs_a, flam_obs, metadata = self._observed_spectrum_from_params(params)
        wave_out, flux_out = _sample_or_clip_spectrum(wave_obs_a, flam_obs, wavelengths, wavelength_range)
        if not np.all(np.isfinite(flux_out)):
            raise FloatingPointError("FSPS spectrum contains non-finite flux values after sampling.")
        return ModelSpectrum(
            wavelength=wave_out,
            flux=flux_out,
            wavelength_unit="angstrom",
            flux_unit="erg/s/cm^2/angstrom",
            metadata=metadata,
        )

    def _observed_spectrum_from_params(self, params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        params = dict(params)
        z = self._get_redshift(params)
        t_gyr, sfr = self._get_tabular_sfh(params)
        self._validate_tabular_sfh(t_gyr, sfr, z)

        if self.mass_normalization == MassNormalization.PER_SOLAR_MASS:
            sfr = normalize_sfh_to_formed_mass(t_gyr, sfr)

        sp = self._stellar_population()
        sp.params["zred"] = z
        for key in FSPS_PARAMETER_KEYS:
            if key in params:
                sp.params[key] = float(params[key])

        sp.set_tabular_sfh(t_gyr, sfr)
        wave_rest_a, llam_lsun_per_a = sp.get_spectrum(tage=float(t_gyr[-1]), peraa=True)
        wave_obs_a, flam_obs = self._rest_spectrum_to_observed_flux(wave_rest_a, llam_lsun_per_a, z)
        metadata = {
            "backend": "fsps",
            "redshift": z,
            "wavelength_unit": "angstrom",
            "flux_unit": "erg/s/cm^2/angstrom",
            "spectrum_frame": "observed",
            "mass_normalization": self.mass_normalization.value,
        }
        return wave_obs_a, flam_obs, metadata

    def _stellar_population(self):
        if self._sp is None:
            import fsps

            kwargs = {
                "zcontinuous": 1,
                "sfh": 3,
                "add_dust_emission": False,
                "add_neb_emission": True,
                "compute_vega_mags": False,
            }
            if self.sp_kwargs is not None:
                kwargs.update(dict(self.sp_kwargs))
            self._sp = fsps.StellarPopulation(**kwargs)
        return self._sp

    def _get_redshift(self, params: Mapping[str, Any]) -> float:
        ordered_keys = (self.default_z_key,) + tuple(key for key in REDSHIFT_KEYS if key != self.default_z_key)
        for key in ordered_keys:
            if key in params:
                z = float(params[key])
                if not np.isfinite(z) or z < 0.0:
                    raise ValueError(f"Redshift parameter {key!r} must be finite and non-negative.")
                return z
        raise ValueError("Missing redshift parameter. Provide one of: z, zred, redshift.")

    @staticmethod
    def _get_tabular_sfh(params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        has_time = "tabular_time_gyr" in params
        has_sfr = "tabular_sfr_msun_per_yr" in params
        if not (has_time and has_sfr):
            raise ValueError("FSPSBackend requires tabular_time_gyr and tabular_sfr_msun_per_yr parameters.")
        return (
            np.asarray(params["tabular_time_gyr"], dtype=float),
            np.asarray(params["tabular_sfr_msun_per_yr"], dtype=float),
        )

    def _validate_tabular_sfh(self, time_gyr: np.ndarray, sfr_msun_per_yr: np.ndarray, z: float) -> None:
        if time_gyr.ndim != 1 or sfr_msun_per_yr.ndim != 1 or time_gyr.shape != sfr_msun_per_yr.shape:
            raise ValueError("tabular_time_gyr and tabular_sfr_msun_per_yr must be matching 1D arrays.")
        if time_gyr.size < 2:
            raise ValueError("tabular_time_gyr must contain at least two time points.")
        if not np.all(np.isfinite(time_gyr)):
            raise ValueError("tabular_time_gyr must be finite.")
        if np.any(time_gyr < 0.0):
            raise ValueError("tabular_time_gyr must be non-negative cosmic time in Gyr.")
        if np.any(np.diff(time_gyr) <= 0.0):
            raise ValueError("tabular_time_gyr must be strictly increasing.")
        if not np.all(np.isfinite(sfr_msun_per_yr)):
            raise ValueError("tabular_sfr_msun_per_yr must be finite.")
        if np.any(sfr_msun_per_yr < 0.0):
            raise ValueError("tabular_sfr_msun_per_yr must be non-negative.")

        age_universe_gyr = self._age_of_universe_gyr(z)
        if time_gyr[-1] > age_universe_gyr + float(self.age_tolerance_gyr):
            raise ValueError(
                "tabular_time_gyr exceeds the age of the Universe at the requested redshift "
                f"({time_gyr[-1]:.6g} Gyr > {age_universe_gyr:.6g} Gyr at z={z:.6g})."
            )

    def _age_of_universe_gyr(self, z: float) -> float:
        cosmo = self._cosmology()
        return float(cosmo.age(float(z)).to("Gyr").value)

    def _luminosity_distance_cm(self, z: float) -> float:
        if z <= 0.0:
            return 10.0 * PARSEC_CM
        cosmo = self._cosmology()
        return float(cosmo.luminosity_distance(float(z)).to("cm").value)

    def _cosmology(self):
        if self.cosmology is not None:
            return self.cosmology
        from astropy.cosmology import Planck18

        return Planck18

    def _rest_spectrum_to_observed_flux(
        self, wave_rest_a: Sequence[float], llam_lsun_per_a: Sequence[float], z: float
    ) -> tuple[np.ndarray, np.ndarray]:
        wave_rest_a = np.asarray(wave_rest_a, dtype=float)
        llam_lsun_per_a = np.asarray(llam_lsun_per_a, dtype=float)
        if wave_rest_a.ndim != 1 or llam_lsun_per_a.ndim != 1 or wave_rest_a.shape != llam_lsun_per_a.shape:
            raise ValueError("FSPS spectrum wavelength and luminosity arrays must be matching 1D arrays.")
        if wave_rest_a.size < 2 or np.any(np.diff(wave_rest_a) <= 0.0):
            raise FloatingPointError("FSPS spectrum wavelength grid must be strictly increasing.")
        if not np.all(np.isfinite(wave_rest_a)) or not np.all(np.isfinite(llam_lsun_per_a)):
            raise FloatingPointError("FSPS spectrum contains non-finite values.")

        d_l_cm = self._luminosity_distance_cm(z)
        wave_obs_a = wave_rest_a * (1.0 + z)
        flam_obs = (llam_lsun_per_a * LSUN_CGS) / (4.0 * np.pi * d_l_cm**2 * (1.0 + z))
        return wave_obs_a, flam_obs


def coerce_filter_set(filters: FilterSet | Sequence[object]) -> FilterSet:
    if filters is None:
        raise ValueError("FSPSBackend requires a FilterSet or sequence of sedpy filters.")
    if isinstance(filters, FilterSet):
        return filters
    return FilterSet(filters=tuple(filters))


def _load_getsed():
    try:
        from sedpy.observate import getSED
    except ImportError as exc:
        raise ImportError("FSPSBackend requires sedpy for filter integration.") from exc
    return getSED


def _module_available(name: str) -> bool:
    module = sys.modules.get(name)
    if module is not None:
        return True
    return importlib.util.find_spec(name) is not None


def _sample_or_clip_spectrum(
    wavelength: Sequence[float],
    flux: Sequence[float],
    requested_wavelengths: Sequence[float] | None,
    wavelength_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(wavelength, dtype=float)
    spec = np.asarray(flux, dtype=float)
    if requested_wavelengths is not None:
        out_wave = np.asarray(requested_wavelengths, dtype=float)
        if out_wave.ndim != 1:
            raise ValueError("Requested wavelengths must be one-dimensional.")
        out_flux = np.interp(out_wave, wave, spec, left=np.nan, right=np.nan)
    else:
        out_wave = wave.copy()
        out_flux = spec.copy()

    if wavelength_range is not None:
        lo, hi = wavelength_range
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("wavelength_range must be a finite increasing (min, max) pair.")
        keep = (out_wave >= lo) & (out_wave <= hi)
        out_wave = out_wave[keep]
        out_flux = out_flux[keep]
    return out_wave, out_flux
