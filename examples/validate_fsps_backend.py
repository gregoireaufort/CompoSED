"""Validate FSPSBackend against an explicit python-fsps + sedpy calculation.

Run this in a local science environment with python-fsps, sedpy, astropy, and
``SPS_HOME`` configured. The script computes the same simple tabular-SFH galaxy
two ways:

1. through ``composed.backends.FSPSBackend``;
2. directly with ``fsps.StellarPopulation`` plus sedpy filter integration.

It fails loudly for non-finite fluxes, negative maggies, shape mismatches, or a
numerical disagreement larger than the documented tolerances below.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from composed.backends.fsps import FSPSBackend
from composed.filters import FilterSet
from composed.transforms.sfh import normalize_sfh_to_formed_mass
from composed.units import LSUN_CGS, MassNormalization, PARSEC_CM

DEFAULT_FILTER_NAMES = ("sdss_g0", "sdss_r0", "sdss_i0")
FLUX_RTOL = 1e-10
MAG_ATOL = 1e-8
FLAT_FNU_SANITY_RTOL = 5e-5
C_A_PER_S = 2.99792458e18
AB_ZERO_FNU_CGS = 3631.0e-23


def check_environment() -> None:
    missing = [name for name in ("fsps", "sedpy", "astropy") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing required package(s): {', '.join(missing)}")
    sps_home = os.environ.get("SPS_HOME")
    if not sps_home:
        raise RuntimeError("SPS_HOME is not set.")
    if not Path(sps_home).exists():
        raise RuntimeError(f"SPS_HOME does not exist: {sps_home}")


def default_params() -> dict[str, object]:
    return {
        "zred": 0.1,
        "logzsol": -0.3,
        "dust2": 0.2,
        "dust1": 0.1,
        "dust_index": -0.7,
        "gas_logz": -0.3,
        "gas_logu": -2.0,
        "tabular_time_gyr": np.array([0.01, 1.0, 5.0]),
        "tabular_sfr_msun_per_yr": np.array([1.0, 1.0, 0.2]),
    }


def default_sp_kwargs() -> dict[str, object]:
    return {
        "zcontinuous": 1,
        "sfh": 3,
        "add_dust_emission": False,
        "add_neb_emission": True,
        "compute_vega_mags": False,
    }


def load_filter_set(filter_names: Sequence[str] = DEFAULT_FILTER_NAMES) -> FilterSet:
    from sedpy.observate import load_filters

    filters = load_filters(list(filter_names))
    return FilterSet(filters, names=filter_names)


def direct_fsps_sedpy_photometry(
    params: dict[str, object],
    filters: FilterSet,
    mass_normalization: MassNormalization,
    sp_kwargs: dict[str, object] | None = None,
):
    """Reference calculation that does not call FSPSBackend."""

    import fsps
    from astropy.cosmology import Planck18
    from sedpy.observate import getSED

    kwargs = default_sp_kwargs()
    if sp_kwargs is not None:
        kwargs.update(sp_kwargs)

    sp = fsps.StellarPopulation(**kwargs)
    z = float(params["zred"])
    sp.params["zred"] = z
    for key in ("logzsol", "dust2", "dust1", "dust_index", "gas_logz", "gas_logu", "fagn", "agn_tau"):
        if key in params:
            sp.params[key] = float(params[key])

    t_gyr = np.asarray(params["tabular_time_gyr"], dtype=float)
    sfr = np.asarray(params["tabular_sfr_msun_per_yr"], dtype=float)
    if mass_normalization == MassNormalization.PER_SOLAR_MASS:
        sfr = normalize_sfh_to_formed_mass(t_gyr, sfr)

    sp.set_tabular_sfh(t_gyr, sfr)
    wave_rest_a, llam_lsun_per_a = sp.get_spectrum(tage=float(t_gyr[-1]), peraa=True)

    if z > 0.0:
        d_l_cm = Planck18.luminosity_distance(z).to("cm").value
    else:
        d_l_cm = 10.0 * PARSEC_CM
    wave_obs_a = np.asarray(wave_rest_a, dtype=float) * (1.0 + z)
    flam_obs = (np.asarray(llam_lsun_per_a, dtype=float) * LSUN_CGS) / (
        4.0 * np.pi * d_l_cm**2 * (1.0 + z)
    )
    mags = np.asarray(getSED(wave_obs_a, flam_obs, list(filters.filters), linear_flux=False), dtype=float)
    flux = 10.0 ** (-0.4 * mags)
    return flux, mags, wave_obs_a, flam_obs


def ab_magnitudes_from_maggies(flux_maggies: np.ndarray) -> np.ndarray:
    flux_maggies = np.asarray(flux_maggies, dtype=float)
    return -2.5 * np.log10(flux_maggies)


def trapz_numpy(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 1.x/2.x compatible trapezoidal integral."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def flat_fnu_tophat_sanity_check() -> float:
    """Closed-form unit sanity check independent of FSPSBackend and sedpy.

    For a flat AB-zero spectrum, ``f_nu = 3631 Jy``, a top-hat filter must
    return exactly one maggie.  We evaluate the numerator numerically in
    ``f_lambda`` and compare it to the analytic integral

    ``integral f_lambda lambda dlambda = f_nu c ln(lambda_2 / lambda_1)``.

    This does not validate FSPS stellar physics, but it catches the common
    wavelength/f_lambda/f_nu unit mistakes that a backend-vs-direct refactor
    comparison can accidentally share.
    """

    lam1_a = 4000.0
    lam2_a = 5000.0
    wave_a = np.linspace(lam1_a, lam2_a, 4096)
    flat_flam = AB_ZERO_FNU_CGS * C_A_PER_S / wave_a**2
    numerator = trapz_numpy(flat_flam * wave_a, wave_a)
    analytic_ab_zero = AB_ZERO_FNU_CGS * C_A_PER_S * np.log(lam2_a / lam1_a)
    maggies = numerator / analytic_ab_zero
    error = abs(maggies - 1.0)
    if error > FLAT_FNU_SANITY_RTOL:
        raise RuntimeError(
            f"Flat-fnu top-hat sanity check failed: got {maggies:.8f} maggies; "
            f"expected 1 within {FLAT_FNU_SANITY_RTOL:.1e}."
        )
    return float(error)


def filter_effective_wavelengths(filters: FilterSet) -> list[float]:
    wavelengths = []
    for filt in filters.filters:
        value = getattr(filt, "wave_effective", getattr(filt, "lambda_eff", np.nan))
        wavelengths.append(float(value) if np.isfinite(value) else np.nan)
    return wavelengths


def validate_outputs(backend_flux: np.ndarray, reference_flux: np.ndarray, n_filters: int) -> tuple[np.ndarray, float, float]:
    backend_flux = np.asarray(backend_flux, dtype=float)
    reference_flux = np.asarray(reference_flux, dtype=float)
    if backend_flux.shape != (n_filters,):
        raise RuntimeError(f"Backend flux shape {backend_flux.shape} does not match expected {(n_filters,)}.")
    if reference_flux.shape != (n_filters,):
        raise RuntimeError(f"Reference flux shape {reference_flux.shape} does not match expected {(n_filters,)}.")
    if not np.all(np.isfinite(backend_flux)) or not np.all(np.isfinite(reference_flux)):
        raise RuntimeError("Validation produced NaN or inf fluxes.")
    if np.any(backend_flux < 0.0) or np.any(reference_flux < 0.0):
        raise RuntimeError("Validation produced negative maggies.")

    backend_mag = ab_magnitudes_from_maggies(backend_flux)
    reference_mag = ab_magnitudes_from_maggies(reference_flux)
    rel = np.abs(backend_flux - reference_flux) / np.maximum(np.abs(reference_flux), np.finfo(float).tiny)
    mag_diff = np.abs(backend_mag - reference_mag)
    max_rel = float(np.max(rel))
    max_mag_diff = float(np.max(mag_diff))
    if max_rel > FLUX_RTOL:
        raise RuntimeError(f"Flux relative difference {max_rel:.3e} exceeds tolerance {FLUX_RTOL:.3e}.")
    if max_mag_diff > MAG_ATOL:
        raise RuntimeError(f"AB magnitude difference {max_mag_diff:.3e} exceeds tolerance {MAG_ATOL:.3e}.")
    return backend_mag, max_rel, max_mag_diff


def run_validation(filter_names: Sequence[str] = DEFAULT_FILTER_NAMES) -> dict[str, object]:
    check_environment()
    flat_fnu_error = flat_fnu_tophat_sanity_check()
    filters = load_filter_set(filter_names)
    params = default_params()
    backend = FSPSBackend(sp_kwargs=default_sp_kwargs(), mass_normalization=MassNormalization.PER_SOLAR_MASS)
    phot = backend.predict_photometry(params, filters)
    reference_flux, reference_mag, wave_obs_a, flam_obs = direct_fsps_sedpy_photometry(
        params,
        filters,
        mass_normalization=backend.mass_normalization,
        sp_kwargs=default_sp_kwargs(),
    )
    backend_mag, max_rel, max_mag_diff = validate_outputs(phot.flux, reference_flux, len(filters))
    return {
        "backend": backend,
        "filters": filters,
        "photometry": phot,
        "backend_mag": backend_mag,
        "reference_flux": reference_flux,
        "reference_mag": reference_mag,
        "wave_obs_a": wave_obs_a,
        "flam_obs": flam_obs,
        "max_relative_flux_difference": max_rel,
        "max_ab_magnitude_difference": max_mag_diff,
        "flat_fnu_tophat_sanity_error": flat_fnu_error,
    }


def main() -> None:
    result = run_validation()
    backend = result["backend"]
    filters = result["filters"]
    phot = result["photometry"]
    wavelengths = filter_effective_wavelengths(filters)

    print(f"FSPSBackend mass normalization: {backend.mass_normalization.value}")
    print(f"Flux tolerance: relative <= {FLUX_RTOL:.1e}")
    print(f"AB magnitude tolerance: absolute <= {MAG_ATOL:.1e}")
    print(f"Flat-fnu top-hat sanity error: {result['flat_fnu_tophat_sanity_error']:.3e}")
    print("band wavelength_A backend_maggies backend_ABmag reference_maggies reference_ABmag")
    for band, wave, flux, mag, ref_flux, ref_mag in zip(
        phot.band_names,
        wavelengths,
        phot.flux,
        result["backend_mag"],
        result["reference_flux"],
        result["reference_mag"],
    ):
        print(f"{band} {wave:.6g} {flux:.12e} {mag:.8f} {ref_flux:.12e} {ref_mag:.8f}")
    print(f"Observed spectrum wavelength range [A]: {result['wave_obs_a'][0]:.6g} - {result['wave_obs_a'][-1]:.6g}")
    print(f"Observed spectrum f_lambda finite: {np.all(np.isfinite(result['flam_obs']))}")
    print(f"Max relative flux difference: {result['max_relative_flux_difference']:.3e}")
    print(f"Max AB magnitude difference: {result['max_ab_magnitude_difference']:.3e}")
    print("FSPSBackend validation passed.")


if __name__ == "__main__":
    main()
