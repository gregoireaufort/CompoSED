import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from composed._numerics import trapezoid
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


def _trapezoid(y, x):
    """Integrate with either NumPy 1.x or NumPy 2.x."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(trapezoid(y, x))


def _flat_fnu_tophat_error():
    """Return the maggie error for an analytic flat-f_nu top-hat case."""

    wave_a = np.linspace(4000.0, 5000.0, 4096)
    flat_flam = AB_ZERO_FNU_CGS * C_A_PER_S / wave_a**2
    numerical = _trapezoid(flat_flam * wave_a, wave_a)
    analytic = AB_ZERO_FNU_CGS * C_A_PER_S * np.log(5000.0 / 4000.0)
    return abs(numerical / analytic - 1.0)


def _load_filters():
    from sedpy.observate import load_filters

    filters = load_filters(list(DEFAULT_FILTER_NAMES))
    return FilterSet(filters, names=DEFAULT_FILTER_NAMES)


def _parameters():
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


def _stellar_population_kwargs():
    return {
        "zcontinuous": 1,
        "sfh": 3,
        "add_dust_emission": False,
        "add_neb_emission": True,
        "compute_vega_mags": False,
    }


def _direct_fsps_sedpy_photometry(params, filters):
    """Independent reference calculation that does not call FSPSBackend."""

    import fsps
    from astropy.cosmology import Planck18
    from sedpy.observate import getSED

    sp = fsps.StellarPopulation(**_stellar_population_kwargs())
    redshift = float(params["zred"])
    sp.params["zred"] = redshift
    for name in (
        "logzsol",
        "dust2",
        "dust1",
        "dust_index",
        "gas_logz",
        "gas_logu",
        "fagn",
        "agn_tau",
    ):
        if name in params:
            sp.params[name] = float(params[name])

    time_gyr = np.asarray(params["tabular_time_gyr"], dtype=float)
    sfr = normalize_sfh_to_formed_mass(
        time_gyr,
        np.asarray(params["tabular_sfr_msun_per_yr"], dtype=float),
    )
    sp.set_tabular_sfh(time_gyr, sfr)
    wave_rest_a, luminosity_lsun_per_a = sp.get_spectrum(
        tage=float(time_gyr[-1]),
        peraa=True,
    )

    formed_mass = float(sp.formed_mass)
    surviving_mass = float(sp.stellar_mass)
    assert np.isfinite(formed_mass) and formed_mass > 0.0
    assert np.isfinite(surviving_mass) and surviving_mass > 0.0
    luminosity_lsun_per_a = (
        np.asarray(luminosity_lsun_per_a, dtype=float) / surviving_mass
    )

    distance_cm = Planck18.luminosity_distance(redshift).to("cm").value
    if redshift <= 0.0:
        distance_cm = 10.0 * PARSEC_CM
    wave_observed_a = np.asarray(wave_rest_a, dtype=float) * (1.0 + redshift)
    flux_lambda = luminosity_lsun_per_a * LSUN_CGS / (
        4.0 * np.pi * distance_cm**2 * (1.0 + redshift)
    )
    magnitudes = np.asarray(
        getSED(
            wave_observed_a,
            flux_lambda,
            list(filters.filters),
            linear_flux=False,
        ),
        dtype=float,
    )
    return 10.0 ** (-0.4 * magnitudes), {
        "formed_mass_msun": formed_mass,
        "surviving_stellar_mass_msun": surviving_mass,
        "surviving_stellar_mass_fraction": surviving_mass / formed_mass,
    }


def test_flat_fnu_tophat_sanity_check_is_independent_of_fsps():
    assert _flat_fnu_tophat_error() <= FLAT_FNU_SANITY_RTOL


@pytest.mark.fsps
def test_validate_fsps_backend_against_direct_calculation():
    for package in ("fsps", "sedpy", "astropy"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{package} is not importable.")
    sps_home = os.environ.get("SPS_HOME")
    if not sps_home or not Path(sps_home).exists():
        pytest.skip("SPS_HOME is not configured or does not exist.")

    try:
        filters = _load_filters()
    except Exception as exc:
        pytest.skip(f"Requested sedpy filters could not be loaded: {exc}")

    params = _parameters()
    backend = FSPSBackend(
        sp_kwargs=_stellar_population_kwargs(),
        mass_normalization=MassNormalization.PER_SOLAR_MASS,
    )
    model = backend.predict_photometry(params, filters)
    reference_flux, reference_mass = _direct_fsps_sedpy_photometry(
        params,
        filters,
    )

    assert model.flux.shape == reference_flux.shape == (len(filters),)
    assert np.all(np.isfinite(model.flux))
    assert np.all(model.flux >= 0.0)
    relative_flux_error = np.abs(model.flux - reference_flux) / np.maximum(
        np.abs(reference_flux),
        np.finfo(float).tiny,
    )
    model_mag = -2.5 * np.log10(model.flux)
    reference_mag = -2.5 * np.log10(reference_flux)
    assert float(np.max(relative_flux_error)) <= FLUX_RTOL
    assert float(np.max(np.abs(model_mag - reference_mag))) <= MAG_ATOL

    for name, expected in reference_mass.items():
        assert float(model.metadata[name]) == pytest.approx(
            expected,
            rel=1e-12,
            abs=1e-12,
        )
    assert 0.0 < model.metadata["surviving_stellar_mass_fraction"] <= 1.0
    assert model.metadata["mass_reference"] == "surviving_stellar_mass"
