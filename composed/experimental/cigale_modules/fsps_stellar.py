"""Experimental FSPS stellar-emission module for CIGALE.

This module is intentionally separate from the stable ``composed`` backends.
It is loaded by calling
``composed.experimental.cigale_fsps_stellar.register_cigale_fsps_stellar_module``
before using a CIGALE module list containing ``"fsps_stellar"``.

The module replaces CIGALE's ``bc03``/``m2005`` stellar module. It consumes the
SFH already stored on the CIGALE ``SED`` object, evaluates a stellar-only FSPS
continuum, and then populates the CIGALE stellar contributions/metadata needed
by downstream CIGALE modules such as ``nebular`` and dust attenuation modules.

This is a research prototype. In particular, stellar mass bookkeeping is
approximated from FSPS attributes when available, and should be validated before
production use.
"""

from __future__ import annotations

import numpy as np

from pcigale.sed_modules import SedModule

from composed.experimental.cigale_fsps_stellar_conventions import (
    FSPS_SOLAR_METALLICITY,
    fsps_imf_type_to_cigale_bc03_imf,
    fsps_imf_type_to_label,
    fsps_logzsol_to_cigale_metallicity,
)

__category__ = "SSP"

LSUN_W = 3.828e26
H_J_S = 6.62607015e-34
C_M_S = 2.99792458e8


class FSPSStellar(SedModule):
    """CIGALE stellar module powered by python-fsps."""

    parameter_list = {
        "imf_type": (
            "cigale_list(dtype=int, options=0 & 1 & 2 & 3)",
            "FSPS IMF type. Common values are 0=Salpeter, 1=Chabrier, 2=Kroupa.",
            1,
        ),
        "logzsol": (
            "cigale_list()",
            "Log10 stellar metallicity relative to solar passed to FSPS.",
            0.0,
        ),
        "z_sun": (
            "cigale_list(minvalue=0.)",
            "Solar metallicity convention used to report absolute CIGALE-style Z.",
            FSPS_SOLAR_METALLICITY,
        ),
        "zcontinuous": (
            "cigale_list(dtype=int, options=0 & 1 & 2)",
            "FSPS metallicity interpolation mode.",
            1,
        ),
        "separation_age": (
            "cigale_list(dtype=int, minvalue=0)",
            "Age in Myr separating young and old stellar populations.",
            10,
        ),
    }
    parameters = parameter_list

    def _init_code(self):
        self.imf_type = int(self.parameters["imf_type"])
        self.logzsol = float(self.parameters["logzsol"])
        self.z_sun = float(self.parameters.get("z_sun", FSPS_SOLAR_METALLICITY))
        self.zcontinuous = int(self.parameters["zcontinuous"])
        self.separation_age = int(self.parameters["separation_age"])
        self.imf_label = fsps_imf_type_to_label(self.imf_type)
        self.equivalent_metallicity = fsps_logzsol_to_cigale_metallicity(self.logzsol, z_sun=self.z_sun)
        self.bc03_imf = _optional_bc03_imf(self.imf_type)
        if not np.isfinite(self.logzsol):
            raise ValueError("logzsol must be finite.")
        if not np.isfinite(self.z_sun) or self.z_sun <= 0.0:
            raise ValueError("z_sun must be finite and positive.")
        if self.zcontinuous not in {0, 1, 2}:
            raise ValueError("zcontinuous must be one of 0, 1, or 2.")
        if self.separation_age < 0:
            raise ValueError("separation_age must be non-negative.")

        try:
            import fsps
        except ImportError as exc:
            raise ImportError("The experimental fsps_stellar CIGALE module requires python-fsps.") from exc

        self._sp = fsps.StellarPopulation(
            zcontinuous=self.zcontinuous,
            sfh=3,
            imf_type=self.imf_type,
            add_neb_emission=False,
            add_dust_emission=False,
            compute_vega_mags=False,
        )
        self._sp.params["logzsol"] = self.logzsol

    def process(self, sed):
        """Add FSPS stellar old/young contributions to the CIGALE SED."""

        if sed.sfh is None:
            raise ValueError("fsps_stellar must be called after a CIGALE SFH module.")
        sfh = _validate_sfh(sed.sfh)

        n_young = min(self.separation_age, sfh.size)
        sfh_young = np.zeros_like(sfh)
        if n_young > 0:
            sfh_young[-n_young:] = sfh[-n_young:]
        sfh_old = sfh - sfh_young

        wave_nm, spec_young, info_young = self._evaluate_component(sfh_young)
        wave_nm_old, spec_old, info_old = self._evaluate_component(sfh_old)
        if wave_nm_old.shape != wave_nm.shape or np.any(wave_nm_old != wave_nm):
            spec_old = np.interp(wave_nm, wave_nm_old, spec_old, left=0.0, right=0.0)

        info_all = {
            "m_star": info_young["m_star"] + info_old["m_star"],
            "m_gas": info_young["m_gas"] + info_old["m_gas"],
            "n_ly": info_young["n_ly"] + info_old["n_ly"],
            "age_mass": _formed_mass_weighted_age_myr(sfh),
        }
        lum_ly_young = _integrated_luminosity(spec_young, wave_nm, max_wave_nm=91.1)
        lum_ly_old = _integrated_luminosity(spec_old, wave_nm, max_wave_nm=91.1)
        lum_young = _integrated_luminosity(spec_young, wave_nm)
        lum_old = _integrated_luminosity(spec_old, wave_nm)

        sed.add_module(self.name, self.parameters)
        sed.add_info("stellar.imf", self.imf_type)
        sed.add_info("stellar.imf_label", self.imf_label)
        if self.bc03_imf is not None:
            sed.add_info("stellar.bc03_equivalent_imf", self.bc03_imf)
        sed.add_info("stellar.metallicity", self.equivalent_metallicity)
        sed.add_info("stellar.logzsol", self.logzsol)
        sed.add_info("stellar.fsps.imf_type", self.imf_type)
        sed.add_info("stellar.fsps.imf_label", self.imf_label)
        sed.add_info("stellar.fsps.logzsol", self.logzsol)
        sed.add_info("stellar.fsps.z_sun", self.z_sun)
        sed.add_info("stellar.fsps.equivalent_metallicity", self.equivalent_metallicity)
        sed.add_info("stellar.fsps.zcontinuous", self.zcontinuous)
        sed.add_info("stellar.old_young_separation_age", self.separation_age, unit="Myr")

        sed.add_info("stellar.m_star_young", info_young["m_star"], True, unit="solMass")
        sed.add_info("stellar.m_gas_young", info_young["m_gas"], True, unit="solMass")
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True, unit="ph/s")
        sed.add_info("stellar.lum_ly_young", lum_ly_young, True, unit="W")
        sed.add_info("stellar.lum_young", lum_young, True, unit="W")

        sed.add_info("stellar.m_star_old", info_old["m_star"], True, unit="solMass")
        sed.add_info("stellar.m_gas_old", info_old["m_gas"], True, unit="solMass")
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True, unit="ph/s")
        sed.add_info("stellar.lum_ly_old", lum_ly_old, True, unit="W")
        sed.add_info("stellar.lum_old", lum_old, True, unit="W")

        sed.add_info("stellar.m_star", info_all["m_star"], True, unit="solMass")
        sed.add_info("stellar.m_gas", info_all["m_gas"], True, unit="solMass")
        sed.add_info("stellar.n_ly", info_all["n_ly"], True, unit="ph/s")
        sed.add_info("stellar.lum_ly", lum_ly_young + lum_ly_old, True, unit="W")
        sed.add_info("stellar.lum", lum_young + lum_old, True, unit="W")
        sed.add_info("stellar.age_m_star", info_all["age_mass"], unit="Myr")

        sed.add_contribution("stellar.old", wave_nm, spec_old)
        sed.add_contribution("stellar.young", wave_nm, spec_young)

    def _evaluate_component(self, sfh_msun_per_yr):
        sfh_msun_per_yr = _validate_sfh(sfh_msun_per_yr)
        formed_mass = _formed_mass_msun(sfh_msun_per_yr)
        if formed_mass == 0.0:
            wave_nm = np.asarray(self._sp.wavelengths, dtype=float) / 10.0
            _validate_wavelength_grid(wave_nm)
            info = {"m_star": 0.0, "m_gas": 0.0, "n_ly": 0.0}
            return wave_nm, np.zeros_like(wave_nm), info

        time_gyr = _cigale_sfh_time_grid_gyr(sfh_msun_per_yr.size)
        self._sp.set_tabular_sfh(time_gyr, np.asarray(sfh_msun_per_yr, dtype=float))
        wave_a, llam_lsun_per_a = self._sp.get_spectrum(tage=float(time_gyr[-1]), peraa=True)
        wave_nm = np.asarray(wave_a, dtype=float) / 10.0
        spec_w_per_nm = np.asarray(llam_lsun_per_a, dtype=float) * LSUN_W * 10.0
        if not np.all(np.isfinite(wave_nm)) or not np.all(np.isfinite(spec_w_per_nm)):
            raise FloatingPointError("FSPS returned non-finite stellar spectrum values.")
        _validate_wavelength_grid(wave_nm)

        m_star = _stellar_mass_from_fsps(self._sp, formed_mass)
        m_gas = max(formed_mass - m_star, 0.0)
        info = {
            "m_star": m_star,
            "m_gas": m_gas,
            "n_ly": _ionizing_photon_rate(wave_nm, spec_w_per_nm),
        }
        return wave_nm, spec_w_per_nm, info


def _optional_bc03_imf(imf_type):
    try:
        return fsps_imf_type_to_cigale_bc03_imf(imf_type)
    except ValueError:
        return None


def _validate_sfh(sfh_msun_per_yr):
    sfh = np.asarray(sfh_msun_per_yr, dtype=float)
    if sfh.ndim != 1 or sfh.size < 2:
        raise ValueError("CIGALE SFH must be a one-dimensional array with at least two bins.")
    if not np.all(np.isfinite(sfh)):
        raise ValueError("CIGALE SFH must contain finite SFR values.")
    if np.any(sfh < 0.0):
        raise ValueError("CIGALE SFH must contain non-negative SFR values.")
    return sfh


def _validate_wavelength_grid(wave_nm):
    if wave_nm.size < 2 or np.any(np.diff(wave_nm) <= 0.0):
        raise FloatingPointError("FSPS wavelength grid is not strictly increasing.")
    if not np.all(np.isfinite(wave_nm)):
        raise FloatingPointError("FSPS wavelength grid contains non-finite values.")


def _cigale_sfh_time_grid_gyr(n_bins):
    # CIGALE SFH arrays are 1 Myr bins from galaxy birth to observation.
    # FSPS tabular SFHs need a strictly increasing time grid in Gyr.
    return np.arange(1, int(n_bins) + 1, dtype=float) / 1000.0


def _formed_mass_msun(sfh_msun_per_yr):
    return float(np.sum(np.asarray(sfh_msun_per_yr, dtype=float)) * 1.0e6)


def _stellar_mass_from_fsps(sp, formed_mass):
    if not hasattr(sp, "stellar_mass"):
        return float(formed_mass)
    value = float(sp.stellar_mass)
    if not np.isfinite(value) or value < 0.0:
        raise FloatingPointError("FSPS returned an invalid stellar_mass value.")
    return value


def _integrated_luminosity(spec_w_per_nm, wave_nm, max_wave_nm=None):
    wave_nm = np.asarray(wave_nm, dtype=float)
    spec_w_per_nm = np.asarray(spec_w_per_nm, dtype=float)
    if max_wave_nm is not None:
        use = wave_nm <= float(max_wave_nm)
        if np.count_nonzero(use) < 2:
            return 0.0
        wave_nm = wave_nm[use]
        spec_w_per_nm = spec_w_per_nm[use]
    return float(np.trapz(spec_w_per_nm, wave_nm))


def _ionizing_photon_rate(wave_nm, spec_w_per_nm):
    wave_nm = np.asarray(wave_nm, dtype=float)
    spec_w_per_nm = np.asarray(spec_w_per_nm, dtype=float)
    use = wave_nm <= 91.1
    if np.count_nonzero(use) < 2:
        return 0.0
    wave_m = wave_nm[use] * 1.0e-9
    photons_per_s_per_nm = spec_w_per_nm[use] * wave_m / (H_J_S * C_M_S)
    return float(np.trapz(photons_per_s_per_nm, wave_nm[use]))


def _formed_mass_weighted_age_myr(sfh_msun_per_yr):
    sfh = np.asarray(sfh_msun_per_yr, dtype=float)
    weights = sfh[::-1]
    if not np.any(weights > 0.0):
        return 0.0
    ages_myr = np.arange(1, sfh.size + 1, dtype=float)
    return float(np.average(ages_myr, weights=weights))


Module = FSPSStellar
