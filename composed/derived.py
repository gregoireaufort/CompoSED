"""Derived physical quantities shared by backend tutorials and analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from composed._numerics import trapezoid
from composed.sfh import SFHModel
from composed.units import MassNormalization, backend_mass_reference


@dataclass(frozen=True)
class DerivedSFHQuantities:
    """Absolute SFH and scalar quantities implied by one parameter vector."""

    time_gyr: np.ndarray
    sfr_msun_per_yr: np.ndarray
    current_sfr_msun_per_yr: float
    log10_sfr: float
    log10_ssfr_per_yr: float
    surviving_stellar_mass_msun: float
    formed_mass_msun: float
    surviving_mass_fraction: float
    mass_weighted_age_gyr: float


def derive_sfh_quantities(
    backend,
    params: Mapping[str, float],
    filters,
    *,
    sfh: SFHModel | None = None,
) -> DerivedSFHQuantities:
    """Derive the absolute SFH implied by surviving stellar mass.

    The backend is evaluated once to obtain its parameter-dependent surviving
    stellar-mass fraction. The named CompoSED SFH supplies a history normalized
    by formed mass. Combining those two pieces converts the fitted
    ``log10_mass`` into an absolute SFR in solar masses per year.
    """

    if MassNormalization(getattr(backend, "mass_normalization", None)) != MassNormalization.PER_SOLAR_MASS:
        raise ValueError("derive_sfh_quantities requires a PER_SOLAR_MASS backend.")
    backend_mass_reference(backend)
    if "log10_mass" not in params:
        raise KeyError("derive_sfh_quantities requires fitted 'log10_mass'.")

    sfh_model = sfh if sfh is not None else getattr(backend, "sfh", None)
    if not isinstance(sfh_model, SFHModel):
        raise TypeError("derive_sfh_quantities requires a named CompoSED SFHModel.")

    backend_params = dict(params)
    log10_mass = float(backend_params.pop("log10_mass"))
    surviving_mass = 10.0**log10_mass
    model = backend.predict_photometry(backend_params, filters)
    fraction = float(model.metadata.get("surviving_stellar_mass_fraction", np.nan))
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("Backend metadata lacks a physical surviving_stellar_mass_fraction.")

    redshift = _redshift_from_params(backend_params, getattr(backend, "default_z_key", None))
    history = sfh_model.evaluate(
        backend_params,
        redshift=redshift,
        cosmology=getattr(backend, "cosmology", None),
    )
    formed_mass = surviving_mass / fraction
    history_scale = formed_mass / history.formed_mass_msun
    absolute_sfr = history.sfr_msun_per_yr * history_scale
    current_sfr = float(absolute_sfr[-1])
    if not np.isfinite(current_sfr) or current_sfr <= 0.0:
        raise FloatingPointError("Derived current SFR is non-positive or non-finite.")

    lookback_gyr = history.time_gyr[-1] - history.time_gyr
    formed_from_history = _trapz(absolute_sfr, history.time_gyr)
    mass_weighted_age = _trapz(absolute_sfr * lookback_gyr, history.time_gyr) / formed_from_history
    return DerivedSFHQuantities(
        time_gyr=history.time_gyr,
        sfr_msun_per_yr=absolute_sfr,
        current_sfr_msun_per_yr=current_sfr,
        log10_sfr=float(np.log10(current_sfr)),
        log10_ssfr_per_yr=float(np.log10(current_sfr / surviving_mass)),
        surviving_stellar_mass_msun=float(surviving_mass),
        formed_mass_msun=float(formed_mass),
        surviving_mass_fraction=float(fraction),
        mass_weighted_age_gyr=float(mass_weighted_age),
    )


def _redshift_from_params(params: Mapping[str, float], preferred: str | None) -> float | None:
    keys = tuple(key for key in (preferred, "z", "zred", "redshift") if key is not None)
    for key in dict.fromkeys(keys):
        if key in params:
            return float(params[key])
    return None


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, grid))
    return float(trapezoid(values, grid))
