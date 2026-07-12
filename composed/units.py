from __future__ import annotations

from enum import Enum

import numpy as np


class MassNormalization(str, Enum):
    PER_SOLAR_MASS = "per_solar_mass"
    ABSOLUTE = "absolute"


LSUN_CGS = 3.828e33
PARSEC_CM = 3.085677581491367e18


_PHOTOMETRIC_UNIT_ALIASES = {
    "maggie": "maggies",
    "maggies": "maggies",
    "jy": "jy",
    "jansky": "jy",
    "janskys": "jy",
    "mjy": "mjy",
    "millijy": "mjy",
    "millijansky": "mjy",
    "ujy": "ujy",
    "microjy": "ujy",
    "microjansky": "ujy",
}

_JY_PER_PHOTOMETRIC_UNIT = {
    "maggies": 3631.0,
    "jy": 1.0,
    "mjy": 1.0e-3,
    "ujy": 1.0e-6,
}


def canonical_photometric_flux_unit(unit: str) -> str:
    """Return a canonical linear photometric flux unit or raise clearly."""

    key = str(unit).strip().lower().replace(" ", "")
    try:
        return _PHOTOMETRIC_UNIT_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(_JY_PER_PHOTOMETRIC_UNIT)
        raise ValueError(f"Unsupported photometric flux unit {unit!r}; expected one of {allowed}.") from exc


def convert_photometric_flux(values, from_unit: str, to_unit: str):
    """Convert linear fluxes among maggies, Jy, mJy, and microJy."""

    source = canonical_photometric_flux_unit(from_unit)
    target = canonical_photometric_flux_unit(to_unit)
    values = np.asarray(values, dtype=float)
    return values * (_JY_PER_PHOTOMETRIC_UNIT[source] / _JY_PER_PHOTOMETRIC_UNIT[target])


def canonical_wavelength_unit(unit: str) -> str:
    key = str(unit).strip().lower().replace("å", "angstrom")
    aliases = {"a": "angstrom", "aa": "angstrom", "angstrom": "angstrom", "nm": "nm"}
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported wavelength unit {unit!r}; expected Angstrom or nm.") from exc


def canonical_spectral_flux_unit(unit: str) -> str:
    key = str(unit).strip().lower().replace(" ", "")
    aliases = {
        "erg/s/cm^2/angstrom": "erg/s/cm^2/angstrom",
        "ergs^-1cm^-2angstrom^-1": "erg/s/cm^2/angstrom",
        "w/nm": "w/nm",
        "wnm^-1": "w/nm",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported spectral flux unit {unit!r}; expected erg/s/cm^2/angstrom or W/nm."
        ) from exc
