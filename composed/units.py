from __future__ import annotations

from enum import Enum

import numpy as np


class MassNormalization(str, Enum):
    """Whether backend outputs still require an explicit mass amplitude.

    ``PER_SOLAR_MASS`` means per one solar mass of present-day surviving
    stellar mass. Backends may use formed mass internally, but must convert
    their output to this reference before returning it.
    """

    PER_SOLAR_MASS = "per_solar_mass"
    ABSOLUTE = "absolute"


class MassReference(str, Enum):
    """Physical mass represented by a per-solar-mass model normalization."""

    SURVIVING_STELLAR_MASS = "surviving_stellar_mass"
    FORMED_MASS = "formed_mass"


MASS_CONVENTION_SCHEMA = "composed.mass.surviving_stellar.v1"


def validate_mass_reference(
    mass_normalization: MassNormalization | str,
    mass_reference: MassReference | str | None,
) -> MassReference | None:
    """Validate the public mass-amplitude convention.

    Absolute model outputs have no separate mass reference. Per-mass outputs
    must be normalized by surviving stellar mass; formed-mass outputs are
    deliberately rejected so ``log10_mass`` cannot silently change meaning.
    """

    normalization = MassNormalization(mass_normalization)
    if normalization == MassNormalization.ABSOLUTE:
        return None
    if mass_reference is None:
        raise ValueError(
            "A PER_SOLAR_MASS backend or model grid must declare mass_reference="
            "MassReference.SURVIVING_STELLAR_MASS."
        )
    reference = MassReference(mass_reference)
    if reference != MassReference.SURVIVING_STELLAR_MASS:
        raise ValueError(
            "CompoSED log10_mass denotes present-day surviving stellar mass, but "
            f"the model declares mass_reference={reference.value!r}."
        )
    return reference


def backend_mass_reference(backend: object) -> MassReference | None:
    """Return and validate the mass reference declared by a backend."""

    normalization = getattr(backend, "mass_normalization", None)
    if normalization is None:
        raise ValueError("Backend must declare mass_normalization.")
    return validate_mass_reference(normalization, getattr(backend, "mass_reference", None))


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
