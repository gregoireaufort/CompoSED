"""Conventions for comparing CIGALE BC03 and FSPS stellar modules.

These helpers keep the intentionally approximate BC03/FSPS matching rules in
one place.  They do not claim that the underlying SPS models are equivalent;
they only make the IMF and metallicity conventions explicit enough that a
comparison cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


FSPS_SOLAR_METALLICITY = 0.02
"""Solar metallicity convention used when mapping absolute ``Z`` to FSPS logZ.

CIGALE's BC03 grid exposes absolute metallicities and historically includes
``Z=0.02`` as the solar template.  FSPS accepts ``logzsol``.  We therefore use
``Z_sun=0.02`` by default for BC03-matched diagnostics, while keeping it an
explicit function argument so future runs can choose another convention.
"""


BC03_METALLICITY_GRID = (0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05)
"""CIGALE v2022 BC03 stellar metallicity grid."""


CIGALE_NEBULAR_ZGAS_GRID = (
    0.0001,
    0.0004,
    0.001,
    0.002,
    0.0025,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.011,
    0.012,
    0.014,
    0.016,
    0.019,
    0.022,
    0.025,
    0.03,
    0.033,
    0.037,
    0.041,
    0.046,
    0.051,
)
"""CIGALE v2022 nebular gas metallicity grid."""


BC03_IMF_TO_FSPS_IMF_TYPE = {
    0: 0,  # Salpeter
    1: 1,  # Chabrier
}
"""Subset of CIGALE BC03 IMF choices with direct FSPS counterparts."""


FSPS_IMF_LABELS = {
    0: "salpeter",
    1: "chabrier",
    2: "kroupa",
    3: "van_dokkum",
}


@dataclass(frozen=True)
class FSPSStellarParameterMapping:
    """Matched FSPS parameters corresponding to a CIGALE BC03 convention."""

    imf_type: int
    logzsol: float
    z_sun: float
    metallicity: float
    zgas: float


def cigale_bc03_imf_to_fsps_imf_type(imf: int) -> int:
    """Return the FSPS ``imf_type`` matching a CIGALE BC03 IMF id.

    CIGALE BC03 exposes only Salpeter and Chabrier in the installed v2022 grid.
    Any other value is rejected because silently mapping it would make BC03/FSPS
    comparisons scientifically ambiguous.
    """

    imf = int(imf)
    try:
        return BC03_IMF_TO_FSPS_IMF_TYPE[imf]
    except KeyError as exc:
        raise ValueError(f"CIGALE BC03 IMF {imf!r} has no explicit FSPS mapping.") from exc


def fsps_imf_type_to_label(imf_type: int) -> str:
    """Return a short label for a supported FSPS IMF type."""

    imf_type = int(imf_type)
    try:
        return FSPS_IMF_LABELS[imf_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported FSPS imf_type {imf_type!r}.") from exc


def fsps_imf_type_to_cigale_bc03_imf(imf_type: int) -> int:
    """Return the CIGALE BC03 IMF id for FSPS IMF types with a direct match."""

    imf_type = int(imf_type)
    for cigale_imf, mapped_imf_type in BC03_IMF_TO_FSPS_IMF_TYPE.items():
        if mapped_imf_type == imf_type:
            return cigale_imf
    raise ValueError(f"FSPS imf_type {imf_type!r} has no direct CIGALE BC03 IMF mapping.")


def cigale_metallicity_to_fsps_logzsol(
    metallicity: float,
    *,
    z_sun: float = FSPS_SOLAR_METALLICITY,
) -> float:
    """Convert absolute CIGALE metallicity ``Z`` to FSPS ``logzsol``."""

    metallicity = _positive_finite_float(metallicity, "metallicity")
    z_sun = _positive_finite_float(z_sun, "z_sun")
    return math.log10(metallicity / z_sun)


def fsps_logzsol_to_cigale_metallicity(
    logzsol: float,
    *,
    z_sun: float = FSPS_SOLAR_METALLICITY,
) -> float:
    """Convert FSPS ``logzsol`` to an absolute CIGALE-style metallicity."""

    logzsol = _finite_float(logzsol, "logzsol")
    z_sun = _positive_finite_float(z_sun, "z_sun")
    return z_sun * (10.0**logzsol)


def nearest_cigale_nebular_zgas(metallicity: float) -> float:
    """Return the nearest CIGALE nebular gas metallicity to absolute ``Z``."""

    metallicity = _positive_finite_float(metallicity, "metallicity")
    return min(CIGALE_NEBULAR_ZGAS_GRID, key=lambda value: abs(value - metallicity))


def fsps_parameters_from_cigale_bc03(
    *,
    imf: int,
    metallicity: float,
    z_sun: float = FSPS_SOLAR_METALLICITY,
) -> FSPSStellarParameterMapping:
    """Build matched FSPS stellar parameters from CIGALE BC03 parameters."""

    metallicity = _positive_finite_float(metallicity, "metallicity")
    return FSPSStellarParameterMapping(
        imf_type=cigale_bc03_imf_to_fsps_imf_type(imf),
        logzsol=cigale_metallicity_to_fsps_logzsol(metallicity, z_sun=z_sun),
        z_sun=_positive_finite_float(z_sun, "z_sun"),
        metallicity=metallicity,
        zgas=nearest_cigale_nebular_zgas(metallicity),
    )


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _positive_finite_float(value: float, name: str) -> float:
    value = _finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


__all__ = [
    "BC03_IMF_TO_FSPS_IMF_TYPE",
    "BC03_METALLICITY_GRID",
    "CIGALE_NEBULAR_ZGAS_GRID",
    "FSPSStellarParameterMapping",
    "FSPS_IMF_LABELS",
    "FSPS_SOLAR_METALLICITY",
    "cigale_bc03_imf_to_fsps_imf_type",
    "cigale_metallicity_to_fsps_logzsol",
    "fsps_imf_type_to_cigale_bc03_imf",
    "fsps_imf_type_to_label",
    "fsps_logzsol_to_cigale_metallicity",
    "fsps_parameters_from_cigale_bc03",
    "nearest_cigale_nebular_zgas",
]
