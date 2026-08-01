"""In-memory tabular SFHs for CIGALE v2022.0.

This private bridge lets CompoSED pass the same canonical ``SFHHistory`` to
CIGALE that it passes to FSPS.  CIGALE expects one SFR value per 1 Myr age bin,
ordered from the onset of star formation to the observation time.  The bridge
therefore has two explicit steps:

1. integrate the canonical, piecewise-linear SFH into 1 Myr bins;
2. register that finite array only while ``SedWarehouse.get_sed`` is running.

The CIGALE module receives a content hash, rather than a NumPy array, because
CIGALE uses module parameter values as cache keys and arrays are not hashable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.machinery
from pathlib import Path
import sys
from threading import RLock
import types
from typing import Iterator

import numpy as np

from composed.errors import ModelDomainError
from composed.sfh import SFHHistory


CIGALE_TABULAR_MODULE = "sfhcomposed_tabular"
CIGALE_TABULAR_IMPORT = f"pcigale.sed_modules.{CIGALE_TABULAR_MODULE}"
CIGALE_TABULAR_SCHEMA = "composed.cigale-tabular-sfh.v1"


@dataclass(frozen=True)
class CigaleTabularSFH:
    """A unit-formed-mass SFH sampled in CIGALE's native 1 Myr bins."""

    sfr_msun_per_yr: np.ndarray
    source_age_gyr: float
    cigale_age_myr: int
    source_formed_mass_msun: float
    time_scale: float

    def metadata(self) -> dict[str, object]:
        return {
            "execution": "composed_tabular_1myr",
            "module": CIGALE_TABULAR_MODULE,
            "schema": CIGALE_TABULAR_SCHEMA,
            "source_age_gyr": float(self.source_age_gyr),
            "cigale_age_myr": int(self.cigale_age_myr),
            "source_formed_mass_msun": float(self.source_formed_mass_msun),
            "projection_time_scale": float(self.time_scale),
            "formed_mass_msun": float(np.sum(self.sfr_msun_per_yr) * 1.0e6),
        }


@dataclass
class _RegistryEntry:
    sfr_msun_per_yr: np.ndarray
    references: int = 1


_REGISTRY: dict[str, _RegistryEntry] = {}
_REGISTRY_LOCK = RLock()


def project_sfh_to_cigale_1myr(history: SFHHistory) -> CigaleTabularSFH:
    """Project one canonical SFH onto CIGALE's chronological 1 Myr grid.

    CIGALE represents a galaxy age as an integer number of 1 Myr SFR bins.
    The source age is rounded to the nearest Myr, matching CIGALE's native
    integer-age convention.  The source time axis is rescaled by at most half
    a Myr at its endpoint, then the piecewise-linear SFR is integrated exactly
    over every CIGALE bin.  Finally, the bins are normalized to one solar mass
    formed, which is the mass contract used by ``CIGALEBackend``.
    """

    time_gyr = np.asarray(history.time_gyr, dtype=np.float64)
    sfr = np.asarray(history.sfr_msun_per_yr, dtype=np.float64)
    if not np.isclose(time_gyr[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ModelDomainError(
            "CIGALE tabular SFHs must start at time 0 Gyr (star-formation onset)."
        )

    source_age_gyr = float(time_gyr[-1])
    cigale_age_myr = int(np.rint(1000.0 * source_age_gyr))
    if cigale_age_myr < 2:
        raise ModelDomainError("CIGALE tabular SFHs require a galaxy age of at least 2 Myr.")

    projected_age_gyr = cigale_age_myr / 1000.0
    time_scale = projected_age_gyr / source_age_gyr
    projected_time_gyr = time_gyr * time_scale
    bin_edges_gyr = np.arange(cigale_age_myr + 1, dtype=np.float64) / 1000.0
    cumulative_mass_msun = _piecewise_linear_cumulative_mass(
        projected_time_gyr,
        sfr,
        bin_edges_gyr,
    )
    bin_mass_msun = np.diff(cumulative_mass_msun)
    if np.any(bin_mass_msun < -1.0e-14 * np.max(np.abs(cumulative_mass_msun))):
        raise FloatingPointError("CIGALE SFH projection produced negative bin masses.")
    bin_mass_msun = np.maximum(bin_mass_msun, 0.0)

    formed_mass = float(np.sum(bin_mass_msun))
    if not np.isfinite(formed_mass) or formed_mass <= 0.0:
        raise ModelDomainError("CIGALE SFH projection produced no finite formed stellar mass.")
    sfr_1myr = np.ascontiguousarray(bin_mass_msun / (formed_mass * 1.0e6), dtype=np.float64)
    normalized_mass = float(np.sum(sfr_1myr) * 1.0e6)
    if not np.isclose(normalized_mass, 1.0, rtol=1.0e-13, atol=1.0e-13):
        raise FloatingPointError("CIGALE SFH projection did not normalize to one solar mass formed.")

    sfr_1myr.setflags(write=False)
    return CigaleTabularSFH(
        sfr_msun_per_yr=sfr_1myr,
        source_age_gyr=source_age_gyr,
        cigale_age_myr=cigale_age_myr,
        source_formed_mass_msun=float(history.formed_mass_msun),
        time_scale=time_scale,
    )


def _piecewise_linear_cumulative_mass(
    time_gyr: np.ndarray,
    sfr_msun_per_yr: np.ndarray,
    query_gyr: np.ndarray,
) -> np.ndarray:
    """Integrate a piecewise-linear SFR exactly at requested times."""

    dt_gyr = np.diff(time_gyr)
    slopes = np.diff(sfr_msun_per_yr) / dt_gyr
    interval_mass = 1.0e9 * 0.5 * (sfr_msun_per_yr[:-1] + sfr_msun_per_yr[1:]) * dt_gyr
    cumulative_nodes = np.concatenate([[0.0], np.cumsum(interval_mass)])

    indices = np.searchsorted(time_gyr, query_gyr, side="right") - 1
    indices = np.clip(indices, 0, time_gyr.size - 2)
    local_dt = query_gyr - time_gyr[indices]
    cumulative = cumulative_nodes[indices] + 1.0e9 * (
        sfr_msun_per_yr[indices] * local_dt + 0.5 * slopes[indices] * local_dt**2
    )
    cumulative = np.where(query_gyr <= time_gyr[0], 0.0, cumulative)
    cumulative = np.where(query_gyr >= time_gyr[-1], cumulative_nodes[-1], cumulative)
    return cumulative


def sfh_content_hash(sfr_msun_per_yr: np.ndarray) -> str:
    """Return the stable cache key used as the CIGALE module parameter."""

    values = np.ascontiguousarray(sfr_msun_per_yr, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(CIGALE_TABULAR_SCHEMA.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


@contextmanager
def registered_cigale_sfh(sfr_msun_per_yr: np.ndarray) -> Iterator[str]:
    """Register one projected SFH for the duration of a warehouse call."""

    values = np.ascontiguousarray(sfr_msun_per_yr, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Registered CIGALE SFHs must be one-dimensional with at least two bins.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ModelDomainError("Registered CIGALE SFHs must be finite and non-negative.")
    formed_mass = float(np.sum(values) * 1.0e6)
    if not np.isfinite(formed_mass) or formed_mass <= 0.0:
        raise ModelDomainError("Registered CIGALE SFHs must form a positive finite mass.")
    history_hash = sfh_content_hash(values)
    stored = values.copy()
    stored.setflags(write=False)

    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(history_hash)
        if entry is None:
            _REGISTRY[history_hash] = _RegistryEntry(stored)
        else:
            if not np.array_equal(entry.sfr_msun_per_yr, stored):  # pragma: no cover - SHA collision guard.
                raise RuntimeError("CIGALE SFH content-hash collision detected.")
            entry.references += 1
    try:
        yield history_hash
    finally:
        with _REGISTRY_LOCK:
            entry = _REGISTRY.get(history_hash)
            if entry is not None:
                entry.references -= 1
                if entry.references == 0:
                    del _REGISTRY[history_hash]


def registry_size() -> int:
    """Return the number of live arrays; exposed only for bounded-memory tests."""

    with _REGISTRY_LOCK:
        return len(_REGISTRY)


def _registered_sfh(history_hash: str) -> np.ndarray:
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(str(history_hash))
        if entry is None:
            raise RuntimeError(
                "CIGALE requested a CompoSED tabular SFH outside its registration context."
            )
        return entry.sfr_msun_per_yr.copy()


def register_cigale_tabular_module() -> None:
    """Register the CompoSED SFH module in CIGALE's import namespace."""

    from pcigale.sed_modules import SedModule

    existing = sys.modules.get(CIGALE_TABULAR_IMPORT)
    if existing is not None:
        if getattr(existing, "COMPOSED_SCHEMA", None) != CIGALE_TABULAR_SCHEMA:
            raise RuntimeError(
                f"CIGALE module name {CIGALE_TABULAR_IMPORT!r} is already registered by other code."
            )
        existing_class = getattr(existing, "Module", None)
        if isinstance(existing_class, type) and issubclass(existing_class, SedModule):
            return
        # A test, notebook reload, or alternate CIGALE environment in the same
        # interpreter may have replaced pcigale.sed_modules. Rebuild the alias
        # against the currently imported SedModule base rather than retaining
        # a class tied to stale process-local state.
        del sys.modules[CIGALE_TABULAR_IMPORT]

    class ComposedTabularSFHModule(SedModule):
        parameter_list = {
            "history_hash": (
                "string()",
                "Content hash of a process-local CompoSED 1 Myr SFH.",
                None,
            ),
            "normalise": (
                "boolean()",
                "Normalize to one solar mass formed; always true in CompoSED.",
                True,
            ),
        }

        def _init_code(self):
            normalise = self.parameters["normalise"]
            if isinstance(normalise, str):
                normalise = normalise.strip().lower() == "true"
            if not bool(normalise):
                raise ValueError("CompoSED's CIGALE tabular SFH requires normalise=True.")
            self.sfr = _registered_sfh(self.parameters["history_hash"])
            formed_mass = float(np.sum(self.sfr) * 1.0e6)
            if not np.isclose(formed_mass, 1.0, rtol=1.0e-12, atol=1.0e-12):
                self.sfr /= formed_mass
            self.sfr_integrated = 1.0

        def process(self, sed):
            sed.add_module(self.name, self.parameters)
            sed.sfh = self.sfr
            sed.add_info("sfh.integrated", self.sfr_integrated, True, unit="solMass")

    ComposedTabularSFHModule.__name__ = "ComposedTabularSFH"
    ComposedTabularSFHModule.__qualname__ = "ComposedTabularSFH"
    ComposedTabularSFHModule.__module__ = CIGALE_TABULAR_IMPORT

    module = types.ModuleType(CIGALE_TABULAR_IMPORT)
    module.__file__ = __file__
    module.__package__ = "pcigale.sed_modules"
    module.__spec__ = importlib.machinery.ModuleSpec(
        CIGALE_TABULAR_IMPORT,
        loader=None,
        origin=__file__,
    )
    module.COMPOSED_SCHEMA = CIGALE_TABULAR_SCHEMA
    module.Module = ComposedTabularSFHModule
    sys.modules[CIGALE_TABULAR_IMPORT] = module


def cigale_tabular_bridge_specification() -> dict[str, object]:
    """Return provenance for the CompoSED-owned CIGALE module source."""

    source_path = Path(__file__).resolve()
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError:
        source_sha256 = None
    return {
        "module": CIGALE_TABULAR_MODULE,
        "schema": CIGALE_TABULAR_SCHEMA,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "time_grid": "chronological 1 Myr bins from star-formation onset",
        "normalization": "sum(SFR [Msun/yr]) * 1e6 yr = 1 Msun formed",
    }


__all__ = [
    "CIGALE_TABULAR_MODULE",
    "CIGALE_TABULAR_SCHEMA",
    "CigaleTabularSFH",
    "cigale_tabular_bridge_specification",
    "project_sfh_to_cigale_1myr",
    "register_cigale_tabular_module",
    "registered_cigale_sfh",
]
