from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class SEDDataset:
    """Observed photometry for one SED.

    Parameters
    ----------
    band_names
        Names of the photometric bands. These names define the data order and
        are used by likelihoods to align model photometry.
    flux, sigma
        One-dimensional observed fluxes and Gaussian 1-sigma uncertainties in
        the same linear flux units. The package does not reinterpret units.
    mask
        Optional boolean array where ``True`` means the band is usable. Invalid
        or non-positive uncertainty values are also excluded from active arrays.
    upper_limit, upper_limit_mask
        Optional one-dimensional upper-limit fluxes and boolean flags. A band
        marked as an upper limit contributes a one-sided Gaussian CDF term to
        likelihoods instead of a residual. Its ``flux`` value may be non-finite
        because it is not used as a detection.
    metadata
        Extra caller-owned context, such as a ``FilterSet`` for backends.
    """

    band_names: Sequence[str]
    flux: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray | None = None
    upper_limit: np.ndarray | None = None
    upper_limit_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.band_names = tuple(str(name) for name in self.band_names)
        self.flux = np.asarray(self.flux, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.flux.shape != self.sigma.shape:
            raise ValueError("flux and sigma must have the same shape.")
        if self.flux.ndim != 1:
            raise ValueError("flux and sigma must be one-dimensional arrays.")
        if len(self.band_names) != self.flux.size:
            raise ValueError("band_names length must match flux length.")
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.flux.shape:
                raise ValueError("mask must have the same shape as flux.")
        if self.upper_limit_mask is None:
            self.upper_limit_mask = np.zeros(self.flux.shape, dtype=bool)
        else:
            self.upper_limit_mask = np.asarray(self.upper_limit_mask, dtype=bool)
            if self.upper_limit_mask.shape != self.flux.shape:
                raise ValueError("upper_limit_mask must have the same shape as flux.")
        if self.upper_limit is None:
            self.upper_limit = np.zeros(self.flux.shape, dtype=float)
        else:
            self.upper_limit = np.asarray(self.upper_limit, dtype=float)
            if self.upper_limit.shape != self.flux.shape:
                raise ValueError("upper_limit must have the same shape as flux.")
        if self.mask is not None and np.any(self.upper_limit_mask & ~self.mask):
            raise ValueError("Upper-limit bands must also be active in mask.")
        if not np.all(np.isfinite(self.upper_limit[self.upper_limit_mask])):
            raise ValueError("Active upper_limit values must be finite.")
        self.metadata = dict(self.metadata)
        if not np.any(self.active_mask):
            raise ValueError(
                "SEDDataset requires at least one active band: a finite detection or finite upper limit "
                "with strictly positive sigma."
            )

    @property
    def active_mask(self) -> np.ndarray:
        sigma_ok = np.isfinite(self.sigma) & (self.sigma > 0.0)
        detection_ok = (~self.upper_limit_mask) & np.isfinite(self.flux)
        upper_ok = self.upper_limit_mask & np.isfinite(self.upper_limit)
        finite = sigma_ok & (detection_ok | upper_ok)
        if self.mask is None:
            return finite
        return np.asarray(self.mask, dtype=bool) & finite

    @property
    def active_indices(self) -> np.ndarray:
        return np.where(self.active_mask)[0]

    @property
    def active_flux(self) -> np.ndarray:
        return self.flux[self.active_mask]

    @property
    def active_sigma(self) -> np.ndarray:
        return self.sigma[self.active_mask]

    @property
    def active_band_names(self) -> tuple[str, ...]:
        idx = self.active_indices
        return tuple(self.band_names[i] for i in idx)

    @property
    def active_upper_limit(self) -> np.ndarray:
        return self.upper_limit[self.active_indices]

    @property
    def active_upper_limit_mask(self) -> np.ndarray:
        return self.upper_limit_mask[self.active_indices]

    def active_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        idx = self.active_indices
        return self.flux[idx], self.sigma[idx], idx, tuple(self.band_names[i] for i in idx)


@dataclass
class SpectrumDataset:
    """Observed spectrum for one object.

    Parameters
    ----------
    wavelength
        One-dimensional observed-frame wavelength array. The default unit is
        Angstrom because FSPS and sedpy naturally operate there.
    flux, sigma
        Observed ``f_lambda`` and Gaussian 1-sigma uncertainty arrays in the
        same linear flux unit. The default unit is
        ``erg s^-1 cm^-2 Angstrom^-1``.
    mask
        Optional boolean array where ``True`` means the spectral pixel is used.
        Non-finite values and non-positive uncertainties are always excluded.
    metadata
        Extra caller-owned context, for example spectral resolution notes or
        mask provenance.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray | None = None
    wavelength_unit: str = "angstrom"
    flux_unit: str = "erg/s/cm^2/angstrom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.wavelength.ndim != 1 or self.flux.ndim != 1 or self.sigma.ndim != 1:
            raise ValueError("wavelength, flux, and sigma must be one-dimensional arrays.")
        if self.wavelength.shape != self.flux.shape or self.flux.shape != self.sigma.shape:
            raise ValueError("wavelength, flux, and sigma must have the same shape.")
        if self.wavelength.size < 2:
            raise ValueError("SpectrumDataset requires at least two spectral pixels.")
        if np.any(np.diff(self.wavelength) <= 0.0):
            raise ValueError("wavelength must be strictly increasing.")
        if not np.all(np.isfinite(self.wavelength)):
            raise ValueError("wavelength must be finite.")
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.flux.shape:
                raise ValueError("mask must have the same shape as flux.")
        self.wavelength_unit = str(self.wavelength_unit)
        self.flux_unit = str(self.flux_unit)
        self.metadata = dict(self.metadata)
        if not np.any(self.active_mask):
            raise ValueError(
                "SpectrumDataset requires at least one active spectral pixel with finite flux and strictly positive sigma."
            )

    @property
    def active_mask(self) -> np.ndarray:
        finite = np.isfinite(self.flux) & np.isfinite(self.sigma) & (self.sigma > 0.0)
        if self.mask is None:
            return finite
        return np.asarray(self.mask, dtype=bool) & finite

    @property
    def active_indices(self) -> np.ndarray:
        return np.where(self.active_mask)[0]

    @property
    def active_wavelength(self) -> np.ndarray:
        return self.wavelength[self.active_mask]

    @property
    def active_flux(self) -> np.ndarray:
        return self.flux[self.active_mask]

    @property
    def active_sigma(self) -> np.ndarray:
        return self.sigma[self.active_mask]

    def active_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.active_indices
        return self.wavelength[idx], self.flux[idx], self.sigma[idx], idx
