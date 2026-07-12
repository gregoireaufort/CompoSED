from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from composed.units import (
    MassNormalization,
    canonical_photometric_flux_unit,
    canonical_spectral_flux_unit,
    canonical_wavelength_unit,
)


@dataclass(frozen=True)
class ModelPhotometry:
    """Predicted photometry from a backend.

    ``flux`` must be a one-dimensional vector in the backend's declared flux
    normalization. ``band_names`` names each element so likelihoods can align
    model and data without assuming positional agreement.
    """

    band_names: Sequence[str]
    flux: np.ndarray
    metadata: Mapping[str, object] | None = None
    flux_unit: str = "maggies"

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_names", tuple(str(name) for name in self.band_names))
        flux = np.asarray(self.flux, dtype=float)
        if flux.ndim != 1:
            raise ValueError("ModelPhotometry.flux must be one-dimensional.")
        if len(self.band_names) != flux.size:
            raise ValueError("band_names length must match flux length.")
        if len(set(self.band_names)) != len(self.band_names):
            raise ValueError("ModelPhotometry band_names must be unique.")
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))
        object.__setattr__(self, "flux_unit", canonical_photometric_flux_unit(self.flux_unit))


@dataclass(frozen=True)
class ModelSpectrum:
    """Predicted observed-frame spectrum from a backend.

    ``wavelength`` and ``flux`` are one-dimensional arrays with matching shape.
    The first spectral likelihood implementation expects observed-frame
    wavelength in Angstrom and observed ``f_lambda`` in
    ``erg s^-1 cm^-2 Angstrom^-1``. Backends should record those units in
    ``wavelength_unit`` and ``flux_unit`` instead of relying on convention.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    wavelength_unit: str = "angstrom"
    flux_unit: str = "erg/s/cm^2/angstrom"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        if wavelength.ndim != 1 or flux.ndim != 1:
            raise ValueError("ModelSpectrum wavelength and flux must be one-dimensional.")
        if wavelength.shape != flux.shape:
            raise ValueError("ModelSpectrum wavelength and flux must have the same shape.")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "wavelength_unit", canonical_wavelength_unit(self.wavelength_unit))
        object.__setattr__(self, "flux_unit", canonical_spectral_flux_unit(self.flux_unit))
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))


class SEDBackend:
    """Common backend interface.

    Backends must declare ``mass_normalization`` and implement
    ``predict_photometry(params, filters)``. They should not apply any
    likelihood-specific mass scaling; that is handled centrally by
    ``GaussianPhotometricLikelihood`` when the normalization is
    ``PER_SOLAR_MASS``.
    """

    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params: Mapping[str, float], filters) -> ModelPhotometry:
        raise NotImplementedError

    def predict_spectrum(
        self,
        params: Mapping[str, float],
        wavelengths: Sequence[float] | None = None,
        wavelength_range: tuple[float, float] | None = None,
        resolution: float | None = None,
    ) -> ModelSpectrum:
        raise NotImplementedError

    def predict_rest_spectrum(
        self,
        params: Mapping[str, float],
        wavelengths: Sequence[float] | None = None,
        wavelength_range: tuple[float, float] | None = None,
    ) -> ModelSpectrum:
        """Predict rest-frame luminosity density for fast catalog projection.

        Implementations should return a ``ModelSpectrum`` whose wavelength unit
        and luminosity-density unit are explicit in the metadata.  The
        CompoSED-native catalog path expects rest-frame wavelength in nm and
        luminosity density in W/nm, but conversion can be done by the grid
        builder when a backend documents another compatible unit.
        """

        raise NotImplementedError
