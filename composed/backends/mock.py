from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from composed.backends.base import ModelPhotometry, ModelSpectrum, SEDBackend
from composed.units import MassNormalization


@dataclass
class MockBackend(SEDBackend):
    """Deterministic backend for tests and examples."""

    flux: Sequence[float]
    band_names: Sequence[str] | None = None
    spectrum_wavelength: Sequence[float] | None = None
    spectrum_flux: Sequence[float] | None = None
    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE
    fail_on_call: bool = False

    def predict_photometry(self, params: Mapping[str, float], filters) -> ModelPhotometry:
        del params
        if self.fail_on_call:
            raise FloatingPointError("Mock backend numerical failure.")
        flux = np.asarray(self.flux, dtype=float)
        if self.band_names is not None:
            names = tuple(str(name) for name in self.band_names)
        elif hasattr(filters, "names"):
            names = tuple(filters.names)
        else:
            names = tuple(str(i) for i in range(flux.size))
        return ModelPhotometry(band_names=names, flux=flux)

    def predict_spectrum(
        self,
        params: Mapping[str, float],
        wavelengths: Sequence[float] | None = None,
        wavelength_range: tuple[float, float] | None = None,
        resolution: float | None = None,
    ) -> ModelSpectrum:
        del params, resolution
        if self.fail_on_call:
            raise FloatingPointError("Mock backend numerical failure.")
        if self.spectrum_flux is None:
            raise NotImplementedError("MockBackend requires spectrum_flux for predict_spectrum.")

        flux = np.asarray(self.spectrum_flux, dtype=float)
        if self.spectrum_wavelength is None:
            if wavelengths is not None and len(wavelengths) == flux.size:
                base_wave = np.asarray(wavelengths, dtype=float)
            else:
                base_wave = np.arange(flux.size, dtype=float)
        else:
            base_wave = np.asarray(self.spectrum_wavelength, dtype=float)
        if base_wave.shape != flux.shape:
            raise ValueError("spectrum_wavelength and spectrum_flux must have matching shape.")

        if wavelengths is not None:
            out_wave = np.asarray(wavelengths, dtype=float)
            out_flux = np.interp(out_wave, base_wave, flux, left=np.nan, right=np.nan)
        else:
            out_wave = base_wave
            out_flux = flux

        if wavelength_range is not None:
            lo, hi = wavelength_range
            use = (out_wave >= float(lo)) & (out_wave <= float(hi))
            out_wave = out_wave[use]
            out_flux = out_flux[use]

        return ModelSpectrum(wavelength=out_wave, flux=out_flux)
