"""Small, explicit photometric noise models for simulation-based inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EmpiricalPhotometricNoise:
    """Draw a catalog uncertainty row and add a fractional model-flux term.

    ``sigma_rows`` has shape ``(n_objects, n_bands)`` in the same flux unit as
    the simulator. Sampling one complete row preserves empirical correlations
    in depth between bands. The returned uncertainty is

    ``sqrt((sigma_scale * empirical_sigma)**2 + (fractional_error * abs(flux))**2)``.

    The object is pickleable, unlike a notebook closure, so process workers can
    use it during expensive FSPS or CIGALE simulation campaigns.
    """

    sigma_rows: np.ndarray
    fractional_error: float = 0.05
    sigma_scale: float = 1.0
    band_names: Sequence[str] | None = None
    flux_unit: str | None = None

    def __post_init__(self) -> None:
        sigma = np.asarray(self.sigma_rows, dtype=float)
        if sigma.ndim != 2 or sigma.shape[0] == 0 or sigma.shape[1] == 0:
            raise ValueError("sigma_rows must have shape (n_objects, n_bands) with non-zero dimensions.")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma_rows must contain finite strictly positive uncertainties.")
        fractional_error = float(self.fractional_error)
        sigma_scale = float(self.sigma_scale)
        if not np.isfinite(fractional_error) or fractional_error < 0.0:
            raise ValueError("fractional_error must be finite and non-negative.")
        if not np.isfinite(sigma_scale) or sigma_scale <= 0.0:
            raise ValueError("sigma_scale must be finite and positive.")
        if self.band_names is None:
            band_names = tuple(f"band_{i}" for i in range(sigma.shape[1]))
        else:
            band_names = tuple(str(name) for name in self.band_names)
            if len(band_names) != sigma.shape[1] or len(set(band_names)) != len(band_names):
                raise ValueError("band_names must be unique and match sigma_rows columns.")

        object.__setattr__(self, "sigma_rows", sigma)
        object.__setattr__(self, "fractional_error", fractional_error)
        object.__setattr__(self, "sigma_scale", sigma_scale)
        object.__setattr__(self, "band_names", band_names)
        if self.flux_unit is not None:
            object.__setattr__(self, "flux_unit", str(self.flux_unit))

    def __call__(self, flux, *, rng: np.random.Generator | None = None, theta=None) -> np.ndarray:
        """Return one sigma vector matching a noiseless one-dimensional flux."""

        del theta
        flux = np.asarray(flux, dtype=float)
        if flux.shape != (self.sigma_rows.shape[1],):
            raise ValueError(
                f"Noise model expected flux shape {(self.sigma_rows.shape[1],)}, got {flux.shape}."
            )
        if not np.all(np.isfinite(flux)):
            raise ValueError("Noise model received non-finite flux.")
        if rng is None:
            rng = np.random.default_rng()
        row = self.sigma_rows[int(rng.integers(0, self.sigma_rows.shape[0]))]
        return np.hypot(self.sigma_scale * row, self.fractional_error * np.abs(flux))

    def specification(self) -> dict[str, object]:
        return {
            "name": type(self).__name__,
            "n_empirical_rows": int(self.sigma_rows.shape[0]),
            "band_names": self.band_names,
            "fractional_error": self.fractional_error,
            "sigma_scale": self.sigma_scale,
            "flux_unit": self.flux_unit,
        }
