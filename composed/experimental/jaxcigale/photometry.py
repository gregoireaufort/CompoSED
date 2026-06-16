from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from composed.experimental.jaxcigale.dependencies import require_jax

C_A_PER_S = 2.99792458e18
AB_FNU_CGS = 3631.0e-23


def trapz_numpy(y, x):
    """NumPy 1.x/2.x compatible trapezoidal integral."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


@dataclass(frozen=True)
class JaxFilterSet:
    """Fixed filter curves padded to a rectangular JAX-friendly array."""

    names: tuple[str, ...]
    wavelength_a: np.ndarray
    transmission: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        wave = np.asarray(self.wavelength_a, dtype=float)
        trans = np.asarray(self.transmission, dtype=float)
        valid = np.asarray(self.valid, dtype=bool)
        if wave.ndim != 2 or trans.shape != wave.shape or valid.shape != wave.shape:
            raise ValueError("Filter wavelength, transmission, and valid arrays must be matching 2D arrays.")
        if wave.shape[0] != len(self.names):
            raise ValueError("Number of filter names must match first filter-array dimension.")
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(self, "wavelength_a", wave)
        object.__setattr__(self, "transmission", trans)
        object.__setattr__(self, "valid", valid)

    @classmethod
    def from_curves(
        cls,
        names: Sequence[str],
        wavelength_a: Sequence[Sequence[float]],
        transmission: Sequence[Sequence[float]],
    ) -> "JaxFilterSet":
        names = tuple(str(name) for name in names)
        if len(names) != len(wavelength_a) or len(names) != len(transmission):
            raise ValueError("names, wavelength_a, and transmission must have the same length.")
        lengths = [len(np.asarray(w, dtype=float)) for w in wavelength_a]
        if min(lengths) < 2:
            raise ValueError("Each filter must contain at least two wavelength samples.")
        n_filter = len(names)
        n_max = max(lengths)
        wave = np.zeros((n_filter, n_max), dtype=float)
        trans = np.zeros((n_filter, n_max), dtype=float)
        valid = np.zeros((n_filter, n_max), dtype=bool)
        for i, (w, t) in enumerate(zip(wavelength_a, transmission)):
            w = np.asarray(w, dtype=float)
            t = np.asarray(t, dtype=float)
            if w.ndim != 1 or t.ndim != 1 or w.shape != t.shape:
                raise ValueError("Each filter curve must have matching one-dimensional wavelength/transmission.")
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(t)):
                raise ValueError("Filter curves must be finite.")
            if np.any(np.diff(w) <= 0.0):
                raise ValueError("Filter wavelengths must be strictly increasing.")
            n = w.size
            wave[i, :n] = w
            if n < n_max:
                wave[i, n:] = w[-1]
            trans[i, :n] = np.maximum(t, 0.0)
            valid[i, :n] = True
        return cls(names=names, wavelength_a=wave, transmission=trans, valid=valid)

    @classmethod
    def from_sedpy(cls, filters: Sequence[object], names: Sequence[str] | None = None) -> "JaxFilterSet":
        if names is None:
            names = [getattr(filt, "name", f"filter_{i}") for i, filt in enumerate(filters)]
        waves = []
        transmissions = []
        for filt in filters:
            wave = getattr(filt, "wavelength", getattr(filt, "wave", None))
            trans = getattr(filt, "transmission", None)
            if wave is None or trans is None:
                raise ValueError("sedpy-like filters must expose wavelength/wave and transmission arrays.")
            waves.append(np.asarray(wave, dtype=float))
            transmissions.append(np.asarray(trans, dtype=float))
        return cls.from_curves(names, waves, transmissions)

    def as_jax(self):
        _, jnp = require_jax()
        return (
            jnp.asarray(self.wavelength_a),
            jnp.asarray(self.transmission),
            jnp.asarray(self.valid),
        )


def trapz_jax(y, x):
    _, jnp = require_jax()
    return jnp.sum(0.5 * (y[..., 1:] + y[..., :-1]) * (x[..., 1:] - x[..., :-1]), axis=-1)


def integrate_maggies(wavelength_obs_a, flux_lambda_cgs, filters: JaxFilterSet):
    """Integrate observed ``f_lambda`` through fixed filters and return maggies."""

    _, jnp = require_jax()
    wave_f, trans_f, valid_f = filters.as_jax()
    trans_f = jnp.where(valid_f, trans_f, 0.0)

    def one_filter(filter_wave, filter_trans):
        flam = jnp.interp(filter_wave, wavelength_obs_a, flux_lambda_cgs, left=0.0, right=0.0)
        numerator = trapz_jax(flam * filter_wave * filter_trans, filter_wave)
        denominator = trapz_jax((C_A_PER_S / filter_wave) * filter_trans, filter_wave)
        fnu_cgs = numerator / jnp.maximum(denominator, 1e-300)
        return fnu_cgs / AB_FNU_CGS

    return jnp.asarray([one_filter(wave_f[i], trans_f[i]) for i in range(len(filters.names))])


def integrate_maggies_numpy(wavelength_obs_a, flux_lambda_cgs, filters: JaxFilterSet) -> np.ndarray:
    """NumPy reference for tests and debugging."""

    wave = np.asarray(wavelength_obs_a, dtype=float)
    flam = np.asarray(flux_lambda_cgs, dtype=float)
    out = []
    for fw, ft, valid in zip(filters.wavelength_a, filters.transmission, filters.valid):
        fw = fw[valid]
        ft = ft[valid]
        model_flam = np.interp(fw, wave, flam, left=0.0, right=0.0)
        numerator = trapz_numpy(model_flam * fw * ft, fw)
        denominator = trapz_numpy((C_A_PER_S / fw) * ft, fw)
        out.append((numerator / denominator) / AB_FNU_CGS)
    return np.asarray(out, dtype=float)
