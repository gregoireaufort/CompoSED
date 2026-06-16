from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from composed.experimental.jaxcigale.cue import CUE_HI_EDGE_A, CueDerivedInputs
from composed.experimental.jaxcigale.dependencies import require_jax
from composed.experimental.jaxcigale.photometry import C_A_PER_S


@dataclass(frozen=True)
class FspsNebularContinuumTable:
    """Static FSPS/CLOUDY nebular-continuum table for JAX interpolation.

    The FSPS ``*.cont`` files store nebular continuum in ``Lsun / Hz / Q`` on
    a grid of wavelength, gas metallicity, SSP age, and ionization parameter.
    ``Q`` is the hydrogen-ionizing photon rate in photons/s.  This table is
    useful as an optional LyC-continuum add-on to Cue because public Cue v0.1
    deliberately predicts nebular continuum only redward of 912 Angstrom.

    This is not a full FSPS nebular module.  The caller chooses one effective
    nebular age for the continuum table, while the normalization still comes
    from the gas-powered ``Q_H`` derived from the actual stellar spectrum.
    """

    wavelength_a: np.ndarray
    logz_grid: np.ndarray
    log_age_yr_grid: np.ndarray
    logu_grid: np.ndarray
    log_lnu_lsun_per_hz_per_q: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path, *, max_wavelength_a: float = CUE_HI_EDGE_A) -> "FspsNebularContinuumTable":
        """Read an FSPS ``nebular/*.cont`` table and keep only LyC wavelengths."""

        path = Path(path)
        with path.open() as handle:
            _header = handle.readline()
            wavelength_a = np.fromstring(handle.readline(), sep=" ")
            rows: list[tuple[float, float, float, np.ndarray]] = []
            while True:
                parameter_line = handle.readline()
                if not parameter_line:
                    break
                values_line = handle.readline()
                if not values_line:
                    raise ValueError(f"Malformed FSPS nebular continuum table: {path}")
                logz, age_yr, logu = np.fromstring(parameter_line, sep=" ")
                spectrum = np.fromstring(values_line, sep=" ")
                rows.append((float(logz), float(age_yr), float(logu), spectrum))

        if not rows:
            raise ValueError(f"Empty FSPS nebular continuum table: {path}")

        logz_grid = np.unique([row[0] for row in rows])
        age_grid_yr = np.unique([row[1] for row in rows])
        logu_grid = np.unique([row[2] for row in rows])
        log_age_yr_grid = np.log10(age_grid_yr)

        keep_wave = wavelength_a < float(max_wavelength_a)
        if not np.any(keep_wave):
            raise ValueError(f"FSPS table has no wavelengths below {max_wavelength_a} Angstrom: {path}")
        wavelength_a = wavelength_a[keep_wave]

        table = np.full(
            (logz_grid.size, log_age_yr_grid.size, logu_grid.size, wavelength_a.size),
            -95.0,
            dtype=float,
        )
        z_index = {float(value): i for i, value in enumerate(logz_grid)}
        age_index = {float(value): i for i, value in enumerate(age_grid_yr)}
        u_index = {float(value): i for i, value in enumerate(logu_grid)}
        for logz, age_yr, logu, spectrum in rows:
            table[z_index[logz], age_index[age_yr], u_index[logu], :] = np.log10(spectrum[keep_wave] + 1.0e-95)

        return cls(
            wavelength_a=np.asarray(wavelength_a, dtype=float),
            logz_grid=np.asarray(logz_grid, dtype=float),
            log_age_yr_grid=np.asarray(log_age_yr_grid, dtype=float),
            logu_grid=np.asarray(logu_grid, dtype=float),
            log_lnu_lsun_per_hz_per_q=table,
        )

    def make_lyc_continuum_apply(self, *, effective_age_yr: float = 1.0e6):
        """Return a pure-JAX LyC continuum function for ``cue_nebular_module``.

        The returned function has signature ``(wave_rest_a, cue_inputs)`` and
        returns ``Lsun / Angstrom`` per solar mass formed.  The gas metallicity
        is read from ``cue_inputs.gas_logoh`` and treated as FSPS ``gas_logz``;
        that is a pragmatic first pass for the CIGALE-like experiments where
        both are logarithmic abundance offsets from the solar reference.
        """

        log_age_yr = float(np.log10(effective_age_yr))
        wavelength_a = np.asarray(self.wavelength_a, dtype=float)
        logz_grid = np.asarray(self.logz_grid, dtype=float)
        log_age_grid = np.asarray(self.log_age_yr_grid, dtype=float)
        logu_grid = np.asarray(self.logu_grid, dtype=float)
        log_table = np.asarray(self.log_lnu_lsun_per_hz_per_q, dtype=float)

        def apply(wave_rest_a, cue_inputs: CueDerivedInputs):
            _, jnp = require_jax()
            wave = jnp.asarray(wave_rest_a)
            table_wave = jnp.asarray(wavelength_a)
            log_lnu_grid = _interp3_log_table(
                jnp.asarray(log_table),
                jnp.asarray(logz_grid),
                jnp.asarray(log_age_grid),
                jnp.asarray(logu_grid),
                cue_inputs.gas_logoh,
                jnp.asarray(log_age_yr),
                cue_inputs.logu,
            )
            lnu_per_q = 10.0**log_lnu_grid
            lnu_per_q_on_model_grid = jnp.interp(wave, table_wave, lnu_per_q, left=0.0, right=0.0)
            q_h_gas = 10.0 ** cue_inputs.log_q_h_gas
            l_lambda = lnu_per_q_on_model_grid * q_h_gas * C_A_PER_S / jnp.maximum(wave, 1.0) ** 2
            return jnp.where(wave < CUE_HI_EDGE_A, l_lambda, jnp.zeros_like(l_lambda))

        return apply


def default_fsps_nebular_continuum_path(
    *,
    sps_home: str | Path | None = None,
    isoc_type: str = "mist",
    cloudy_dust: bool = False,
) -> Path:
    """Return the conventional FSPS nebular continuum table path."""

    root = Path(sps_home or os.environ.get("SPS_HOME", ""))
    prefix = "ZAU_WD" if cloudy_dust else "ZAU_ND"
    return root / "nebular" / f"{prefix}_{isoc_type}.cont"


def load_fsps_lyc_continuum_apply(
    path: str | Path | None = None,
    *,
    sps_home: str | Path | None = None,
    isoc_type: str = "mist",
    cloudy_dust: bool = False,
    effective_age_yr: float = 1.0e6,
):
    """Load an FSPS table and return a Cue-compatible LyC continuum function."""

    table_path = default_fsps_nebular_continuum_path(
        sps_home=sps_home,
        isoc_type=isoc_type,
        cloudy_dust=cloudy_dust,
    ) if path is None else Path(path)
    table = FspsNebularContinuumTable.from_file(table_path)
    return table.make_lyc_continuum_apply(effective_age_yr=effective_age_yr)


def _interp3_log_table(table, x_grid, y_grid, z_grid, x, y, z):
    _, jnp = require_jax()
    ix0, ix1, wx = _bracket_axis(x_grid, x)
    iy0, iy1, wy = _bracket_axis(y_grid, y)
    iz0, iz1, wz = _bracket_axis(z_grid, z)

    c000 = table[ix0, iy0, iz0]
    c001 = table[ix0, iy0, iz1]
    c010 = table[ix0, iy1, iz0]
    c011 = table[ix0, iy1, iz1]
    c100 = table[ix1, iy0, iz0]
    c101 = table[ix1, iy0, iz1]
    c110 = table[ix1, iy1, iz0]
    c111 = table[ix1, iy1, iz1]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz


def _bracket_axis(grid, value):
    _, jnp = require_jax()
    grid = jnp.asarray(grid)
    value = jnp.clip(jnp.asarray(value), grid[0], grid[-1])
    upper = jnp.clip(jnp.searchsorted(grid, value, side="right"), 1, grid.size - 1)
    lower = upper - 1
    span = jnp.maximum(grid[upper] - grid[lower], 1.0e-30)
    weight = (value - grid[lower]) / span
    return lower, upper, weight
