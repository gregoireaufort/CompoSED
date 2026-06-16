from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from composed.experimental.jaxcigale.core import (
    ModuleSpec,
    SEDState,
    flat_lcdm_age_gyr,
    flat_lcdm_age_gyr_numpy,
    observed_flux_from_luminosity,
)
from composed.experimental.jaxcigale.dependencies import require_dsps, require_jax
from composed.experimental.jaxcigale.photometry import C_A_PER_S, trapz_jax


@dataclass(frozen=True)
class SFHState:
    age_grid_gyr: np.ndarray
    tau_parameter: str
    tage_parameter: str
    alpha_parameter: str
    softness_gyr: float
    mode: str


def delayed_sfh_module(
    age_grid_gyr: Sequence[float],
    tau_parameter: str = "tau_gyr",
    tage_parameter: str = "tage_gyr",
    softness_gyr: float = 0.05,
) -> ModuleSpec:
    """Smooth delayed-tau SFH normalized to one solar mass formed."""

    return ModuleSpec(
        name="delayed_sfh",
        setup_fn=lambda config: SFHState(
            age_grid_gyr=np.asarray(age_grid_gyr, dtype=float),
            tau_parameter=tau_parameter,
            tage_parameter=tage_parameter,
            alpha_parameter="sfh_alpha",
            softness_gyr=float(softness_gyr),
            mode="delayed",
        ),
        apply_fn=_apply_parametric_sfh,
    )


def exponential_sfh_module(
    age_grid_gyr: Sequence[float],
    tau_parameter: str = "tau_gyr",
    tage_parameter: str = "tage_gyr",
    softness_gyr: float = 0.05,
) -> ModuleSpec:
    """Smooth exponentially declining SFH normalized to one solar mass formed."""

    return ModuleSpec(
        name="exponential_sfh",
        setup_fn=lambda config: SFHState(
            age_grid_gyr=np.asarray(age_grid_gyr, dtype=float),
            tau_parameter=tau_parameter,
            tage_parameter=tage_parameter,
            alpha_parameter="sfh_alpha",
            softness_gyr=float(softness_gyr),
            mode="exponential",
        ),
        apply_fn=_apply_parametric_sfh,
    )


def constant_sfh_module(
    age_grid_gyr: Sequence[float],
    tage_parameter: str = "tage_gyr",
    softness_gyr: float = 0.05,
) -> ModuleSpec:
    """Constant SFH normalized to one solar mass formed."""

    return ModuleSpec(
        name="constant_sfh",
        setup_fn=lambda config: SFHState(
            age_grid_gyr=np.asarray(age_grid_gyr, dtype=float),
            tau_parameter="tau_gyr",
            tage_parameter=tage_parameter,
            alpha_parameter="sfh_alpha",
            softness_gyr=float(softness_gyr),
            mode="constant",
        ),
        apply_fn=_apply_parametric_sfh,
    )


def powerlaw_sfh_module(
    age_grid_gyr: Sequence[float],
    alpha_parameter: str = "sfh_alpha",
    tage_parameter: str = "tage_gyr",
    softness_gyr: float = 0.05,
) -> ModuleSpec:
    """Power-law SFH, ``SFR(t) proportional to t**alpha``.

    This is useful for high-redshift toy models where a simple rising or
    falling history is easier to audit than a delayed-tau model.
    """

    return ModuleSpec(
        name="powerlaw_sfh",
        setup_fn=lambda config: SFHState(
            age_grid_gyr=np.asarray(age_grid_gyr, dtype=float),
            tau_parameter="tau_gyr",
            tage_parameter=tage_parameter,
            alpha_parameter=alpha_parameter,
            softness_gyr=float(softness_gyr),
            mode="powerlaw",
        ),
        apply_fn=_apply_parametric_sfh,
    )


@dataclass(frozen=True)
class CosmicParametricSFHState:
    n_time: int
    min_age_since_onset_gyr: float
    tau_parameter: str
    tage_parameter: str
    alpha_parameter: str
    redshift_parameter: str
    mode: str
    tage_is_fraction_of_universe_age: bool
    age_table_z: np.ndarray
    age_table_gyr: np.ndarray


def delayed_sfh_cosmic_time_module(
    n_time: int = 512,
    min_age_since_onset_gyr: float = 0.02,
    tau_parameter: str = "tau_gyr",
    tage_parameter: str = "tage_gyr",
    redshift_parameter: str = "z",
    tage_is_fraction_of_universe_age: bool = False,
    age_table_z: Sequence[float] | None = None,
    age_table_gyr: Sequence[float] | None = None,
) -> ModuleSpec:
    """Delayed-tau SFH tabulated on cosmic time for DSPS/FSPS-style CSPs.

    The fitted ``tage_gyr`` is the galaxy age at observation.  The module
    constructs a fixed-length table from galaxy onset to observation:

    ``cosmic_time = age_of_universe(z) - tage_gyr + age_since_onset``.

    When ``tage_is_fraction_of_universe_age=True``, the fitted
    ``tage_parameter`` is interpreted as ``tage / age_of_universe(z)``.  The
    age of the Universe is then read from a fixed interpolation table prepared
    during setup.  This keeps the NUTS graph cheap and avoids parameter
    proposals with galaxy ages older than the Universe by construction.

    This is the convention expected by DSPS and python-fsps tabular SFHs.
    The SFR is normalized so the time integral is one solar mass formed.
    """

    return ModuleSpec(
        name="delayed_sfh_cosmic_time",
        setup_fn=lambda config: _make_cosmic_parametric_sfh_state(
            n_time=int(n_time),
            min_age_since_onset_gyr=float(min_age_since_onset_gyr),
            tau_parameter=tau_parameter,
            tage_parameter=tage_parameter,
            alpha_parameter="sfh_alpha",
            redshift_parameter=redshift_parameter,
            mode="delayed",
            tage_is_fraction_of_universe_age=bool(tage_is_fraction_of_universe_age),
            age_table_z=age_table_z,
            age_table_gyr=age_table_gyr,
        ),
        apply_fn=_apply_cosmic_parametric_sfh,
    )


def exponential_sfh_cosmic_time_module(
    n_time: int = 512,
    min_age_since_onset_gyr: float = 0.02,
    tau_parameter: str = "tau_gyr",
    tage_parameter: str = "tage_gyr",
    redshift_parameter: str = "z",
    tage_is_fraction_of_universe_age: bool = False,
    age_table_z: Sequence[float] | None = None,
    age_table_gyr: Sequence[float] | None = None,
) -> ModuleSpec:
    """Exponentially declining SFH tabulated on cosmic time.

    See :func:`delayed_sfh_cosmic_time_module` for the time convention.
    """

    return ModuleSpec(
        name="exponential_sfh_cosmic_time",
        setup_fn=lambda config: _make_cosmic_parametric_sfh_state(
            n_time=int(n_time),
            min_age_since_onset_gyr=float(min_age_since_onset_gyr),
            tau_parameter=tau_parameter,
            tage_parameter=tage_parameter,
            alpha_parameter="sfh_alpha",
            redshift_parameter=redshift_parameter,
            mode="exponential",
            tage_is_fraction_of_universe_age=bool(tage_is_fraction_of_universe_age),
            age_table_z=age_table_z,
            age_table_gyr=age_table_gyr,
        ),
        apply_fn=_apply_cosmic_parametric_sfh,
    )


def constant_sfh_cosmic_time_module(
    n_time: int = 512,
    min_age_since_onset_gyr: float = 0.02,
    tage_parameter: str = "tage_gyr",
    redshift_parameter: str = "z",
    tage_is_fraction_of_universe_age: bool = False,
    age_table_z: Sequence[float] | None = None,
    age_table_gyr: Sequence[float] | None = None,
) -> ModuleSpec:
    """Constant SFH tabulated on cosmic time and normalized to one solar mass."""

    return ModuleSpec(
        name="constant_sfh_cosmic_time",
        setup_fn=lambda config: _make_cosmic_parametric_sfh_state(
            n_time=int(n_time),
            min_age_since_onset_gyr=float(min_age_since_onset_gyr),
            tau_parameter="tau_gyr",
            tage_parameter=tage_parameter,
            alpha_parameter="sfh_alpha",
            redshift_parameter=redshift_parameter,
            mode="constant",
            tage_is_fraction_of_universe_age=bool(tage_is_fraction_of_universe_age),
            age_table_z=age_table_z,
            age_table_gyr=age_table_gyr,
        ),
        apply_fn=_apply_cosmic_parametric_sfh,
    )


def powerlaw_sfh_cosmic_time_module(
    n_time: int = 512,
    min_age_since_onset_gyr: float = 0.02,
    alpha_parameter: str = "sfh_alpha",
    tage_parameter: str = "tage_gyr",
    redshift_parameter: str = "z",
    tage_is_fraction_of_universe_age: bool = False,
    age_table_z: Sequence[float] | None = None,
    age_table_gyr: Sequence[float] | None = None,
) -> ModuleSpec:
    """Power-law SFH on cosmic time, ``SFR(age_since_onset) ~ age**alpha``."""

    return ModuleSpec(
        name="powerlaw_sfh_cosmic_time",
        setup_fn=lambda config: _make_cosmic_parametric_sfh_state(
            n_time=int(n_time),
            min_age_since_onset_gyr=float(min_age_since_onset_gyr),
            tau_parameter="tau_gyr",
            tage_parameter=tage_parameter,
            alpha_parameter=alpha_parameter,
            redshift_parameter=redshift_parameter,
            mode="powerlaw",
            tage_is_fraction_of_universe_age=bool(tage_is_fraction_of_universe_age),
            age_table_z=age_table_z,
            age_table_gyr=age_table_gyr,
        ),
        apply_fn=_apply_cosmic_parametric_sfh,
    )


def _make_cosmic_parametric_sfh_state(
    *,
    n_time: int,
    min_age_since_onset_gyr: float,
    tau_parameter: str,
    tage_parameter: str,
    alpha_parameter: str,
    redshift_parameter: str,
    mode: str,
    tage_is_fraction_of_universe_age: bool,
    age_table_z: Sequence[float] | None,
    age_table_gyr: Sequence[float] | None,
) -> CosmicParametricSFHState:
    if age_table_z is None:
        z_grid = np.linspace(0.0, 20.0, 4096)
    else:
        z_grid = np.asarray(age_table_z, dtype=float)
    if age_table_gyr is None:
        age_grid = flat_lcdm_age_gyr_numpy(z_grid)
    else:
        age_grid = np.asarray(age_table_gyr, dtype=float)
    if z_grid.ndim != 1 or age_grid.shape != z_grid.shape or z_grid.size < 2:
        raise ValueError("age_table_z and age_table_gyr must be matching one-dimensional arrays.")
    if np.any(np.diff(z_grid) <= 0.0) or not np.all(np.isfinite(z_grid)) or not np.all(np.isfinite(age_grid)):
        raise ValueError("age-of-universe interpolation table must be finite and strictly increasing in redshift.")
    return CosmicParametricSFHState(
        n_time=int(n_time),
        min_age_since_onset_gyr=float(min_age_since_onset_gyr),
        tau_parameter=tau_parameter,
        tage_parameter=tage_parameter,
        alpha_parameter=alpha_parameter,
        redshift_parameter=redshift_parameter,
        mode=mode,
        tage_is_fraction_of_universe_age=bool(tage_is_fraction_of_universe_age),
        age_table_z=z_grid,
        age_table_gyr=age_grid,
    )


def _apply_parametric_sfh(params: Mapping[str, object], sed_state: SEDState, state: SFHState) -> SEDState:
    _, jnp = require_jax()
    age = jnp.asarray(state.age_grid_gyr)
    tage = jnp.maximum(params[state.tage_parameter], 1e-4)
    smooth_cut = 1.0 / (1.0 + jnp.exp((age - tage) / state.softness_gyr))
    if state.mode == "delayed":
        tau = jnp.maximum(params[state.tau_parameter], 1e-4)
        raw_sfr = age * jnp.exp(-age / tau) * smooth_cut
    elif state.mode == "exponential":
        tau = jnp.maximum(params[state.tau_parameter], 1e-4)
        raw_sfr = jnp.exp(-age / tau) * smooth_cut
    elif state.mode == "constant":
        raw_sfr = jnp.ones_like(age) * smooth_cut
    elif state.mode == "powerlaw":
        alpha = params[state.alpha_parameter]
        raw_sfr = (jnp.maximum(age, 1.0e-4) / jnp.maximum(tage, 1.0e-4)) ** alpha * smooth_cut
    else:
        raise ValueError(f"Unsupported SFH mode: {state.mode!r}")
    formed_mass = trapz_jax(raw_sfr, age * 1.0e9)
    sfr = raw_sfr / jnp.maximum(formed_mass, 1e-300)
    return sed_state._replace(sfh_time_gyr=age, sfr_msun_per_yr=sfr, formed_mass_msun=jnp.asarray(1.0))


def _apply_cosmic_parametric_sfh(
    params: Mapping[str, object], sed_state: SEDState, state: CosmicParametricSFHState
) -> SEDState:
    _, jnp = require_jax()
    z = params[state.redshift_parameter]
    if state.tage_is_fraction_of_universe_age:
        t_obs = jnp.interp(z, jnp.asarray(state.age_table_z), jnp.asarray(state.age_table_gyr))
    else:
        t_obs = flat_lcdm_age_gyr(z)
    raw_tage = jnp.maximum(params[state.tage_parameter], 1e-4)
    if state.tage_is_fraction_of_universe_age:
        tage = jnp.clip(raw_tage, 1e-4, 0.999) * t_obs
    else:
        tage = raw_tage
    valid_age = jnp.isfinite(t_obs) & jnp.isfinite(tage) & (t_obs > 0.0) & (tage > 0.0) & (tage <= t_obs)

    # Keep a fixed table shape for JIT/NUTS, but let the physical time span
    # depend on tage.  The support variable is time since star formation onset.
    u = jnp.linspace(0.0, 1.0, state.n_time)
    first_age = jnp.minimum(jnp.asarray(state.min_age_since_onset_gyr), 0.25 * tage)
    first_age = jnp.maximum(first_age, 1e-4)
    age_since_onset = first_age + u * (tage - first_age)
    cosmic_time = t_obs - tage + age_since_onset

    if state.mode == "delayed":
        tau = jnp.maximum(params[state.tau_parameter], 1e-4)
        raw_sfr = age_since_onset * jnp.exp(-age_since_onset / tau)
    elif state.mode == "exponential":
        tau = jnp.maximum(params[state.tau_parameter], 1e-4)
        raw_sfr = jnp.exp(-age_since_onset / tau)
    elif state.mode == "constant":
        raw_sfr = jnp.ones_like(age_since_onset)
    elif state.mode == "powerlaw":
        alpha = params[state.alpha_parameter]
        raw_sfr = (age_since_onset / jnp.maximum(tage, 1.0e-4)) ** alpha
    else:
        raise ValueError(f"Unsupported SFH mode: {state.mode!r}")
    formed_mass = trapz_jax(raw_sfr, cosmic_time * 1.0e9)
    sfr = raw_sfr / jnp.maximum(formed_mass, 1e-300)
    cosmic_time = jnp.where(valid_age, cosmic_time, jnp.nan)
    sfr = jnp.where(valid_age, sfr, jnp.nan)
    formed_mass_out = jnp.where(valid_age, jnp.asarray(1.0), jnp.nan)
    return sed_state._replace(sfh_time_gyr=cosmic_time, sfr_msun_per_yr=sfr, formed_mass_msun=formed_mass_out)


@dataclass(frozen=True)
class ContinuitySFHState:
    bin_edges_gyr: np.ndarray
    parameter_prefix: str
    softness_gyr: float


def continuity_sfh_module(
    bin_edges_gyr: Sequence[float],
    parameter_prefix: str = "logsfr",
    softness_gyr: float = 0.03,
) -> ModuleSpec:
    """Piecewise-constant continuity SFH with smooth bin edges."""

    edges = np.asarray(bin_edges_gyr, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges_gyr must be a strictly increasing one-dimensional array.")
    return ModuleSpec(
        name="continuity_sfh",
        setup_fn=lambda config: ContinuitySFHState(edges, parameter_prefix, float(softness_gyr)),
        apply_fn=_apply_continuity_sfh,
    )


def _apply_continuity_sfh(params: Mapping[str, object], sed_state: SEDState, state: ContinuitySFHState) -> SEDState:
    _, jnp = require_jax()
    edges = jnp.asarray(state.bin_edges_gyr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    log_sfr = jnp.asarray([params[f"{state.parameter_prefix}_{i}"] for i in range(centers.size)])
    sfr_bins = 10.0**log_sfr
    age = centers
    raw_sfr = sfr_bins
    formed_mass = jnp.sum(raw_sfr * widths * 1.0e9)
    sfr = raw_sfr / jnp.maximum(formed_mass, 1e-300)
    return sed_state._replace(sfh_time_gyr=age, sfr_msun_per_yr=sfr, formed_mass_msun=jnp.asarray(1.0))


@dataclass(frozen=True)
class CosmicContinuitySFHState:
    lookback_edges_gyr: np.ndarray
    parameter_prefix: str
    redshift_parameter: str
    min_bin_width_gyr: float
    age_table_z: np.ndarray
    age_table_gyr: np.ndarray


def continuity_sfh_cosmic_time_module(
    lookback_edges_gyr: Sequence[float],
    parameter_prefix: str = "logsfr",
    redshift_parameter: str = "z",
    min_bin_width_gyr: float = 1.0e-4,
    age_table_z: Sequence[float] | None = None,
    age_table_gyr: Sequence[float] | None = None,
) -> ModuleSpec:
    """Redshift-aware piecewise SFH whose oldest bin reaches ``age_universe(z)``.

    ``lookback_edges_gyr`` are lookback-time bin edges measured backward from
    the observed galaxy time.  They must be strictly increasing and start at
    zero.  The final edge is not supplied by the user: at runtime the module
    appends ``age_universe(z)`` so the oldest bin always ends at the Big Bang
    for the current redshift proposal.

    Example
    -------
    ``lookback_edges_gyr=[0.0, 0.03, 0.1, 0.3, 1.0]`` creates five bins:
    0-30 Myr, 30-100 Myr, 100-300 Myr, 0.3-1 Gyr, and
    1 Gyr-``age_universe(z)``.

    The free parameters are ``logsfr_0`` ... ``logsfr_N`` in recent-to-old
    order.  The output table is reversed into increasing cosmic time because
    DSPS/FSPS-style tabular SFHs expect cosmic time to increase.
    """

    edges = np.asarray(lookback_edges_gyr, dtype=float)
    if edges.ndim != 1 or edges.size < 1:
        raise ValueError("lookback_edges_gyr must be a one-dimensional array with at least one edge.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("lookback_edges_gyr must be finite and strictly increasing.")
    if not np.isclose(edges[0], 0.0):
        raise ValueError("lookback_edges_gyr must start at 0.0 Gyr.")
    if min_bin_width_gyr <= 0.0:
        raise ValueError("min_bin_width_gyr must be positive.")
    return ModuleSpec(
        name="continuity_sfh_cosmic_time",
        setup_fn=lambda config: _make_cosmic_continuity_sfh_state(
            lookback_edges_gyr=edges,
            parameter_prefix=parameter_prefix,
            redshift_parameter=redshift_parameter,
            min_bin_width_gyr=float(min_bin_width_gyr),
            age_table_z=age_table_z,
            age_table_gyr=age_table_gyr,
        ),
        apply_fn=_apply_cosmic_continuity_sfh,
    )


def _make_cosmic_continuity_sfh_state(
    *,
    lookback_edges_gyr: np.ndarray,
    parameter_prefix: str,
    redshift_parameter: str,
    min_bin_width_gyr: float,
    age_table_z: Sequence[float] | None,
    age_table_gyr: Sequence[float] | None,
) -> CosmicContinuitySFHState:
    z_grid, age_grid = _prepare_age_table(age_table_z, age_table_gyr)
    return CosmicContinuitySFHState(
        lookback_edges_gyr=lookback_edges_gyr,
        parameter_prefix=parameter_prefix,
        redshift_parameter=redshift_parameter,
        min_bin_width_gyr=float(min_bin_width_gyr),
        age_table_z=z_grid,
        age_table_gyr=age_grid,
    )


def _apply_cosmic_continuity_sfh(
    params: Mapping[str, object], sed_state: SEDState, state: CosmicContinuitySFHState
) -> SEDState:
    _, jnp = require_jax()
    z = params[state.redshift_parameter]
    t_obs = jnp.interp(z, jnp.asarray(state.age_table_z), jnp.asarray(state.age_table_gyr))

    fixed_edges = jnp.asarray(state.lookback_edges_gyr)
    n_fixed_edges = int(state.lookback_edges_gyr.size)
    # If a high-redshift proposal makes some fixed lookback edges older than
    # the Universe, gently compress those oldest fixed edges below t_obs while
    # preserving a fixed number of bins and a strictly increasing edge order.
    min_width = jnp.minimum(jnp.asarray(state.min_bin_width_gyr), t_obs / (10.0 * n_fixed_edges))
    edge_index = jnp.arange(n_fixed_edges)
    max_fixed_edges = t_obs - min_width * (n_fixed_edges - edge_index)
    clipped_fixed_edges = jnp.minimum(fixed_edges, max_fixed_edges)
    clipped_fixed_edges = clipped_fixed_edges.at[0].set(0.0)

    lookback_edges = jnp.concatenate([clipped_fixed_edges, jnp.asarray([t_obs])])
    lookback_widths = lookback_edges[1:] - lookback_edges[:-1]
    lookback_centers = 0.5 * (lookback_edges[:-1] + lookback_edges[1:])

    log_sfr = jnp.asarray([params[f"{state.parameter_prefix}_{i}"] for i in range(n_fixed_edges)])
    sfr_recent_to_old = 10.0**log_sfr
    formed_mass = jnp.sum(sfr_recent_to_old * lookback_widths * 1.0e9)
    sfr_recent_to_old = sfr_recent_to_old / jnp.maximum(formed_mass, 1e-300)

    cosmic_time_recent_to_old = t_obs - lookback_centers
    cosmic_time_old_to_recent = cosmic_time_recent_to_old[::-1]
    sfr_old_to_recent = sfr_recent_to_old[::-1]
    return sed_state._replace(
        sfh_time_gyr=cosmic_time_old_to_recent,
        sfr_msun_per_yr=sfr_old_to_recent,
        formed_mass_msun=jnp.asarray(1.0),
    )


def _prepare_age_table(
    age_table_z: Sequence[float] | None,
    age_table_gyr: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if age_table_z is None:
        z_grid = np.linspace(0.0, 20.0, 4096)
    else:
        z_grid = np.asarray(age_table_z, dtype=float)
    if age_table_gyr is None:
        age_grid = flat_lcdm_age_gyr_numpy(z_grid)
    else:
        age_grid = np.asarray(age_table_gyr, dtype=float)
    if z_grid.ndim != 1 or age_grid.shape != z_grid.shape or z_grid.size < 2:
        raise ValueError("age_table_z and age_table_gyr must be matching one-dimensional arrays.")
    if np.any(np.diff(z_grid) <= 0.0) or not np.all(np.isfinite(z_grid)) or not np.all(np.isfinite(age_grid)):
        raise ValueError("age-of-universe interpolation table must be finite and strictly increasing in redshift.")
    return z_grid, age_grid


@dataclass(frozen=True)
class AnalyticStellarState:
    z_sun: float
    metallicity_parameter: str


def analytic_stellar_module(z_sun: float = 0.0142, metallicity_parameter: str = "logzsol") -> ModuleSpec:
    """Small differentiable stellar continuum used for plumbing tests."""

    return ModuleSpec(
        name="analytic_stellar",
        setup_fn=lambda config: AnalyticStellarState(float(z_sun), metallicity_parameter),
        apply_fn=_apply_analytic_stellar,
    )


def _apply_analytic_stellar(params: Mapping[str, object], sed_state: SEDState, state: AnalyticStellarState) -> SEDState:
    _, jnp = require_jax()
    wave = sed_state.wave_rest_a
    logzsol = params[state.metallicity_parameter]
    z_factor = 10.0**logzsol
    slope = -1.4 - 0.35 * logzsol
    continuum = 2.0e-7 * (wave / 5500.0) ** slope * jnp.exp(-wave / 25000.0)
    break_4000 = 1.0 - 0.25 * z_factor / (1.0 + z_factor) / (1.0 + jnp.exp(-(wave - 4000.0) / 80.0))
    lum = continuum * break_4000
    return sed_state._replace(
        intrinsic_lum_lsun_per_a=lum,
        stellar_lum_lsun_per_a=lum,
        stellar_young_lum_lsun_per_a=jnp.zeros_like(lum),
        stellar_old_lum_lsun_per_a=lum,
        total_lum_lsun_per_a=lum,
    )


@dataclass(frozen=True)
class DSPSStellarState:
    ssp_wave_a: np.ndarray
    ssp_lgmet: np.ndarray
    ssp_lg_age_gyr: np.ndarray
    ssp_flux: np.ndarray
    z_sun: float
    metallicity_parameter: str
    metallicity_scatter: float
    separation_age_myr: float


def dsps_stellar_module(
    ssp_data,
    z_sun: float = 0.0142,
    metallicity_parameter: str = "logzsol",
    metallicity_scatter: float = 0.05,
    separation_age_myr: float = 10.0,
) -> ModuleSpec:
    """DSPS CSP stellar module using continuous metallicity.

    The module evaluates the CSP twice: once for stars younger than
    ``separation_age_myr`` and once for older stars.  This mirrors the
    young/old bookkeeping used by CIGALE dust attenuation modules and keeps the
    later dust calculation explicit.
    """

    require_dsps()
    return ModuleSpec(
        name="dsps_stellar",
        setup_fn=lambda config: DSPSStellarState(
            ssp_wave_a=np.asarray(ssp_data.ssp_wave, dtype=float),
            ssp_lgmet=np.asarray(ssp_data.ssp_lgmet, dtype=float),
            ssp_lg_age_gyr=np.asarray(ssp_data.ssp_lg_age_gyr, dtype=float),
            ssp_flux=np.asarray(ssp_data.ssp_flux, dtype=float),
            z_sun=float(z_sun),
            metallicity_parameter=metallicity_parameter,
            metallicity_scatter=float(metallicity_scatter),
            separation_age_myr=float(separation_age_myr),
        ),
        apply_fn=_apply_dsps_stellar,
    )


def _apply_dsps_stellar(params: Mapping[str, object], sed_state: SEDState, state: DSPSStellarState) -> SEDState:
    _, jnp = require_jax()

    z = params["z"]
    t_obs = flat_lcdm_age_gyr(z)
    stellar_age_gyr = t_obs - sed_state.sfh_time_gyr
    separation_age_gyr = jnp.asarray(state.separation_age_myr / 1000.0)
    young_mask = stellar_age_gyr <= separation_age_gyr
    sfr_young = jnp.where(young_mask, sed_state.sfr_msun_per_yr, 0.0)
    sfr_old = jnp.where(young_mask, 0.0, sed_state.sfr_msun_per_yr)

    young_l_lambda_lsun_per_a = _evaluate_dsps_luminosity_per_a(params, sed_state, state, sfr_young, t_obs)
    old_l_lambda_lsun_per_a = _evaluate_dsps_luminosity_per_a(params, sed_state, state, sfr_old, t_obs)
    l_lambda_lsun_per_a = young_l_lambda_lsun_per_a + old_l_lambda_lsun_per_a
    return sed_state._replace(
        intrinsic_lum_lsun_per_a=l_lambda_lsun_per_a,
        stellar_lum_lsun_per_a=l_lambda_lsun_per_a,
        stellar_young_lum_lsun_per_a=young_l_lambda_lsun_per_a,
        stellar_old_lum_lsun_per_a=old_l_lambda_lsun_per_a,
        total_lum_lsun_per_a=l_lambda_lsun_per_a,
    )


def _evaluate_dsps_luminosity_per_a(
    params: Mapping[str, object],
    sed_state: SEDState,
    state: DSPSStellarState,
    sfr_msun_per_yr,
    t_obs,
):
    _, jnp = require_jax()
    from dsps import calc_rest_sed_sfh_table_lognormal_mdf

    logzsol = params[state.metallicity_parameter]
    gal_lgmet = jnp.log10(state.z_sun) + logzsol
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf(
        sed_state.sfh_time_gyr,
        sfr_msun_per_yr,
        gal_lgmet,
        state.metallicity_scatter,
        jnp.asarray(state.ssp_lgmet),
        jnp.asarray(state.ssp_lg_age_gyr),
        jnp.asarray(state.ssp_flux),
        t_obs,
    )
    # DSPS rest_sed convention follows Lsun/Hz for FSPS-like SSP data.
    lnu_lsun_per_hz = jnp.interp(sed_state.wave_rest_a, jnp.asarray(state.ssp_wave_a), sed_info.rest_sed)
    return lnu_lsun_per_hz * C_A_PER_S / sed_state.wave_rest_a**2


@dataclass(frozen=True)
class NoNebularState:
    q_h_parameter: str


def no_nebular_module(q_h_parameter: str = "q_h") -> ModuleSpec:
    """Zero nebular module with the same graph slot as a future emulator."""

    return ModuleSpec(
        name="no_nebular",
        setup_fn=lambda config: NoNebularState(q_h_parameter),
        apply_fn=_apply_no_nebular,
    )


def _apply_no_nebular(params: Mapping[str, object], sed_state: SEDState, state: NoNebularState) -> SEDState:
    del params, state
    _, jnp = require_jax()
    zeros = jnp.zeros_like(sed_state.wave_rest_a)
    total = sed_state.stellar_lum_lsun_per_a
    return sed_state._replace(
        nebular_lum_lsun_per_a=zeros,
        nebular_continuum_lum_lsun_per_a=zeros,
        nebular_line_lum_lsun_per_a=zeros,
        total_lum_lsun_per_a=total,
    )


@dataclass(frozen=True)
class NebularEmulatorState:
    emulator_apply: object
    parameter_names: tuple[str, ...]


def nebular_emulator_module(emulator_apply, parameter_names=("logu", "density", "f_esc", "logzsol")) -> ModuleSpec:
    """Adapter for a future Cue/CLOUDY emulator."""

    return ModuleSpec(
        name="nebular_emulator",
        setup_fn=lambda config: NebularEmulatorState(emulator_apply, tuple(parameter_names)),
        apply_fn=_apply_nebular_emulator,
    )


def _apply_nebular_emulator(params: Mapping[str, object], sed_state: SEDState, state: NebularEmulatorState) -> SEDState:
    args = {name: params[name] for name in state.parameter_names if name in params}
    continuum, lines = state.emulator_apply(sed_state.wave_rest_a, **args)
    neb = continuum + lines
    return sed_state._replace(
        nebular_lum_lsun_per_a=neb,
        nebular_continuum_lum_lsun_per_a=continuum,
        nebular_line_lum_lsun_per_a=lines,
        total_lum_lsun_per_a=sed_state.stellar_lum_lsun_per_a + neb,
    )


@dataclass(frozen=True)
class CalzettiState:
    av_parameter: str
    slope_parameter: str | None
    bump_amplitude_parameter: str | None
    rv: float


def calzetti_attenuation_module(
    av_parameter: str = "dust2",
    slope_parameter: str | None = "dust_slope",
    bump_amplitude_parameter: str | None = "uv_bump",
    rv: float = 4.05,
) -> ModuleSpec:
    """Calzetti-like attenuation with optional slope and UV bump parameters."""

    return ModuleSpec(
        name="calzetti_attenuation",
        setup_fn=lambda config: CalzettiState(av_parameter, slope_parameter, bump_amplitude_parameter, float(rv)),
        apply_fn=_apply_calzetti,
    )


def _apply_calzetti(params: Mapping[str, object], sed_state: SEDState, state: CalzettiState) -> SEDState:
    _, jnp = require_jax()
    wave_um = sed_state.wave_rest_a / 1.0e4
    k_short = 2.659 * (-2.156 + 1.509 / wave_um - 0.198 / wave_um**2 + 0.011 / wave_um**3) + state.rv
    k_long = 2.659 * (-1.857 + 1.040 / wave_um) + state.rv
    k_lambda = jnp.where(wave_um < 0.63, k_short, k_long)
    slope = params[state.slope_parameter] if state.slope_parameter is not None and state.slope_parameter in params else 0.0
    bump = (
        params[state.bump_amplitude_parameter]
        if state.bump_amplitude_parameter is not None and state.bump_amplitude_parameter in params
        else 0.0
    )
    drude = bump * (wave_um**2 * 0.035**2) / ((wave_um**2 - 0.2175**2) ** 2 + wave_um**2 * 0.035**2)
    a_lambda = params[state.av_parameter] * (k_lambda / state.rv) * (wave_um / 0.55) ** slope + drude
    transmission = 10.0 ** (-0.4 * jnp.maximum(a_lambda, 0.0))
    before = sed_state.total_lum_lsun_per_a
    after = before * transmission
    absorbed = trapz_jax(jnp.maximum(before - after, 0.0), sed_state.wave_rest_a)
    return sed_state._replace(attenuated_lum_lsun_per_a=after, total_lum_lsun_per_a=after, absorbed_lum_lsun=absorbed)


@dataclass(frozen=True)
class ModifiedStarburstAttenuationState:
    ebv_young_parameter: str
    ebv_old_factor_parameter: str
    powerlaw_slope_parameter: str | None
    uv_bump_amplitude_parameter: str | None
    nebular_ebv_parameter: str | None
    nebular_extinction_law: str
    nebular_rv: float
    uv_bump_wavelength_nm: float
    uv_bump_width_nm: float


def modified_starburst_attenuation_module(
    ebv_young_parameter: str = "E_BV_young",
    ebv_old_factor_parameter: str = "E_BV_old_factor",
    powerlaw_slope_parameter: str | None = "powerlaw_slope",
    uv_bump_amplitude_parameter: str | None = "uv_bump_amplitude",
    nebular_ebv_parameter: str | None = None,
    nebular_extinction_law: str = "mw_ccm89",
    nebular_rv: float = 3.1,
    uv_bump_wavelength_nm: float = 217.5,
    uv_bump_width_nm: float = 35.0,
) -> ModuleSpec:
    """CIGALE-like modified starburst attenuation with young/old splitting.

    The module uses the Calzetti+Leitherer continuum curve modified by a UV
    bump and power-law slope.  It attenuates the young and old stellar spectra
    separately:

    ``E(B-V)_old = E_BV_old_factor * E(B-V)_young``.

    Nebular emission is attenuated with ``nebular_ebv_parameter`` when supplied;
    otherwise it uses the young-star colour excess.  The nebular curve is a
    CIGALE-style emission-line extinction law evaluated at each wavelength grid
    point. Since Cue currently broadens lines onto the spectrum grid before the
    dust step, this is equivalent to line-by-line attenuation for narrow line
    profiles whose width is small compared with variations in the extinction
    curve.
    """

    allowed_laws = {"mw_ccm89", "smc_pei92", "lmc_pei92"}
    if nebular_extinction_law not in allowed_laws:
        raise ValueError(f"nebular_extinction_law must be one of {sorted(allowed_laws)}.")
    return ModuleSpec(
        name="modified_starburst_attenuation",
        setup_fn=lambda config: ModifiedStarburstAttenuationState(
            ebv_young_parameter=ebv_young_parameter,
            ebv_old_factor_parameter=ebv_old_factor_parameter,
            powerlaw_slope_parameter=powerlaw_slope_parameter,
            uv_bump_amplitude_parameter=uv_bump_amplitude_parameter,
            nebular_ebv_parameter=nebular_ebv_parameter,
            nebular_extinction_law=nebular_extinction_law,
            nebular_rv=float(nebular_rv),
            uv_bump_wavelength_nm=float(uv_bump_wavelength_nm),
            uv_bump_width_nm=float(uv_bump_width_nm),
        ),
        apply_fn=_apply_modified_starburst_attenuation,
    )


def _apply_modified_starburst_attenuation(
    params: Mapping[str, object],
    sed_state: SEDState,
    state: ModifiedStarburstAttenuationState,
) -> SEDState:
    _, jnp = require_jax()

    ebv_young = jnp.maximum(params[state.ebv_young_parameter], 0.0)
    old_factor = jnp.clip(params[state.ebv_old_factor_parameter], 0.0, 1.0)
    ebv_old = ebv_young * old_factor
    if state.nebular_ebv_parameter is not None and state.nebular_ebv_parameter in params:
        ebv_nebular = jnp.maximum(params[state.nebular_ebv_parameter], 0.0)
    else:
        ebv_nebular = ebv_young
    powerlaw_slope = (
        params[state.powerlaw_slope_parameter]
        if state.powerlaw_slope_parameter is not None and state.powerlaw_slope_parameter in params
        else 0.0
    )
    bump_amplitude = (
        params[state.uv_bump_amplitude_parameter]
        if state.uv_bump_amplitude_parameter is not None and state.uv_bump_amplitude_parameter in params
        else 0.0
    )
    stellar_a_over_ebv = _modified_starburst_a_over_ebv(
        sed_state.wave_rest_a,
        bump_wave_nm=state.uv_bump_wavelength_nm,
        bump_width_nm=state.uv_bump_width_nm,
        bump_amplitude=bump_amplitude,
        powerlaw_slope=powerlaw_slope,
    )
    nebular_a_over_ebv = _nebular_a_over_ebv(sed_state.wave_rest_a, law=state.nebular_extinction_law, rv=state.nebular_rv)

    stellar_split = sed_state.stellar_young_lum_lsun_per_a + sed_state.stellar_old_lum_lsun_per_a
    has_split = jnp.sum(stellar_split) > 0.0
    young = jnp.where(has_split, sed_state.stellar_young_lum_lsun_per_a, jnp.zeros_like(sed_state.stellar_lum_lsun_per_a))
    old = jnp.where(has_split, sed_state.stellar_old_lum_lsun_per_a, sed_state.stellar_lum_lsun_per_a)
    nebular_split = sed_state.nebular_continuum_lum_lsun_per_a + sed_state.nebular_line_lum_lsun_per_a
    has_nebular_split = jnp.sum(nebular_split) > 0.0
    nebular_continuum = jnp.where(
        has_nebular_split,
        sed_state.nebular_continuum_lum_lsun_per_a,
        sed_state.nebular_lum_lsun_per_a,
    )
    nebular_lines = jnp.where(
        has_nebular_split,
        sed_state.nebular_line_lum_lsun_per_a,
        jnp.zeros_like(sed_state.nebular_lum_lsun_per_a),
    )
    nebular = nebular_continuum + nebular_lines
    other = sed_state.total_lum_lsun_per_a - sed_state.stellar_lum_lsun_per_a - nebular

    trans_young = 10.0 ** (-0.4 * ebv_young * stellar_a_over_ebv)
    trans_old = 10.0 ** (-0.4 * ebv_old * stellar_a_over_ebv)
    trans_nebular = 10.0 ** (-0.4 * ebv_nebular * nebular_a_over_ebv)

    attenuated_young = young * trans_young
    attenuated_old = old * trans_old
    attenuated_nebular_continuum = nebular_continuum * trans_nebular
    attenuated_nebular_lines = nebular_lines * trans_nebular
    attenuated_nebular = attenuated_nebular_continuum + attenuated_nebular_lines
    after = attenuated_young + attenuated_old + attenuated_nebular + other
    before = sed_state.total_lum_lsun_per_a
    absorbed = trapz_jax(jnp.maximum(before - after, 0.0), sed_state.wave_rest_a)
    return sed_state._replace(
        stellar_young_lum_lsun_per_a=attenuated_young,
        stellar_old_lum_lsun_per_a=attenuated_old,
        stellar_lum_lsun_per_a=attenuated_young + attenuated_old,
        nebular_continuum_lum_lsun_per_a=attenuated_nebular_continuum,
        nebular_line_lum_lsun_per_a=attenuated_nebular_lines,
        nebular_lum_lsun_per_a=attenuated_nebular,
        attenuated_lum_lsun_per_a=after,
        total_lum_lsun_per_a=after,
        absorbed_lum_lsun=absorbed,
    )


@dataclass(frozen=True)
class Gordon16RvFaExtinctionState:
    av_parameter: str
    rv_parameter: str
    fa_parameter: str
    old_av_factor_parameter: str | None
    nebular_av_parameter: str | None
    apply_to_nebular: bool


def gordon16_rvfa_extinction_module(
    av_parameter: str = "A_V",
    rv_parameter: str = "R_V",
    fa_parameter: str = "f_A",
    old_av_factor_parameter: str | None = None,
    nebular_av_parameter: str | None = None,
    apply_to_nebular: bool = True,
) -> ModuleSpec:
    """Gordon et al. (2016) BEAST ``R(V), f_A`` extinction screen.

    This is the original BEAST dust family: a linear mixture of a Milky-Way
    Fitzpatrick99 component and a Gordon03 SMC-Bar component.  The public
    parameters follow the BEAST convention:

    ``A_V``
        Extinction amplitude in magnitudes.
    ``R_V``
        Total mixture value of ``A_V / E(B-V)``.
    ``f_A``
        Fraction of the Milky-Way-like component.  ``f_A=1`` is pure
        Fitzpatrick99; ``f_A=0`` is pure SMC Bar.

    Internally the module derives ``R(V)_A`` for the Milky-Way component so
    that the mixture has the requested total ``R_V``.  By default the same
    screen is applied to all stellar light.  If ``old_av_factor_parameter`` is
    supplied and the stellar young/old split is populated, young stars use
    ``A_V`` and old stars use ``A_V_old_factor * A_V``.  Nebular emission is
    attenuated by the same law when ``apply_to_nebular`` is true; a separate
    ``nebular_av_parameter`` may be supplied.
    """

    return ModuleSpec(
        name="gordon16_rvfa_extinction",
        setup_fn=lambda config: Gordon16RvFaExtinctionState(
            av_parameter=av_parameter,
            rv_parameter=rv_parameter,
            fa_parameter=fa_parameter,
            old_av_factor_parameter=old_av_factor_parameter,
            nebular_av_parameter=nebular_av_parameter,
            apply_to_nebular=bool(apply_to_nebular),
        ),
        apply_fn=_apply_gordon16_rvfa_extinction,
    )


def _apply_gordon16_rvfa_extinction(
    params: Mapping[str, object],
    sed_state: SEDState,
    state: Gordon16RvFaExtinctionState,
) -> SEDState:
    _, jnp = require_jax()

    av = jnp.maximum(params[state.av_parameter], 0.0)
    rv = jnp.maximum(params[state.rv_parameter], 1.0e-6)
    f_a = jnp.clip(params[state.fa_parameter], 0.0, 1.0)
    a_over_av = _gordon16_rvfa_a_over_av(sed_state.wave_rest_a, rv=rv, f_a=f_a)

    stellar_split = sed_state.stellar_young_lum_lsun_per_a + sed_state.stellar_old_lum_lsun_per_a
    has_split = jnp.sum(stellar_split) > 0.0
    young = jnp.where(has_split, sed_state.stellar_young_lum_lsun_per_a, jnp.zeros_like(sed_state.stellar_lum_lsun_per_a))
    old = jnp.where(has_split, sed_state.stellar_old_lum_lsun_per_a, sed_state.stellar_lum_lsun_per_a)

    if state.old_av_factor_parameter is not None and state.old_av_factor_parameter in params:
        old_factor = jnp.clip(params[state.old_av_factor_parameter], 0.0, 1.0)
    else:
        old_factor = 1.0
    av_young = av
    av_old = av * old_factor

    nebular_split = sed_state.nebular_continuum_lum_lsun_per_a + sed_state.nebular_line_lum_lsun_per_a
    has_nebular_split = jnp.sum(nebular_split) > 0.0
    nebular_continuum = jnp.where(
        has_nebular_split,
        sed_state.nebular_continuum_lum_lsun_per_a,
        sed_state.nebular_lum_lsun_per_a,
    )
    nebular_lines = jnp.where(
        has_nebular_split,
        sed_state.nebular_line_lum_lsun_per_a,
        jnp.zeros_like(sed_state.nebular_lum_lsun_per_a),
    )
    nebular = nebular_continuum + nebular_lines
    other = sed_state.total_lum_lsun_per_a - sed_state.stellar_lum_lsun_per_a - nebular

    trans_young = 10.0 ** (-0.4 * av_young * a_over_av)
    trans_old = 10.0 ** (-0.4 * av_old * a_over_av)
    if state.nebular_av_parameter is not None and state.nebular_av_parameter in params:
        av_nebular = jnp.maximum(params[state.nebular_av_parameter], 0.0)
    else:
        av_nebular = av
    trans_nebular = jnp.where(
        state.apply_to_nebular,
        10.0 ** (-0.4 * av_nebular * a_over_av),
        jnp.ones_like(a_over_av),
    )

    attenuated_young = young * trans_young
    attenuated_old = old * trans_old
    attenuated_nebular_continuum = nebular_continuum * trans_nebular
    attenuated_nebular_lines = nebular_lines * trans_nebular
    attenuated_nebular = attenuated_nebular_continuum + attenuated_nebular_lines
    after = attenuated_young + attenuated_old + attenuated_nebular + other
    before = sed_state.total_lum_lsun_per_a
    absorbed = trapz_jax(jnp.maximum(before - after, 0.0), sed_state.wave_rest_a)
    return sed_state._replace(
        stellar_young_lum_lsun_per_a=attenuated_young,
        stellar_old_lum_lsun_per_a=attenuated_old,
        stellar_lum_lsun_per_a=attenuated_young + attenuated_old,
        nebular_continuum_lum_lsun_per_a=attenuated_nebular_continuum,
        nebular_line_lum_lsun_per_a=attenuated_nebular_lines,
        nebular_lum_lsun_per_a=attenuated_nebular,
        attenuated_lum_lsun_per_a=after,
        total_lum_lsun_per_a=after,
        absorbed_lum_lsun=absorbed,
    )


@dataclass(frozen=True)
class SMCScreenAttenuationState:
    av_parameter: str
    old_av_factor_parameter: str | None
    nebular_av_parameter: str | None
    apply_to_nebular: bool


def smc_screen_attenuation_module(
    av_parameter: str = "A_V",
    old_av_factor_parameter: str | None = None,
    nebular_av_parameter: str | None = None,
    apply_to_nebular: bool = True,
) -> ModuleSpec:
    """Gordon03 SMC-Bar screen attenuation.

    This is intentionally simpler than the CIGALE modified-starburst law: one
    SMC-Bar curve, parameterized by ``A_V``.  If the stellar module populated
    young/old spectra, young stars receive ``A_V`` and old stars receive
    ``A_V_old_factor * A_V`` when ``old_av_factor_parameter`` is supplied.
    Nebular emission receives either ``A_V`` or a separate ``nebular_A_V``.
    """

    return ModuleSpec(
        name="smc_screen_attenuation",
        setup_fn=lambda config: SMCScreenAttenuationState(
            av_parameter=av_parameter,
            old_av_factor_parameter=old_av_factor_parameter,
            nebular_av_parameter=nebular_av_parameter,
            apply_to_nebular=bool(apply_to_nebular),
        ),
        apply_fn=_apply_smc_screen_attenuation,
    )


def _apply_smc_screen_attenuation(
    params: Mapping[str, object],
    sed_state: SEDState,
    state: SMCScreenAttenuationState,
) -> SEDState:
    _, jnp = require_jax()

    av = jnp.maximum(params[state.av_parameter], 0.0)
    a_over_av = _gordon03_smcbar_a_over_av(sed_state.wave_rest_a)

    stellar_split = sed_state.stellar_young_lum_lsun_per_a + sed_state.stellar_old_lum_lsun_per_a
    has_split = jnp.sum(stellar_split) > 0.0
    young = jnp.where(has_split, sed_state.stellar_young_lum_lsun_per_a, jnp.zeros_like(sed_state.stellar_lum_lsun_per_a))
    old = jnp.where(has_split, sed_state.stellar_old_lum_lsun_per_a, sed_state.stellar_lum_lsun_per_a)

    if state.old_av_factor_parameter is not None and state.old_av_factor_parameter in params:
        old_factor = jnp.clip(params[state.old_av_factor_parameter], 0.0, 1.0)
    else:
        old_factor = 1.0

    nebular_split = sed_state.nebular_continuum_lum_lsun_per_a + sed_state.nebular_line_lum_lsun_per_a
    has_nebular_split = jnp.sum(nebular_split) > 0.0
    nebular_continuum = jnp.where(
        has_nebular_split,
        sed_state.nebular_continuum_lum_lsun_per_a,
        sed_state.nebular_lum_lsun_per_a,
    )
    nebular_lines = jnp.where(
        has_nebular_split,
        sed_state.nebular_line_lum_lsun_per_a,
        jnp.zeros_like(sed_state.nebular_lum_lsun_per_a),
    )
    nebular = nebular_continuum + nebular_lines
    other = sed_state.total_lum_lsun_per_a - sed_state.stellar_lum_lsun_per_a - nebular

    if state.nebular_av_parameter is not None and state.nebular_av_parameter in params:
        av_nebular = jnp.maximum(params[state.nebular_av_parameter], 0.0)
    else:
        av_nebular = av

    trans_young = 10.0 ** (-0.4 * av * a_over_av)
    trans_old = 10.0 ** (-0.4 * av * old_factor * a_over_av)
    trans_nebular = jnp.where(
        state.apply_to_nebular,
        10.0 ** (-0.4 * av_nebular * a_over_av),
        jnp.ones_like(a_over_av),
    )

    attenuated_young = young * trans_young
    attenuated_old = old * trans_old
    attenuated_nebular_continuum = nebular_continuum * trans_nebular
    attenuated_nebular_lines = nebular_lines * trans_nebular
    attenuated_nebular = attenuated_nebular_continuum + attenuated_nebular_lines
    after = attenuated_young + attenuated_old + attenuated_nebular + other
    before = sed_state.total_lum_lsun_per_a
    absorbed = trapz_jax(jnp.maximum(before - after, 0.0), sed_state.wave_rest_a)
    return sed_state._replace(
        stellar_young_lum_lsun_per_a=attenuated_young,
        stellar_old_lum_lsun_per_a=attenuated_old,
        stellar_lum_lsun_per_a=attenuated_young + attenuated_old,
        nebular_continuum_lum_lsun_per_a=attenuated_nebular_continuum,
        nebular_line_lum_lsun_per_a=attenuated_nebular_lines,
        nebular_lum_lsun_per_a=attenuated_nebular,
        attenuated_lum_lsun_per_a=after,
        total_lum_lsun_per_a=after,
        absorbed_lum_lsun=absorbed,
    )


def _gordon16_rvfa_a_over_av(wave_a, *, rv, f_a):
    """BEAST Gordon16 ``R(V), f_A`` law as ``A(lambda) / A(V)``.

    The curve is valid over the component-law range
    ``0.3 <= 1/lambda[um] <= 10``.  Outside that range we return the closest
    boundary value by clipping the wavenumber.  This keeps the JAX graph finite;
    scientific runs should use wavelength grids covering the intended domain.
    """

    _, jnp = require_jax()
    f_a = jnp.clip(f_a, 0.0, 1.0)
    rv_a = _gordon16_rv_a_from_mixture_rv(rv, f_a)
    mw = _fitzpatrick99_a_over_av(wave_a, rv=rv_a)
    smc = _gordon03_smcbar_a_over_av(wave_a)
    return jnp.maximum(f_a * mw + (1.0 - f_a) * smc, 0.0)


def _gordon16_rv_a_from_mixture_rv(rv, f_a):
    """Milky-Way component ``R(V)_A`` implied by mixture ``R_V`` and ``f_A``.

    The scientific prior should keep the requested ``R_V, f_A`` pair in the
    physical domain.  The final clip is only a numerical guard matching the
    usual Gordon16/Fitzpatrick99 support for ``R(V)_A``.
    """

    _, jnp = require_jax()
    rv_b = jnp.asarray(2.74)
    rv = jnp.maximum(rv, 1.0e-6)
    f_a = jnp.clip(f_a, 0.0, 1.0)
    denom = 1.0 / (rv * jnp.maximum(f_a, 1.0e-12)) - (1.0 - f_a) / (jnp.maximum(f_a, 1.0e-12) * rv_b)
    rv_a = jnp.where(f_a > 0.0, 1.0 / jnp.maximum(denom, 1.0e-12), 3.1)
    return jnp.clip(rv_a, 2.0, 6.0)


def _gordon16_mixture_rv_from_rv_a(rv_a, f_a):
    """Total mixture ``R_V`` from component ``R(V)_A`` and ``f_A``."""

    _, jnp = require_jax()
    rv_b = jnp.asarray(2.74)
    rv_a = jnp.maximum(rv_a, 1.0e-6)
    f_a = jnp.clip(f_a, 0.0, 1.0)
    return 1.0 / (f_a / rv_a + (1.0 - f_a) / rv_b)


def _fitzpatrick99_a_over_av(wave_a, *, rv):
    """Fitzpatrick99 Milky-Way curve, BEAST convention, returning A(lambda)/A(V)."""

    _, jnp = require_jax()
    rv = jnp.maximum(rv, 1.0e-6)
    x = _safe_inverse_micron(wave_a)

    c2 = -0.824 + 4.717 / rv
    c1 = 2.030 - 3.007 * c2
    c3 = 3.23
    c4 = 0.41
    x0 = 4.596
    gamma = 0.99
    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / jnp.asarray([2700.0, 2600.0])

    uv = c1 + c2 * x + c3 * x**2 / ((x**2 - x0**2) ** 2 + gamma**2 * x**2)
    uv = uv + jnp.where(x >= 5.9, c4 * (0.5392 * (x - 5.9) ** 2 + 0.05644 * (x - 5.9) ** 3), 0.0)
    uv = uv + rv

    yspluv = c1 + c2 * xspluv + c3 * xspluv**2 / ((xspluv**2 - x0**2) ** 2 + gamma**2 * xspluv**2) + rv
    xsplopir = jnp.asarray([0.0, 10000.0 / 26500.0, 10000.0 / 12200.0, 10000.0 / 6000.0, 10000.0 / 5470.0, 10000.0 / 4670.0, 10000.0 / 4110.0])
    y_ir0 = jnp.asarray([0.0, 0.26469, 0.82925]) * rv / 3.1
    y_ir1 = jnp.asarray(
        [
            2.13572e-04 * rv**2 + 1.00270 * rv - 4.22809e-01,
            -7.35778e-05 * rv**2 + 1.00216 * rv - 5.13540e-02,
            -3.32598e-05 * rv**2 + 1.00184 * rv + 7.00127e-01,
            1.19456 + 1.01707 * rv - 5.46959e-03 * rv**2 + 7.97809e-04 * rv**3 - 4.45636e-05 * rv**4,
        ]
    )
    xspline = jnp.concatenate([xsplopir, xspluv])
    yspline = jnp.concatenate([y_ir0, y_ir1, yspluv])
    optical_ir = _not_a_knot_cubic_spline(xspline, yspline, x)
    return jnp.where(x >= xcutuv, uv / rv, optical_ir / rv)


def _gordon03_smcbar_a_over_av(wave_a):
    """Gordon03 SMC-Bar average curve, BEAST convention, returning A(lambda)/A(V)."""

    _, jnp = require_jax()
    x = _safe_inverse_micron(wave_a)

    rv = 2.74
    c1 = -4.959 / rv
    c2 = 2.264 / rv
    c3 = 0.389 / rv
    c4 = 0.461 / rv
    x0 = 4.6
    gamma = 1.0
    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / jnp.asarray([2700.0, 2600.0])

    uv = 1.0 + c1 + c2 * x + c3 * x**2 / ((x**2 - x0**2) ** 2 + gamma**2 * x**2)
    uv = uv + jnp.where(x >= 5.9, c4 * (0.5392 * (x - 5.9) ** 2 + 0.05644 * (x - 5.9) ** 3), 0.0)

    yspluv = 1.0 + c1 + c2 * xspluv + c3 * xspluv**2 / ((xspluv**2 - x0**2) ** 2 + gamma**2 * xspluv**2)
    xsplopir = jnp.asarray([0.0, 1.0 / 2.198, 1.0 / 1.65, 1.0 / 1.25, 1.0 / 0.81, 1.0 / 0.65, 1.0 / 0.55, 1.0 / 0.44, 1.0 / 0.37])
    ysplopir = jnp.asarray([0.0, 0.11, 0.169, 0.25, 0.567, 0.801, 1.00, 1.374, 1.672])
    xspline = jnp.concatenate([xsplopir, xspluv])
    yspline = jnp.concatenate([ysplopir, yspluv])
    optical_ir = _not_a_knot_cubic_spline(xspline, yspline, x)
    return jnp.where(x >= xcutuv, uv, optical_ir)


def _safe_inverse_micron(wave_a):
    _, jnp = require_jax()
    wave = jnp.maximum(wave_a, 1.0e-6)
    x = 1.0e4 / wave
    return jnp.clip(x, 0.3, 10.0)


def _not_a_knot_cubic_spline(x_nodes, y_nodes, x_eval):
    """Small JAX cubic spline matching SciPy's default not-a-knot boundary."""

    _, jnp = require_jax()
    n = int(x_nodes.shape[0])
    h = x_nodes[1:] - x_nodes[:-1]
    matrix = jnp.zeros((n, n), dtype=y_nodes.dtype)
    rhs = jnp.zeros((n,), dtype=y_nodes.dtype)

    matrix = matrix.at[0, 0].set(-h[1])
    matrix = matrix.at[0, 1].set(h[0] + h[1])
    matrix = matrix.at[0, 2].set(-h[0])
    matrix = matrix.at[n - 1, n - 3].set(h[n - 2])
    matrix = matrix.at[n - 1, n - 2].set(-(h[n - 3] + h[n - 2]))
    matrix = matrix.at[n - 1, n - 1].set(h[n - 3])

    for i in range(1, n - 1):
        matrix = matrix.at[i, i - 1].set(h[i - 1])
        matrix = matrix.at[i, i].set(2.0 * (h[i - 1] + h[i]))
        matrix = matrix.at[i, i + 1].set(h[i])
        rhs = rhs.at[i].set(6.0 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1]))

    second = jnp.linalg.solve(matrix, rhs)
    interval = jnp.clip(jnp.searchsorted(x_nodes, x_eval, side="right") - 1, 0, n - 2)
    x0 = x_nodes[interval]
    x1 = x_nodes[interval + 1]
    y0 = y_nodes[interval]
    y1 = y_nodes[interval + 1]
    m0 = second[interval]
    m1 = second[interval + 1]
    hi = x1 - x0
    left = x1 - x_eval
    right = x_eval - x0
    return (
        m0 * left**3 / (6.0 * hi)
        + m1 * right**3 / (6.0 * hi)
        + (y0 - m0 * hi**2 / 6.0) * left / hi
        + (y1 - m1 * hi**2 / 6.0) * right / hi
    )


def _modified_starburst_a_over_ebv(
    wave_a,
    *,
    bump_wave_nm: float,
    bump_width_nm: float,
    bump_amplitude,
    powerlaw_slope,
):
    """JAX version of CIGALE's Calzetti+Leitherer modified starburst curve."""

    _, jnp = require_jax()
    wave_nm = wave_a / 10.0
    wave_safe = jnp.maximum(wave_nm, 1.0e-6)
    k_leitherer = 5.472 + 671.0 / wave_safe - 9218.0 / wave_safe**2 + 2.620e6 / wave_safe**3
    k_calz_short = 2.659 * (-2.156 + 1509.0 / wave_safe - 198000.0 / wave_safe**2 + 1.1e7 / wave_safe**3) + 4.05
    k_calz_long = 2.659 * (-1.857 + 1040.0 / wave_safe) + 4.05
    k_calz = jnp.where(wave_safe < 630.0, k_calz_short, k_calz_long)
    attenuation = jnp.where((wave_safe > 91.2) & (wave_safe < 150.0), k_leitherer, 0.0)
    attenuation = jnp.where(wave_safe >= 150.0, k_calz, attenuation)
    attenuation = jnp.maximum(attenuation, 0.0)
    attenuation = attenuation * (wave_safe / 550.0) ** powerlaw_slope
    attenuation = attenuation + _uv_bump_drude_nm(wave_safe, bump_wave_nm, bump_width_nm, bump_amplitude)

    wl_bv = jnp.asarray([440.0, 550.0])
    ebv_calz = _calzetti_k_nm(wl_bv) + _uv_bump_drude_nm(wl_bv, bump_wave_nm, bump_width_nm, bump_amplitude)
    ebv = _calzetti_k_nm(wl_bv) * (wl_bv / 550.0) ** powerlaw_slope
    ebv = ebv + _uv_bump_drude_nm(wl_bv, bump_wave_nm, bump_width_nm, bump_amplitude)
    ebv_delta = ebv[1] - ebv[0]
    ebv_delta_sign = jnp.where(ebv_delta >= 0.0, 1.0, -1.0)
    ebv_delta = jnp.where(jnp.abs(ebv_delta) > 1.0e-12, ebv_delta, ebv_delta_sign * 1.0e-12)
    correction = (ebv_calz[1] - ebv_calz[0]) / ebv_delta
    return jnp.maximum(attenuation * correction, 0.0)


def _calzetti_k_nm(wave_nm):
    _, jnp = require_jax()
    wave_safe = jnp.maximum(wave_nm, 1.0e-6)
    k_short = 2.659 * (-2.156 + 1509.0 / wave_safe - 198000.0 / wave_safe**2 + 1.1e7 / wave_safe**3) + 4.05
    k_long = 2.659 * (-1.857 + 1040.0 / wave_safe) + 4.05
    return jnp.where(wave_safe < 630.0, k_short, k_long)


def _uv_bump_drude_nm(wave_nm, central_nm: float, width_nm: float, amplitude):
    _, jnp = require_jax()
    central = jnp.asarray(central_nm)
    width = jnp.asarray(width_nm)
    return amplitude * wave_nm**2 * width**2 / ((wave_nm**2 - central**2) ** 2 + wave_nm**2 * width**2)


def _nebular_a_over_ebv(wave_a, *, law: str, rv: float):
    """Emission-line attenuation curve A(lambda)/E(B-V) on the model grid."""

    wave_nm = wave_a / 10.0
    if law == "mw_ccm89":
        return _ccm89_a_over_ebv_nm(wave_nm, rv=rv)
    if law == "smc_pei92":
        return _pei92_a_over_ebv_nm(wave_nm, law="smc")
    if law == "lmc_pei92":
        return _pei92_a_over_ebv_nm(wave_nm, law="lmc")
    raise ValueError(f"Unsupported nebular extinction law: {law!r}")


def _ccm89_a_over_ebv_nm(wave_nm, *, rv: float = 3.1):
    """JAX port of CIGALE's CCM89 emission-line extinction curve."""

    _, jnp = require_jax()
    x = 1.0e3 / jnp.maximum(wave_nm, 1.0e-6)
    rv = jnp.asarray(rv)

    y = x - 1.82
    cond1 = x < 1.1
    cond2 = (x >= 1.1) & (x < 3.3)
    cond3 = (x >= 3.3) & (x < 5.9)
    cond4 = (x >= 5.9) & (x < 8.0)
    cond5 = (x >= 8.0) & (x <= 11.0)

    value1 = rv * 0.574 * x**1.61 - 0.527 * x**1.61
    value2 = rv * _polyval(jnp.asarray([-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0]), y)
    value2 = value2 + _polyval(jnp.asarray([3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0]), y)
    value3 = rv * (1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341))
    value3 = value3 + (-3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263))
    value4 = rv * (
        1.752
        - 0.316 * x
        - 0.104 / ((x - 4.67) ** 2 + 0.341)
        + _polyval(jnp.asarray([-0.009779, -0.04473, 0.0, 0.0]), x - 5.9)
    )
    value4 = value4 + (
        -3.090
        + 1.825 * x
        + 1.206 / ((x - 4.62) ** 2 + 0.263)
        + _polyval(jnp.asarray([0.1207, 0.2130, 0.0, 0.0]), x - 5.9)
    )
    value5 = rv * _polyval(jnp.asarray([-0.070, 0.137, -0.628, -1.073]), x - 8.0)
    value5 = value5 + _polyval(jnp.asarray([0.374, -0.420, 4.257, 13.670]), x - 8.0)

    out = jnp.zeros_like(x)
    out = jnp.where(cond1, value1, out)
    out = jnp.where(cond2, value2, out)
    out = jnp.where(cond3, value3, out)
    out = jnp.where(cond4, value4, out)
    out = jnp.where(cond5, value5, out)
    return jnp.maximum(out, 0.0)


def _pei92_a_over_ebv_nm(wave_nm, *, law: str):
    """JAX port of CIGALE's Pei92 SMC/LMC emission-line extinction curves."""

    _, jnp = require_jax()
    law = law.lower()
    if law == "smc":
        rv = 2.93
        a_coeff = jnp.asarray([185.0, 27.0, 0.005, 0.010, 0.012, 0.03])
        b_coeff = jnp.asarray([90.0, 5.50, -1.95, -1.95, -1.80, 0.0])
        n_coeff = jnp.asarray([2.0, 4.0, 2.0, 2.0, 2.0, 2.0])
    elif law == "lmc":
        rv = 3.16
        a_coeff = jnp.asarray([175.0, 19.0, 0.023, 0.005, 0.006, 0.02])
        b_coeff = jnp.asarray([90.0, 5.5, -1.95, -1.95, -1.8, 0.0])
        n_coeff = jnp.asarray([2.0, 4.5, 2.0, 2.0, 2.0, 2.0])
    elif law == "mw":
        rv = 3.08
        a_coeff = jnp.asarray([165.0, 14.0, 0.045, 0.002, 0.002, 0.012])
        b_coeff = jnp.asarray([90.0, 4.0, -1.95, -1.95, -1.8, 0.0])
        n_coeff = jnp.asarray([2.0, 6.5, 2.0, 2.0, 2.0, 2.0])
    else:
        raise ValueError(f"Unsupported Pei92 law: {law!r}")

    wvl_um = jnp.maximum(wave_nm * 1.0e-3, 1.0e-12)
    wvl_coeff = jnp.asarray([0.046, 0.08, 0.22, 9.7, 18.0, 25.0])
    wvl = wvl_um[:, None]
    terms = a_coeff[None, :] / ((wvl / wvl_coeff[None, :]) ** n_coeff[None, :] + (wvl_coeff[None, :] / wvl) ** n_coeff[None, :] + b_coeff[None, :])
    a_lambda_over_ab = jnp.sum(terms, axis=1)
    a_lambda_over_av = (1.0 / rv + 1.0) * a_lambda_over_ab
    a_lambda_over_ebv = rv * a_lambda_over_av
    in_domain = (wvl_um >= 0.0912) & (wvl_um <= 30.0)
    return jnp.where(in_domain, jnp.maximum(a_lambda_over_ebv, 0.0), 0.0)


def _polyval(coefficients, x):
    _, jnp = require_jax()
    value = jnp.zeros_like(x) + coefficients[0]
    for coefficient in coefficients[1:]:
        value = value * x + coefficient
    return value


@dataclass(frozen=True)
class ModifiedBlackbodyState:
    temperature_parameter: str
    beta_parameter: str
    default_temperature: float
    default_beta: float
    normalization_wavelength_grid_a: np.ndarray | None


def modified_blackbody_dust_module(
    temperature_parameter: str = "dust_temperature",
    beta_parameter: str = "dust_beta",
    default_temperature: float = 35.0,
    default_beta: float = 1.8,
    normalization_wavelength_grid_a: Sequence[float] | None = None,
) -> ModuleSpec:
    """Simple differentiable IR re-emission normalized by absorbed luminosity.

    ``default_temperature`` and ``default_beta`` are used when the corresponding
    names are absent from the parameter dictionary.  This keeps the default
    JAX-CIGALE dust mode energy-balanced without forcing every photometric run
    to fit far-IR dust parameters.  If those parameters are present in the
    ``JaxParameterSpace`` or ``fixed_parameters``, their values override the
    defaults.
    """

    norm_grid = None if normalization_wavelength_grid_a is None else np.asarray(normalization_wavelength_grid_a, dtype=float)
    if norm_grid is not None:
        if norm_grid.ndim != 1 or norm_grid.size < 2 or np.any(np.diff(norm_grid) <= 0.0):
            raise ValueError("normalization_wavelength_grid_a must be a strictly increasing one-dimensional grid.")

    return ModuleSpec(
        name="modified_blackbody_dust",
        setup_fn=lambda config: ModifiedBlackbodyState(
            temperature_parameter=temperature_parameter,
            beta_parameter=beta_parameter,
            default_temperature=float(default_temperature),
            default_beta=float(default_beta),
            normalization_wavelength_grid_a=norm_grid,
        ),
        apply_fn=_apply_modified_blackbody,
    )


def _apply_modified_blackbody(params: Mapping[str, object], sed_state: SEDState, state: ModifiedBlackbodyState) -> SEDState:
    _, jnp = require_jax()
    temperature = jnp.maximum(params.get(state.temperature_parameter, state.default_temperature), 1.0)
    beta = params.get(state.beta_parameter, state.default_beta)
    shape = _modified_blackbody_shape(sed_state.wave_rest_a, temperature, beta)
    if state.normalization_wavelength_grid_a is None:
        norm_wave = sed_state.wave_rest_a
    else:
        norm_wave = jnp.asarray(state.normalization_wavelength_grid_a)
    norm_shape = _modified_blackbody_shape(norm_wave, temperature, beta)
    norm = trapz_jax(norm_shape, norm_wave)
    dust = sed_state.absorbed_lum_lsun * shape / jnp.maximum(norm, 1e-300)
    return sed_state._replace(dust_lum_lsun_per_a=dust, total_lum_lsun_per_a=sed_state.total_lum_lsun_per_a + dust)


@dataclass(frozen=True)
class TwoTemperatureDustState:
    cold_temperature_parameter: str
    warm_temperature_parameter: str
    beta_parameter: str
    warm_fraction_parameter: str


def two_temperature_dust_module(
    cold_temperature_parameter: str = "dust_cold_temperature",
    warm_temperature_parameter: str = "dust_warm_temperature",
    beta_parameter: str = "dust_beta",
    warm_fraction_parameter: str = "dust_warm_fraction",
) -> ModuleSpec:
    """Two modified blackbodies sharing the absorbed stellar luminosity.

    ``dust_warm_fraction`` is the fraction of absorbed luminosity reradiated
    by the warm component.  The rest goes to the cold component.  Each shape is
    normalized by integrating ``L_lambda`` over the model rest-frame wavelength
    grid, so the total dust luminosity equals ``absorbed_lum_lsun`` up to the
    wavelength coverage of that grid.
    """

    return ModuleSpec(
        name="two_temperature_dust",
        setup_fn=lambda config: TwoTemperatureDustState(
            cold_temperature_parameter=cold_temperature_parameter,
            warm_temperature_parameter=warm_temperature_parameter,
            beta_parameter=beta_parameter,
            warm_fraction_parameter=warm_fraction_parameter,
        ),
        apply_fn=_apply_two_temperature_dust,
    )


def _apply_two_temperature_dust(
    params: Mapping[str, object],
    sed_state: SEDState,
    state: TwoTemperatureDustState,
) -> SEDState:
    _, jnp = require_jax()

    cold_temperature = jnp.maximum(params[state.cold_temperature_parameter], 1.0)
    warm_temperature = jnp.maximum(params[state.warm_temperature_parameter], 1.0)
    beta = params[state.beta_parameter]
    warm_fraction = jnp.clip(params[state.warm_fraction_parameter], 0.0, 1.0)

    cold_shape = _modified_blackbody_shape(sed_state.wave_rest_a, cold_temperature, beta)
    warm_shape = _modified_blackbody_shape(sed_state.wave_rest_a, warm_temperature, beta)
    cold_norm = trapz_jax(cold_shape, sed_state.wave_rest_a)
    warm_norm = trapz_jax(warm_shape, sed_state.wave_rest_a)

    cold_dust = (1.0 - warm_fraction) * sed_state.absorbed_lum_lsun * cold_shape / jnp.maximum(cold_norm, 1e-300)
    warm_dust = warm_fraction * sed_state.absorbed_lum_lsun * warm_shape / jnp.maximum(warm_norm, 1e-300)
    dust = cold_dust + warm_dust
    return sed_state._replace(dust_lum_lsun_per_a=dust, total_lum_lsun_per_a=sed_state.total_lum_lsun_per_a + dust)


def _modified_blackbody_shape(wave_a, temperature, beta):
    _, jnp = require_jax()
    wave_m = wave_a * 1.0e-10
    hc_over_k = 1.438776877e-2
    x = hc_over_k / (wave_m * temperature)
    shape = wave_m ** (-(5.0 + beta)) / jnp.expm1(jnp.clip(x, 1.0e-6, 700.0))
    return jnp.where(jnp.isfinite(shape), shape, 0.0)


@dataclass(frozen=True)
class IGMState:
    igm_factor: float
    igm_factor_parameter: str | None


def madau_igm_module(igm_factor: float = 1.0, igm_factor_parameter: str | None = None) -> ModuleSpec:
    """Approximate differentiable Madau-like IGM attenuation.

    Parameters
    ----------
    igm_factor:
        Static multiplicative factor on the optical depth. A value of one is
        the fiducial Madau-like attenuation used by the current toy JAX-CIGALE
        module.
    igm_factor_parameter:
        Optional parameter name. If provided, the value in the sampled
        parameter dictionary multiplies the optical depth instead of the static
        ``igm_factor``. This is useful near the Ly-alpha forest where the exact
        line-of-sight attenuation is a nuisance quantity.
    """

    return ModuleSpec(
        name="madau_igm",
        setup_fn=lambda config: IGMState(float(igm_factor), igm_factor_parameter),
        apply_fn=_apply_madau_igm,
    )


def _apply_madau_igm(params: Mapping[str, object], sed_state: SEDState, state: IGMState) -> SEDState:
    _, jnp = require_jax()
    z = params["z"]
    wave = sed_state.wave_rest_a
    tau_lya = jnp.where(wave < 1216.0, 0.0036 * (1.0 + z) ** 3.46 * (wave / 1216.0) ** 1.5, 0.0)
    tau_ll = jnp.where(wave < 912.0, 0.25 * (1.0 + z) ** 3.0 * (912.0 / jnp.maximum(wave, 1.0)) ** 3, 0.0)
    if state.igm_factor_parameter is None:
        factor = state.igm_factor
    else:
        factor = params[state.igm_factor_parameter]
    factor = jnp.maximum(jnp.asarray(factor), 0.0)
    transmission = jnp.exp(-factor * (tau_lya + tau_ll))
    lum = sed_state.total_lum_lsun_per_a * transmission
    return sed_state._replace(total_lum_lsun_per_a=lum)


@dataclass(frozen=True)
class RedshiftState:
    redshift_parameter: str


def redshift_module(redshift_parameter: str = "z") -> ModuleSpec:
    """Convert rest-frame luminosity density to observed f_lambda per solar mass."""

    return ModuleSpec(
        name="redshift",
        setup_fn=lambda config: RedshiftState(redshift_parameter),
        apply_fn=_apply_redshift,
    )


def _apply_redshift(params: Mapping[str, object], sed_state: SEDState, state: RedshiftState) -> SEDState:
    wave_obs, flux = observed_flux_from_luminosity(
        sed_state.wave_rest_a,
        sed_state.total_lum_lsun_per_a,
        params[state.redshift_parameter],
    )
    return sed_state._replace(wave_obs_a=wave_obs, flux_lambda_cgs=flux)
