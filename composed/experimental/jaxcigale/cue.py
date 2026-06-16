from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import numpy as np

from composed.experimental.jaxcigale.core import ModuleSpec, SEDState
from composed.experimental.jaxcigale.dependencies import require_jax
from composed.experimental.jaxcigale.photometry import C_A_PER_S, trapz_jax
from composed.units import LSUN_CGS

H_ERG_S = 6.62607015e-27
CM_PER_S = 2.99792458e10

# Cue follows ionization-edge segments in the Lyman-continuum regime.
# The values match the public Cue constants to the precision needed here.
CUE_HEII_EDGE_A = 1.0e8 / 438908.8789
CUE_OII_EDGE_A = 1.0e8 / 283270.9
CUE_HEI_EDGE_A = 1.0e8 / 198310.66637
CUE_HI_EDGE_A = 911.6
CUE_IONIZING_EDGES_A = np.asarray(
    [1.0, CUE_HEII_EDGE_A, CUE_OII_EDGE_A, CUE_HEI_EDGE_A, CUE_HI_EDGE_A],
    dtype=float,
)

# Public Cue v0.1 training ranges from Li et al. 2024, Table 1.
# The last five entries are the physical gas/emulator parameters:
# logU, log10(n_H / cm^-3), [O/H], log10(N/O), log10(C/O).
# Cue's text table quotes abundance-ratio ranges as linear ratios, while the
# public package variables are named log_NO_ratio/log_CO_ratio and examples use
# values such as -0.134. Store the log10 form here.
CUE_THETA_BOUNDS = np.asarray(
    [
        [1.0, 42.0],
        [-0.3, 30.0],
        [-1.1, 14.0],
        [-1.7, 8.0],
        [-0.1, 10.1],
        [-0.5, 1.9],
        [-0.4, 2.2],
        [-4.0, -1.0],
        [1.0, 4.0],
        [-2.2, 0.5],
        [np.log10(0.1), np.log10(5.4)],
        [np.log10(0.1), np.log10(5.4)],
    ],
    dtype=float,
)


class CueDerivedInputs(NamedTuple):
    """Quantities derived inside the Cue nebular block from the stellar spectrum.

    All luminosities are per solar mass formed because the upstream stellar
    module is per solar mass. The galaxy mass scaling is still applied once by
    ``JaxSedModel.predict_photometry`` after redshifting, exactly as for the
    non-nebular graph.
    """

    theta12: object
    ionizing_slopes: object
    ionizing_log_luminosity_ratios: object
    log_q_h_intrinsic: object
    log_q_h_gas: object
    gas_photon_fraction: object
    lyc_escape_fraction: object
    lyc_dust_fraction: object
    logu: object
    logn_h: object
    gas_logoh: object
    log_no: object
    log_co: object
    segment_log_luminosity_lsun: object
    segment_log_q: object


@dataclass(frozen=True)
class CueNebularState:
    emulator_apply: Callable[[object, object, CueDerivedInputs], tuple[object, object]]
    lyc_continuum_apply: Callable[[object, CueDerivedInputs], object] | None
    logu_parameter: str
    logn_h_parameter: str
    gas_logoh_parameter: str | None
    gas_stellar_logoh_offset_parameter: str | None
    stellar_metallicity_parameter: str
    log_no_parameter: str | None
    log_co_parameter: str | None
    f_esc_parameter: str | None
    f_dust_parameter: str | None
    default_logu: float
    default_logn_h: float
    default_gas_logoh_offset: float
    default_log_no: float
    default_log_co: float
    default_f_esc: float
    default_f_dust: float
    inner_radius_cm: float
    min_positive_luminosity: float
    min_gas_photon_fraction: float
    lyc_numerical_floor_fraction: float
    clip_derived_ionizing_shape: bool
    absorb_lyc: bool


def cue_nebular_module(
    emulator_apply: Callable[[object, object, CueDerivedInputs], tuple[object, object]],
    *,
    lyc_continuum_apply: Callable[[object, CueDerivedInputs], object] | None = None,
    logu_parameter: str = "gas_logu",
    logn_h_parameter: str = "gas_logn_h",
    gas_logoh_parameter: str | None = "gas_logoh",
    gas_stellar_logoh_offset_parameter: str | None = "gas_stellar_logoh_offset",
    stellar_metallicity_parameter: str = "logzsol",
    log_no_parameter: str | None = "gas_logno",
    log_co_parameter: str | None = "gas_logco",
    f_esc_parameter: str | None = "gas_f_esc",
    f_dust_parameter: str | None = "gas_f_dust",
    default_logu: float = -2.5,
    default_logn_h: float = 2.0,
    default_gas_logoh_offset: float = 0.0,
    default_log_no: float = -0.134,
    default_log_co: float = -0.134,
    default_f_esc: float = 0.0,
    default_f_dust: float = 0.0,
    inner_radius_cm: float = 1.0e19,
    min_positive_luminosity: float = 1.0e-300,
    min_gas_photon_fraction: float = 1.0e-12,
    lyc_numerical_floor_fraction: float = 1.0e-12,
    clip_derived_ionizing_shape: bool = True,
    absorb_lyc: bool = True,
) -> ModuleSpec:
    """Cue-style nebular module deriving ionizing inputs from the stellar spectrum.

    This module reads the upstream stellar spectrum to derive Cue's ionizing
    power-law parameters and gas-powered ``Q_H``, then delegates the actual
    CLOUDY emulator to ``emulator_apply``.  By default it also applies the
    standard nebular bookkeeping convention to the emergent stellar spectrum:
    below 912 Angstrom, only the escaped LyC fraction ``f_esc`` remains in the
    output stellar spectrum.  The remaining photons either power gas emission
    or are absorbed by dust, according to ``f_dust``.

    ``emulator_apply`` must be a pure JAX-compatible callable with signature::

        continuum_lsun_per_a, lines_lsun_per_a = emulator_apply(
            wave_rest_a, cue_theta12, cue_inputs
        )

    The returned arrays must be rest-frame luminosity densities in
    ``Lsun / Angstrom`` per solar mass formed.

    ``lyc_numerical_floor_fraction`` controls one deliberately non-physical
    cleanup step: sub-Lyman nebular values with absolute luminosity below this
    fraction of the model's own non-ionizing continuum scale are set to exactly
    zero.  This removes interpolation/PCA floors from the Cue emulator output
    while preserving genuine partial LyC escape.

    ``lyc_continuum_apply`` is an explicit extension hook for nebular-continuum
    tables that do predict emission below 912 Angstrom, such as FSPS/CLOUDY
    tables.  It is off by default so the public Cue path remains pure.  When
    supplied, it is called as ``lyc_continuum_apply(wave_rest_a, cue_inputs)``
    and added after the Cue numerical-floor cleanup, so a physically intended
    LyC continuum is not mistaken for a PCA floor.
    """

    return ModuleSpec(
        name="cue_nebular",
        setup_fn=lambda config: CueNebularState(
            emulator_apply=emulator_apply,
            lyc_continuum_apply=lyc_continuum_apply,
            logu_parameter=logu_parameter,
            logn_h_parameter=logn_h_parameter,
            gas_logoh_parameter=gas_logoh_parameter,
            gas_stellar_logoh_offset_parameter=gas_stellar_logoh_offset_parameter,
            stellar_metallicity_parameter=stellar_metallicity_parameter,
            log_no_parameter=log_no_parameter,
            log_co_parameter=log_co_parameter,
            f_esc_parameter=f_esc_parameter,
            f_dust_parameter=f_dust_parameter,
            default_logu=float(default_logu),
            default_logn_h=float(default_logn_h),
            default_gas_logoh_offset=float(default_gas_logoh_offset),
            default_log_no=float(default_log_no),
            default_log_co=float(default_log_co),
            default_f_esc=float(default_f_esc),
            default_f_dust=float(default_f_dust),
            inner_radius_cm=float(inner_radius_cm),
            min_positive_luminosity=float(min_positive_luminosity),
            min_gas_photon_fraction=float(min_gas_photon_fraction),
            lyc_numerical_floor_fraction=float(lyc_numerical_floor_fraction),
            clip_derived_ionizing_shape=bool(clip_derived_ionizing_shape),
            absorb_lyc=bool(absorb_lyc),
        ),
        apply_fn=_apply_cue_nebular,
    )


def _parameter_or_default(params: Mapping[str, object], name: str | None, default):
    if name is not None and name in params:
        return params[name]
    return default


def _gas_logoh(params: Mapping[str, object], state: CueNebularState):
    _, jnp = require_jax()
    if state.gas_logoh_parameter is not None and state.gas_logoh_parameter in params:
        return params[state.gas_logoh_parameter]
    stellar_logz = _parameter_or_default(params, state.stellar_metallicity_parameter, 0.0)
    offset = _parameter_or_default(
        params,
        state.gas_stellar_logoh_offset_parameter,
        state.default_gas_logoh_offset,
    )
    return jnp.asarray(stellar_logz) + jnp.asarray(offset)


def _apply_cue_nebular(params: Mapping[str, object], sed_state: SEDState, state: CueNebularState) -> SEDState:
    _, jnp = require_jax()
    cue_inputs = derive_cue_inputs_from_stellar_spectrum(
        sed_state.wave_rest_a,
        sed_state.stellar_lum_lsun_per_a,
        logu=_parameter_or_default(params, state.logu_parameter, state.default_logu),
        logn_h=_parameter_or_default(params, state.logn_h_parameter, state.default_logn_h),
        gas_logoh=_gas_logoh(params, state),
        log_no=_parameter_or_default(params, state.log_no_parameter, state.default_log_no),
        log_co=_parameter_or_default(params, state.log_co_parameter, state.default_log_co),
        f_esc=_parameter_or_default(params, state.f_esc_parameter, state.default_f_esc),
        f_dust=_parameter_or_default(params, state.f_dust_parameter, state.default_f_dust),
        inner_radius_cm=state.inner_radius_cm,
        min_positive_luminosity=state.min_positive_luminosity,
        min_gas_photon_fraction=state.min_gas_photon_fraction,
        clip_derived_ionizing_shape=state.clip_derived_ionizing_shape,
    )
    continuum, lines = state.emulator_apply(sed_state.wave_rest_a, cue_inputs.theta12, cue_inputs)
    continuum = jnp.asarray(continuum)
    lines = jnp.asarray(lines)
    if continuum.shape != sed_state.wave_rest_a.shape or lines.shape != sed_state.wave_rest_a.shape:
        raise ValueError("Cue emulator must return continuum and line arrays matching wave_rest_a.")
    continuum = zero_numerical_lyc_floor(
        sed_state.wave_rest_a,
        continuum,
        floor_fraction=state.lyc_numerical_floor_fraction,
    )
    lines = zero_numerical_lyc_floor(
        sed_state.wave_rest_a,
        lines,
        floor_fraction=state.lyc_numerical_floor_fraction,
    )
    if state.lyc_continuum_apply is not None:
        lyc_continuum = jnp.asarray(state.lyc_continuum_apply(sed_state.wave_rest_a, cue_inputs))
        if lyc_continuum.shape != sed_state.wave_rest_a.shape:
            raise ValueError("LyC continuum extension must return an array matching wave_rest_a.")
        continuum = continuum + lyc_continuum
    nebular = continuum + lines
    stellar = _apply_emergent_lyc_fraction(
        sed_state.wave_rest_a,
        sed_state.stellar_lum_lsun_per_a,
        cue_inputs.lyc_escape_fraction,
        absorb_lyc=state.absorb_lyc,
    )
    stellar_young = _apply_emergent_lyc_fraction(
        sed_state.wave_rest_a,
        sed_state.stellar_young_lum_lsun_per_a,
        cue_inputs.lyc_escape_fraction,
        absorb_lyc=state.absorb_lyc,
    )
    stellar_old = _apply_emergent_lyc_fraction(
        sed_state.wave_rest_a,
        sed_state.stellar_old_lum_lsun_per_a,
        cue_inputs.lyc_escape_fraction,
        absorb_lyc=state.absorb_lyc,
    )
    return sed_state._replace(
        stellar_lum_lsun_per_a=stellar,
        stellar_young_lum_lsun_per_a=stellar_young,
        stellar_old_lum_lsun_per_a=stellar_old,
        nebular_lum_lsun_per_a=nebular,
        nebular_continuum_lum_lsun_per_a=continuum,
        nebular_line_lum_lsun_per_a=lines,
        total_lum_lsun_per_a=stellar + nebular,
    )


def _apply_emergent_lyc_fraction(wave_rest_a, luminosity_lsun_per_a, f_esc, *, absorb_lyc: bool):
    _, jnp = require_jax()
    lum = jnp.asarray(luminosity_lsun_per_a)
    if not absorb_lyc:
        return lum
    transmission = jnp.where(jnp.asarray(wave_rest_a) < CUE_HI_EDGE_A, jnp.clip(jnp.asarray(f_esc), 0.0, 1.0), 1.0)
    return lum * transmission


def zero_numerical_lyc_floor(wave_rest_a, luminosity_lsun_per_a, *, floor_fraction: float = 1.0e-12):
    """Set non-physical Cue LyC numerical floors to exactly zero.

    Cue is used here as a nebular emitter.  Once the stellar LyC photons have
    been assigned to escape, gas, or dust, any residual nebular continuum far
    below the non-ionizing continuum scale is bookkeeping noise rather than a
    measurable prediction.  The reference scale is the mean absolute luminosity
    over 1500--9000 A, with fallbacks for restricted wavelength grids.

    The default threshold, ``1e-12`` of that scale, is intentionally far below
    any broadband-relevant flux but high enough to remove PCA/interpolation
    values such as ``1e-250`` that otherwise pollute downstream arrays and log
    plots.  Real partial LyC escape is many orders above this threshold and is
    left untouched.
    """

    _, jnp = require_jax()
    wave = jnp.asarray(wave_rest_a)
    lum = jnp.asarray(luminosity_lsun_per_a)
    if floor_fraction <= 0.0:
        return lum

    finite_abs_lum = jnp.where(jnp.isfinite(lum), jnp.abs(lum), 0.0)

    def masked_mean(mask):
        weights = jnp.where(mask, 1.0, 0.0)
        return jnp.sum(finite_abs_lum * weights) / jnp.maximum(jnp.sum(weights), 1.0)

    continuum_reference = masked_mean((wave >= 1500.0) & (wave <= 9000.0))
    above_lyman_reference = masked_mean(wave >= CUE_HI_EDGE_A)
    global_reference = jnp.max(finite_abs_lum)
    reference = jnp.where(continuum_reference > 0.0, continuum_reference, above_lyman_reference)
    reference = jnp.where(reference > 0.0, reference, global_reference)

    floor = jnp.asarray(floor_fraction) * reference
    is_numerical_lyc_floor = (wave < CUE_HI_EDGE_A) & (finite_abs_lum < floor)
    return jnp.where(is_numerical_lyc_floor, jnp.zeros_like(lum), lum)


def derive_cue_inputs_from_stellar_spectrum(
    wave_rest_a,
    stellar_lum_lsun_per_a,
    *,
    logu,
    logn_h,
    gas_logoh,
    log_no,
    log_co,
    f_esc=0.0,
    f_dust=0.0,
    inner_radius_cm: float = 1.0e19,
    min_positive_luminosity: float = 1.0e-300,
    min_gas_photon_fraction: float = 1.0e-12,
    clip_derived_ionizing_shape: bool = True,
) -> CueDerivedInputs:
    """Derive Cue's 12 input parameters from a rest-frame stellar spectrum.

    Parameters
    ----------
    wave_rest_a
        Rest-frame wavelength grid in Angstrom.
    stellar_lum_lsun_per_a
        Stellar luminosity density in ``Lsun / Angstrom`` per solar mass formed.
    logu, logn_h, gas_logoh, log_no, log_co
        Cue gas parameters. ``gas_logoh`` is [O/H]. The abundance-ratio
        parameters follow Cue's public API names ``log_NO_ratio`` and
        ``log_CO_ratio``.
    f_esc, f_dust
        Scalar Lyman-continuum fractions that do not power nebular emission.
        The gas-powered budget is ``max(1 - f_esc - f_dust, floor) * Q_H``.

    Returns
    -------
    CueDerivedInputs
        The Cue 12-vector plus diagnostic quantities useful for auditing.
    """

    _, jnp = require_jax()
    wave = jnp.asarray(wave_rest_a)
    l_lambda = jnp.maximum(jnp.asarray(stellar_lum_lsun_per_a), 0.0)
    l_lambda_safe = jnp.maximum(l_lambda, min_positive_luminosity)
    lnu_lsun_per_hz = l_lambda_safe * wave**2 / C_A_PER_S

    slopes, log_luminosities, log_q_segments = _fit_cue_power_law_segments(
        wave,
        l_lambda_safe,
        lnu_lsun_per_hz,
        min_positive_luminosity=min_positive_luminosity,
    )
    log_luminosity_ratios = log_luminosities[1:] - log_luminosities[:-1]
    if clip_derived_ionizing_shape:
        lo = jnp.asarray(CUE_THETA_BOUNDS[:7, 0])
        hi = jnp.asarray(CUE_THETA_BOUNDS[:7, 1])
        shape = jnp.clip(jnp.concatenate([slopes, log_luminosity_ratios]), lo, hi)
        slopes = shape[:4]
        log_luminosity_ratios = shape[4:]

    q_h_intrinsic = _ionizing_photon_rate(wave, l_lambda_safe, CUE_HI_EDGE_A)
    f_esc = jnp.clip(jnp.asarray(f_esc), 0.0, 1.0)
    f_dust = jnp.clip(jnp.asarray(f_dust), 0.0, 1.0)
    gas_fraction = jnp.maximum(1.0 - f_esc - f_dust, min_gas_photon_fraction)
    q_h_gas = jnp.maximum(q_h_intrinsic * gas_fraction, min_positive_luminosity)
    log_q_h_intrinsic = jnp.log10(jnp.maximum(q_h_intrinsic, min_positive_luminosity))
    log_q_h_gas = jnp.log10(q_h_gas)

    # Cue's public package uses logQ rather than logU internally. We keep the
    # requested logU in the 12-vector convention for the JAX adapter; a wrapper
    # around the original TensorFlow Cue can convert using this helper.
    theta12 = jnp.concatenate(
        [
            slopes,
            log_luminosity_ratios,
            jnp.asarray(
                [
                    logu,
                    logn_h,
                    gas_logoh,
                    log_no,
                    log_co,
                ]
            ),
        ]
    )
    return CueDerivedInputs(
        theta12=theta12,
        ionizing_slopes=slopes,
        ionizing_log_luminosity_ratios=log_luminosity_ratios,
        log_q_h_intrinsic=log_q_h_intrinsic,
        log_q_h_gas=log_q_h_gas,
        gas_photon_fraction=gas_fraction,
        lyc_escape_fraction=f_esc,
        lyc_dust_fraction=f_dust,
        logu=jnp.asarray(logu),
        logn_h=jnp.asarray(logn_h),
        gas_logoh=jnp.asarray(gas_logoh),
        log_no=jnp.asarray(log_no),
        log_co=jnp.asarray(log_co),
        segment_log_luminosity_lsun=log_luminosities,
        segment_log_q=log_q_segments,
    )


def cue_logq_from_logu(logu, logn_h, inner_radius_cm: float = 1.0e19):
    """Convert Cue-style ``logU`` and ``log n_H`` to the package's ``logQ``."""

    _, jnp = require_jax()
    return (
        jnp.asarray(logu)
        + jnp.log10(4.0 * jnp.pi)
        + 2.0 * jnp.log10(jnp.asarray(inner_radius_cm))
        + jnp.asarray(logn_h)
        + jnp.log10(jnp.asarray(CM_PER_S))
    )


def cue_theta12_to_public_package_theta(cue_inputs: CueDerivedInputs, inner_radius_cm: float = 1.0e19):
    """Convert the composed Cue convention to the public TensorFlow package input.

    The public Cue package expects:

    ``gamma1..gamma4, logLratio1..3, logQ, n_H, [O/H], log(N/O), log(C/O)``.

    This helper is deliberately separate from the JAX module because the public
    package is TensorFlow/PCA based and not directly differentiable by JAX.
    """

    _, jnp = require_jax()
    logq_for_logu = cue_logq_from_logu(cue_inputs.logu, cue_inputs.logn_h, inner_radius_cm=inner_radius_cm)
    return jnp.concatenate(
        [
            cue_inputs.ionizing_slopes,
            cue_inputs.ionizing_log_luminosity_ratios,
            jnp.asarray(
                [
                    logq_for_logu,
                    10.0 ** cue_inputs.logn_h,
                    cue_inputs.gas_logoh,
                    cue_inputs.log_no,
                    cue_inputs.log_co,
                ]
            ),
        ]
    )


def cue_theta_in_training_bounds(theta12):
    """Return a boolean mask for Cue's documented training-domain bounds."""

    _, jnp = require_jax()
    theta = jnp.asarray(theta12)
    bounds = jnp.asarray(CUE_THETA_BOUNDS)
    return (theta >= bounds[:, 0]) & (theta <= bounds[:, 1])


def reconstruct_cue_piecewise_lnu(wave_rest_a, slopes, segment_log_luminosity_lsun):
    """Reconstruct Cue's four-segment ionizing ``L_nu`` approximation.

    This is a diagnostic for auditing the information loss between the stellar
    ionizing continuum and the Cue parameterization. It is not used by the
    nebular module unless a user-supplied emulator chooses to inspect it.
    """

    _, jnp = require_jax()
    wave = jnp.asarray(wave_rest_a)
    slopes = jnp.asarray(slopes)
    log_lum = jnp.asarray(segment_log_luminosity_lsun)
    edges = jnp.asarray(CUE_IONIZING_EDGES_A)

    pieces = []
    for i in range(4):
        lo = edges[i]
        hi = edges[i + 1]
        alpha = slopes[i]
        # Convert integrated segment luminosity in Lsun to a power-law
        # normalization in Lsun/Hz for Lnu = A lambda^alpha.
        integral = _segment_lnu_energy_integral(alpha, lo, hi)
        log_a = log_lum[i] - jnp.log10(jnp.maximum(integral, 1.0e-300))
        pieces.append(10.0 ** (log_a + alpha * jnp.log10(jnp.maximum(wave, 1.0e-300))))
    out = jnp.zeros_like(wave)
    for i in range(4):
        mask = (wave >= edges[i]) & (wave <= edges[i + 1])
        out = jnp.where(mask, pieces[i], out)
    return out


def _fit_cue_power_law_segments(wave, l_lambda_lsun_per_a, lnu_lsun_per_hz, *, min_positive_luminosity):
    _, jnp = require_jax()
    edges = jnp.asarray(CUE_IONIZING_EDGES_A)
    slopes = []
    log_luminosities = []
    log_q_segments = []
    for i in range(4):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (wave >= lo) & (wave <= hi)
        slopes.append(_weighted_loglog_slope(wave, lnu_lsun_per_hz, mask))
        seg_lum = trapz_jax(jnp.where(mask, l_lambda_lsun_per_a, 0.0), wave)
        seg_q = _ionizing_photon_rate_segment(wave, l_lambda_lsun_per_a, lo, hi)
        log_luminosities.append(jnp.log10(jnp.maximum(seg_lum, min_positive_luminosity)))
        log_q_segments.append(jnp.log10(jnp.maximum(seg_q, min_positive_luminosity)))
    return jnp.asarray(slopes), jnp.asarray(log_luminosities), jnp.asarray(log_q_segments)


def _weighted_loglog_slope(wave, luminosity, mask):
    _, jnp = require_jax()
    x = jnp.log10(jnp.maximum(wave, 1.0e-300))
    y = jnp.log10(jnp.maximum(luminosity, 1.0e-300))
    weights = jnp.where(mask, 1.0, 0.0)
    wsum = jnp.sum(weights)
    xbar = jnp.sum(weights * x) / jnp.maximum(wsum, 1.0)
    ybar = jnp.sum(weights * y) / jnp.maximum(wsum, 1.0)
    dx = x - xbar
    dy = y - ybar
    denom = jnp.sum(weights * dx * dx)
    slope = jnp.sum(weights * dx * dy) / jnp.maximum(denom, 1.0e-300)
    return jnp.where((wsum >= 2.0) & (denom > 0.0), slope, 0.0)


def _ionizing_photon_rate(wave, l_lambda_lsun_per_a, upper_a):
    return _ionizing_photon_rate_segment(wave, l_lambda_lsun_per_a, 1.0, upper_a)


def _ionizing_photon_rate_segment(wave, l_lambda_lsun_per_a, lower_a, upper_a):
    _, jnp = require_jax()
    mask = (wave >= lower_a) & (wave <= upper_a)
    integrand = jnp.where(mask, l_lambda_lsun_per_a * LSUN_CGS * wave / (H_ERG_S * C_A_PER_S), 0.0)
    return trapz_jax(integrand, wave)


def _segment_lnu_energy_integral(alpha, lower_a, upper_a):
    """Integral converting ``Lnu = A lambda^alpha`` to segment luminosity.

    The luminosity in a wavelength segment is ``int Lnu dnu``. Since
    ``dnu = -c / lambda^2 dlambda``, the positive segment integral is
    ``A * c * int lambda^(alpha - 2) dlambda``.
    """

    _, jnp = require_jax()
    exponent = alpha - 1.0
    power_integral = jnp.where(
        jnp.abs(exponent) < 1.0e-6,
        jnp.log(upper_a / lower_a),
        (upper_a**exponent - lower_a**exponent) / exponent,
    )
    return C_A_PER_S * jnp.abs(power_integral)


def toy_cue_emulator(wave_rest_a, theta12, cue_inputs: CueDerivedInputs):
    """Tiny differentiable placeholder useful for tests and examples.

    This is not a physical nebular model. It only checks that the plumbing,
    normalization, and parameter flow are correct before the real Cue weights
    are ported to JAX.
    """

    _, jnp = require_jax()
    wave = jnp.asarray(wave_rest_a)
    q_scale = 10.0 ** (cue_inputs.log_q_h_gas - 53.0)
    metallicity_scale = 10.0 ** (0.15 * cue_inputs.gas_logoh)
    continuum_shape = (wave / 3646.0) ** -0.15 * (1.0 + 0.35 / (1.0 + jnp.exp((wave - 3646.0) / 30.0)))
    continuum = 1.0e8 * q_scale * metallicity_scale * continuum_shape

    def gaussian(center, sigma, amp):
        return amp * jnp.exp(-0.5 * ((wave - center) / sigma) ** 2)

    lines = q_scale * metallicity_scale * (
        gaussian(1215.67, 2.0, 2.0e11)
        + gaussian(5007.0, 3.0, 1.2e10 * 10.0 ** (-0.4 * (theta12[7] + 3.0)))
        + gaussian(6562.8, 3.0, 8.0e9)
    )
    return continuum, lines
