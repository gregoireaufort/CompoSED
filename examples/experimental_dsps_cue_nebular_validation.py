#!/usr/bin/env python
"""Validate DSPS + JAX-Cue nebular spectra against CIGALE/FSPS references.

This is a diagnostic script, not production API. It makes the computation
auditable by keeping the scientific steps visible:

1. define a small deterministic parameter sweep;
2. compute rest-frame DSPS stellar spectra plus Cue nebular emission;
3. optionally compute CIGALE ``fsps_stellar + nebular`` and direct FSPS nebular;
4. redshift each rest-frame spectrum through the same simple filters;
5. plot spectra and broadband magnitude differences.

The stages can be run in different Python environments:

    # JAX/DSPS/Cue environment
    PYTHONPATH=/path/to/CompoSED \
    DSPS_CONTINUUM_SSP_FILE=/path/to/fsps_continuum_ssp_data.h5 \
    CUE_DATA_DIR=/path/to/cue/src/cue/data \
    python examples/experimental_dsps_cue_nebular_validation.py --stage dsps-cue

    # CIGALE/FSPS environment
    PYTHONPATH=/path/to/CompoSED \
    SPS_HOME=/path/to/fsps \
    python examples/experimental_dsps_cue_nebular_validation.py --stage references

    # Either environment with matplotlib/numpy
    python examples/experimental_dsps_cue_nebular_validation.py --stage plots
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from composed.experimental.jaxcigale.ssp_data import (
    default_continuum_ssp_path,
    require_continuum_ssp_path,
)

LSUN_CGS = 3.828e33
LSUN_W = 3.828e26
MPC_CM = 3.0856775814913673e24
C_A_PER_S = 2.99792458e18
AB_FNU_CGS = 3631.0e-23
C_KM_PER_S = 299792.458

REST_WAVE_A = np.geomspace(50.0, 30000.0, 2200)
REST_WAVE_NM = REST_WAVE_A / 10.0
# DSPS/FSPS SSP tables are not well behaved at arbitrarily tiny ages. Keep the
# first SFH support point at 20 Myr, matching the other JAX-CIGALE demos.
AGE_GRID_GYR = np.linspace(0.02, 10.0, 512)

FILTER_SPECS = [
    ("u_like", 3600.0, 350.0),
    ("g_like", 4800.0, 520.0),
    ("r_like", 6200.0, 560.0),
    ("i_like", 7600.0, 650.0),
    ("z_like", 9000.0, 720.0),
    ("Y_like", 10200.0, 800.0),
    ("J_like", 12500.0, 1100.0),
    ("H_like", 16500.0, 1400.0),
]

CIGALE_NEBULAR_Z_GRID = np.asarray(
    [0.0001, 0.0004, 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.03],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "dsps-cue", "references", "plots"], default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimental_dsps_cue_nebular_validation"),
    )
    parser.add_argument(
        "--ssp-file",
        type=Path,
        default=Path(os.environ.get("DSPS_CONTINUUM_SSP_FILE", default_continuum_ssp_path())),
        help="Continuum-only DSPS/FSPS SSP table. Required before adding Cue nebular emission.",
    )
    parser.add_argument(
        "--allow-nebular-included-ssp",
        action="store_true",
        help="Diagnostic escape hatch allowing DSPS' default nebular-included SSP table.",
    )
    parser.add_argument(
        "--cue-data-dir",
        type=Path,
        default=Path(os.environ.get("CUE_DATA_DIR", "/private/tmp/cue/src/cue/data")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    draws = load_or_create_draws(args.output_dir / "parameter_draws.json")

    if args.stage in {"all", "dsps-cue"}:
        run_dsps_cue_stage(draws, args)
    if args.stage in {"all", "references"}:
        run_reference_stage(draws, args)
    if args.stage in {"all", "plots"}:
        make_plots_and_summary(args.output_dir)


def load_or_create_draws(path: Path) -> list[dict[str, float]]:
    if path.exists():
        return json.loads(path.read_text())
    draws = [
        {"label": "young_subsolar", "z": 0.4, "log10_mass": 10.0, "tage_gyr": 0.8, "tau_gyr": 1.6, "logzsol": -0.5, "gas_logoh": -0.5, "gas_logu": -2.5, "gas_logn_h": 2.0, "gas_logno": -0.3, "gas_logco": -0.2, "gas_f_esc": 0.0, "gas_f_dust": 0.0},
        {"label": "main_sequence_solar", "z": 1.0, "log10_mass": 10.3, "tage_gyr": 2.2, "tau_gyr": 3.0, "logzsol": 0.0, "gas_logoh": 0.0, "gas_logu": -2.0, "gas_logn_h": 2.0, "gas_logno": -0.134, "gas_logco": -0.134, "gas_f_esc": 0.0, "gas_f_dust": 0.0},
        {"label": "dust_free_high_u", "z": 2.0, "log10_mass": 9.7, "tage_gyr": 0.45, "tau_gyr": 0.8, "logzsol": -1.0, "gas_logoh": -0.8, "gas_logu": -1.5, "gas_logn_h": 2.5, "gas_logno": -0.5, "gas_logco": -0.4, "gas_f_esc": 0.0, "gas_f_dust": 0.0},
        {"label": "old_metal_rich", "z": 0.7, "log10_mass": 10.8, "tage_gyr": 5.0, "tau_gyr": 1.0, "logzsol": 0.25, "gas_logoh": 0.1, "gas_logu": -3.0, "gas_logn_h": 2.0, "gas_logno": 0.0, "gas_logco": 0.0, "gas_f_esc": 0.0, "gas_f_dust": 0.0},
        {"label": "partial_escape", "z": 1.5, "log10_mass": 10.1, "tage_gyr": 1.2, "tau_gyr": 2.5, "logzsol": -0.3, "gas_logoh": -0.2, "gas_logu": -2.2, "gas_logn_h": 2.0, "gas_logno": -0.2, "gas_logco": -0.15, "gas_f_esc": 0.1, "gas_f_dust": 0.1},
        {"label": "low_z_blue", "z": 0.12, "log10_mass": 9.5, "tage_gyr": 1.8, "tau_gyr": 4.0, "logzsol": -0.7, "gas_logoh": -0.6, "gas_logu": -2.4, "gas_logn_h": 2.0, "gas_logno": -0.4, "gas_logco": -0.3, "gas_f_esc": 0.0, "gas_f_dust": 0.0},
    ]
    path.write_text(json.dumps(draws, indent=2))
    return draws


def run_dsps_cue_stage(draws: list[dict[str, float]], args: argparse.Namespace) -> None:
    from dsps import load_ssp_templates

    from composed.experimental.jaxcigale import (
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        cue_nebular_module,
        delayed_sfh_cosmic_time_module,
        dsps_stellar_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.cue_port import CueJaxPort
    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.photometry import JaxFilterSet

    jax, jnp = require_jax()
    print("Running DSPS + Cue stage with JAX backend:", jax.default_backend())
    ssp_file = require_continuum_ssp_path(
        args.ssp_file,
        allow_nebular_included=args.allow_nebular_included_ssp,
    )
    print("SSP file:", ssp_file)
    print("Cue data:", args.cue_data_dir)

    ssp_data = load_ssp_templates(fn=str(ssp_file))
    cue_port = CueJaxPort.from_public_cue_data_dir(args.cue_data_dir)
    filters = jax_filter_set()
    names = list(draws[0].keys())
    names.remove("label")
    space = JaxParameterSpace(names=names, priors={name: UniformJaxPrior(-50.0, 60.0) for name in names})
    model = build_jax_sed_model(
        [
            delayed_sfh_cosmic_time_module(n_time=AGE_GRID_GYR.size),
            dsps_stellar_module(ssp_data),
            cue_nebular_module(cue_port.make_nebular_apply(line_sigma_a=1.5)),
            redshift_module(),
        ],
        REST_WAVE_A,
        filters,
        space,
    )

    stellar = []
    nebular = []
    total = []
    phot = []
    cue_theta = []
    for draw in draws:
        theta = jnp.asarray([draw[name] for name in names])
        state = model.run_modules(theta)
        stellar_lum = np.asarray(state.stellar_lum_lsun_per_a)
        neb_lum = np.asarray(state.nebular_lum_lsun_per_a)
        total_lum = np.asarray(state.total_lum_lsun_per_a)
        mass = 10.0 ** draw["log10_mass"]
        stellar.append(stellar_lum)
        nebular.append(neb_lum)
        total.append(total_lum)
        phot.append(photometry_from_rest_spectrum(REST_WAVE_A, total_lum * mass, draw["z"]))
        print(f"  DSPS+Cue {draw['label']}")

    np.savez(
        args.output_dir / "dsps_cue_spectra.npz",
        rest_wave_nm=REST_WAVE_NM,
        stellar_lsun_per_a=np.asarray(stellar),
        nebular_lsun_per_a=np.asarray(nebular),
        total_lsun_per_a=np.asarray(total),
        phot_maggies=np.asarray(phot),
        filter_names=np.asarray([name for name, _, _ in FILTER_SPECS]),
    )
    print("Saved:", args.output_dir / "dsps_cue_spectra.npz")


def run_reference_stage(draws: list[dict[str, float]], args: argparse.Namespace) -> None:
    print("Running CIGALE/FSPS reference stage")
    cigale_stellar = np.full((len(draws), REST_WAVE_A.size), np.nan)
    cigale_total = np.full((len(draws), REST_WAVE_A.size), np.nan)
    direct_fsps_stellar = np.full((len(draws), REST_WAVE_A.size), np.nan)
    direct_fsps_total = np.full((len(draws), REST_WAVE_A.size), np.nan)
    cigale_phot = np.full((len(draws), len(FILTER_SPECS)), np.nan)
    direct_fsps_phot = np.full((len(draws), len(FILTER_SPECS)), np.nan)

    try:
        from pcigale.warehouse import SedWarehouse

        from composed.experimental.cigale_fsps_stellar import register_cigale_fsps_stellar_module

        register_cigale_fsps_stellar_module()
        warehouse = SedWarehouse(nocache=["fsps_stellar", "nebular"])
        for i, draw in enumerate(draws):
            try:
                stellar_sed = warehouse.get_sed(
                    ["sfhdelayed", "fsps_stellar"],
                    [cigale_sfh_params(draw), cigale_fsps_stellar_params(draw)],
                )
                total_sed = warehouse.get_sed(
                    ["sfhdelayed", "fsps_stellar", "nebular"],
                    [cigale_sfh_params(draw), cigale_fsps_stellar_params(draw), cigale_nebular_params(draw)],
                )
                wave_a, lum_lsun_per_a = cigale_spectrum_to_lsun_per_a(stellar_sed)
                cigale_stellar[i] = np.interp(REST_WAVE_A, wave_a, lum_lsun_per_a, left=np.nan, right=np.nan)
                wave_a, lum_lsun_per_a = cigale_spectrum_to_lsun_per_a(total_sed)
                cigale_total[i] = np.interp(REST_WAVE_A, wave_a, lum_lsun_per_a, left=np.nan, right=np.nan)
                cigale_phot[i] = photometry_from_rest_spectrum(REST_WAVE_A, cigale_total[i] * 10.0 ** draw["log10_mass"], draw["z"])
                print(f"  CIGALE {draw['label']}")
            except Exception as exc:
                print(f"  CIGALE skipped {draw['label']}: {exc!r}")
    except Exception as exc:
        print("CIGALE reference skipped:", repr(exc))

    try:
        import fsps

        for i, draw in enumerate(draws):
            wave_a, lum_lsun_per_a = direct_fsps_spectrum(fsps, draw, add_nebular=False)
            direct_fsps_stellar[i] = np.interp(REST_WAVE_A, wave_a, lum_lsun_per_a, left=np.nan, right=np.nan)
            wave_a, lum_lsun_per_a = direct_fsps_spectrum(fsps, draw, add_nebular=True)
            direct_fsps_total[i] = np.interp(REST_WAVE_A, wave_a, lum_lsun_per_a, left=np.nan, right=np.nan)
            direct_fsps_phot[i] = photometry_from_rest_spectrum(REST_WAVE_A, direct_fsps_total[i] * 10.0 ** draw["log10_mass"], draw["z"])
            print(f"  direct FSPS {draw['label']}")
    except Exception as exc:
        print("Direct FSPS reference skipped:", repr(exc))

    np.savez(
        args.output_dir / "reference_spectra.npz",
        rest_wave_nm=REST_WAVE_NM,
        cigale_stellar_lsun_per_a=cigale_stellar,
        cigale_total_lsun_per_a=cigale_total,
        direct_fsps_stellar_lsun_per_a=direct_fsps_stellar,
        direct_fsps_total_lsun_per_a=direct_fsps_total,
        cigale_phot_maggies=cigale_phot,
        direct_fsps_phot_maggies=direct_fsps_phot,
        filter_names=np.asarray([name for name, _, _ in FILTER_SPECS]),
    )
    print("Saved:", args.output_dir / "reference_spectra.npz")


def make_plots_and_summary(output_dir: Path) -> None:
    draws = json.loads((output_dir / "parameter_draws.json").read_text())
    dsps = np.load(output_dir / "dsps_cue_spectra.npz", allow_pickle=True)
    ref_path = output_dir / "reference_spectra.npz"
    ref = np.load(ref_path, allow_pickle=True) if ref_path.exists() else None

    rest_wave_nm = dsps["rest_wave_nm"]
    dsps_total = dsps["total_lsun_per_a"]
    dsps_stellar = dsps["stellar_lsun_per_a"]
    dsps_nebular = dsps["nebular_lsun_per_a"]
    dsps_phot = dsps["phot_maggies"]
    cigale_stellar = get_npz_array(ref, "cigale_stellar_lsun_per_a", dsps_total)
    cigale_total = ref["cigale_total_lsun_per_a"] if ref is not None else np.full_like(dsps_total, np.nan)
    fsps_stellar = get_npz_array(ref, "direct_fsps_stellar_lsun_per_a", dsps_total)
    fsps_total = ref["direct_fsps_total_lsun_per_a"] if ref is not None else np.full_like(dsps_total, np.nan)
    cigale_phot = ref["cigale_phot_maggies"] if ref is not None else np.full_like(dsps_phot, np.nan)
    fsps_phot = ref["direct_fsps_phot_maggies"] if ref is not None else np.full_like(dsps_phot, np.nan)

    plot_spectra_grid(output_dir, draws, rest_wave_nm, dsps_total, cigale_total, fsps_total)
    plot_normalized_spectra_grid(output_dir, draws, rest_wave_nm, dsps_total, cigale_total, fsps_total)
    plot_stellar_vs_total_grid(
        output_dir,
        draws,
        rest_wave_nm,
        dsps_stellar,
        dsps_total,
        cigale_stellar,
        cigale_total,
        fsps_stellar,
        fsps_total,
    )
    plot_nebular_contribution_grid(output_dir, draws, rest_wave_nm, dsps_nebular, cigale_total - cigale_stellar, fsps_total - fsps_stellar)
    plot_broadband_differences(output_dir, draws, dsps_phot, cigale_phot, fsps_phot)

    summary = {
        "n_draws": len(draws),
        "filter_names": [name for name, _, _ in FILTER_SPECS],
        "median_abs_dmag_cigale_minus_dsps_cue": finite_median_abs_delta_mag(cigale_phot, dsps_phot),
        "median_abs_dmag_direct_fsps_minus_dsps_cue": finite_median_abs_delta_mag(fsps_phot, dsps_phot),
        "median_spectral_ratio_cigale_over_dsps_cue_120_900nm": median_spectral_ratio(cigale_total, dsps_total, rest_wave_nm, 120.0, 900.0),
        "median_spectral_ratio_direct_fsps_over_dsps_cue_120_900nm": median_spectral_ratio(fsps_total, dsps_total, rest_wave_nm, 120.0, 900.0),
        "median_550nm_scale_cigale_over_dsps_cue": median_continuum_scale(cigale_total, dsps_total, rest_wave_nm, 550.0),
        "median_550nm_scale_direct_fsps_over_dsps_cue": median_continuum_scale(fsps_total, dsps_total, rest_wave_nm, 550.0),
        "median_550nm_stellar_scale_cigale_over_dsps": median_continuum_scale(cigale_stellar, dsps_stellar, rest_wave_nm, 550.0),
        "median_550nm_stellar_scale_direct_fsps_over_dsps": median_continuum_scale(fsps_stellar, dsps_stellar, rest_wave_nm, 550.0),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved plots under:", output_dir)


def cigale_sfh_params(draw: dict[str, float]) -> dict[str, object]:
    return {
        "age_main": int(round(draw["tage_gyr"] * 1000.0)),
        "tau_main": float(draw["tau_gyr"] * 1000.0),
        "normalise": True,
    }


def cigale_fsps_stellar_params(draw: dict[str, float]) -> dict[str, object]:
    return {
        "imf_type": 1,
        "logzsol": float(draw["logzsol"]),
        "zcontinuous": 1,
        "separation_age": 10,
    }


def cigale_nebular_params(draw: dict[str, float]) -> dict[str, object]:
    requested_zgas = 0.02 * 10.0 ** float(draw["gas_logoh"])
    # CIGALE's nebular module is table based. Use the nearest available grid
    # value for the reference run rather than pretending it is continuous.
    zgas = float(CIGALE_NEBULAR_Z_GRID[np.argmin(np.abs(np.log10(CIGALE_NEBULAR_Z_GRID / requested_zgas)))])
    return {
        "logU": float(draw["gas_logu"]),
        "zgas": zgas,
        # Public CIGALE nebular tables in this installation are sparse in
        # density. Keep the reference at the standard 100 cm^-3; the DSPS+Cue
        # branch still uses the requested gas_logn_h.
        "ne": 100.0,
        "f_esc": float(draw["gas_f_esc"]),
        "f_dust": float(draw["gas_f_dust"]),
        "lines_width": 300.0,
        "emission": True,
    }


def cigale_spectrum_to_lsun_per_a(sed) -> tuple[np.ndarray, np.ndarray]:
    wave_nm = np.asarray(sed.wavelength_grid, dtype=float)
    lum_w_per_nm = np.asarray(sed.luminosity, dtype=float)
    wave_a = wave_nm * 10.0
    lum_lsun_per_a = lum_w_per_nm / (LSUN_W * 10.0)
    return wave_a, lum_lsun_per_a


def direct_fsps_spectrum(fsps, draw: dict[str, float], *, add_nebular: bool) -> tuple[np.ndarray, np.ndarray]:
    sp = fsps.StellarPopulation(
        zcontinuous=1,
        sfh=3,
        imf_type=1,
        add_neb_emission=bool(add_nebular),
        add_neb_continuum=bool(add_nebular),
    )
    sp.params["logzsol"] = float(draw["logzsol"])
    sp.params["gas_logu"] = float(draw["gas_logu"])
    sp.params["gas_logz"] = float(draw["gas_logoh"])
    time_gyr, sfr = delayed_sfh_grid_for_draw(draw)
    sp.set_tabular_sfh(time_gyr, sfr)
    wave_a, lum_lsun_per_a = sp.get_spectrum(tage=float(draw["tage_gyr"]), peraa=True)
    return np.asarray(wave_a, dtype=float), np.asarray(lum_lsun_per_a, dtype=float)


def delayed_sfh_grid_for_draw(draw: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    age = np.linspace(0.001, float(draw["tage_gyr"]), 512)
    tau = max(float(draw["tau_gyr"]), 1e-4)
    raw = age * np.exp(-age / tau)
    formed_mass = trapezoid(raw, age * 1.0e9)
    return age, raw / formed_mass


def photometry_from_rest_spectrum(wave_rest_a: np.ndarray, lum_lsun_per_a: np.ndarray, z: float) -> np.ndarray:
    wave_obs_a = wave_rest_a * (1.0 + z)
    d_l_cm = flat_lcdm_luminosity_distance_mpc(z) * MPC_CM
    flux_lambda = lum_lsun_per_a * LSUN_CGS / (4.0 * np.pi * d_l_cm**2 * (1.0 + z))
    out = []
    for _, center, width in FILTER_SPECS:
        fw, ft = gaussian_filter_curve(center, width)
        flam = np.interp(fw, wave_obs_a, flux_lambda, left=0.0, right=0.0)
        numerator = trapezoid(flam * fw * ft, fw)
        denominator = trapezoid((C_A_PER_S / fw) * ft, fw)
        out.append((numerator / denominator) / AB_FNU_CGS)
    return np.asarray(out, dtype=float)


def jax_filter_set():
    from composed.experimental.jaxcigale.photometry import JaxFilterSet

    waves = []
    transmissions = []
    names = []
    for name, center, width in FILTER_SPECS:
        wave, trans = gaussian_filter_curve(center, width)
        names.append(name)
        waves.append(wave)
        transmissions.append(trans)
    return JaxFilterSet.from_curves(names, waves, transmissions)


def gaussian_filter_curve(center_a: float, sigma_a: float) -> tuple[np.ndarray, np.ndarray]:
    wave = np.linspace(center_a - 3.5 * sigma_a, center_a + 3.5 * sigma_a, 160)
    trans = np.exp(-0.5 * ((wave - center_a) / sigma_a) ** 2)
    return wave, trans


def flat_lcdm_luminosity_distance_mpc(z: float, omega_m: float = 0.3075, h: float = 0.6774) -> float:
    zz = np.linspace(0.0, float(z), 512)
    e_z = np.sqrt(omega_m * (1.0 + zz) ** 3 + (1.0 - omega_m))
    integral = trapezoid(1.0 / e_z, zz)
    return (1.0 + z) * (C_KM_PER_S / (100.0 * h)) * integral


def trapezoid(y, x):
    """NumPy 1.x/2.x compatible trapezoidal integral."""

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def plot_spectra_grid(output_dir: Path, draws, wave_nm, dsps_total, cigale_total, fsps_total) -> None:
    n = len(draws)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), constrained_layout=True)
    for i, draw in enumerate(draws):
        ax = axes[i, 0]
        plot_one_spectrum(ax, wave_nm, dsps_total[i], "DSPS + JAX-Cue", "black", lw=1.6)
        plot_one_spectrum(ax, wave_nm, cigale_total[i], "CIGALE fsps_stellar + nebular", "tab:blue", alpha=0.8)
        plot_one_spectrum(ax, wave_nm, fsps_total[i], "direct FSPS nebular", "tab:orange", alpha=0.8)
        ax.axvline(91.16, color="0.7", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10.0, 3000.0)
        ax.set_ylabel(r"$L_\lambda$ [L$_\odot$ A$^{-1}$]")
        ax.set_title(f"{draw['label']}  z={draw['z']:.2f}, logZ={draw['logzsol']:.2f}, logU={draw['gas_logu']:.1f}")
        if i == 0:
            ax.legend(fontsize=8)

        rax = axes[i, 1]
        ratio_plot(rax, wave_nm, cigale_total[i], dsps_total[i], "CIGALE / DSPS+Cue", "tab:blue")
        ratio_plot(rax, wave_nm, fsps_total[i], dsps_total[i], "FSPS / DSPS+Cue", "tab:orange")
        rax.axhline(1.0, color="black", lw=0.8)
        rax.axvline(91.16, color="0.7", ls=":", lw=1)
        rax.set_xscale("log")
        rax.set_yscale("log")
        rax.set_ylim(1e-3, 1e3)
        rax.set_xlim(10.0, 3000.0)
        if i == 0:
            rax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Rest wavelength [nm]")
    axes[-1, 1].set_xlabel("Rest wavelength [nm]")
    fig.savefig(output_dir / "dsps_cue_vs_references_spectra.png", dpi=180)
    plt.close(fig)


def plot_one_spectrum(ax, wave_nm, lum, label, color, **kwargs) -> None:
    good = spectrum_plot_mask(wave_nm, lum)
    if np.any(good):
        ax.plot(wave_nm, np.where(good, lum, np.nan), label=label, color=color, **kwargs)


def ratio_plot(ax, wave_nm, numerator, denominator, label, color) -> None:
    good = (
        spectrum_plot_mask(wave_nm, numerator)
        & spectrum_plot_mask(wave_nm, denominator)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    if np.any(good):
        ratio = np.where(good, numerator / denominator, np.nan)
        ax.plot(wave_nm, ratio, label=label, color=color, lw=1.2)


def spectrum_plot_mask(wave_nm, luminosity) -> np.ndarray:
    """Mask numerical zero-floors in log SED diagnostic plots.

    When ``f_esc=0``, the physically intended emergent LyC continuum is zero.
    The arrays can still contain tiny positive interpolation/emulator floors.
    On a log axis those floors look like real 200-dex plunges, so we suppress
    values far below the non-ionizing continuum scale.
    """

    wave_nm = np.asarray(wave_nm, dtype=float)
    luminosity = np.asarray(luminosity, dtype=float)
    positive = np.isfinite(luminosity) & (luminosity > 0.0)
    if not np.any(positive):
        return positive
    continuum_window = positive & (wave_nm >= 150.0) & (wave_nm <= 900.0)
    if np.any(continuum_window):
        reference = float(np.nanmedian(luminosity[continuum_window]))
    else:
        reference = float(np.nanmax(luminosity[positive]))
    floor = max(reference * 1.0e-12, np.finfo(float).tiny)
    return positive & (luminosity > floor)


def plot_broadband_differences(output_dir: Path, draws, dsps_phot, cigale_phot, fsps_phot) -> None:
    dsps_mag = maggies_to_ab(dsps_phot)
    cigale_mag = maggies_to_ab(cigale_phot)
    fsps_mag = maggies_to_ab(fsps_phot)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, delta, title in [
        (axes[0], cigale_mag - dsps_mag, "CIGALE - DSPS+Cue [AB mag]"),
        (axes[1], fsps_mag - dsps_mag, "direct FSPS - DSPS+Cue [AB mag]"),
    ]:
        image = ax.imshow(delta, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)
        ax.set_xticks(np.arange(len(FILTER_SPECS)), [name for name, _, _ in FILTER_SPECS], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(draws)), [draw["label"] for draw in draws])
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(output_dir / "dsps_cue_vs_references_broadband_dmag.png", dpi=180)
    plt.close(fig)


def get_npz_array(npz, name, shape_like):
    if npz is not None and name in npz.files:
        return npz[name]
    return np.full_like(shape_like, np.nan)


def plot_stellar_vs_total_grid(
    output_dir: Path,
    draws,
    wave_nm,
    dsps_stellar,
    dsps_total,
    cigale_stellar,
    cigale_total,
    fsps_stellar,
    fsps_total,
) -> None:
    labels = [
        ("DSPS + JAX-Cue", dsps_stellar, dsps_total, "black"),
        ("CIGALE fsps_stellar + nebular", cigale_stellar, cigale_total, "tab:blue"),
        ("direct FSPS", fsps_stellar, fsps_total, "tab:orange"),
    ]
    fig, axes = plt.subplots(len(draws), 3, figsize=(16, 2.8 * len(draws)), sharex=True, constrained_layout=True)
    for row, draw in enumerate(draws):
        for col, (label, stellar, total, color) in enumerate(labels):
            ax = axes[row, col]
            plot_one_spectrum(ax, wave_nm, stellar[row], "stellar before nebular", color, lw=1.3, ls="--", alpha=0.8)
            plot_one_spectrum(ax, wave_nm, total[row], "stellar + nebular", color, lw=1.5)
            ax.axvline(91.16, color="0.75", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(10.0, 3000.0)
            ax.set_title(f"{draw['label']}\n{label}", fontsize=9)
            if col == 0:
                ax.set_ylabel(r"$L_\lambda$ [L$_\odot$ A$^{-1}$]")
            if row == 0:
                ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Rest wavelength [nm]")
    fig.savefig(output_dir / "stellar_before_after_nebular_by_pipeline.png", dpi=180)
    plt.close(fig)


def plot_nebular_contribution_grid(output_dir: Path, draws, wave_nm, dsps_nebular, cigale_nebular, fsps_nebular) -> None:
    labels = [
        ("DSPS + JAX-Cue nebular", dsps_nebular, "black"),
        ("CIGALE nebular increment", cigale_nebular, "tab:blue"),
        ("direct FSPS nebular increment", fsps_nebular, "tab:orange"),
    ]
    fig, axes = plt.subplots(len(draws), 1, figsize=(10, 2.5 * len(draws)), sharex=True, constrained_layout=True)
    for row, draw in enumerate(draws):
        ax = axes[row]
        for label, nebular, color in labels:
            positive = np.where(np.isfinite(nebular[row]) & (nebular[row] > 0.0), nebular[row], np.nan)
            plot_one_spectrum(ax, wave_nm, positive, label, color, lw=1.2)
        ax.axvline(91.16, color="0.75", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10.0, 3000.0)
        ax.set_ylabel(r"$\Delta L_\lambda$")
        ax.set_title(f"Nebular-only contribution: {draw['label']}")
        if row == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Rest wavelength [nm]")
    fig.savefig(output_dir / "nebular_only_contribution_by_pipeline.png", dpi=180)
    plt.close(fig)


def plot_normalized_spectra_grid(output_dir: Path, draws, wave_nm, dsps_total, cigale_total, fsps_total) -> None:
    n = len(draws)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.5 * n), constrained_layout=True)
    for i, draw in enumerate(draws):
        ax = axes[i]
        for label, spectra, color in [
            ("DSPS + JAX-Cue", dsps_total, "black"),
            ("CIGALE fsps_stellar + nebular", cigale_total, "tab:blue"),
            ("direct FSPS nebular", fsps_total, "tab:orange"),
        ]:
            normalized = normalize_at_wavelength(spectra[i], wave_nm, 550.0)
            plot_one_spectrum(ax, wave_nm, normalized, label, color, lw=1.2, alpha=0.9)
        ax.axvline(91.16, color="0.7", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10.0, 3000.0)
        ax.set_ylabel("normalized")
        ax.set_title(f"{draw['label']} normalized at 550 nm")
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Rest wavelength [nm]")
    fig.savefig(output_dir / "dsps_cue_vs_references_spectra_normalized_550nm.png", dpi=180)
    plt.close(fig)


def normalize_at_wavelength(spectrum, wave_nm, wavelength_nm):
    spectrum = np.asarray(spectrum, dtype=float)
    good = np.isfinite(spectrum) & (spectrum > 0.0)
    if np.count_nonzero(good) < 2:
        return np.full_like(spectrum, np.nan)
    norm = np.interp(wavelength_nm, wave_nm[good], spectrum[good], left=np.nan, right=np.nan)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.full_like(spectrum, np.nan)
    return spectrum / norm


def maggies_to_ab(maggies: np.ndarray) -> np.ndarray:
    mag = np.full_like(maggies, np.nan, dtype=float)
    good = np.isfinite(maggies) & (maggies > 0.0)
    mag[good] = -2.5 * np.log10(maggies[good])
    return mag


def finite_median_abs_delta_mag(reference_phot, dsps_phot):
    delta = maggies_to_ab(reference_phot) - maggies_to_ab(dsps_phot)
    good = np.isfinite(delta)
    return float(np.nanmedian(np.abs(delta[good]))) if np.any(good) else None


def median_spectral_ratio(reference, dsps, wave_nm, lo, hi):
    ratios = []
    band = (wave_nm >= lo) & (wave_nm <= hi)
    for ref, base in zip(reference, dsps):
        good = band & np.isfinite(ref) & np.isfinite(base) & (ref > 0.0) & (base > 0.0)
        if np.any(good):
            ratios.append(np.nanmedian(ref[good] / base[good]))
    return float(np.nanmedian(ratios)) if ratios else None


def median_continuum_scale(reference, dsps, wave_nm, wavelength_nm):
    scales = []
    for ref, base in zip(reference, dsps):
        ref_norm = normalize_at_wavelength(ref, wave_nm, wavelength_nm)
        base_norm = normalize_at_wavelength(base, wave_nm, wavelength_nm)
        if np.isfinite(ref_norm).any() and np.isfinite(base_norm).any():
            ref_value = np.interp(wavelength_nm, wave_nm, ref)
            base_value = np.interp(wavelength_nm, wave_nm, base)
            if np.isfinite(ref_value) and np.isfinite(base_value) and ref_value > 0.0 and base_value > 0.0:
                scales.append(ref_value / base_value)
    return float(np.nanmedian(scales)) if scales else None


if __name__ == "__main__":
    main()
