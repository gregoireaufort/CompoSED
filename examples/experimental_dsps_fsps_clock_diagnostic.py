#!/usr/bin/env python
"""Diagnose DSPS/FSPS differences caused by SFH time conventions.

This is intentionally a small, auditable diagnostic.  It asks one question:

    Are we giving DSPS the same tabular SFH clock that FSPS sees?

Both DSPS and python-fsps document tabular SFH time as *cosmic time*, i.e. the
age of the Universe in Gyr.  Many SED-fitting notebooks instead write a
galaxy-age grid from 0 to ``tage`` and ask FSPS for ``get_spectrum(tage=tage)``.
That is fine as a local CSP convention, but it is not the same thing as giving
DSPS an age grid while also setting ``t_obs = age_of_universe(z)``.

Run the stages in the environments that have the required dependencies:

    # FSPS environment: build a continuum-only FSPS SSP table and direct FSPS CSPs
    SPS_HOME=/Users/gregoire/Work/FSPS \
    PYTHONPATH=/Users/gregoire/Documents/Sedfitting/CompoSED \
    /Users/gregoire/opt/anaconda3/envs/sbi_candide/bin/python \
        examples/experimental_dsps_fsps_clock_diagnostic.py --stage make-ssp

    SPS_HOME=/Users/gregoire/Work/FSPS \
    PYTHONPATH=/Users/gregoire/Documents/Sedfitting/CompoSED \
    /Users/gregoire/opt/anaconda3/envs/sbi_candide/bin/python \
        examples/experimental_dsps_fsps_clock_diagnostic.py --stage fsps

    # DSPS/JAX environment
    PYTHONPATH=/Users/gregoire/Documents/Sedfitting/CompoSED \
    /Users/gregoire/miniforge3/envs/dsps_nuts/bin/python \
        examples/experimental_dsps_fsps_clock_diagnostic.py --stage dsps

    # Either environment with matplotlib
    python examples/experimental_dsps_fsps_clock_diagnostic.py --stage plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

C_A_PER_S = 2.99792458e18
Z_SUN_FSPS = 0.019

DEFAULT_OUTPUT_DIR = Path("outputs/experimental_dsps_fsps_clock_diagnostic")
DEFAULT_SSP_PATH = DEFAULT_OUTPUT_DIR / "fsps_continuum_ssp_data.h5"
REST_WAVE_A = np.geomspace(95.0, 30000.0, 2400)
REST_WAVE_NM = REST_WAVE_A / 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "make-ssp", "fsps", "dsps", "plots"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ssp-file", type=Path, default=DEFAULT_SSP_PATH)
    parser.add_argument("--metallicity-scatter", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draws = load_or_create_draws(args.output_dir / "parameter_draws.json")

    if args.stage in {"all", "make-ssp"}:
        make_continuum_fsps_ssp_table(args.ssp_file)
    if args.stage in {"all", "fsps"}:
        run_direct_fsps_stage(draws, args)
    if args.stage in {"all", "dsps"}:
        run_dsps_stage(draws, args)
    if args.stage in {"all", "plots"}:
        make_plots_and_summary(args.output_dir)


def load_or_create_draws(path: Path) -> list[dict[str, float]]:
    """Small deterministic draw set covering young, old, low-z, and high-z CSPs."""

    if path.exists():
        return json.loads(path.read_text())
    draws = [
        {"label": "young_z04", "z": 0.4, "tage_gyr": 0.8, "tau_gyr": 1.6, "logzsol": -0.5},
        {"label": "solar_z10", "z": 1.0, "tage_gyr": 2.2, "tau_gyr": 3.0, "logzsol": 0.0},
        {"label": "very_young_z20", "z": 2.0, "tage_gyr": 0.45, "tau_gyr": 0.8, "logzsol": -1.0},
        {"label": "old_z07", "z": 0.7, "tage_gyr": 5.0, "tau_gyr": 1.0, "logzsol": 0.25},
        {"label": "blue_lowz", "z": 0.12, "tage_gyr": 1.8, "tau_gyr": 4.0, "logzsol": -0.7},
    ]
    path.write_text(json.dumps(draws, indent=2))
    return draws


def make_continuum_fsps_ssp_table(path: Path) -> None:
    """Write the exact FSPS SSP grid DSPS should consume for a fair comparison."""

    import fsps
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing continuum-only FSPS SSP table:", path)

    sp0 = fsps.StellarPopulation(
        zcontinuous=0,
        imf_type=1,
        add_neb_emission=False,
        add_dust_emission=False,
        compute_vega_mags=False,
    )
    ssp_lgmet = np.log10(np.asarray(sp0.zlegend, dtype=float))
    ssp_lg_age_gyr = np.asarray(sp0.log_age, dtype=float) - 9.0

    spectra_lsun_per_hz_per_msun = []
    wave_a = None
    for zmet_index in range(1, ssp_lgmet.size + 1):
        sp = fsps.StellarPopulation(
            zcontinuous=0,
            zmet=zmet_index,
            imf_type=1,
            add_neb_emission=False,
            add_dust_emission=False,
            compute_vega_mags=False,
        )
        wave_a, spec_lsun_per_hz = sp.get_spectrum(peraa=False)
        spectra_lsun_per_hz_per_msun.append(np.asarray(spec_lsun_per_hz, dtype=float))
        print(f"  zmet {zmet_index:02d}/{ssp_lgmet.size}")

    with h5py.File(path, "w") as h5:
        h5.attrs["composed_ssp_kind"] = "stellar_continuum"
        h5.attrs["fsps_add_neb_emission"] = False
        h5.attrs["fsps_add_dust_emission"] = False
        h5["ssp_lgmet"] = ssp_lgmet
        h5["ssp_lg_age_gyr"] = ssp_lg_age_gyr
        h5["ssp_wave"] = np.asarray(wave_a, dtype=float)
        h5["ssp_flux"] = np.asarray(spectra_lsun_per_hz_per_msun, dtype=float)
    print("Saved:", path)


def run_direct_fsps_stage(draws: list[dict[str, float]], args: argparse.Namespace) -> None:
    """Compute direct python-fsps CSPs under two equivalent clock conventions."""

    import fsps

    age_axis = []
    cosmic_axis = []
    for draw in draws:
        wave_a, lum = direct_fsps_csp(fsps, draw, clock="age")
        age_axis.append(np.interp(REST_WAVE_A, wave_a, lum, left=np.nan, right=np.nan))

        wave_a, lum = direct_fsps_csp(fsps, draw, clock="cosmic")
        cosmic_axis.append(np.interp(REST_WAVE_A, wave_a, lum, left=np.nan, right=np.nan))
        print(f"  FSPS {draw['label']}")

    np.savez(
        args.output_dir / "direct_fsps_clock_spectra.npz",
        rest_wave_nm=REST_WAVE_NM,
        fsps_age_axis_lsun_per_a=np.asarray(age_axis),
        fsps_cosmic_axis_lsun_per_a=np.asarray(cosmic_axis),
    )
    print("Saved:", args.output_dir / "direct_fsps_clock_spectra.npz")


def direct_fsps_csp(fsps, draw: dict[str, float], *, clock: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a stellar-only FSPS CSP spectrum in Lsun / Angstrom."""

    sp = fsps.StellarPopulation(
        zcontinuous=1,
        sfh=3,
        imf_type=1,
        add_neb_emission=False,
        add_dust_emission=False,
        compute_vega_mags=False,
    )
    sp.params["logzsol"] = float(draw["logzsol"])
    age_gyr, sfr = delayed_sfh_on_galaxy_age_grid(draw)
    if clock == "age":
        time_gyr = age_gyr
        tage_gyr = float(draw["tage_gyr"])
    elif clock == "cosmic":
        t_obs = flat_lcdm_age_gyr_numpy(draw["z"])
        time_gyr = t_obs - float(draw["tage_gyr"]) + age_gyr
        tage_gyr = t_obs
    else:
        raise ValueError("clock must be 'age' or 'cosmic'.")
    sp.set_tabular_sfh(time_gyr, sfr)
    wave_a, lum_lsun_per_a = sp.get_spectrum(tage=tage_gyr, peraa=True)
    return np.asarray(wave_a, dtype=float), np.asarray(lum_lsun_per_a, dtype=float)


def run_dsps_stage(draws: list[dict[str, float]], args: argparse.Namespace) -> None:
    """Compute DSPS CSPs with correct and intentionally wrong clocks."""

    from dsps import calc_rest_sed_sfh_table_lognormal_mdf, load_ssp_templates

    ssp = load_ssp_templates(fn=str(args.ssp_file))
    print("Loaded DSPS SSP table:", args.ssp_file)

    current_wrong = []
    age_axis = []
    cosmic_axis = []
    for draw in draws:
        age_gyr, sfr = delayed_sfh_on_galaxy_age_grid(draw)
        t_obs_cosmic = flat_lcdm_age_gyr_numpy(draw["z"])
        t_obs_age = float(draw["tage_gyr"])
        cosmic_time_gyr = t_obs_cosmic - float(draw["tage_gyr"]) + age_gyr

        # This is the suspicious convention used by the first experimental
        # JAX-CIGALE pass: age-since-onset grid, but cosmological t_obs.
        current_wrong.append(
            dsps_spectrum_lsun_per_a(
                calc_rest_sed_sfh_table_lognormal_mdf,
                ssp,
                age_gyr,
                sfr,
                t_obs_cosmic,
                draw["logzsol"],
                args.metallicity_scatter,
            )
        )

        # This matches the notebook/FSPS local-CSP convention:
        # galaxy age grid and t_obs = tage.
        age_axis.append(
            dsps_spectrum_lsun_per_a(
                calc_rest_sed_sfh_table_lognormal_mdf,
                ssp,
                age_gyr,
                sfr,
                t_obs_age,
                draw["logzsol"],
                args.metallicity_scatter,
            )
        )

        # This matches the cosmological tabular-SFH convention:
        # cosmic time grid and t_obs = age_of_universe(z).
        cosmic_axis.append(
            dsps_spectrum_lsun_per_a(
                calc_rest_sed_sfh_table_lognormal_mdf,
                ssp,
                cosmic_time_gyr,
                sfr,
                t_obs_cosmic,
                draw["logzsol"],
                args.metallicity_scatter,
            )
        )
        print(f"  DSPS {draw['label']}")

    np.savez(
        args.output_dir / "dsps_clock_spectra.npz",
        rest_wave_nm=REST_WAVE_NM,
        dsps_current_wrong_lsun_per_a=np.asarray(current_wrong),
        dsps_age_axis_lsun_per_a=np.asarray(age_axis),
        dsps_cosmic_axis_lsun_per_a=np.asarray(cosmic_axis),
    )
    print("Saved:", args.output_dir / "dsps_clock_spectra.npz")


def dsps_spectrum_lsun_per_a(
    calc_rest_sed,
    ssp,
    time_gyr: np.ndarray,
    sfr_msun_per_yr: np.ndarray,
    t_obs_gyr: float,
    logzsol: float,
    metallicity_scatter: float,
) -> np.ndarray:
    """Evaluate DSPS and convert Lnu [Lsun/Hz] to Llambda [Lsun/A]."""

    import jax.numpy as jnp

    gal_lgmet = np.log10(Z_SUN_FSPS) + float(logzsol)
    sed = calc_rest_sed(
        jnp.asarray(time_gyr),
        jnp.asarray(sfr_msun_per_yr),
        jnp.asarray(gal_lgmet),
        jnp.asarray(float(metallicity_scatter)),
        jnp.asarray(ssp.ssp_lgmet),
        jnp.asarray(ssp.ssp_lg_age_gyr),
        jnp.asarray(ssp.ssp_flux),
        jnp.asarray(float(t_obs_gyr)),
    )
    lnu_lsun_per_hz = np.asarray(sed.rest_sed, dtype=float)
    lnu_on_grid = np.interp(REST_WAVE_A, np.asarray(ssp.ssp_wave, dtype=float), lnu_lsun_per_hz)
    return lnu_on_grid * C_A_PER_S / REST_WAVE_A**2


def delayed_sfh_on_galaxy_age_grid(draw: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Delayed-tau SFR on age since SF onset, normalized to 1 Msun formed."""

    tage = float(draw["tage_gyr"])
    tau = max(float(draw["tau_gyr"]), 1.0e-4)
    # DSPS assumes the tabulated cosmic time grid starts at or after its
    # internal minimum birth time.  A 20 Myr first support point is also the
    # convention used in the current JAX-CIGALE demos.
    age = np.linspace(0.02, tage, 512)
    raw = age * np.exp(-age / tau)
    formed_mass = trapezoid(raw, age * 1.0e9)
    return age, raw / formed_mass


def make_plots_and_summary(output_dir: Path) -> None:
    draws = json.loads((output_dir / "parameter_draws.json").read_text())
    dsps = np.load(output_dir / "dsps_clock_spectra.npz")
    fsps = np.load(output_dir / "direct_fsps_clock_spectra.npz")
    wave_nm = dsps["rest_wave_nm"]

    products = {
        "FSPS age-axis": fsps["fsps_age_axis_lsun_per_a"],
        "FSPS cosmic-axis": fsps["fsps_cosmic_axis_lsun_per_a"],
        "DSPS current wrong": dsps["dsps_current_wrong_lsun_per_a"],
        "DSPS age-axis": dsps["dsps_age_axis_lsun_per_a"],
        "DSPS cosmic-axis": dsps["dsps_cosmic_axis_lsun_per_a"],
    }

    plot_spectra(output_dir, draws, wave_nm, products)
    plot_ratios(output_dir, draws, wave_nm, products)

    summary = {
        "n_draws": len(draws),
        "ratios_at_550nm": {
            "dsps_current_wrong_over_fsps_cosmic": continuum_ratios(
                products["DSPS current wrong"], products["FSPS cosmic-axis"], wave_nm, 550.0
            ),
            "dsps_cosmic_over_fsps_cosmic": continuum_ratios(
                products["DSPS cosmic-axis"], products["FSPS cosmic-axis"], wave_nm, 550.0
            ),
            "dsps_age_over_fsps_age": continuum_ratios(
                products["DSPS age-axis"], products["FSPS age-axis"], wave_nm, 550.0
            ),
            "fsps_cosmic_over_fsps_age": continuum_ratios(
                products["FSPS cosmic-axis"], products["FSPS age-axis"], wave_nm, 550.0
            ),
        },
        "median_abs_log10_ratio_120_900nm": {
            "dsps_current_wrong_vs_fsps_cosmic": median_abs_log_ratio(
                products["DSPS current wrong"], products["FSPS cosmic-axis"], wave_nm, 120.0, 900.0
            ),
            "dsps_cosmic_vs_fsps_cosmic": median_abs_log_ratio(
                products["DSPS cosmic-axis"], products["FSPS cosmic-axis"], wave_nm, 120.0, 900.0
            ),
            "dsps_age_vs_fsps_age": median_abs_log_ratio(
                products["DSPS age-axis"], products["FSPS age-axis"], wave_nm, 120.0, 900.0
            ),
        },
    }
    (output_dir / "clock_diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved plots under:", output_dir)


def plot_spectra(output_dir: Path, draws, wave_nm, products) -> None:
    n = len(draws)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    colors = {
        "FSPS age-axis": "tab:blue",
        "FSPS cosmic-axis": "tab:cyan",
        "DSPS current wrong": "tab:red",
        "DSPS age-axis": "tab:green",
        "DSPS cosmic-axis": "black",
    }
    linestyles = {
        "FSPS age-axis": "-",
        "FSPS cosmic-axis": "--",
        "DSPS current wrong": "-",
        "DSPS age-axis": "-",
        "DSPS cosmic-axis": "--",
    }
    for i, draw in enumerate(draws):
        ax = axes[i]
        for label, spectra in products.items():
            ax.plot(wave_nm, spectra[i], label=label, color=colors[label], ls=linestyles[label], lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(90.0, 3000.0)
        ax.set_title(
            f"{draw['label']}: z={draw['z']}, tage={draw['tage_gyr']} Gyr, "
            f"tau={draw['tau_gyr']} Gyr, logZ/Zsun={draw['logzsol']}"
        )
        ax.set_ylabel(r"$L_\lambda$ [L$_\odot$ A$^{-1}$]")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Rest wavelength [nm]")
    fig.savefig(output_dir / "clock_diagnostic_spectra.png", dpi=180)
    plt.close(fig)


def plot_ratios(output_dir: Path, draws, wave_nm, products) -> None:
    n = len(draws)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    ratio_specs = [
        ("DSPS current wrong / FSPS cosmic", products["DSPS current wrong"], products["FSPS cosmic-axis"], "tab:red"),
        ("DSPS cosmic / FSPS cosmic", products["DSPS cosmic-axis"], products["FSPS cosmic-axis"], "black"),
        ("DSPS age / FSPS age", products["DSPS age-axis"], products["FSPS age-axis"], "tab:green"),
        ("FSPS cosmic / FSPS age", products["FSPS cosmic-axis"], products["FSPS age-axis"], "tab:cyan"),
    ]
    for i, draw in enumerate(draws):
        ax = axes[i]
        for label, numerator, denominator, color in ratio_specs:
            ratio = safe_ratio(numerator[i], denominator[i])
            ax.plot(wave_nm, ratio, label=label, color=color, lw=1.2)
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(90.0, 3000.0)
        ax.set_ylim(1.0e-2, 1.0e2)
        ax.set_title(draw["label"])
        ax.set_ylabel("ratio")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Rest wavelength [nm]")
    fig.savefig(output_dir / "clock_diagnostic_ratios.png", dpi=180)
    plt.close(fig)


def continuum_ratios(numerator: np.ndarray, denominator: np.ndarray, wave_nm: np.ndarray, wavelength_nm: float) -> list[float]:
    idx = int(np.argmin(np.abs(wave_nm - wavelength_nm)))
    return [float(safe_ratio(num[idx], den[idx])) for num, den in zip(numerator, denominator)]


def median_abs_log_ratio(numerator: np.ndarray, denominator: np.ndarray, wave_nm: np.ndarray, lo_nm: float, hi_nm: float) -> float:
    mask = (wave_nm >= lo_nm) & (wave_nm <= hi_nm)
    values = []
    for num, den in zip(numerator, denominator):
        ratio = safe_ratio(num[mask], den[mask])
        ok = np.isfinite(ratio) & (ratio > 0.0)
        values.extend(np.abs(np.log10(ratio[ok])))
    return float(np.median(values))


def safe_ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(np.asarray(numerator, dtype=float), np.nan), where=np.asarray(denominator) > 0.0)


def flat_lcdm_age_gyr_numpy(z: float, omega_m: float = 0.3075, h: float = 0.6774) -> float:
    """Age of a flat matter+Lambda universe in Gyr, matching JAX-CIGALE."""

    z = float(z)
    omega_l = 1.0 - omega_m
    hubble_time_gyr = 9.778 / h
    arg = np.sqrt(omega_l / omega_m) / (1.0 + z) ** 1.5
    return float((2.0 / (3.0 * np.sqrt(omega_l))) * np.arcsinh(arg) * hubble_time_gyr)


def trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


if __name__ == "__main__":
    main()
