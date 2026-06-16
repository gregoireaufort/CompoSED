"""Small BC03-vs-FSPS sweep inside the same CIGALE pipeline.

This is the first broad failure-mode check for ``fsps_stellar``. It samples a
handful of physically plausible CIGALE parameter vectors and compares two
otherwise identical pipelines:

    sfhdelayed -> bc03          -> nebular -> dustatt_modified_starburst -> redshifting
    sfhdelayed -> fsps_stellar  -> nebular -> dustatt_modified_starburst -> redshifting

The intent is not to make BC03 and FSPS agree exactly. It is to check that the
experimental FSPS-backed CIGALE stellar module behaves sensibly across a small
region of parameter space, and to localize differences to the SPS ingredient.

The script saves the sampled parameter vectors, per-filter AB magnitude
differences, spectra, and heatmaps so each odd case can be traced back to the
exact physical inputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from composed.experimental.cigale_fsps_stellar import register_cigale_fsps_stellar_module
from composed.experimental.cigale_fsps_stellar_conventions import (
    fsps_parameters_from_cigale_bc03,
)


OUTPUT_DIR = Path("outputs/experimental_cigale_fsps_bc03_parameter_sweep")
N_DRAWS = 20
RNG_SEED = 91723
SEPARATION_AGE_MYR = 10
Z_SUN_FOR_MAPPING = 0.02

BC03_METALLICITIES = np.asarray([0.0004, 0.004, 0.008, 0.02, 0.05], dtype=float)
FILTERS = [
    "galex.FUV",
    "galex.NUV",
    "LSST_u",
    "LSST_g",
    "LSST_r",
    "LSST_i",
    "LSST_z",
    "LSST_y",
    "vista.vircam.J",
    "vista.vircam.H",
    "vista.vircam.Ks",
]

MJY_PER_MAGGIE = 3631.0e3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_cigale_fsps_stellar_module()

    from pcigale.data import SimpleDatabase as Database
    from pcigale.utils.cosmology import age as universe_age_myr
    from pcigale.warehouse import SedWarehouse

    filter_pivots = load_filter_pivots(Database)
    draws = sample_parameter_draws(np.random.default_rng(RNG_SEED), universe_age_myr)

    # Several CIGALE modules cache wavelength-grid-dependent objects. The
    # BC03 and FSPS stellar grids are different, so keep the downstream modules
    # fresh for this mixed-grid diagnostic.
    warehouse = SedWarehouse(nocache=["fsps_stellar", "nebular", "dustatt_modified_starburst", "redshifting"])

    rows = []
    bc03_seds = []
    fsps_seds = []
    for idx, draw in enumerate(draws):
        bc03_sed = warehouse.get_sed(
            ["sfhdelayed", "bc03", "nebular", "dustatt_modified_starburst", "redshifting"],
            [
                draw["sfhdelayed"],
                draw["bc03"],
                draw["nebular"],
                draw["dustatt_modified_starburst"],
                draw["redshifting"],
            ],
        )
        fsps_sed = warehouse.get_sed(
            ["sfhdelayed", "fsps_stellar", "nebular", "dustatt_modified_starburst", "redshifting"],
            [
                draw["sfhdelayed"],
                draw["fsps_stellar"],
                draw["nebular"],
                draw["dustatt_modified_starburst"],
                draw["redshifting"],
            ],
        )
        bc03_seds.append(bc03_sed)
        fsps_seds.append(fsps_sed)
        rows.append(build_row(idx, draw, bc03_sed, fsps_sed, filter_pivots))
        print(f"Finished draw {idx + 1:02d}/{len(draws)}", flush=True)

    write_outputs(draws, rows, filter_pivots)
    plot_spectra_grid(rows, bc03_seds, fsps_seds)
    plot_broadband_sed_grid(rows, filter_pivots)
    plot_delta_heatmap(rows)
    print_summary(rows)


def sample_parameter_draws(rng, universe_age_myr):
    draws = []
    logu_grid = np.round(np.arange(-3.5, -1.49, 0.1), 1)
    for _ in range(N_DRAWS):
        redshift = float(rng.uniform(0.02, 4.0))
        max_age = max(120.0, min(12000.0, 0.9 * float(universe_age_myr(redshift))))
        age_main = int(rng.integers(100, int(max_age) + 1))
        age_burst = int(min(age_main, rng.integers(10, 301)))
        tau_main = float(10.0 ** rng.uniform(np.log10(150.0), np.log10(8000.0)))
        tau_burst = float(10.0 ** rng.uniform(np.log10(10.0), np.log10(300.0)))
        f_burst = float(rng.choice([0.0, rng.uniform(0.005, 0.25)], p=[0.45, 0.55]))

        metallicity = float(rng.choice(BC03_METALLICITIES))
        mapping = fsps_parameters_from_cigale_bc03(imf=1, metallicity=metallicity, z_sun=Z_SUN_FOR_MAPPING)

        ebv_lines = float(rng.uniform(0.0, 0.6))
        dust = {
            "E_BV_lines": ebv_lines,
            "E_BV_factor": 0.44,
            "uv_bump_wavelength": 217.5,
            "uv_bump_width": 35.0,
            "uv_bump_amplitude": float(rng.uniform(0.0, 3.0)),
            "powerlaw_slope": float(rng.uniform(-0.5, 0.2)),
            "Ext_law_emission_lines": 1,
            "Rv": 3.1,
            "filters": " & ".join(FILTERS),
        }
        draws.append(
            {
                "sfhdelayed": {
                    "age_main": age_main,
                    "tau_main": tau_main,
                    "age_burst": age_burst,
                    "tau_burst": tau_burst,
                    "f_burst": f_burst,
                    "normalise": True,
                },
                "bc03": {"imf": 1, "metallicity": mapping.metallicity, "separation_age": SEPARATION_AGE_MYR},
                "fsps_stellar": {
                    "imf_type": mapping.imf_type,
                    "logzsol": mapping.logzsol,
                    "z_sun": mapping.z_sun,
                    "zcontinuous": 1,
                    "separation_age": SEPARATION_AGE_MYR,
                },
                "nebular": {
                    "logU": float(rng.choice(logu_grid)),
                    "zgas": mapping.zgas,
                    "ne": 100,
                    "f_esc": 0.0,
                    "f_dust": 0.0,
                    "lines_width": 300.0,
                    "emission": True,
                },
                "dustatt_modified_starburst": dust,
                "redshifting": {"redshift": redshift},
            }
        )
    return draws


def load_filter_pivots(Database):
    pivots = {}
    with Database("filters") as db:
        for name in FILTERS:
            filt = db.get(name=name)
            pivots[name] = float(filt.pivot)
    return pivots


def build_row(idx, draw, bc03_sed, fsps_sed, filter_pivots):
    bc03_mags = compute_ab_mags(bc03_sed)
    fsps_mags = compute_ab_mags(fsps_sed)
    dmag = {name: safe_difference(fsps_mags[name], bc03_mags[name]) for name in FILTERS}
    ratio = spectral_ratio(fsps_sed, bc03_sed)
    finite_dmag = np.asarray([dmag[name] for name in FILTERS], dtype=float)
    finite_dmag = finite_dmag[np.isfinite(finite_dmag)]

    row = {
        "index": idx,
        "z": draw["redshifting"]["redshift"],
        "age_main_myr": draw["sfhdelayed"]["age_main"],
        "tau_main_myr": draw["sfhdelayed"]["tau_main"],
        "f_burst": draw["sfhdelayed"]["f_burst"],
        "age_burst_myr": draw["sfhdelayed"]["age_burst"],
        "tau_burst_myr": draw["sfhdelayed"]["tau_burst"],
        "metallicity": draw["bc03"]["metallicity"],
        "fsps_logzsol": draw["fsps_stellar"]["logzsol"],
        "logU": draw["nebular"]["logU"],
        "zgas": draw["nebular"]["zgas"],
        "E_BV_lines": draw["dustatt_modified_starburst"]["E_BV_lines"],
        "uv_bump_amplitude": draw["dustatt_modified_starburst"]["uv_bump_amplitude"],
        "powerlaw_slope": draw["dustatt_modified_starburst"]["powerlaw_slope"],
        "stellar_n_ly_ratio": safe_ratio(fsps_sed.info.get("stellar.n_ly"), bc03_sed.info.get("stellar.n_ly")),
        "stellar_lum_ratio": safe_ratio(fsps_sed.info.get("stellar.lum"), bc03_sed.info.get("stellar.lum")),
        "median_dmag": json_float(np.nanmedian(finite_dmag)) if finite_dmag.size else None,
        "max_abs_dmag": json_float(np.nanmax(np.abs(finite_dmag))) if finite_dmag.size else None,
        "median_ratio_120_900nm": window_median_ratio(ratio, 120.0, 900.0),
        "median_ratio_900_2500nm": window_median_ratio(ratio, 900.0, 2500.0),
        "median_ratio_10_91nm": window_median_ratio(ratio, 10.0, 91.1),
        "bc03_mags": bc03_mags,
        "fsps_mags": fsps_mags,
        "dmag_fsps_minus_bc03": dmag,
        "filter_pivots_nm": dict(filter_pivots),
    }
    return row


def compute_ab_mags(sed):
    mags = {}
    for filt in FILTERS:
        try:
            f_mjy = float(sed.compute_fnu(filt))
        except Exception:
            f_mjy = np.nan
        maggies = f_mjy / MJY_PER_MAGGIE if np.isfinite(f_mjy) else np.nan
        mags[filt] = json_float(-2.5 * np.log10(maggies)) if maggies > 0.0 else None
    return mags


def write_outputs(draws, rows, filter_pivots):
    with (OUTPUT_DIR / "parameter_draws.json").open("w") as f:
        json.dump(draws, f, indent=2, sort_keys=True)
    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(
            {
                "n_draws": len(rows),
                "filters": FILTERS,
                "filter_pivots_nm": filter_pivots,
                "rows": rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    fieldnames = [
        "index",
        "z",
        "age_main_myr",
        "tau_main_myr",
        "f_burst",
        "metallicity",
        "fsps_logzsol",
        "logU",
        "E_BV_lines",
        "stellar_n_ly_ratio",
        "stellar_lum_ratio",
        "median_dmag",
        "max_abs_dmag",
        "median_ratio_120_900nm",
        "median_ratio_900_2500nm",
        "median_ratio_10_91nm",
    ] + [f"dmag_{name}" for name in FILTERS]
    with (OUTPUT_DIR / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fieldnames if not key.startswith("dmag_")}
            for name in FILTERS:
                flat[f"dmag_{name}"] = row["dmag_fsps_minus_bc03"][name]
            writer.writerow(flat)


def plot_spectra_grid(rows, bc03_seds, fsps_seds):
    fig, axes = plt.subplots(5, 4, figsize=(16, 15), sharex=True, sharey=False)
    for row, bc03_sed, fsps_sed, ax in zip(rows, bc03_seds, fsps_seds, axes.flat):
        plot_sed_spectrum(ax, bc03_sed, color="tab:blue", label="BC03")
        plot_sed_spectrum(ax, fsps_sed, color="tab:orange", label="FSPS")
        autoscale_spectrum_axis(ax, bc03_sed, fsps_sed)
        ax.set_title(
            f"#{row['index']} z={row['z']:.2f}, Z={row['metallicity']:.4g}\n"
            f"age={row['age_main_myr']} Myr, E(B-V)l={row['E_BV_lines']:.2f}",
            fontsize=8,
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Observed wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$L_\lambda$ [W nm$^{-1}$]")
    fig.suptitle("Final CIGALE spectra: BC03 vs experimental fsps_stellar")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spectra_grid.png", dpi=170)
    plt.close(fig)


def plot_broadband_sed_grid(rows, filter_pivots):
    x = np.asarray([filter_pivots[name] for name in FILTERS], dtype=float)
    fig, axes = plt.subplots(5, 4, figsize=(16, 15), sharex=True, sharey=True)
    for row, ax in zip(rows, axes.flat):
        bc03 = np.asarray([row["bc03_mags"][name] for name in FILTERS], dtype=float)
        fsps = np.asarray([row["fsps_mags"][name] for name in FILTERS], dtype=float)
        ax.plot(x, bc03, "o-", color="tab:blue", lw=1.1, ms=3, label="BC03")
        ax.plot(x, fsps, "o-", color="tab:orange", lw=1.1, ms=3, label="FSPS")
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_title(f"#{row['index']} z={row['z']:.2f}; max |Δm|={row['max_abs_dmag']:.2f}", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Filter pivot wavelength [nm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("AB mag")
    fig.suptitle("Broadband SEDs from CIGALE filter integration")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "broadband_sed_grid.png", dpi=170)
    plt.close(fig)


def plot_delta_heatmap(rows):
    dmag = np.asarray([[row["dmag_fsps_minus_bc03"][name] for name in FILTERS] for row in rows], dtype=float)
    order = np.argsort([row["z"] for row in rows])
    dmag = dmag[order]
    labels = [f"#{rows[i]['index']} z={rows[i]['z']:.2f}" for i in order]
    vmax = max(0.1, float(np.nanpercentile(np.abs(dmag), 95)))

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(dmag, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(FILTERS)), FILTERS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Broadband ΔAB = FSPS - BC03, sorted by redshift")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mag")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "broadband_delta_heatmap.png", dpi=180)
    plt.close(fig)


def plot_sed_spectrum(ax, sed, color, label):
    wave = np.asarray(sed.wavelength_grid, dtype=float)
    lum = np.asarray(sed.luminosity, dtype=float)
    good = np.isfinite(wave) & np.isfinite(lum) & (wave > 0.0) & (lum > 0.0)
    ax.loglog(wave[good], lum[good], color=color, lw=1.0, alpha=0.9, label=label)
    ax.set_xlim(50.0, 5000.0)


def autoscale_spectrum_axis(ax, *seds):
    values = []
    for sed in seds:
        wave = np.asarray(sed.wavelength_grid, dtype=float)
        lum = np.asarray(sed.luminosity, dtype=float)
        use = (wave >= 50.0) & (wave <= 5000.0) & np.isfinite(lum) & (lum > 0.0)
        if np.any(use):
            values.append(lum[use])
    if not values:
        return
    log_lum = np.log10(np.concatenate(values))
    lo, hi = np.nanpercentile(log_lum, [2.0, 99.8])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi - lo < 1.0:
        mid = 0.5 * (hi + lo)
        lo, hi = mid - 0.5, mid + 0.5
    ax.set_ylim(10.0 ** (lo - 0.4), 10.0 ** (hi + 0.3))


def spectral_ratio(numerator_sed, denominator_sed):
    wave = np.asarray(denominator_sed.wavelength_grid, dtype=float)
    denom = np.asarray(denominator_sed.luminosity, dtype=float)
    num = np.interp(
        wave,
        np.asarray(numerator_sed.wavelength_grid, dtype=float),
        np.asarray(numerator_sed.luminosity, dtype=float),
        left=np.nan,
        right=np.nan,
    )
    ratio = np.full_like(wave, np.nan, dtype=float)
    good = np.isfinite(num) & np.isfinite(denom) & (denom > 0.0)
    ratio[good] = num[good] / denom[good]
    return np.column_stack([wave, ratio])


def window_median_ratio(ratio, lo, hi):
    use = (ratio[:, 0] >= lo) & (ratio[:, 0] <= hi) & np.isfinite(ratio[:, 1]) & (ratio[:, 1] > 0.0)
    if np.count_nonzero(use) == 0:
        return None
    return json_float(np.nanmedian(ratio[use, 1]))


def safe_ratio(num, den):
    if num is None or den is None:
        return None
    den = float(den)
    if den == 0.0 or not np.isfinite(den):
        return None
    return json_float(float(num) / den)


def safe_difference(num, den):
    if num is None or den is None:
        return None
    return json_float(float(num) - float(den))


def json_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def print_summary(rows):
    median_abs = np.nanmedian([row["max_abs_dmag"] for row in rows])
    max_abs = np.nanmax([row["max_abs_dmag"] for row in rows])
    nly = np.asarray([row["stellar_n_ly_ratio"] for row in rows], dtype=float)
    print(f"Ran {len(rows)} CIGALE parameter draws.")
    print(f"Median over objects of max |ΔAB| across filters: {median_abs:.3f} mag")
    print(f"Largest object-level max |ΔAB| across filters: {max_abs:.3f} mag")
    print(f"Median stellar.n_ly FSPS/BC03: {np.nanmedian(nly):.3g}")
    print(f"Saved outputs under {OUTPUT_DIR}")
    print(f"  {OUTPUT_DIR / 'spectra_grid.png'}")
    print(f"  {OUTPUT_DIR / 'broadband_sed_grid.png'}")
    print(f"  {OUTPUT_DIR / 'broadband_delta_heatmap.png'}")
    print(f"  {OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
