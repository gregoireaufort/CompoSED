"""CIGALE escape-fraction sweep for the experimental FSPS stellar module.

This script asks a practical UV question: how much do GALEX FUV/NUV fluxes
change when the CIGALE nebular escape fraction changes? It compares direct FSPS
``add_neb_emission=True`` with CIGALE ``fsps_stellar + nebular`` and then
redshifts the same model through GALEX filters.

Saved outputs are the rest-frame spectrum plot, the GALEX flux plot, a CSV
table of fluxes, and a JSON summary. The CSV is meant to be the audit trail.
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


OUTPUT_DIR = Path("outputs/experimental_cigale_escape_fraction_sweep")

LSUN_W = 3.828e26
H_J_S = 6.62607015e-34
C_M_S = 2.99792458e8
MJY_PER_MAGGIE = 3631.0e3

SFH_PARAMS = {"age_main": 1000, "tau_main": 3000.0, "normalise": True}
FSPS_STELLAR_PARAMS = {"imf_type": 1, "logzsol": 0.0, "zcontinuous": 1, "separation_age": 10}
BASE_NEBULAR_PARAMS = {"logU": -2.0, "zgas": 0.014, "f_dust": 0.0, "emission": True}
ESCAPE_FRACTIONS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
REDSHIFTS = [0.05, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
FILTERS = ["galex.FUV", "galex.NUV"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_cigale_fsps_stellar_module()

    from pcigale.warehouse import SedWarehouse

    warehouse = SedWarehouse(nocache=["fsps_stellar", "nebular", "redshifting"])
    sfh_sed = warehouse.get_sed(["sfhdelayed"], [SFH_PARAMS])
    sfh = np.asarray(sfh_sed.sfh, dtype=float)

    stellar = warehouse.get_sed(["sfhdelayed", "fsps_stellar"], [SFH_PARAMS, FSPS_STELLAR_PARAMS])
    cigale_by_fesc = {
        fesc: warehouse.get_sed(
            ["sfhdelayed", "fsps_stellar", "nebular"],
            [SFH_PARAMS, FSPS_STELLAR_PARAMS, nebular_params(fesc)],
        )
        for fesc in ESCAPE_FRACTIONS
    }
    direct_fsps = direct_fsps_nebular_spectrum(sfh)

    photometry_rows = compute_galex_photometry(warehouse)
    summary = build_summary(stellar, cigale_by_fesc, direct_fsps, photometry_rows)
    write_outputs(summary, photometry_rows)
    plot_rest_spectra(stellar, cigale_by_fesc, direct_fsps)
    plot_galex_photometry(photometry_rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved rest-spectrum plot: {OUTPUT_DIR / 'escape_fraction_rest_spectra.png'}")
    print(f"Saved GALEX plot: {OUTPUT_DIR / 'escape_fraction_galex_fluxes.png'}")
    print(f"Saved photometry table: {OUTPUT_DIR / 'galex_fluxes.csv'}")


def nebular_params(fesc):
    params = dict(BASE_NEBULAR_PARAMS)
    params["f_esc"] = float(fesc)
    return params


def direct_fsps_nebular_spectrum(sfh_msun_per_yr):
    import fsps

    sp = fsps.StellarPopulation(
        zcontinuous=FSPS_STELLAR_PARAMS["zcontinuous"],
        sfh=3,
        imf_type=FSPS_STELLAR_PARAMS["imf_type"],
        add_neb_emission=True,
        add_neb_continuum=True,
        compute_vega_mags=False,
    )
    sp.params["logzsol"] = FSPS_STELLAR_PARAMS["logzsol"]
    sp.params["gas_logu"] = BASE_NEBULAR_PARAMS["logU"]
    sp.params["gas_logz"] = 0.0
    time_gyr = np.arange(1, sfh_msun_per_yr.size + 1, dtype=float) / 1000.0
    sp.set_tabular_sfh(time_gyr, sfh_msun_per_yr)
    wave_a, llam_lsun_per_a = sp.get_spectrum(tage=float(time_gyr[-1]), peraa=True)
    return {
        "wave_nm": np.asarray(wave_a, dtype=float) / 10.0,
        "lum_w_per_nm": np.asarray(llam_lsun_per_a, dtype=float) * LSUN_W * 10.0,
    }


def compute_galex_photometry(warehouse):
    rows = []
    for fesc in ESCAPE_FRACTIONS:
        for z in REDSHIFTS:
            sed = warehouse.get_sed(
                ["sfhdelayed", "fsps_stellar", "nebular", "redshifting"],
                [SFH_PARAMS, FSPS_STELLAR_PARAMS, nebular_params(fesc), {"redshift": z}],
            )
            row = {"f_esc": float(fesc), "redshift": float(z)}
            for filt in FILTERS:
                try:
                    f_mjy = float(sed.compute_fnu(filt))
                except Exception:
                    f_mjy = np.nan
                maggies = f_mjy / MJY_PER_MAGGIE if np.isfinite(f_mjy) else np.nan
                abmag = -2.5 * np.log10(maggies) if maggies > 0.0 else np.nan
                row[f"{filt}_mJy"] = f_mjy
                row[f"{filt}_maggies"] = maggies
                row[f"{filt}_ABmag"] = abmag
            rows.append(row)
    return rows


def build_summary(stellar, cigale_by_fesc, direct_fsps, photometry_rows):
    lyman = {}
    for fesc, sed in cigale_by_fesc.items():
        spec = spectrum_dict_from_cigale(sed)
        lyman[str(fesc)] = {
            "n_ly_like": nly_from_spectrum(spec),
            "luminosity_w": lum_from_spectrum(spec),
        }
    direct = {
        "n_ly_like": nly_from_spectrum(direct_fsps),
        "luminosity_w": lum_from_spectrum(direct_fsps),
    }
    stellar_spec = spectrum_dict_from_cigale(stellar)
    stellar_pre = {
        "n_ly_like": nly_from_spectrum(stellar_spec),
        "luminosity_w": lum_from_spectrum(stellar_spec),
        "stellar.n_ly_info": _json_float(stellar.info.get("stellar.n_ly")),
    }
    return {
        "parameters": {
            "sfhdelayed": SFH_PARAMS,
            "fsps_stellar": FSPS_STELLAR_PARAMS,
            "nebular_base": BASE_NEBULAR_PARAMS,
            "escape_fractions": ESCAPE_FRACTIONS,
            "redshifts": REDSHIFTS,
        },
        "lyman_continuum_integrals": {
            "cigale_stellar_before_nebular": stellar_pre,
            "cigale_after_nebular_by_f_esc": lyman,
            "direct_fsps_add_neb_emission": direct,
        },
        "galex_flux_dynamic_range": galex_dynamic_range(photometry_rows),
    }


def galex_dynamic_range(rows):
    out = {}
    for z in REDSHIFTS:
        subset = [row for row in rows if row["redshift"] == z]
        for filt in FILTERS:
            vals = np.asarray([row[f"{filt}_maggies"] for row in subset], dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size == 0:
                out[f"{filt}_z{z}"] = None
            else:
                out[f"{filt}_z{z}"] = {
                    "max_over_min": _json_float(vals.max() / vals.min()),
                    "mag_range": _json_float(-2.5 * np.log10(vals.max() / vals.min())),
                }
    return out


def write_outputs(summary, rows):
    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (OUTPUT_DIR / "galex_fluxes.csv").open("w", newline="") as f:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rest_spectra(stellar, cigale_by_fesc, direct_fsps) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax, rax = axes
    stellar_spec = spectrum_dict_from_cigale(stellar)
    _plot_dict(ax, stellar_spec, "CIGALE fsps_stellar before nebular", "0.55", ls="--")
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ESCAPE_FRACTIONS)))
    fesc0 = spectrum_dict_from_cigale(cigale_by_fesc[0.0])
    for color, fesc in zip(colors, ESCAPE_FRACTIONS):
        spec = spectrum_dict_from_cigale(cigale_by_fesc[fesc])
        _plot_dict(ax, spec, f"CIGALE nebular fesc={fesc:g}", color)
        ratio = spectrum_ratio(spec, fesc0)
        good = np.isfinite(ratio[:, 1]) & (ratio[:, 1] > 0.0)
        rax.semilogx(ratio[good, 0], ratio[good, 1], color=color, lw=1.2)
    _plot_dict(ax, direct_fsps, "Direct FSPS add_neb_emission=True", "tab:orange", lw=2.0)
    rax.axhline(1.0, color="0.5", ls="--", lw=1)
    rax.set_yscale("log")
    rax.set_ylim(1e-3, 1e6)
    rax.set_ylabel("CIGALE fesc / fesc=0")
    rax.set_xlabel("Rest wavelength [nm]")
    for axis in axes:
        axis.axvline(91.1, color="0.7", ls=":", lw=1)
        axis.set_xlim(10.0, 3000.0)
        axis.grid(alpha=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$L_\lambda$ [W nm$^{-1}$]")
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.set_title("CIGALE escape fraction sweep with experimental fsps_stellar")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "escape_fraction_rest_spectra.png", dpi=180)
    plt.close(fig)


def plot_galex_photometry(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, filt in zip(axes, FILTERS):
        for z in REDSHIFTS:
            subset = [row for row in rows if row["redshift"] == z]
            fesc = np.asarray([row["f_esc"] for row in subset], dtype=float)
            flux = np.asarray([row[f"{filt}_maggies"] for row in subset], dtype=float)
            good = np.isfinite(flux) & (flux > 0.0)
            if np.any(good):
                ax.plot(fesc[good], flux[good], marker="o", lw=1.4, label=f"z={z:g}")
        ax.set_yscale("log")
        ax.set_xlabel(r"CIGALE nebular $f_{\rm esc}$")
        ax.set_title(filt)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed flux [maggies]")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Native CIGALE GALEX fluxes after redshifting/IGM")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "escape_fraction_galex_fluxes.png", dpi=180)
    plt.close(fig)


def spectrum_dict_from_cigale(sed):
    return {"wave_nm": np.asarray(sed.wavelength_grid, dtype=float), "lum_w_per_nm": np.asarray(sed.luminosity, dtype=float)}


def _plot_dict(ax, spectrum, label, color, ls="-", lw=1.3):
    wave = spectrum["wave_nm"]
    lum = spectrum["lum_w_per_nm"]
    good = np.isfinite(wave) & np.isfinite(lum) & (wave > 0.0) & (lum > 0.0)
    ax.loglog(wave[good], lum[good], label=label, color=color, ls=ls, lw=lw)


def spectrum_ratio(numerator, denominator):
    wave = denominator["wave_nm"]
    denom = denominator["lum_w_per_nm"]
    num = np.interp(wave, numerator["wave_nm"], numerator["lum_w_per_nm"], left=np.nan, right=np.nan)
    ratio = np.full_like(wave, np.nan, dtype=float)
    good = np.isfinite(num) & np.isfinite(denom) & (denom > 0.0)
    ratio[good] = num[good] / denom[good]
    return np.column_stack([wave, ratio])


def nly_from_spectrum(spectrum, lo=0.0, hi=91.1):
    wave = spectrum["wave_nm"]
    lum = spectrum["lum_w_per_nm"]
    good = (wave >= lo) & (wave <= hi) & np.isfinite(wave) & np.isfinite(lum)
    if np.count_nonzero(good) < 2:
        return 0.0
    wave_m = wave[good] * 1.0e-9
    return _json_float(np.trapz(lum[good] * wave_m / (H_J_S * C_M_S), wave[good]))


def lum_from_spectrum(spectrum, lo=0.0, hi=91.1):
    wave = spectrum["wave_nm"]
    lum = spectrum["lum_w_per_nm"]
    good = (wave >= lo) & (wave <= hi) & np.isfinite(wave) & np.isfinite(lum)
    if np.count_nonzero(good) < 2:
        return 0.0
    return _json_float(np.trapz(lum[good], wave[good]))


def _json_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
