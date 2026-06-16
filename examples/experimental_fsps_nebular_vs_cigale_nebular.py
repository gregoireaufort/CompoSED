"""Compare direct FSPS nebular emission to CIGALE nebular after fsps_stellar.

This diagnostic isolates one modeling question: if the same SFH is fed to FSPS
directly and to CIGALE through ``fsps_stellar``, how different is the nebular
spectrum? The SFH is defined once by CIGALE, then evaluated as:

1. direct python-FSPS with ``add_neb_emission=True``;
2. CIGALE ``fsps_stellar`` followed by CIGALE ``nebular``.

The script saves the spectrum plot and a JSON summary of Lyman-continuum
integrals so the disagreement can be inspected without rerunning anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from composed.experimental.cigale_fsps_stellar import register_cigale_fsps_stellar_module


OUTPUT_DIR = Path("outputs/experimental_fsps_nebular_vs_cigale_nebular")

LSUN_W = 3.828e26
H_J_S = 6.62607015e-34
C_M_S = 2.99792458e8

SFH_PARAMS = {
    "age_main": 1000,  # Myr
    "tau_main": 3000.0,  # Myr
    "normalise": True,
}
FSPS_STELLAR_PARAMS = {
    "imf_type": 1,
    "logzsol": 0.0,
    "zcontinuous": 1,
    "separation_age": 10,
}
CIGALE_NEBULAR_PARAMS = {
    "logU": -2.0,
    "zgas": 0.014,
    "f_esc": 0.0,
    "f_dust": 0.0,
    "emission": True,
}
DIRECT_FSPS_KWARGS = {
    "zcontinuous": 1,
    "sfh": 3,
    "imf_type": 1,
    "logzsol": 0.0,
    "add_neb_emission": True,
    "add_neb_continuum": True,
    "gas_logu": -2.0,
    "gas_logz": 0.0,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_cigale_fsps_stellar_module()

    from pcigale.warehouse import SedWarehouse

    warehouse = SedWarehouse(nocache=["fsps_stellar", "nebular"])
    sfh_sed = warehouse.get_sed(["sfhdelayed"], [SFH_PARAMS])
    sfh = np.asarray(sfh_sed.sfh, dtype=float)

    cigale_stellar = warehouse.get_sed(["sfhdelayed", "fsps_stellar"], [SFH_PARAMS, FSPS_STELLAR_PARAMS])
    cigale_nebular = warehouse.get_sed(
        ["sfhdelayed", "fsps_stellar", "nebular"],
        [SFH_PARAMS, FSPS_STELLAR_PARAMS, CIGALE_NEBULAR_PARAMS],
    )
    fsps_nebular = direct_fsps_nebular_spectrum(sfh)

    summary = build_summary(cigale_stellar, cigale_nebular, fsps_nebular)
    write_summary(summary)
    plot_spectra(cigale_stellar, cigale_nebular, fsps_nebular)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved plot: {OUTPUT_DIR / 'direct_fsps_nebular_vs_cigale_nebular.png'}")
    print(f"Saved summary: {OUTPUT_DIR / 'summary.json'}")


def direct_fsps_nebular_spectrum(sfh_msun_per_yr):
    import fsps

    kwargs = dict(DIRECT_FSPS_KWARGS)
    sp = fsps.StellarPopulation(**{k: v for k, v in kwargs.items() if k not in {"logzsol", "gas_logu", "gas_logz"}})
    sp.params["logzsol"] = kwargs["logzsol"]
    sp.params["gas_logu"] = kwargs["gas_logu"]
    sp.params["gas_logz"] = kwargs["gas_logz"]
    time_gyr = np.arange(1, sfh_msun_per_yr.size + 1, dtype=float) / 1000.0
    sp.set_tabular_sfh(time_gyr, sfh_msun_per_yr)
    wave_a, llam_lsun_per_a = sp.get_spectrum(tage=float(time_gyr[-1]), peraa=True)
    return {
        "wave_nm": np.asarray(wave_a, dtype=float) / 10.0,
        "lum_w_per_nm": np.asarray(llam_lsun_per_a, dtype=float) * LSUN_W * 10.0,
    }


def build_summary(cigale_stellar, cigale_nebular, fsps_nebular):
    cig_stellar = spectrum_dict_from_cigale(cigale_stellar)
    cig_neb = spectrum_dict_from_cigale(cigale_nebular)
    fsps = fsps_nebular
    ratio = spectrum_ratio(fsps, cig_neb)
    summary = {
        "parameters": {
            "sfhdelayed": SFH_PARAMS,
            "cigale_fsps_stellar": FSPS_STELLAR_PARAMS,
            "cigale_nebular": CIGALE_NEBULAR_PARAMS,
            "direct_fsps": DIRECT_FSPS_KWARGS,
        },
        "n_ly_like_integrals": {
            "cigale_stellar_before_nebular": nly_from_spectrum(cig_stellar),
            "cigale_after_nebular": nly_from_spectrum(cig_neb),
            "direct_fsps_add_neb_emission": nly_from_spectrum(fsps),
        },
        "lyman_continuum_luminosity": {
            "cigale_stellar_before_nebular": lum_from_spectrum(cig_stellar),
            "cigale_after_nebular": lum_from_spectrum(cig_neb),
            "direct_fsps_add_neb_emission": lum_from_spectrum(fsps),
        },
        "cigale_info": {
            "stellar.n_ly": _json_float(cigale_nebular.info.get("stellar.n_ly")),
            "stellar.n_ly_young": _json_float(cigale_nebular.info.get("stellar.n_ly_young")),
            "stellar.n_ly_old": _json_float(cigale_nebular.info.get("stellar.n_ly_old")),
            "nebular.f_esc": _json_float(cigale_nebular.info.get("nebular.f_esc")),
            "nebular.f_dust": _json_float(cigale_nebular.info.get("nebular.f_dust")),
        },
        "median_direct_fsps_over_cigale_nebular": {
            "10_91nm": _median_ratio(ratio, 10.0, 91.1),
            "91_120nm": _median_ratio(ratio, 91.1, 120.0),
            "120_900nm": _median_ratio(ratio, 120.0, 900.0),
            "900_2500nm": _median_ratio(ratio, 900.0, 2500.0),
        },
    }
    return summary


def write_summary(summary) -> None:
    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def plot_spectra(cigale_stellar, cigale_nebular, fsps_nebular) -> None:
    cig_stellar = spectrum_dict_from_cigale(cigale_stellar)
    cig_neb = spectrum_dict_from_cigale(cigale_nebular)
    fsps = fsps_nebular

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax, rax = axes
    _plot_dict(ax, cig_stellar, "CIGALE fsps_stellar before nebular", "0.55", ls="--")
    _plot_dict(ax, cig_neb, "CIGALE fsps_stellar + CIGALE nebular", "tab:blue")
    _plot_dict(ax, fsps, "Direct FSPS add_neb_emission=True", "tab:orange")

    ratio = spectrum_ratio(fsps, cig_neb)
    good = np.isfinite(ratio[:, 1]) & (ratio[:, 1] > 0.0)
    rax.semilogx(ratio[good, 0], ratio[good, 1], color="black", lw=1.4)
    rax.axhline(1.0, color="0.5", ls="--", lw=1)
    rax.set_yscale("log")
    rax.set_ylim(1e-4, 1e4)
    rax.set_ylabel("Direct FSPS / CIGALE nebular")
    rax.set_xlabel("Rest wavelength [nm]")

    for axis in axes:
        axis.axvline(91.1, color="0.7", ls=":", lw=1)
        axis.set_xlim(10.0, 3000.0)
        axis.grid(alpha=0.25)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel(r"$L_\lambda$ [W nm$^{-1}$]")
    ax.legend(loc="best")
    ax.set_title("Direct FSPS nebular vs CIGALE nebular after fsps_stellar")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "direct_fsps_nebular_vs_cigale_nebular.png", dpi=180)
    plt.close(fig)


def spectrum_dict_from_cigale(sed):
    return {
        "wave_nm": np.asarray(sed.wavelength_grid, dtype=float),
        "lum_w_per_nm": np.asarray(sed.luminosity, dtype=float),
    }


def _plot_dict(ax, spectrum, label, color, ls="-"):
    wave = spectrum["wave_nm"]
    lum = spectrum["lum_w_per_nm"]
    good = np.isfinite(wave) & np.isfinite(lum) & (wave > 0.0) & (lum > 0.0)
    ax.loglog(wave[good], lum[good], label=label, color=color, ls=ls, lw=1.4)


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


def _median_ratio(ratio, lo, hi):
    good = (ratio[:, 0] >= lo) & (ratio[:, 0] <= hi) & np.isfinite(ratio[:, 1]) & (ratio[:, 1] > 0.0)
    if not np.any(good):
        return None
    return _json_float(np.nanmedian(ratio[good, 1]))


def _json_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
