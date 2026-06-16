"""One-galaxy sanity check for the experimental FSPS stellar CIGALE module.

The calculation is deliberately small and inspectable. For one matched-ish
parameter vector it runs:

1. ``sfhdelayed -> bc03``;
2. ``sfhdelayed -> fsps_stellar``;
3. both of the above followed by CIGALE ``nebular``.

The useful outputs are a JSON file with CIGALE ``sed.info`` quantities and a
before/after-nebular spectrum plot. This is a diagnostic, not a proof that BC03
and FSPS should agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from composed.experimental.cigale_fsps_stellar import register_cigale_fsps_stellar_module


OUTPUT_DIR = Path("outputs/experimental_cigale_fsps_stellar_sanity")


SFH_PARAMS = {
    "age_main": 1000,  # Myr
    "tau_main": 3000.0,  # Myr
    "normalise": True,
}
SEPARATION_AGE_MYR = 10
BC03_PARAMS = {
    "imf": 1,  # Chabrier
    "metallicity": 0.02,
    "separation_age": SEPARATION_AGE_MYR,
}
FSPS_PARAMS = {
    "imf_type": 1,  # Chabrier in python-fsps
    "logzsol": 0.0,
    "zcontinuous": 1,
    "separation_age": SEPARATION_AGE_MYR,
}
NEBULAR_PARAMS = {
    "logU": -2.0,
    "zgas": 0.014,
    "emission": True,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_cigale_fsps_stellar_module()

    from pcigale.warehouse import SedWarehouse

    # CIGALE's nebular module stores wavelength-grid-dependent arrays on the
    # module instance. Because BC03 and FSPS use different stellar wavelength
    # grids, keep nebular out of the module cache for mixed-grid diagnostics.
    warehouse = SedWarehouse(nocache=["fsps_stellar", "nebular"])

    runs = {
        "bc03_stellar": warehouse.get_sed(["sfhdelayed", "bc03"], [SFH_PARAMS, BC03_PARAMS]),
        "bc03_nebular": warehouse.get_sed(
            ["sfhdelayed", "bc03", "nebular"], [SFH_PARAMS, BC03_PARAMS, NEBULAR_PARAMS]
        ),
        "fsps_stellar": warehouse.get_sed(["sfhdelayed", "fsps_stellar"], [SFH_PARAMS, FSPS_PARAMS]),
        "fsps_nebular": warehouse.get_sed(
            ["sfhdelayed", "fsps_stellar", "nebular"], [SFH_PARAMS, FSPS_PARAMS, NEBULAR_PARAMS]
        ),
    }

    summary = build_summary(runs)
    write_summary(summary)
    plot_spectra(runs)
    print_summary(summary)
    print(f"Saved plot: {OUTPUT_DIR / 'bc03_vs_fsps_before_after_nebular.png'}")
    print(f"Saved summary: {OUTPUT_DIR / 'summary.json'}")


def build_summary(runs):
    summary = {
        "parameters": {
            "sfhdelayed": SFH_PARAMS,
            "bc03": BC03_PARAMS,
            "fsps_stellar": FSPS_PARAMS,
            "nebular": NEBULAR_PARAMS,
        },
        "sed_info": {},
        "ratios": {},
        "spectral_ratios": {},
    }
    keys = [
        "sfh.integrated",
        "stellar.m_star",
        "stellar.m_gas",
        "stellar.n_ly",
        "stellar.n_ly_young",
        "stellar.n_ly_old",
        "stellar.lum",
        "stellar.lum_young",
        "stellar.lum_old",
        "stellar.lum_ly",
        "stellar.lum_ly_young",
        "stellar.lum_ly_old",
    ]
    for name, sed in runs.items():
        summary["sed_info"][name] = {key: _json_float(sed.info.get(key)) for key in keys if key in sed.info}

    for stage in ["stellar", "nebular"]:
        bc03 = summary["sed_info"][f"bc03_{stage}"]
        fsps = summary["sed_info"][f"fsps_{stage}"]
        summary["ratios"][f"fsps_over_bc03_{stage}"] = {
            key: _safe_ratio(fsps.get(key), bc03.get(key)) for key in sorted(set(bc03) & set(fsps))
        }

    for stage in ["stellar", "nebular"]:
        bc03_sed = runs[f"bc03_{stage}"]
        fsps_sed = runs[f"fsps_{stage}"]
        ratio = spectrum_ratio(fsps_sed, bc03_sed)
        summary["spectral_ratios"][f"fsps_over_bc03_{stage}"] = {
            "median_120_900nm": _json_float(np.nanmedian(ratio[(ratio[:, 0] >= 120.0) & (ratio[:, 0] <= 900.0), 1])),
            "median_900_2500nm": _json_float(
                np.nanmedian(ratio[(ratio[:, 0] >= 900.0) & (ratio[:, 0] <= 2500.0), 1])
            ),
            "median_10_91nm": _json_float(np.nanmedian(ratio[(ratio[:, 0] >= 10.0) & (ratio[:, 0] <= 91.1), 1])),
        }
    return summary


def write_summary(summary) -> None:
    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def print_summary(summary) -> None:
    print("Parameter set:")
    print(json.dumps(summary["parameters"], indent=2, sort_keys=True))
    print("\nKey FSPS/BC03 ratios:")
    for stage in ["stellar", "nebular"]:
        ratios = summary["ratios"][f"fsps_over_bc03_{stage}"]
        print(f"  {stage}:")
        for key in ["stellar.m_star", "stellar.n_ly", "stellar.n_ly_young", "stellar.lum", "stellar.lum_ly"]:
            if key in ratios:
                print(f"    {key:20s} {ratios[key]: .4g}")
    print("\nMedian spectral ratios FSPS/BC03:")
    for stage, values in summary["spectral_ratios"].items():
        print(f"  {stage}: {values}")


def plot_spectra(runs) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    stages = [
        ("stellar", "Before nebular"),
        ("nebular", "After nebular"),
    ]
    for col, (stage, title) in enumerate(stages):
        ax = axes[0, col]
        rax = axes[1, col]
        bc03 = runs[f"bc03_{stage}"]
        fsps = runs[f"fsps_{stage}"]
        _plot_sed(ax, bc03, "BC03", color="tab:blue")
        _plot_sed(ax, fsps, "FSPS stellar", color="tab:orange")
        ratio = spectrum_ratio(fsps, bc03)
        mask = np.isfinite(ratio[:, 1]) & (ratio[:, 1] > 0.0)
        rax.semilogx(ratio[mask, 0], ratio[mask, 1], color="black", lw=1.4)
        rax.axhline(1.0, color="0.5", ls="--", lw=1)
        rax.axvline(91.1, color="0.7", ls=":", lw=1)
        rax.set_ylim(1e-2, 1e2)
        rax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(r"$L_\lambda$ [W nm$^{-1}$]")
        rax.set_ylabel("FSPS / BC03")
        rax.set_xlabel("Rest wavelength [nm]")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        rax.grid(alpha=0.25)

    for ax in axes.flat:
        ax.set_xlim(10.0, 3000.0)
    fig.suptitle("CIGALE BC03 vs experimental FSPS stellar module")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bc03_vs_fsps_before_after_nebular.png", dpi=180)
    plt.close(fig)


def _plot_sed(ax, sed, label, color):
    wave = np.asarray(sed.wavelength_grid, dtype=float)
    lum = np.asarray(sed.luminosity, dtype=float)
    mask = np.isfinite(wave) & np.isfinite(lum) & (wave > 0.0) & (lum > 0.0)
    ax.loglog(wave[mask], lum[mask], label=label, color=color, lw=1.5)
    ax.axvline(91.1, color="0.7", ls=":", lw=1)


def spectrum_ratio(numerator_sed, denominator_sed):
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


def _safe_ratio(num, den):
    if num is None or den is None:
        return None
    den = float(den)
    if den == 0.0 or not np.isfinite(den):
        return None
    return _json_float(float(num) / den)


def _json_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
