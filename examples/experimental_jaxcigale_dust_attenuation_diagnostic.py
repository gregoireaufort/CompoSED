#!/usr/bin/env python
"""Plot the first JAX-CIGALE young/old dust attenuation diagnostic.

This is not a production example. It is a sanity check for the scientific
bookkeeping:

1. build a delayed-tau SFH on the correct DSPS cosmic-time clock;
2. evaluate DSPS twice, once for young stars and once for old stars;
3. apply a modified-starburst attenuation curve with different young/old E(B-V);
4. plot the before/after spectra and the attenuation ratio.

The intended run environment is the JAX/DSPS environment:

    PYTHONPATH=/Users/gregoire/Documents/Sedfitting/CompoSED \
    /Users/gregoire/miniforge3/envs/dsps_nuts/bin/python \
        examples/experimental_jaxcigale_dust_attenuation_diagnostic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from composed.experimental.jaxcigale.ssp_data import default_continuum_ssp_path, require_continuum_ssp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssp-file",
        type=Path,
        default=default_continuum_ssp_path(),
        help="Continuum-only DSPS/FSPS SSP table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimental_jaxcigale_dust_attenuation_diagnostic"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from dsps import load_ssp_templates

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        delayed_sfh_cosmic_time_module,
        dsps_stellar_module,
        modified_starburst_attenuation_module,
        no_nebular_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    ssp_file = require_continuum_ssp_path(args.ssp_file)
    ssp_data = load_ssp_templates(fn=str(ssp_file))
    wave_rest_a = np.geomspace(900.0, 30000.0, 1400)
    filters = JaxFilterSet.from_curves(
        ["dummy"],
        [np.linspace(5000.0, 7000.0, 32)],
        [np.ones(32)],
    )
    space = JaxParameterSpace(
        names=[
            "z",
            "tau_gyr",
            "tage_gyr",
            "logzsol",
            "E_BV_young",
            "E_BV_old_factor",
            "powerlaw_slope",
            "uv_bump_amplitude",
        ],
        priors={
            "z": UniformJaxPrior(0.0, 5.0),
            "tau_gyr": UniformJaxPrior(0.1, 10.0),
            "tage_gyr": UniformJaxPrior(0.1, 10.0),
            "logzsol": UniformJaxPrior(-2.0, 0.5),
            "E_BV_young": UniformJaxPrior(0.0, 2.0),
            "E_BV_old_factor": UniformJaxPrior(0.0, 1.0),
            "powerlaw_slope": UniformJaxPrior(-1.0, 1.0),
            "uv_bump_amplitude": UniformJaxPrior(0.0, 5.0),
        },
    )

    common_modules = [
        delayed_sfh_cosmic_time_module(n_time=512),
        dsps_stellar_module(ssp_data, separation_age_myr=100.0),
        no_nebular_module(),
    ]
    model_no_dust = build_jax_sed_model(common_modules, wave_rest_a, filters, space)
    model_dust = build_jax_sed_model(
        common_modules + [modified_starburst_attenuation_module()],
        wave_rest_a,
        filters,
        space,
    )
    theta = jnp.asarray([0.8, 2.0, 3.0, -0.3, 0.45, 0.35, -0.2, 1.5])
    state_no_dust = model_no_dust.run_modules(theta)
    state_dust = model_dust.run_modules(theta)

    wave_nm = np.asarray(state_no_dust.wave_rest_a) / 10.0
    young = np.asarray(state_no_dust.stellar_young_lum_lsun_per_a)
    old = np.asarray(state_no_dust.stellar_old_lum_lsun_per_a)
    total = np.asarray(state_no_dust.total_lum_lsun_per_a)
    attenuated = np.asarray(state_dust.total_lum_lsun_per_a)
    absorbed_lsun = float(state_dust.absorbed_lum_lsun)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(wave_nm, young, label="young stellar before dust", color="tab:blue", lw=1.1)
    axes[0].plot(wave_nm, old, label="old stellar before dust", color="tab:orange", lw=1.1)
    axes[0].plot(wave_nm, total, label="total before dust", color="0.2", lw=1.6)
    axes[0].plot(wave_nm, attenuated, label="total after dust", color="crimson", lw=1.6)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$L_\lambda$ [L$_\odot$ A$^{-1}$]")
    axes[0].set_title(f"JAX-CIGALE dust sanity check, absorbed luminosity = {absorbed_lsun:.3g} Lsun")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    ratio = np.divide(attenuated, total, out=np.full_like(total, np.nan), where=total > 0.0)
    axes[1].plot(wave_nm, ratio, color="crimson", lw=1.5)
    axes[1].axhline(1.0, color="0.5", ls=":", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Rest wavelength [nm]")
    axes[1].set_ylabel("attenuated / intrinsic")
    axes[1].grid(alpha=0.25)

    output_path = args.output_dir / "young_old_modified_starburst_dust.png"
    fig.savefig(output_path, dpi=180)
    print("Saved:", output_path)
    print("Absorbed luminosity [Lsun per Msun formed]:", absorbed_lsun)
    plot_gordon16_rvfa_extinction_curves(args.output_dir)
    plot_nebular_line_extinction(args.output_dir)


def plot_gordon16_rvfa_extinction_curves(output_dir: Path) -> None:
    """Plot the BEAST/Gordon16 R(V), f_A extinction family we use in JAX."""

    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.modules import _gordon16_rvfa_a_over_av

    _, jnp = require_jax()

    wave_a = np.geomspace(1000.0, 30000.0, 800)
    wave_nm = wave_a / 10.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for f_a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        curve = _gordon16_rvfa_a_over_av(jnp.asarray(wave_a), rv=jnp.asarray(3.1), f_a=jnp.asarray(f_a))
        axes[0].plot(wave_nm, np.asarray(curve), label=fr"$f_A={f_a:g}$")
    axes[0].set_title(r"Gordon16 mixture, fixed $R_V=3.1$")
    axes[0].set_xlabel("Rest wavelength [nm]")
    axes[0].set_ylabel(r"$A_\lambda / A_V$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for rv in [2.5, 3.1, 4.0, 5.0]:
        curve = _gordon16_rvfa_a_over_av(jnp.asarray(wave_a), rv=jnp.asarray(rv), f_a=jnp.asarray(0.75))
        axes[1].plot(wave_nm, np.asarray(curve), label=fr"$R_V={rv:g}$")
    axes[1].set_title(r"Gordon16 mixture, fixed $f_A=0.75$")
    axes[1].set_xlabel("Rest wavelength [nm]")
    axes[1].set_ylabel(r"$A_\lambda / A_V$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    output_path = output_dir / "gordon16_rvfa_extinction_curves.png"
    fig.savefig(output_path, dpi=180)
    print("Saved:", output_path)


def plot_nebular_line_extinction(output_dir: Path) -> None:
    """Toy Balmer/OIII line grid showing wavelength-dependent nebular dust."""

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        modified_starburst_attenuation_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    line_names = np.asarray(["[OII] 3727", "Hbeta 4861", "[OIII] 5007", "Halpha 6563"])
    wave_a = np.asarray([3727.0, 4861.0, 5007.0, 6563.0])
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(3500.0, 7000.0, 16)], [np.ones(16)])
    space = JaxParameterSpace(
        names=["E_BV_young", "E_BV_old_factor", "E_BV_lines"],
        priors={
            "E_BV_young": UniformJaxPrior(0.0, 2.0),
            "E_BV_old_factor": UniformJaxPrior(0.0, 1.0),
            "E_BV_lines": UniformJaxPrior(0.0, 2.0),
        },
    )
    model = build_jax_sed_model(
        [modified_starburst_attenuation_module(nebular_ebv_parameter="E_BV_lines", nebular_extinction_law="mw_ccm89")],
        wave_a,
        filters,
        space,
    )
    line_lum = jnp.ones_like(model.initial_state().wave_rest_a)
    pre_dust = model.initial_state()._replace(
        nebular_lum_lsun_per_a=line_lum,
        nebular_line_lum_lsun_per_a=line_lum,
        total_lum_lsun_per_a=line_lum,
    )
    state_dust = model.modules[0].apply(model.params_from_theta(jnp.asarray([0.0, 0.0, 0.5])), pre_dust)
    attenuated = np.asarray(state_dust.nebular_line_lum_lsun_per_a)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(np.arange(line_names.size), np.ones_like(attenuated), alpha=0.4, label="before dust")
    axes[0].bar(np.arange(line_names.size), attenuated, alpha=0.8, label="after MW/CCM89 dust")
    axes[0].set_xticks(np.arange(line_names.size), line_names, rotation=30, ha="right")
    axes[0].set_ylabel("relative line luminosity")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(wave_a / 10.0, attenuated, marker="o")
    axes[1].set_xlabel("Rest wavelength [nm]")
    axes[1].set_ylabel("nebular transmission")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Shorter-wavelength lines are attenuated more")

    output_path = output_dir / "nebular_line_extinction_mw_ccm89.png"
    fig.savefig(output_path, dpi=180)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
