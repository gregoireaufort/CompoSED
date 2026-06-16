#!/usr/bin/env python
"""Cross-validate effective stellar+nebular backend implementations.

This is a validation script, not a production API.  It documents where several
SED implementations agree, where they disagree, and whether the disagreement is
expected, suspicious, or still unresolved.

The staged design is deliberate because the local Python environments differ:

    # CIGALE + python-FSPS environment
    SPS_HOME=/Users/gregoire/Work/FSPS \
    PYTHONPATH=/Users/gregoire/Documents/Sedfitting/CompoSED \
    /Users/gregoire/opt/anaconda3/envs/sbi_candide/bin/python \
        examples/validation_backend_cross_validation.py --stage references --n-draws 80

    # JAX + DSPS + Cue environment
    PYTHONPATH=/Users/gregoire/Documents/Sedfitting/CompoSED \
    DSPS_CONTINUUM_SSP_FILE="outputs/experimental_dsps_fsps_clock_diagnostic/fsps_continuum_ssp_data.h5" \
    CUE_DATA_DIR=/private/tmp/cue/src/cue/data \
    /Users/gregoire/miniforge3/envs/dsps_nuts/bin/python \
        examples/validation_backend_cross_validation.py --stage cue --n-draws 80

    # Either environment with numpy/matplotlib
    python examples/validation_backend_cross_validation.py --stage plots

The compared effective implementations are:

- CIGALE + BC03 stellar;
- CIGALE + BC03 stellar + CIGALE nebular;
- CIGALE + experimental fsps_stellar;
- CIGALE + experimental fsps_stellar + CIGALE nebular;
- direct python-FSPS stellar;
- direct python-FSPS stellar + FSPS nebular;
- CIGALE + BC03 stellar + Cue nebular;
- CIGALE + BC03 stellar + Cue nebular without LyC removal, as a diagnostic;
- JAX-CIGALE DSPS stellar + Cue nebular, when the Cue stage is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

if hasattr(np, "trapezoid"):
    _NP_TRAPEZOID = np.trapezoid
else:
    _NP_TRAPEZOID = np.trapz

LSUN_W = 3.828e26
LSUN_CGS = 3.828e33
MPC_CM = 3.0856775814913673e24
C_A_PER_S = 2.99792458e18
AB_FNU_CGS = 3631.0e-23
C_KM_PER_S = 299792.458

REST_WAVE_NM = np.geomspace(10.0, 3000.0, 1800)
REST_WAVE_A = REST_WAVE_NM * 10.0

MODEL_NAMES = (
    "cigale_bc03_stellar",
    "cigale_bc03_cigale_nebular",
    "cigale_bc03_cue_nebular",
    "cigale_bc03_cue_no_lyc_absorption",
    "cigale_fsps_stellar",
    "cigale_fsps_cigale_nebular",
    "direct_fsps_stellar",
    "direct_fsps_fsps_nebular",
    "jax_dsps_cue_nebular",
    "jax_dsps_cue_no_lyc_absorption",
)

FILTER_SPECS = (
    ("FUV_like", 1550.0, 160.0),
    ("NUV_like", 2300.0, 260.0),
    ("u_like", 3600.0, 340.0),
    ("g_like", 4800.0, 520.0),
    ("r_like", 6200.0, 560.0),
    ("i_like", 7600.0, 650.0),
    ("z_like", 9000.0, 720.0),
    ("Y_like", 10200.0, 850.0),
    ("J_like", 12500.0, 1100.0),
    ("H_like", 16500.0, 1400.0),
)

WINDOWS_NM = (
    ("lyc_10_91nm", 10.0, 91.1),
    ("ly_edge_91_121nm", 91.1, 121.6),
    ("uv_121_300nm", 121.6, 300.0),
    ("optical_300_900nm", 300.0, 900.0),
    ("nir_900_2500nm", 900.0, 2500.0),
)

BC03_METALLICITIES = np.asarray([0.0004, 0.004, 0.008, 0.02, 0.05], dtype=float)
NEBULAR_LOGU_GRID = np.round(np.arange(-3.5, -1.49, 0.1), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "references", "cue", "plots"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation_backend_cross_validation"))
    parser.add_argument("--n-draws", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument(
        "--ssp-file",
        type=Path,
        default=Path(os.environ.get("DSPS_CONTINUUM_SSP_FILE", default_continuum_ssp_path())),
        help=(
            "Continuum-only DSPS/FSPS SSP table.  Do not use DSPS' default "
            "ssp_data_fsps_v3.2_lgmet_age.h5 for Cue runs."
        ),
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
    draws = load_or_create_draws(args.output_dir / "parameter_draws.json", n_draws=args.n_draws, seed=args.seed)

    if args.stage in {"all", "references"}:
        run_reference_stage(draws, args.output_dir)
    if args.stage in {"all", "cue"}:
        run_cue_stage(
            draws,
            args.output_dir,
            args.ssp_file,
            args.cue_data_dir,
            allow_nebular_included_ssp=args.allow_nebular_included_ssp,
        )
    if args.stage in {"all", "plots"}:
        run_plot_stage(draws, args.output_dir)


def load_or_create_draws(path: Path, *, n_draws: int, seed: int) -> list[dict[str, float]]:
    if path.exists():
        draws = json.loads(path.read_text())
        if len(draws) >= n_draws:
            return draws[:n_draws]
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(int(n_draws)):
        redshift = float(rng.uniform(0.02, 4.0))
        t_universe_myr = flat_lcdm_age_gyr(redshift) * 1000.0
        max_age = max(80.0, min(9000.0, 0.85 * t_universe_myr))
        age_main = int(rng.uniform(80.0, max_age))
        tau_main = float(10.0 ** rng.uniform(np.log10(120.0), np.log10(6000.0)))
        metallicity = float(rng.choice(BC03_METALLICITIES))
        logzsol = math.log10(metallicity / 0.02)
        draws.append(
            {
                "index": i,
                "label": f"draw_{i:04d}",
                "z": redshift,
                "log10_mass": float(rng.uniform(9.0, 11.0)),
                "age_main_myr": age_main,
                "tau_main_myr": tau_main,
                "metallicity": metallicity,
                "logzsol": logzsol,
                "logu": float(rng.choice(NEBULAR_LOGU_GRID)),
                "zgas": float(nearest_cigale_nebular_zgas(metallicity)),
                "f_esc": 0.0,
                "f_dust": 0.0,
            }
        )
    path.write_text(json.dumps(draws, indent=2, sort_keys=True) + "\n")
    return draws


def run_reference_stage(draws: list[dict[str, float]], output_dir: Path) -> None:
    """Run CIGALE+BC03, CIGALE+FSPS, and direct FSPS spectra."""

    from composed.experimental.cigale_fsps_stellar import register_cigale_fsps_stellar_module
    from composed.experimental.cigale_fsps_stellar_conventions import fsps_parameters_from_cigale_bc03

    register_cigale_fsps_stellar_module()
    from pcigale.warehouse import SedWarehouse
    import fsps

    warehouse = SedWarehouse(nocache=["fsps_stellar", "nebular"])
    spectra = empty_spectra(len(draws))
    phot = empty_photometry(len(draws))
    info_rows = []

    for i, draw in enumerate(draws):
        mapping = fsps_parameters_from_cigale_bc03(imf=1, metallicity=draw["metallicity"], z_sun=0.02)
        sfh_params = cigale_sfh_params(draw)
        bc03_params = {"imf": 1, "metallicity": mapping.metallicity, "separation_age": 10}
        fsps_stellar_params = {
            "imf_type": mapping.imf_type,
            "logzsol": mapping.logzsol,
            "z_sun": mapping.z_sun,
            "zcontinuous": 1,
            "separation_age": 10,
        }
        neb_params = cigale_nebular_params(draw)

        sed_sfh = warehouse.get_sed(["sfhdelayed"], [sfh_params])
        sfh = np.asarray(sed_sfh.sfh, dtype=float)

        sed_objects = {
            "cigale_bc03_stellar": warehouse.get_sed(["sfhdelayed", "bc03"], [sfh_params, bc03_params]),
            "cigale_bc03_cigale_nebular": warehouse.get_sed(
                ["sfhdelayed", "bc03", "nebular"],
                [sfh_params, bc03_params, neb_params],
            ),
            "cigale_fsps_stellar": warehouse.get_sed(
                ["sfhdelayed", "fsps_stellar"],
                [sfh_params, fsps_stellar_params],
            ),
            "cigale_fsps_cigale_nebular": warehouse.get_sed(
                ["sfhdelayed", "fsps_stellar", "nebular"],
                [sfh_params, fsps_stellar_params, neb_params],
            ),
        }
        for name, sed in sed_objects.items():
            wave_nm, lum_w_per_nm = cigale_sed_to_rest_luminosity(sed)
            spectra[name][i] = resample_spectrum(wave_nm, lum_w_per_nm)
            phot[name][i] = observed_maggies_from_rest_luminosity(REST_WAVE_NM, spectra[name][i], draw["z"], draw["log10_mass"])

        wave_nm, lum_w_per_nm = direct_fsps_spectrum(fsps, draw, sfh, add_nebular=False)
        spectra["direct_fsps_stellar"][i] = resample_spectrum(wave_nm, lum_w_per_nm)
        phot["direct_fsps_stellar"][i] = observed_maggies_from_rest_luminosity(
            REST_WAVE_NM, spectra["direct_fsps_stellar"][i], draw["z"], draw["log10_mass"]
        )

        wave_nm, lum_w_per_nm = direct_fsps_spectrum(fsps, draw, sfh, add_nebular=True)
        spectra["direct_fsps_fsps_nebular"][i] = resample_spectrum(wave_nm, lum_w_per_nm)
        phot["direct_fsps_fsps_nebular"][i] = observed_maggies_from_rest_luminosity(
            REST_WAVE_NM, spectra["direct_fsps_fsps_nebular"][i], draw["z"], draw["log10_mass"]
        )

        info_rows.append(
            {
                "index": draw["index"],
                "label": draw["label"],
                "cigale_bc03_stellar_n_ly": json_float(sed_objects["cigale_bc03_stellar"].info.get("stellar.n_ly")),
                "cigale_fsps_stellar_n_ly": json_float(sed_objects["cigale_fsps_stellar"].info.get("stellar.n_ly")),
                "cigale_bc03_nebular_n_ly": json_float(sed_objects["cigale_bc03_cigale_nebular"].info.get("stellar.n_ly")),
                "cigale_fsps_nebular_n_ly": json_float(sed_objects["cigale_fsps_cigale_nebular"].info.get("stellar.n_ly")),
            }
        )
        print(f"[references] {i + 1:04d}/{len(draws):04d} {draw['label']}", flush=True)

    save_stage(output_dir / "reference_spectra.npz", spectra, phot)
    (output_dir / "reference_info.json").write_text(json.dumps(info_rows, indent=2, sort_keys=True) + "\n")
    print("Saved reference stage:", output_dir / "reference_spectra.npz")


def run_cue_stage(
    draws: list[dict[str, float]],
    output_dir: Path,
    ssp_file: Path,
    cue_data_dir: Path,
    *,
    allow_nebular_included_ssp: bool,
) -> None:
    """Run JAX-CIGALE DSPS + Cue spectra in the JAX environment."""

    from dsps import load_ssp_templates

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        cue_nebular_module,
        derive_cue_inputs_from_stellar_spectrum,
        delayed_sfh_cosmic_time_module,
        dsps_stellar_module,
    )
    from composed.experimental.jaxcigale.cue_port import CueJaxPort
    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()
    print("[cue] JAX backend:", jax.default_backend())
    ssp_file = require_continuum_ssp_path(ssp_file, allow_nebular_included=allow_nebular_included_ssp)
    print("[cue] DSPS SSP file:", ssp_file)
    ssp_data = load_ssp_templates(fn=str(ssp_file))
    cue_port = CueJaxPort.from_public_cue_data_dir(cue_data_dir)
    cue_apply = cue_port.make_nebular_apply(line_sigma_a=1.5)

    # Optional hybrid path: CIGALE+BC03 stellar spectra are produced in the
    # CIGALE/FSPS environment and cached by the references stage.  In this JAX
    # stage we reuse that exact stellar continuum, derive Cue's ionizing inputs
    # from it, and add Cue nebular emission.  This keeps the stellar provenance
    # unambiguous: no DSPS/FSPS stellar calculation enters this hybrid model.
    reference_path = output_dir / "reference_spectra.npz"
    if reference_path.exists():
        reference = np.load(reference_path, allow_pickle=True)
        if "spectrum_cigale_bc03_stellar" in reference.files:
            bc03_stellar_w_per_nm = reference["spectrum_cigale_bc03_stellar"]
        else:
            bc03_stellar_w_per_nm = None
    else:
        reference = None
        bc03_stellar_w_per_nm = None

    names = (
        "z",
        "log10_mass",
        "tau_gyr",
        "tage_gyr",
        "logzsol",
        "gas_logu",
        "gas_logoh",
        "gas_logn_h",
        "gas_logno",
        "gas_logco",
        "gas_f_esc",
        "gas_f_dust",
    )
    space = JaxParameterSpace(names=names, priors={name: UniformJaxPrior(-100.0, 100.0) for name in names})
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(4000.0, 5000.0, 8)], [np.ones(8)])
    model_stellar = build_jax_sed_model(
        [
            delayed_sfh_cosmic_time_module(n_time=256),
            dsps_stellar_module(ssp_data),
        ],
        REST_WAVE_A,
        filters,
        space,
    )
    model_cue = build_jax_sed_model(
        [
            delayed_sfh_cosmic_time_module(n_time=256),
            dsps_stellar_module(ssp_data),
            cue_nebular_module(cue_apply),
        ],
        REST_WAVE_A,
        filters,
        space,
    )
    model_cue_no_lyc_absorption = build_jax_sed_model(
        [
            delayed_sfh_cosmic_time_module(n_time=256),
            dsps_stellar_module(ssp_data),
            cue_nebular_module(cue_apply, absorb_lyc=False),
        ],
        REST_WAVE_A,
        filters,
        space,
    )

    spectra = empty_spectra(len(draws))
    phot = empty_photometry(len(draws))
    cue_bc03_info = []
    for i, draw in enumerate(draws):
        theta = jnp.asarray(theta_for_cue(draw, names))
        stellar_state = model_stellar.run_modules(theta)
        cue_state = model_cue.run_modules(theta)
        cue_no_abs_state = model_cue_no_lyc_absorption.run_modules(theta)
        spectra["jax_dsps_stellar"][i] = np.asarray(stellar_state.stellar_lum_lsun_per_a) * LSUN_W * 10.0
        spectra["jax_dsps_cue_nebular"][i] = np.asarray(cue_state.total_lum_lsun_per_a) * LSUN_W * 10.0
        spectra["jax_dsps_cue_no_lyc_absorption"][i] = np.asarray(cue_no_abs_state.total_lum_lsun_per_a) * LSUN_W * 10.0
        phot["jax_dsps_stellar"][i] = observed_maggies_from_rest_luminosity(
            REST_WAVE_NM, spectra["jax_dsps_stellar"][i], draw["z"], draw["log10_mass"]
        )
        phot["jax_dsps_cue_nebular"][i] = observed_maggies_from_rest_luminosity(
            REST_WAVE_NM, spectra["jax_dsps_cue_nebular"][i], draw["z"], draw["log10_mass"]
        )
        phot["jax_dsps_cue_no_lyc_absorption"][i] = observed_maggies_from_rest_luminosity(
            REST_WAVE_NM, spectra["jax_dsps_cue_no_lyc_absorption"][i], draw["z"], draw["log10_mass"]
        )

        if bc03_stellar_w_per_nm is not None and i < bc03_stellar_w_per_nm.shape[0]:
            # reference_spectra.npz stores W/nm per solar mass. Cue works in
            # Lsun/Angstrom per solar mass, so divide by (Lsun W) and by 10.
            stellar_lsun_per_a = jnp.asarray(bc03_stellar_w_per_nm[i] / (LSUN_W * 10.0))
            cue_inputs = derive_cue_inputs_from_stellar_spectrum(
                jnp.asarray(REST_WAVE_A),
                stellar_lsun_per_a,
                logu=draw["logu"],
                logn_h=2.0,
                gas_logoh=draw["logzsol"],
                log_no=-0.134,
                log_co=-0.134,
                f_esc=draw["f_esc"],
                f_dust=draw["f_dust"],
            )
            continuum, lines = cue_apply(jnp.asarray(REST_WAVE_A), cue_inputs.theta12, cue_inputs)
            nebular_lsun_per_a = jnp.asarray(continuum) + jnp.asarray(lines)
            stellar_absorbed_lsun_per_a = jnp.where(
                jnp.asarray(REST_WAVE_A) < 911.6,
                jnp.clip(jnp.asarray(draw["f_esc"]), 0.0, 1.0) * stellar_lsun_per_a,
                stellar_lsun_per_a,
            )
            total_lsun_per_a = stellar_absorbed_lsun_per_a + nebular_lsun_per_a
            total_no_abs_lsun_per_a = stellar_lsun_per_a + nebular_lsun_per_a
            spectra["cigale_bc03_cue_nebular"][i] = np.asarray(total_lsun_per_a) * LSUN_W * 10.0
            spectra["cigale_bc03_cue_no_lyc_absorption"][i] = np.asarray(total_no_abs_lsun_per_a) * LSUN_W * 10.0
            phot["cigale_bc03_cue_nebular"][i] = observed_maggies_from_rest_luminosity(
                REST_WAVE_NM,
                spectra["cigale_bc03_cue_nebular"][i],
                draw["z"],
                draw["log10_mass"],
            )
            phot["cigale_bc03_cue_no_lyc_absorption"][i] = observed_maggies_from_rest_luminosity(
                REST_WAVE_NM,
                spectra["cigale_bc03_cue_no_lyc_absorption"][i],
                draw["z"],
                draw["log10_mass"],
            )
            cue_bc03_info.append(
                {
                    "index": draw["index"],
                    "label": draw["label"],
                    "log_q_h_intrinsic": json_float(cue_inputs.log_q_h_intrinsic),
                    "log_q_h_gas": json_float(cue_inputs.log_q_h_gas),
                    "gas_photon_fraction": json_float(cue_inputs.gas_photon_fraction),
                }
            )
        print(f"[cue] {i + 1:04d}/{len(draws):04d} {draw['label']}", flush=True)

    np.savez(
        output_dir / "cue_spectra.npz",
        rest_wave_nm=REST_WAVE_NM,
        filter_names=np.asarray([name for name, _, _ in FILTER_SPECS]),
        filter_centers_a=np.asarray([center for _, center, _ in FILTER_SPECS]),
        **{f"spectrum_{name}": value for name, value in spectra.items()},
        **{f"phot_{name}": value for name, value in phot.items()},
    )
    if cue_bc03_info:
        (output_dir / "cue_bc03_info.json").write_text(json.dumps(cue_bc03_info, indent=2, sort_keys=True) + "\n")
    print("Saved Cue stage:", output_dir / "cue_spectra.npz")


def run_plot_stage(draws: list[dict[str, float]], output_dir: Path) -> None:
    """Merge saved stages, make plots, and write discrepancy metrics."""

    reference = np.load(output_dir / "reference_spectra.npz", allow_pickle=True)
    cue_path = output_dir / "cue_spectra.npz"
    cue = np.load(cue_path, allow_pickle=True) if cue_path.exists() else None

    spectra = {}
    phot = {}
    for name in MODEL_NAMES:
        spectrum_key = f"spectrum_{name}"
        phot_key = f"phot_{name}"
        if spectrum_key in reference:
            spectra[name] = reference[spectrum_key]
            phot[name] = reference[phot_key]
        elif cue is not None and spectrum_key in cue:
            spectra[name] = cue[spectrum_key]
            phot[name] = cue[phot_key]

    metrics = compute_discrepancy_metrics(draws, spectra, phot)
    write_metrics(output_dir, metrics)
    plot_single_sed_atlas(output_dir, draws, spectra, phot, index=0)
    plot_sweep_heatmaps(output_dir, metrics)
    print_summary(metrics, output_dir)


def empty_spectra(n_draws: int) -> dict[str, np.ndarray]:
    return {name: np.full((int(n_draws), REST_WAVE_NM.size), np.nan, dtype=float) for name in MODEL_NAMES + ("jax_dsps_stellar",)}


def empty_photometry(n_draws: int) -> dict[str, np.ndarray]:
    return {name: np.full((int(n_draws), len(FILTER_SPECS)), np.nan, dtype=float) for name in MODEL_NAMES + ("jax_dsps_stellar",)}


def save_stage(path: Path, spectra: dict[str, np.ndarray], phot: dict[str, np.ndarray]) -> None:
    np.savez(
        path,
        rest_wave_nm=REST_WAVE_NM,
        filter_names=np.asarray([name for name, _, _ in FILTER_SPECS]),
        filter_centers_a=np.asarray([center for _, center, _ in FILTER_SPECS]),
        **{f"spectrum_{name}": value for name, value in spectra.items() if np.isfinite(value).any()},
        **{f"phot_{name}": value for name, value in phot.items() if np.isfinite(value).any()},
    )


def cigale_sfh_params(draw: dict[str, float]) -> dict[str, object]:
    return {
        "age_main": int(draw["age_main_myr"]),
        "tau_main": float(draw["tau_main_myr"]),
        "age_burst": 10,
        "tau_burst": 10.0,
        "f_burst": 0.0,
        "normalise": True,
    }


def cigale_nebular_params(draw: dict[str, float]) -> dict[str, object]:
    return {
        "logU": float(draw["logu"]),
        "zgas": float(draw["zgas"]),
        "ne": 100,
        "f_esc": float(draw["f_esc"]),
        "f_dust": float(draw["f_dust"]),
        "lines_width": 300.0,
        "emission": True,
    }


def cigale_sed_to_rest_luminosity(sed) -> tuple[np.ndarray, np.ndarray]:
    """Return CIGALE rest wavelength in nm and L_lambda in W/nm."""

    return np.asarray(sed.wavelength_grid, dtype=float), np.asarray(sed.luminosity, dtype=float)


def direct_fsps_spectrum(fsps_module, draw: dict[str, float], sfh_msun_per_yr: np.ndarray, *, add_nebular: bool):
    sp = fsps_module.StellarPopulation(
        zcontinuous=1,
        sfh=3,
        imf_type=1,
        add_neb_emission=bool(add_nebular),
        add_neb_continuum=bool(add_nebular),
        add_dust_emission=False,
        compute_vega_mags=False,
    )
    sp.params["logzsol"] = float(draw["logzsol"])
    if add_nebular:
        sp.params["gas_logu"] = float(draw["logu"])
        sp.params["gas_logz"] = float(draw["logzsol"])
    time_gyr = np.arange(1, sfh_msun_per_yr.size + 1, dtype=float) / 1000.0
    sp.set_tabular_sfh(time_gyr, sfh_msun_per_yr)
    wave_a, lum_lsun_per_a = sp.get_spectrum(tage=float(time_gyr[-1]), peraa=True)
    return np.asarray(wave_a) / 10.0, np.asarray(lum_lsun_per_a) * LSUN_W * 10.0


def theta_for_cue(draw: dict[str, float], names: tuple[str, ...]) -> list[float]:
    values = {
        "z": draw["z"],
        "log10_mass": draw["log10_mass"],
        "tau_gyr": draw["tau_main_myr"] / 1000.0,
        "tage_gyr": draw["age_main_myr"] / 1000.0,
        "logzsol": draw["logzsol"],
        "gas_logu": draw["logu"],
        "gas_logoh": draw["logzsol"],
        "gas_logn_h": 2.0,
        "gas_logno": -0.134,
        "gas_logco": -0.134,
        "gas_f_esc": draw["f_esc"],
        "gas_f_dust": draw["f_dust"],
    }
    return [float(values[name]) for name in names]


def resample_spectrum(wave_nm: np.ndarray, lum_w_per_nm: np.ndarray) -> np.ndarray:
    return np.interp(REST_WAVE_NM, wave_nm, lum_w_per_nm, left=np.nan, right=np.nan)


def observed_maggies_from_rest_luminosity(rest_wave_nm, lum_w_per_nm, z, log10_mass) -> np.ndarray:
    """Integrate observed synthetic filters from rest L_lambda in W/nm per Msun."""

    rest_wave_nm = np.asarray(rest_wave_nm, dtype=float)
    lum_w_per_nm = np.asarray(lum_w_per_nm, dtype=float)
    mass = 10.0 ** float(log10_mass)
    wave_obs_a = rest_wave_nm * 10.0 * (1.0 + float(z))
    lum_erg_s_a = lum_w_per_nm * 1.0e6 * mass
    d_l_cm = flat_lcdm_luminosity_distance_mpc(float(z)) * MPC_CM
    flux_lambda_cgs_a = lum_erg_s_a / (4.0 * np.pi * d_l_cm**2 * (1.0 + float(z)))
    out = []
    for _, center_a, width_a in FILTER_SPECS:
        wave_filter = np.linspace(center_a - 3.5 * width_a, center_a + 3.5 * width_a, 220)
        trans = np.exp(-0.5 * ((wave_filter - center_a) / width_a) ** 2)
        model = np.interp(wave_filter, wave_obs_a, flux_lambda_cgs_a, left=0.0, right=0.0)
        numerator = _NP_TRAPEZOID(model * wave_filter * trans, wave_filter)
        denominator = _NP_TRAPEZOID((C_A_PER_S / wave_filter) * trans, wave_filter)
        out.append((numerator / denominator) / AB_FNU_CGS if denominator > 0.0 else np.nan)
    return np.asarray(out, dtype=float)


def compute_discrepancy_metrics(draws, spectra, phot) -> list[dict[str, object]]:
    comparisons = (
        ("cigale_fsps_stellar", "cigale_bc03_stellar", "FSPS_stellar_over_BC03_stellar"),
        ("direct_fsps_stellar", "cigale_fsps_stellar", "direct_FSPS_over_CIGALE_fsps_stellar"),
        ("cigale_fsps_cigale_nebular", "cigale_fsps_stellar", "CIGALE_nebular_effect_on_FSPS_stellar"),
        ("cigale_bc03_cue_nebular", "cigale_bc03_stellar", "Cue_nebular_effect_on_BC03_stellar"),
        ("cigale_bc03_cue_no_lyc_absorption", "cigale_bc03_cue_nebular", "BC03_Cue_LyC_absorption_effect"),
        ("cigale_bc03_cue_nebular", "cigale_bc03_cigale_nebular", "Cue_over_CIGALE_nebular_on_BC03"),
        ("direct_fsps_fsps_nebular", "direct_fsps_stellar", "FSPS_nebular_effect"),
        ("direct_fsps_fsps_nebular", "cigale_fsps_cigale_nebular", "direct_FSPS_nebular_over_CIGALE_nebular"),
        ("jax_dsps_cue_no_lyc_absorption", "jax_dsps_cue_nebular", "DSPS_Cue_LyC_absorption_effect"),
        ("jax_dsps_cue_nebular", "cigale_fsps_cigale_nebular", "Cue_DSPS_over_CIGALE_FSPS_nebular"),
    )
    rows = []
    for i, draw in enumerate(draws):
        for numerator, denominator, label in comparisons:
            if numerator not in spectra or denominator not in spectra:
                continue
            spec_num = spectra[numerator][i]
            spec_den = spectra[denominator][i]
            if not np.isfinite(spec_num).any() or not np.isfinite(spec_den).any():
                continue
            ratio = safe_ratio_array(spec_num, spec_den)
            row = {
                "index": draw["index"],
                "label": draw["label"],
                "comparison": label,
                "numerator": numerator,
                "denominator": denominator,
                "z": draw["z"],
                "age_main_myr": draw["age_main_myr"],
                "metallicity": draw["metallicity"],
                "logu": draw["logu"],
            }
            for window, lo, hi in WINDOWS_NM:
                row[f"median_log10_ratio_{window}"] = median_log10_ratio(ratio, lo, hi)
            if numerator in phot and denominator in phot:
                row["median_delta_mag"] = median_delta_mag(phot[numerator][i], phot[denominator][i])
                row["max_abs_delta_mag"] = max_abs_delta_mag(phot[numerator][i], phot[denominator][i])
                row["worst_band"] = worst_delta_mag_band(phot[numerator][i], phot[denominator][i])
            else:
                row["median_delta_mag"] = None
                row["max_abs_delta_mag"] = None
                row["worst_band"] = None
            row["status"] = classify_difference(row)
            rows.append(row)
    return rows


def classify_difference(row: dict[str, object]) -> str:
    comparison = str(row["comparison"])
    max_abs_mag = value_or_nan(row.get("max_abs_delta_mag"))
    optical = value_or_nan(row.get("median_log10_ratio_optical_300_900nm"))
    lyc = value_or_nan(row.get("median_log10_ratio_lyc_10_91nm"))
    if "LyC_absorption_effect" in comparison:
        return "expected: LyC escape convention difference"
    if "FSPS_stellar_over_BC03" in comparison and np.isfinite(lyc) and abs(lyc) > 0.5:
        return "expected: SPS ionizing-continuum model difference"
    if "direct_FSPS_over_CIGALE_fsps_stellar" in comparison:
        if (np.isfinite(optical) and abs(optical) > 0.1) or (np.isfinite(max_abs_mag) and max_abs_mag > 0.25):
            return "suspicious: direct FSPS and CIGALE fsps_stellar should broadly agree"
        return "expected: implementation-close"
    if "nebular" in comparison and np.isfinite(max_abs_mag) and max_abs_mag > 0.5:
        return "unresolved: nebular implementation has large broadband impact"
    if np.isfinite(max_abs_mag) and max_abs_mag > 1.0:
        return "suspicious: very large broadband difference"
    return "expected/model-dependent"


def write_metrics(output_dir: Path, metrics: list[dict[str, object]]) -> None:
    (output_dir / "cross_validation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in metrics for key in row})
    with (output_dir / "cross_validation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def plot_single_sed_atlas(output_dir: Path, draws, spectra, phot, *, index: int = 0) -> None:
    draw = draws[index]
    colors = {
        "cigale_bc03_stellar": "tab:blue",
        "cigale_bc03_cigale_nebular": "navy",
        "cigale_bc03_cue_nebular": "deepskyblue",
        "cigale_bc03_cue_no_lyc_absorption": "lightskyblue",
        "cigale_fsps_stellar": "tab:orange",
        "cigale_fsps_cigale_nebular": "darkorange",
        "direct_fsps_stellar": "tab:green",
        "direct_fsps_fsps_nebular": "darkgreen",
        "jax_dsps_cue_nebular": "crimson",
        "jax_dsps_cue_no_lyc_absorption": "lightcoral",
    }
    linestyles = {
        "cigale_bc03_stellar": (0, (7, 2)),
        "cigale_bc03_cigale_nebular": "-",
        "cigale_bc03_cue_nebular": (0, (1, 1)),
        "cigale_bc03_cue_no_lyc_absorption": (0, (5, 1)),
        "cigale_fsps_stellar": (0, (5, 2, 1, 2)),
        "cigale_fsps_cigale_nebular": "-",
        "direct_fsps_stellar": (0, (3, 1, 1, 1, 1, 1)),
        "direct_fsps_fsps_nebular": (0, (9, 2, 2, 2)),
        "jax_dsps_cue_nebular": (0, (2, 1)),
        "jax_dsps_cue_no_lyc_absorption": (0, (4, 2)),
    }
    linewidths = {
        "cigale_bc03_cigale_nebular": 1.7,
        "cigale_fsps_cigale_nebular": 1.7,
        "direct_fsps_fsps_nebular": 1.5,
        "jax_dsps_cue_nebular": 1.5,
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    ax, rax = axes
    baseline = "cigale_fsps_cigale_nebular"
    for name in MODEL_NAMES:
        if name not in spectra:
            continue
        spec = spectra[name][index]
        good = spectrum_plot_mask(REST_WAVE_NM, spec)
        if not np.any(good):
            continue
        ls = linestyles.get(name, "-")
        lw = linewidths.get(name, 1.25)
        spec_for_plot = np.where(good, spec, np.nan)
        ax.loglog(REST_WAVE_NM, spec_for_plot, color=colors.get(name), lw=lw, ls=ls, alpha=0.92, label=name)
        if baseline in spectra:
            ratio = safe_ratio_array(spec, spectra[baseline][index])
            rg = (
                np.isfinite(ratio)
                & (ratio > 0.0)
                & spectrum_plot_mask(REST_WAVE_NM, spec)
                & spectrum_plot_mask(REST_WAVE_NM, spectra[baseline][index])
            )
            ratio_for_plot = np.where(rg, ratio, np.nan)
            rax.semilogx(REST_WAVE_NM, ratio_for_plot, color=colors.get(name), lw=max(0.95, lw - 0.25), ls=ls, alpha=0.92)
    for axis in axes:
        axis.axvline(91.1, color="0.7", ls=":", lw=1.0)
        axis.axvline(121.6, color="0.75", ls=":", lw=1.0)
        axis.grid(alpha=0.25)
        axis.set_xlim(10.0, 3000.0)
    ax.set_ylabel(r"$L_\lambda$ [W nm$^{-1}$ per M$_\odot$ formed]")
    ax.set_title(
        f"Single-SED backend atlas: {draw['label']} z={draw['z']:.2f}, "
        f"age={draw['age_main_myr']} Myr, Z={draw['metallicity']:.4g}, logU={draw['logu']:.1f}"
    )
    ax.legend(fontsize=8, ncol=2)
    rax.axhline(1.0, color="black", lw=0.8)
    rax.set_yscale("log")
    rax.set_ylim(1e-3, 1e3)
    rax.set_xlabel("Rest wavelength [nm]")
    rax.set_ylabel(f"ratio / {baseline}")
    fig.savefig(output_dir / "single_sed_cross_validation_atlas.png", dpi=180)
    plt.close(fig)

    plot_single_photometry(output_dir, draw, phot, index=index)


def spectrum_plot_mask(wave_nm: np.ndarray, luminosity: np.ndarray) -> np.ndarray:
    """Mask non-physical numerical floors in diagnostic log-spectrum plots.

    Several nebular conventions intentionally leave zero escaping LyC when
    ``f_esc=0``.  On a log plot those zeros often appear as tiny positive
    machine floors from interpolation/emulator bookkeeping.  Mask them so the
    diagnostic figure shows "no emergent flux" rather than a fake 200-dex line.
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
    floor = max(reference * 1.0e-40, np.finfo(float).tiny)
    return positive & (luminosity > floor)


def plot_single_photometry(output_dir: Path, draw, phot, *, index: int = 0) -> None:
    x = np.arange(len(FILTER_SPECS))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for name in MODEL_NAMES:
        if name not in phot:
            continue
        flux = phot[name][index]
        good = np.isfinite(flux) & (flux > 0.0)
        if not np.any(good):
            continue
        mag = np.full_like(flux, np.nan)
        mag[good] = -2.5 * np.log10(flux[good])
        ax.plot(x, mag, "o-", label=name, lw=1.0, ms=3)
    ax.set_xticks(x, [name for name, _, _ in FILTER_SPECS], rotation=35, ha="right")
    ax.invert_yaxis()
    ax.set_ylabel("Observed synthetic AB magnitude")
    ax.set_title(f"Single-SED synthetic photometry: {draw['label']}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "single_sed_photometry_atlas.png", dpi=180)
    plt.close(fig)


def plot_sweep_heatmaps(output_dir: Path, metrics: list[dict[str, object]]) -> None:
    comparisons = sorted(set(row["comparison"] for row in metrics))
    labels = sorted(set(row["label"] for row in metrics))
    matrix = np.full((len(labels), len(comparisons)), np.nan)
    for row in metrics:
        i = labels.index(row["label"])
        j = comparisons.index(row["comparison"])
        matrix[i, j] = value_or_nan(row.get("max_abs_delta_mag"))
    vmax = max(0.1, np.nanpercentile(np.abs(matrix), 95)) if np.isfinite(matrix).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(comparisons)), max(6, 0.22 * len(labels))), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(comparisons)), comparisons, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Sweep max |ΔAB| by comparison")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("max |ΔAB| [mag]")
    fig.savefig(output_dir / "sweep_max_abs_delta_mag_heatmap.png", dpi=180)
    plt.close(fig)

    plot_worst_band_heatmap(output_dir, metrics, comparisons, labels)

    status_counts = {}
    for row in metrics:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(list(status_counts), list(status_counts.values()))
    ax.set_ylabel("count")
    ax.set_title("Automated discrepancy labels")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output_dir / "discrepancy_status_counts.png", dpi=180)
    plt.close(fig)


def plot_worst_band_heatmap(output_dir: Path, metrics: list[dict[str, object]], comparisons, labels) -> None:
    """Show which synthetic filter is responsible for each max-|delta AB| cell."""

    band_names = [name for name, _, _ in FILTER_SPECS]
    band_to_index = {name: i for i, name in enumerate(band_names)}
    matrix = np.full((len(labels), len(comparisons)), np.nan)
    for row in metrics:
        band = row.get("worst_band")
        if band not in band_to_index:
            continue
        i = labels.index(row["label"])
        j = comparisons.index(row["comparison"])
        matrix[i, j] = band_to_index[band]

    cmap = plt.get_cmap("tab10", len(band_names))
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(comparisons)), max(6, 0.22 * len(labels))), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(band_names) - 0.5)
    ax.set_xticks(np.arange(len(comparisons)), comparisons, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Band responsible for max |delta AB|")
    cbar = fig.colorbar(image, ax=ax, ticks=np.arange(len(band_names)))
    cbar.ax.set_yticklabels(band_names)
    cbar.set_label("worst synthetic band")
    fig.savefig(output_dir / "sweep_worst_band_heatmap.png", dpi=180)
    plt.close(fig)


def print_summary(metrics, output_dir: Path) -> None:
    status_counts = {}
    for row in metrics:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print("Saved cross-validation outputs under", output_dir)
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


def safe_ratio_array(num, den):
    out = np.full_like(num, np.nan, dtype=float)
    good = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    out[good] = num[good] / den[good]
    return out


def median_log10_ratio(ratio, lo_nm, hi_nm):
    good = (REST_WAVE_NM >= lo_nm) & (REST_WAVE_NM < hi_nm) & np.isfinite(ratio) & (ratio > 0.0)
    if np.count_nonzero(good) == 0:
        return None
    return json_float(np.nanmedian(np.log10(ratio[good])))


def median_delta_mag(num_flux, den_flux):
    num = np.asarray(num_flux, dtype=float)
    den = np.asarray(den_flux, dtype=float)
    good = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    if not np.any(good):
        return None
    return json_float(np.nanmedian(-2.5 * np.log10(num[good] / den[good])))


def max_abs_delta_mag(num_flux, den_flux):
    num = np.asarray(num_flux, dtype=float)
    den = np.asarray(den_flux, dtype=float)
    good = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    if not np.any(good):
        return None
    return json_float(np.nanmax(np.abs(-2.5 * np.log10(num[good] / den[good]))))


def worst_delta_mag_band(num_flux, den_flux):
    num = np.asarray(num_flux, dtype=float)
    den = np.asarray(den_flux, dtype=float)
    good = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    if not np.any(good):
        return None
    delta = np.full(num.shape, np.nan)
    delta[good] = np.abs(-2.5 * np.log10(num[good] / den[good]))
    return [name for name, _, _ in FILTER_SPECS][int(np.nanargmax(delta))]


def nearest_cigale_nebular_zgas(metallicity: float) -> float:
    grid = np.asarray(
        [
            0.0001,
            0.0004,
            0.001,
            0.002,
            0.0025,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.011,
            0.012,
            0.014,
            0.016,
            0.019,
            0.022,
            0.025,
            0.03,
            0.033,
            0.037,
            0.041,
            0.046,
            0.051,
        ],
        dtype=float,
    )
    return float(grid[np.argmin(np.abs(grid - float(metallicity)))])


def flat_lcdm_age_gyr(z, omega_m=0.3075, h=0.6774):
    omega_l = 1.0 - omega_m
    hubble_time_gyr = 9.778 / h
    arg = math.sqrt(omega_l / omega_m) / (1.0 + float(z)) ** 1.5
    return (2.0 / (3.0 * math.sqrt(omega_l))) * math.asinh(arg) * hubble_time_gyr


def flat_lcdm_luminosity_distance_mpc(z, omega_m=0.3075, h=0.6774, n_grid=512):
    z = float(z)
    if z <= 0.0:
        return 1.0e-5
    zz = np.linspace(0.0, z, int(n_grid))
    e_z = np.sqrt(omega_m * (1.0 + zz) ** 3 + (1.0 - omega_m))
    integral = _NP_TRAPEZOID(1.0 / e_z, zz)
    d_comoving_mpc = (C_KM_PER_S / (100.0 * h)) * integral
    return (1.0 + z) * d_comoving_mpc


def json_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def value_or_nan(value):
    if value is None:
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan


if __name__ == "__main__":
    main()
