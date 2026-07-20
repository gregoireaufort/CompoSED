"""Held-out calibration for the two COSMOS2020 tutorial MAFs.

This script does not retrain either neural posterior. It reconstructs the same
CompoSED Problem used by the selected tutorial, draws a fresh test catalog from
that backend and noise model, loads the saved MAF checkpoint, and compares
posterior samples with the known simulated parameters.

The calculation is:

1. sample physical parameters from the tutorial prior;
2. forward model ugrizYJH fluxes with FSPS or CIGALE;
3. draw catalog-like Gaussian measurement noise;
4. sample q(theta | measured flux, sigma) from the saved MAF;
5. calculate marginal ranks, coverage, and posterior-median residuals.

Run notebook 00 and the selected MAF tutorial before this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from composed import (
    ContinuitySFH,
    DelayedTauSFH,
    EmpiricalPhotometricNoise,
    Gaussian,
    ParameterSpace,
    PhotometricContext,
    Problem,
    SEDDataset,
    Simulate,
    StudentTPrior,
    TrainedMAFSBI,
    UniformPrior,
    collect_run_provenance,
    save_npz_with_provenance,
    write_provenance,
)
from composed.filters import FilterSet
from composed.sbi import simulate_sbi_training_set
from inftools.diagnostics import (
    plot_coverage_curve,
    plot_prediction_scatter,
    plot_rank_histograms,
    run_sbi_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "notebooks/tutorials/data/cosmos2020_ugrizYJH_100k.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("fsps", "cigale"))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_catalog_noise(catalog_path: Path):
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Missing prepared tutorial catalog {catalog_path}. "
            "Run notebooks/tutorials/00_prepare_cosmos2020_ugrizYJH.ipynb first."
        )
    with np.load(catalog_path, allow_pickle=False) as catalog:
        flux = np.asarray(catalog["flux_maggies"], dtype=float)
        sigma = np.asarray(catalog["sigma_maggies"], dtype=float)
        band_names = tuple(catalog["band_names"].astype(str))
        filter_names = tuple(catalog["sedpy_filter_names"].astype(str))

    from sedpy.observate import load_filters

    filters = FilterSet(load_filters(list(filter_names)), names=band_names)
    noise_model = EmpiricalPhotometricNoise(
        sigma,
        fractional_error=0.05,
        band_names=band_names,
        flux_unit="maggies",
    )
    placeholder_sigma = np.sqrt(sigma[0] ** 2 + (0.05 * np.abs(flux[0])) ** 2)
    data = SEDDataset(band_names, flux[0], placeholder_sigma, flux_unit="maggies")
    return filters, noise_model, data


def build_fsps_problem(filters, data):
    from composed.backends.fsps import FSPSBackend

    sfh = ContinuitySFH(
        age="age_fraction",
        age_kind="fraction_of_universe",
        lookback_edges_gyr=(0.0, 0.01, 0.03, 0.1, 0.3),
        samples_per_bin=8,
    )
    parameters = ParameterSpace(
        names=("zred", "log10_mass", "logzsol", "dust2", "age_fraction", *sfh.ratio_names),
        priors={
            "zred": UniformPrior(0.05, 5.0),
            "log10_mass": UniformPrior(6.0, 13.0),
            "logzsol": UniformPrior(-1.5, 0.3),
            "dust2": UniformPrior(0.0, 2.0),
            "age_fraction": UniformPrior(0.30, 0.95),
            **{
                name: StudentTPrior(df=2.0, loc=0.0, scale=0.3)
                for name in sfh.ratio_names
            },
        },
    )
    backend = FSPSBackend(
        sfh=sfh,
        sp_kwargs={
            "sfh": 3,
            "imf_type": 1,
            "zcontinuous": 1,
            "dust_type": 2,
            "add_neb_emission": True,
            "add_igm_absorption": True,
            "add_dust_emission": False,
        },
        default_z_key="zred",
    )
    return Problem(backend, parameters, data, Gaussian(), filters=filters)


def build_cigale_problem(filters, data):
    from composed.backends.cigale import build_cigale_backend_and_parameter_space

    sfh = DelayedTauSFH(age="age_fraction", age_kind="fraction_of_universe", tau="tau_gyr")
    modules = ["bc03", "nebular", "dustatt_modified_starburst", "redshifting"]
    module_parameters = {
        "bc03": {"imf": 1, "metallicity": 0.02, "separation_age": 10},
        "nebular": {
            "logU": -2.0,
            "zgas": 0.019,
            "ne": 100.0,
            "f_esc": 0.0,
            "f_dust": 0.0,
            "lines_width": 300.0,
            "emission": True,
        },
        "dustatt_modified_starburst": {
            "E_BV_lines": {"range": [0.0, 0.8]},
            "E_BV_factor": 0.44,
            "uv_bump_wavelength": 217.5,
            "uv_bump_width": 35.0,
            "uv_bump_amplitude": 0.0,
            "powerlaw_slope": 0.0,
            "Ext_law_emission_lines": 1,
            "Rv": 3.1,
            "filters": "B_B90 & V_B90",
        },
        "redshifting": {"redshift": {"range": [0.05, 5.0]}},
    }
    backend, parameters = build_cigale_backend_and_parameter_space(
        modules,
        module_parameters,
        additional_priors={
            "log10_mass": UniformPrior(6.0, 13.0),
            "age_fraction": UniformPrior(0.30, 0.95),
            "tau_gyr": UniformPrior(0.1, 8.0),
        },
        sfh=sfh,
        photometry_mode="sedpy",
        default_z_key="redshift",
    )
    return Problem(backend, parameters, data, Gaussian(), filters=filters)


def calibration_metrics(diagnostics, theta_names):
    summary = diagnostics["summary"]
    ranks = diagnostics["ranks"]["rank_percentiles"]
    levels = diagnostics["coverage"]["levels"]
    coverage = diagnostics["coverage"]["coverage"]
    truth = diagnostics["sample_set"].theta_true

    parameter_metrics = {}
    for index, name in enumerate(theta_names):
        scale = float(np.std(truth[:, index]))
        median_residual = summary["residual_median"][:, index]
        parameter_metrics[name] = {
            "median_bias": float(np.median(median_residual)),
            "median_absolute_error": float(np.median(np.abs(median_residual))),
            "median_absolute_error_over_truth_std": (
                float(np.median(np.abs(median_residual)) / scale) if scale > 0.0 else None
            ),
            "mean_rank_percentile": float(np.mean(ranks[:, index])),
            "coverage": {
                f"{level:.2f}": float(coverage[level_index, index])
                for level_index, level in enumerate(levels)
            },
        }

    return {
        "coverage_levels": levels.tolist(),
        "mean_coverage": np.mean(coverage, axis=1).tolist(),
        "mean_absolute_coverage_error": float(np.mean(np.abs(coverage - levels[:, None]))),
        "maximum_absolute_coverage_error": float(np.max(np.abs(coverage - levels[:, None]))),
        "parameters": parameter_metrics,
    }


def save_diagnostic_plots(output_dir, diagnostics, theta_names):
    figures = {
        "rank_histograms.png": plot_rank_histograms(diagnostics["ranks"], theta_names),
        "coverage_curve.png": plot_coverage_curve(diagnostics["coverage"], theta_names),
        "prediction_scatter.png": plot_prediction_scatter(
            diagnostics["sample_set"].samples,
            diagnostics["sample_set"].theta_true,
            theta_names,
        ),
    }
    for filename, figure in figures.items():
        figure.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or (
        REPO_ROOT / f"outputs/tutorial_{args.backend}_maf_cosmos2020/maf"
    )
    output_dir = args.output_dir or (
        REPO_ROOT / f"outputs/validate_{args.backend}_maf_calibration"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    filters, noise_model, data = load_catalog_noise(args.catalog)
    problem = (
        build_fsps_problem(filters, data)
        if args.backend == "fsps"
        else build_cigale_problem(filters, data)
    )
    posterior = TrainedMAFSBI.load(checkpoint, device=args.device)
    if posterior.theta_names != tuple(problem.parameters.names):
        raise ValueError(
            f"Checkpoint parameters {posterior.theta_names} do not match "
            f"{args.backend} Problem parameters {tuple(problem.parameters.names)}."
        )

    simulation = Simulate(
        n=args.n_test,
        noise_fn=noise_model,
        infer=problem.parameters.names,
        context=PhotometricContext("snr_logsigma", flux_unit="maggies"),
        n_workers=args.n_workers,
        batch_size=128,
        executor="process" if args.n_workers > 1 else "serial",
        mp_context="spawn" if args.n_workers > 1 else None,
        max_retries=max(1000, args.n_test),
    )
    start = time.perf_counter()
    held_out = simulate_sbi_training_set(problem, simulation, rng=args.seed)
    simulation_seconds = time.perf_counter() - start

    start = time.perf_counter()
    posterior_samples = posterior.sample(
        held_out.x_native,
        sigma=held_out.sigma_native,
        input_units="native",
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )
    inference_seconds = time.perf_counter() - start
    if not np.all(np.isfinite(posterior_samples)):
        raise FloatingPointError("Held-out MAF posterior contains NaN or inf values.")

    levels = np.asarray([0.50, 0.68, 0.90, 0.95])
    diagnostics = run_sbi_diagnostics(
        posterior_samples=posterior_samples,
        theta_true=held_out.theta,
        x_test=held_out.x,
        theta_names=posterior.theta_names,
        levels=levels,
        make_plots=False,
    )
    metrics = {
        "backend": args.backend,
        "checkpoint": str(checkpoint),
        "device": str(posterior.estimator.device),
        "n_test": args.n_test,
        "num_samples_per_object": args.num_samples,
        "simulation_seconds": simulation_seconds,
        "inference_seconds": inference_seconds,
        "posterior_draws_per_second": (
            args.n_test * args.num_samples / inference_seconds
        ),
        **calibration_metrics(diagnostics, posterior.theta_names),
    }

    provenance = collect_run_provenance(
        paths={
            "prepared_cosmos2020_catalog": args.catalog,
            "maf_checkpoint": checkpoint,
        },
        seed=args.seed,
        command_args=vars(args),
        extra={"validation": "held_out_cosmos2020_tutorial_maf", "metrics": metrics},
        repo_root=REPO_ROOT,
    )
    save_npz_with_provenance(
        output_dir / "calibration_samples.npz",
        provenance=provenance,
        compressed=True,
        theta_true=held_out.theta,
        measured_flux=held_out.x_native,
        measured_sigma=held_out.sigma_native,
        posterior_samples=posterior_samples,
        rank_percentiles=diagnostics["ranks"]["rank_percentiles"],
        coverage_levels=diagnostics["coverage"]["levels"],
        coverage=diagnostics["coverage"]["coverage"],
    )
    write_provenance(provenance, output_dir / "run.provenance.json")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    save_diagnostic_plots(output_dir, diagnostics, posterior.theta_names)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
