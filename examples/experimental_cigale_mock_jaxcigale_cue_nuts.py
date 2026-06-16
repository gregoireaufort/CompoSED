#!/usr/bin/env python
"""Fit a CIGALE mock SED with the experimental JAX-CIGALE + Cue model.

This script is intentionally scientist-readable rather than framework-like.
It runs one controlled cross-model test:

1. CIGALE generates one mock SED with
   ``sfhdelayed -> bc03 -> nebular -> dustatt_modified_starburst``.
2. The rest-frame CIGALE spectrum is redshifted to z=0.2 and integrated
   through simple UV + ugriz + YJH Gaussian filters.
3. A small Gaussian noise vector is added in maggies.
4. JAX-CIGALE fits the saved mock photometry with
   ``delayed SFH -> DSPS stellar -> Cue nebular -> modified starburst dust``.

The two stages can be run in different Python environments:

    # CIGALE environment
    PYTHONPATH=/path/to/CompoSED:/path/to/cigale \
    python examples/experimental_cigale_mock_jaxcigale_cue_nuts.py --stage mock

    # JAX/DSPS/NumPyro/Cue environment
    PYTHONPATH=/path/to/CompoSED \
    python examples/experimental_cigale_mock_jaxcigale_cue_nuts.py --stage fit --no-progress

Units and conventions:

- CIGALE wavelength is nm and luminosity is W/nm.
- The saved rest spectrum is converted to Angstrom and Lsun/Angstrom.
- Observed broadband fluxes are maggies.
- The SED is generated per solar mass formed; ``log10_mass`` is applied once
  when turning luminosity into observed photometry.
- The likelihood and NUTS fit consume exactly the noisy active-band vector
  saved by the mock stage.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from composed.experimental.jaxcigale.ssp_data import (
    default_continuum_ssp_path,
    require_continuum_ssp_path,
)
from composed.experimental.jaxcigale.core import flat_lcdm_age_gyr_numpy


LSUN_W = 3.828e26
LSUN_CGS = 3.828e33
MPC_CM = 3.0856775814913673e24
C_A_PER_S = 2.99792458e18
C_KM_PER_S = 299792.458
AB_FNU_CGS = 3631.0e-23

OUTPUT_DIR = Path("outputs/experimental_cigale_mock_jaxcigale_cue_nuts")
MOCK_FILE = "cigale_bc03_nebular_dust_mock.npz"
FIT_FILE = "jaxcigale_cue_nuts_fit.npz"

# Smooth synthetic filters keep the test independent of local filter database
# names. They are broad enough to behave like the requested UV + ugriz + YJH
# bands, but simple enough that the photometry integral is transparent.
FILTER_SPECS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mock", "fit", "benchmark", "plots", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument("--relative-error", type=float, default=0.10)
    parser.add_argument("--noise-floor", type=float, default=1.0e-13, help="Absolute noise floor in maggies.")
    parser.add_argument("--redshift", type=float, default=0.2)
    parser.add_argument("--log10-mass", type=float, default=10.2)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--nuts-target-accept", type=float, default=0.8)
    parser.add_argument("--nuts-max-tree-depth", type=int, default=10)
    parser.add_argument("--nuts-dense-mass", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--benchmark-nuts-warmup", type=int, default=20)
    parser.add_argument("--benchmark-nuts-samples", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--ssp-file",
        type=Path,
        default=Path(os.environ.get("DSPS_CONTINUUM_SSP_FILE", default_continuum_ssp_path())),
    )
    parser.add_argument(
        "--cue-data-dir",
        type=Path,
        default=Path(os.environ.get("CUE_DATA_DIR", "/private/tmp/cue/src/cue/data")),
    )
    parser.add_argument("--jax-platform", choices=("auto", "cpu", "cuda", "gpu", "mps", "metal"), default="auto")
    parser.add_argument("--precision", choices=("auto", "float64", "float32"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"mock", "all"}:
        generate_cigale_mock(args)
    if args.stage in {"fit", "all"}:
        fit_with_jaxcigale_cue(args)
    if args.stage in {"benchmark", "all"}:
        benchmark_jaxcigale_timing(args)
    if args.stage in {"plots", "all"}:
        make_audit_plots(args.output_dir)


def generate_cigale_mock(args: argparse.Namespace) -> None:
    """Generate one noisy photometric SED from CIGALE BC03 + nebular + dust."""

    from pcigale.warehouse import SedWarehouse

    truth = truth_parameters(args.redshift, args.log10_mass)
    modules = ["sfhdelayed", "bc03", "nebular", "dustatt_modified_starburst"]
    module_parameters = [
        cigale_sfh_parameters(truth),
        cigale_bc03_parameters(truth),
        cigale_nebular_parameters(truth),
        cigale_dust_parameters(truth),
    ]

    warehouse = SedWarehouse(nocache=["nebular", "dustatt_modified_starburst"])
    sed = warehouse.get_sed(modules, module_parameters)
    wave_rest_a, luminosity_lsun_per_a = cigale_spectrum_to_lsun_per_a(sed)

    # The CIGALE SED is per solar mass formed because sfhdelayed.normalise=True.
    # Apply the true mass exactly once before redshifting/filter integration.
    mass = 10.0 ** truth["log10_mass"]
    noiseless_flux = photometry_from_rest_spectrum(wave_rest_a, luminosity_lsun_per_a * mass, truth["z"])

    rng = np.random.default_rng(args.seed)
    sigma = args.relative_error * np.abs(noiseless_flux) + args.noise_floor
    observed_flux = noiseless_flux + rng.normal(0.0, sigma)

    check_photometry_vector("CIGALE noiseless flux", noiseless_flux)
    check_photometry_vector("CIGALE noisy flux", observed_flux)
    check_sigma_vector(sigma)

    output_path = args.output_dir / MOCK_FILE
    np.savez(
        output_path,
        rest_wave_a=wave_rest_a,
        rest_luminosity_lsun_per_a=luminosity_lsun_per_a,
        noiseless_flux_maggies=noiseless_flux,
        observed_flux_maggies=observed_flux,
        sigma_maggies=sigma,
        filter_names=np.asarray([name for name, _, _ in FILTER_SPECS]),
        filter_centers_a=np.asarray([center for _, center, _ in FILTER_SPECS], dtype=float),
        filter_widths_a=np.asarray([width for _, _, width in FILTER_SPECS], dtype=float),
        true_theta_for_fit=np.asarray(
            [truth[name] for name in fit_parameter_names()],
            dtype=float,
        ),
        true_theta_names=np.asarray(fit_parameter_names()),
        true_theta_reported=np.asarray(
            [truth[name] for name in reported_parameter_names()],
            dtype=float,
        ),
        true_theta_reported_names=np.asarray(reported_parameter_names()),
    )
    (args.output_dir / "cigale_mock_truth.json").write_text(
        json.dumps(
            {
                "generator": "CIGALE sfhdelayed + bc03 + nebular + dustatt_modified_starburst",
                "truth": truth,
                "cigale_modules": modules,
                "cigale_module_parameters": module_parameters,
                "noise": {
                    "relative_error": args.relative_error,
                    "noise_floor_maggies": args.noise_floor,
                    "seed": args.seed,
                },
                "filter_specs": FILTER_SPECS,
                "units": {
                    "rest_wave_a": "Angstrom",
                    "rest_luminosity_lsun_per_a": "Lsun / Angstrom / formed Msun before mass scaling",
                    "flux_maggies": "maggies after multiplying by 10**log10_mass",
                },
            },
            indent=2,
        )
        + "\n"
    )

    print("\nGenerated CIGALE mock:", output_path)
    print("Truth reported in physical coordinates:")
    saved = output_path_to_np(output_path)
    for name, value in zip(saved["true_theta_reported_names"], saved["true_theta_reported"]):
        print(f"  {name:12s} {float(value): .5f}")
    print("Internal NUTS coordinate:")
    for name, value in zip(saved["true_theta_names"], saved["true_theta_for_fit"]):
        print(f"  {name:12s} {float(value): .5f}")
    print("\nBands:")
    for band, f0, fobs, sig in zip([name for name, _, _ in FILTER_SPECS], noiseless_flux, observed_flux, sigma):
        print(f"  {band:8s} noiseless={f0:.6e} observed={fobs:.6e} sigma={sig:.3e} maggies")


def fit_with_jaxcigale_cue(args: argparse.Namespace) -> None:
    """Fit the saved mock with JAX-CIGALE + Cue using NumPyro NUTS."""

    configure_jax_environment(args.jax_platform, args.precision)

    from dsps import load_ssp_templates

    from composed.experimental.jaxcigale import (
        CueJaxPort,
        GaussianPhotometricData,
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        cue_nebular_module,
        delayed_sfh_cosmic_time_module,
        dsps_stellar_module,
        madau_igm_module,
        modified_starburst_attenuation_module,
        redshift_module,
        run_numpyro_nuts,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()
    mock = np.load(args.output_dir / MOCK_FILE, allow_pickle=True)
    filters = make_jax_filter_set(JaxFilterSet)
    parameter_space = make_fit_parameter_space(JaxParameterSpace, UniformJaxPrior)
    fixed_parameters = fixed_jaxcigale_parameters()

    ssp_file = require_continuum_ssp_path(args.ssp_file)
    ssp_data = load_ssp_templates(fn=str(ssp_file))
    cue_port = CueJaxPort.from_public_cue_data_dir(args.cue_data_dir)

    rest_wave_a = np.geomspace(50.0, 30000.0, 1800)
    modules = [
        delayed_sfh_cosmic_time_module(
            n_time=180,
            tage_parameter="tage_fraction",
            tage_is_fraction_of_universe_age=True,
        ),
        dsps_stellar_module(ssp_data, z_sun=0.02, separation_age_myr=10.0),
        cue_nebular_module(cue_port.make_nebular_apply(line_sigma_a=1.5)),
        modified_starburst_attenuation_module(
            ebv_young_parameter="E_BV_young",
            ebv_old_factor_parameter="E_BV_old_factor",
            powerlaw_slope_parameter="powerlaw_slope",
            uv_bump_amplitude_parameter="uv_bump_amplitude",
            nebular_ebv_parameter="E_BV_nebular",
            nebular_extinction_law="mw_ccm89",
            nebular_rv=3.1,
        ),
        madau_igm_module(),
        redshift_module(),
    ]
    model = build_jax_sed_model(
        modules,
        rest_wave_a,
        filters,
        parameter_space,
        fixed_parameters=fixed_parameters,
    )
    data = GaussianPhotometricData(mock["observed_flux_maggies"], mock["sigma_maggies"])

    true_theta = np.asarray(mock["true_theta_for_fit"], dtype=float)
    initial_theta = parameter_space.from_dict(
        {
            "log10_mass": 10.05,
            "z": 0.24,
            "logzsol": -0.45,
            "E_BV_young": 0.08,
            "tau_gyr": 2.0,
            "tage_fraction": 0.36,
        }
    )

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("JAX default float dtype:", np.asarray(jnp.asarray(1.0)).dtype)
    print("DSPS SSP file:", ssp_file)
    print("Cue data directory:", args.cue_data_dir)
    print("Fixed JAX-CIGALE nuisance parameters:")
    for key, value in fixed_parameters.items():
        print(f"  {key:18s} {value: .5f}")

    t0 = time.perf_counter()
    result = run_numpyro_nuts(
        model,
        data,
        initial_theta=initial_theta,
        num_warmup=args.warmup,
        num_samples=args.samples,
        rng_seed=args.seed + 100,
        progress_bar=not args.no_progress,
        transform_bounds=True,
        target_accept_prob=args.nuts_target_accept,
        max_tree_depth=args.nuts_max_tree_depth,
        dense_mass=args.nuts_dense_mass,
    )
    nuts_seconds = time.perf_counter() - t0

    samples = np.asarray(result.samples)
    nuts_diagnostics = summarize_nuts_extra_fields(result.extra_fields.get("numpyro", {}))
    model_flux_at_truth = np.asarray(jax.jit(model.predict_photometry)(jnp.asarray(true_theta)))
    posterior_flux = posterior_predictive_fluxes(model, samples)
    flux_median = np.nanmedian(posterior_flux, axis=0)
    flux_q16, flux_q84 = np.nanquantile(posterior_flux, [0.16, 0.84], axis=0)
    reported_samples, reported_names = samples_to_reported_parameters(samples, result.theta_names)
    reported_truth, reported_truth_names = theta_to_reported_parameters(true_theta, result.theta_names)
    summary = summarize_samples(reported_samples, reported_names, reported_truth)

    output_path = args.output_dir / FIT_FILE
    np.savez(
        output_path,
        samples=reported_samples,
        theta_names=np.asarray(reported_names),
        true_theta=reported_truth,
        samples_internal=samples,
        theta_names_internal=np.asarray(result.theta_names),
        true_theta_internal=true_theta,
        log_prob=result.log_prob,
        model_flux_at_truth_maggies=model_flux_at_truth,
        posterior_flux_maggies=posterior_flux,
        posterior_flux_median_maggies=flux_median,
        posterior_flux_q16_maggies=flux_q16,
        posterior_flux_q84_maggies=flux_q84,
        initial_theta=initial_theta,
        initial_theta_reported=theta_to_reported_parameters(initial_theta, result.theta_names)[0],
    )
    (args.output_dir / "jaxcigale_cue_nuts_summary.json").write_text(
        json.dumps(
            {
                "fitter": "JAX-CIGALE DSPS stellar + Cue nebular + modified starburst dust",
                "theta_names": list(reported_names),
                "sampled_theta_names_internal": list(result.theta_names),
                "fixed_parameters": fixed_parameters,
                "truth_coordinates": dict(zip(reported_truth_names, map(float, reported_truth))),
                "truth_coordinates_internal": dict(zip(result.theta_names, map(float, true_theta))),
                "summary": summary,
                "timings": {
                    "nuts_seconds": nuts_seconds,
                    "posterior_samples": args.samples,
                    "posterior_samples_per_second": args.samples / max(nuts_seconds, 1.0e-12),
                },
                "nuts_settings": {
                    "target_accept_prob": args.nuts_target_accept,
                    "max_tree_depth": args.nuts_max_tree_depth,
                    "dense_mass": args.nuts_dense_mass,
                },
                "nuts_diagnostics": nuts_diagnostics,
                "jax_backend": jax.default_backend(),
                "jax_default_float_dtype": str(np.asarray(jnp.asarray(1.0)).dtype),
            },
            indent=2,
        )
        + "\n"
    )

    print("\nPosterior summary:")
    for name, item in summary.items():
        print(
            f"  {name:12s} truth={item['truth']: .5f} "
            f"median={item['median']: .5f} "
            f"[{item['q16']: .5f}, {item['q84']: .5f}] "
            f"delta={item['median_minus_truth']: .5f}"
        )
    print("\nInternal sampled coordinate was tage_fraction; reported summary converts it to tage_gyr.")
    print("\nFit timing:")
    print(f"  NUTS seconds: {nuts_seconds:.2f}")
    print(f"  samples/sec : {args.samples / max(nuts_seconds, 1.0e-12):.3f}")
    if nuts_diagnostics:
        print("NUTS diagnostics:")
        for key, item in nuts_diagnostics.items():
            print(f"  {key:24s} {item}")
    print("Saved fit:", output_path)


def benchmark_jaxcigale_timing(args: argparse.Namespace) -> None:
    """Time the expensive scientific pieces in the JAX-CIGALE likelihood."""

    configure_jax_environment(args.jax_platform, args.precision)

    from dsps import load_ssp_templates

    from composed.experimental.jaxcigale import (
        CueJaxPort,
        GaussianPhotometricData,
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        cue_nebular_module,
        delayed_sfh_cosmic_time_module,
        dsps_stellar_module,
        madau_igm_module,
        modified_starburst_attenuation_module,
        no_nebular_module,
        redshift_module,
        run_numpyro_nuts,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()
    mock = np.load(args.output_dir / MOCK_FILE, allow_pickle=True)
    data = GaussianPhotometricData(mock["observed_flux_maggies"], mock["sigma_maggies"])
    theta = jnp.asarray(np.asarray(mock["true_theta_for_fit"], dtype=float))
    filters = make_jax_filter_set(JaxFilterSet)
    parameter_space = make_fit_parameter_space(JaxParameterSpace, UniformJaxPrior)
    fixed_parameters = fixed_jaxcigale_parameters()

    ssp_file = require_continuum_ssp_path(args.ssp_file)
    t0 = time.perf_counter()
    ssp_data = load_ssp_templates(fn=str(ssp_file))
    ssp_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    cue_port = CueJaxPort.from_public_cue_data_dir(args.cue_data_dir)
    cue_load_seconds = time.perf_counter() - t0

    rest_wave_a = np.geomspace(50.0, 30000.0, 1800)
    sfh = delayed_sfh_cosmic_time_module(
        n_time=180,
        tage_parameter="tage_fraction",
        tage_is_fraction_of_universe_age=True,
    )
    stellar = dsps_stellar_module(ssp_data, z_sun=0.02, separation_age_myr=10.0)
    no_nebular = no_nebular_module()
    cue = cue_nebular_module(cue_port.make_nebular_apply(line_sigma_a=1.5))
    dust = modified_starburst_attenuation_module(
        ebv_young_parameter="E_BV_young",
        ebv_old_factor_parameter="E_BV_old_factor",
        powerlaw_slope_parameter="powerlaw_slope",
        uv_bump_amplitude_parameter="uv_bump_amplitude",
        nebular_ebv_parameter="E_BV_nebular",
        nebular_extinction_law="mw_ccm89",
        nebular_rv=3.1,
    )
    igm = madau_igm_module()
    redshift = redshift_module()

    variants = {
        "dsps_stellar_only": [sfh, stellar, redshift],
        "dsps_stellar_plus_dust": [sfh, stellar, no_nebular, dust, igm, redshift],
        "dsps_plus_cue_no_dust": [sfh, stellar, cue, redshift],
        "full_dsps_cue_dust_igm": [sfh, stellar, cue, dust, igm, redshift],
    }

    print("Timing JAX-CIGALE graph variants")
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("JAX default float dtype:", np.asarray(jnp.asarray(1.0)).dtype)
    print("SSP load seconds:", f"{ssp_load_seconds:.3f}")
    print("Cue load seconds:", f"{cue_load_seconds:.3f}")
    print("Benchmark repeats after compilation:", args.benchmark_repeats)

    rows = []
    full_model = None
    for label, modules in variants.items():
        model = build_jax_sed_model(
            modules,
            rest_wave_a,
            filters,
            parameter_space,
            fixed_parameters=fixed_parameters,
        )
        if label == "full_dsps_cue_dust_igm":
            full_model = model
        row = benchmark_one_model(jax, model, data, theta, label, repeats=args.benchmark_repeats)
        rows.append(row)
        print(
            f"  {label:24s} "
            f"compile_predict={row['compile_predict_seconds']:.3f}s "
            f"predict={row['predict_seconds_mean']:.4f}s "
            f"log_prob={row['log_prob_seconds_mean']:.4f}s "
            f"value_and_grad={row['value_and_grad_seconds_mean']:.4f}s"
        )

    nuts_timing = None
    if full_model is not None and args.benchmark_nuts_samples > 0:
        print(
            "Running short NUTS diagnostic:",
            f"warmup={args.benchmark_nuts_warmup}",
            f"samples={args.benchmark_nuts_samples}",
        )
        t0 = time.perf_counter()
        nuts_result = run_numpyro_nuts(
            full_model,
            data,
            initial_theta=np.asarray(theta),
            num_warmup=args.benchmark_nuts_warmup,
            num_samples=args.benchmark_nuts_samples,
            rng_seed=args.seed + 991,
            progress_bar=False,
            transform_bounds=True,
            target_accept_prob=args.nuts_target_accept,
            max_tree_depth=args.nuts_max_tree_depth,
            dense_mass=args.nuts_dense_mass,
        )
        nuts_seconds = time.perf_counter() - t0
        nuts_diagnostics = summarize_nuts_extra_fields(nuts_result.extra_fields.get("numpyro", {}))
        nuts_timing = {
            "warmup": int(args.benchmark_nuts_warmup),
            "samples": int(args.benchmark_nuts_samples),
            "seconds": float(nuts_seconds),
            "seconds_per_posterior_sample": float(nuts_seconds / max(args.benchmark_nuts_samples, 1)),
            "diagnostics": nuts_diagnostics,
            "target_accept_prob": float(args.nuts_target_accept),
            "max_tree_depth": int(args.nuts_max_tree_depth),
            "dense_mass": bool(args.nuts_dense_mass),
        }
        print(
            f"  short_nuts seconds={nuts_seconds:.3f} "
            f"seconds/sample={nuts_timing['seconds_per_posterior_sample']:.3f}"
        )
        if nuts_diagnostics:
            for key, item in nuts_diagnostics.items():
                print(f"  {key:24s} {item}")

    output_path = args.output_dir / "jaxcigale_timing_benchmark.json"
    output_path.write_text(
        json.dumps(
            {
                "jax_backend": jax.default_backend(),
                "jax_default_float_dtype": str(np.asarray(jnp.asarray(1.0)).dtype),
                "ssp_file": str(ssp_file),
                "cue_data_dir": str(args.cue_data_dir),
                "ssp_load_seconds": ssp_load_seconds,
                "cue_load_seconds": cue_load_seconds,
                "benchmark_repeats": args.benchmark_repeats,
                "rows": rows,
                "short_nuts": nuts_timing,
            },
            indent=2,
        )
        + "\n"
    )
    print("Saved timing benchmark:", output_path)


def benchmark_one_model(jax, model, data, theta, label: str, repeats: int) -> dict[str, float | str]:
    """Compile and time predict/log_prob/value_and_grad for one graph."""

    def block(value):
        if isinstance(value, tuple):
            for item in value:
                block(item)
            return value
        value.block_until_ready()
        return value

    predict_fn = jax.jit(model.predict_photometry)
    log_prob_fn = jax.jit(lambda x: model.log_prob(x, data))
    value_and_grad_fn = jax.jit(jax.value_and_grad(lambda x: model.log_prob(x, data)))

    t0 = time.perf_counter()
    block(predict_fn(theta))
    compile_predict_seconds = time.perf_counter() - t0

    predict_seconds = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        block(predict_fn(theta))
        predict_seconds.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    block(log_prob_fn(theta))
    compile_log_prob_seconds = time.perf_counter() - t0

    log_prob_seconds = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        block(log_prob_fn(theta))
        log_prob_seconds.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    block(value_and_grad_fn(theta))
    compile_value_and_grad_seconds = time.perf_counter() - t0

    value_and_grad_seconds = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        block(value_and_grad_fn(theta))
        value_and_grad_seconds.append(time.perf_counter() - t0)

    return {
        "label": label,
        "compile_predict_seconds": float(compile_predict_seconds),
        "predict_seconds_mean": float(np.mean(predict_seconds)),
        "predict_seconds_min": float(np.min(predict_seconds)),
        "compile_log_prob_seconds": float(compile_log_prob_seconds),
        "log_prob_seconds_mean": float(np.mean(log_prob_seconds)),
        "log_prob_seconds_min": float(np.min(log_prob_seconds)),
        "compile_value_and_grad_seconds": float(compile_value_and_grad_seconds),
        "value_and_grad_seconds_mean": float(np.mean(value_and_grad_seconds)),
        "value_and_grad_seconds_min": float(np.min(value_and_grad_seconds)),
    }


def summarize_nuts_extra_fields(extra_fields: dict[str, object]) -> dict[str, dict[str, float]]:
    """Make NumPyro sampler diagnostics JSON-friendly and human-readable."""

    summary: dict[str, dict[str, float]] = {}
    for key, value in extra_fields.items():
        array = np.asarray(value)
        if array.size == 0:
            continue
        if array.dtype == bool:
            summary[key] = {
                "n_true": int(np.sum(array)),
                "fraction_true": float(np.mean(array)),
            }
            continue
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    return summary


def truth_parameters(redshift: float, log10_mass: float) -> dict[str, float]:
    """One explicit physical input vector for the CIGALE mock."""

    metallicity = 0.008
    e_bv_lines = 0.25
    e_bv_factor = 0.44
    return {
        "z": float(redshift),
        "log10_mass": float(log10_mass),
        "tage_gyr": 5.0,
        "tage_fraction": float(5.0 / flat_lcdm_age_gyr_numpy(float(redshift))),
        "tau_gyr": 2.5,
        "metallicity": metallicity,
        # Use CIGALE/BC03's solar metallicity convention for this coordinate.
        "logzsol": float(np.log10(metallicity / 0.02)),
        "gas_logu": -2.5,
        "gas_logn_h": 2.0,
        "E_BV_lines": e_bv_lines,
        "E_BV_factor": e_bv_factor,
        "E_BV_young": e_bv_lines * e_bv_factor,
        "uv_bump_amplitude": 0.0,
        "powerlaw_slope": 0.0,
    }


def cigale_sfh_parameters(truth: dict[str, float]) -> dict[str, object]:
    return {
        "age_main": int(round(truth["tage_gyr"] * 1000.0)),
        "tau_main": float(truth["tau_gyr"] * 1000.0),
        "age_burst": 10,
        "tau_burst": 10.0,
        "f_burst": 0.0,
        "normalise": True,
    }


def cigale_bc03_parameters(truth: dict[str, float]) -> dict[str, object]:
    return {
        "imf": 1,
        "metallicity": float(truth["metallicity"]),
        "separation_age": 10,
    }


def cigale_nebular_parameters(truth: dict[str, float]) -> dict[str, object]:
    return {
        "logU": float(truth["gas_logu"]),
        "zgas": float(truth["metallicity"]),
        "ne": 100.0,
        "f_esc": 0.0,
        "f_dust": 0.0,
        "lines_width": 300.0,
        "emission": True,
    }


def cigale_dust_parameters(truth: dict[str, float]) -> dict[str, object]:
    return {
        "E_BV_lines": float(truth["E_BV_lines"]),
        "E_BV_factor": float(truth["E_BV_factor"]),
        "uv_bump_wavelength": 217.5,
        "uv_bump_width": 35.0,
        "uv_bump_amplitude": float(truth["uv_bump_amplitude"]),
        "powerlaw_slope": float(truth["powerlaw_slope"]),
        "Ext_law_emission_lines": 1,
        "Rv": 3.1,
        # CIGALE's dust module stores attenuation diagnostics in these filters.
        # Our final photometry below is still computed with the explicit
        # synthetic filters in FILTER_SPECS.
        "filters": "B_B90 & V_B90",
    }


def fixed_jaxcigale_parameters() -> dict[str, float]:
    """Nuisance parameters held fixed in the JAX/Cue fit."""

    return {
        "E_BV_old_factor": 1.0,
        "E_BV_nebular": 0.25,
        "powerlaw_slope": 0.0,
        "uv_bump_amplitude": 0.0,
        "gas_logu": -2.5,
        "gas_logn_h": 2.0,
        "gas_stellar_logoh_offset": 0.0,
        "gas_logno": -0.134,
        "gas_logco": -0.134,
        "gas_f_esc": 0.0,
        "gas_f_dust": 0.0,
    }


def fit_parameter_names() -> tuple[str, ...]:
    """Internal order of parameters sampled by NUTS."""

    return ("log10_mass", "z", "logzsol", "E_BV_young", "tau_gyr", "tage_fraction")


def reported_parameter_names() -> tuple[str, ...]:
    """Scientist-facing parameter order written to summaries and plots."""

    return ("log10_mass", "z", "logzsol", "E_BV_young", "tau_gyr", "tage_gyr")


def make_fit_parameter_space(JaxParameterSpace, UniformJaxPrior):
    return JaxParameterSpace(
        names=fit_parameter_names(),
        priors={
            "log10_mass": UniformJaxPrior(8.0, 12.0),
            # Keep the first SFH-fitting diagnostic in a redshift range where
            # the fitted galaxy age prior cannot exceed the Universe age.
            "z": UniformJaxPrior(0.01, 0.6),
            "logzsol": UniformJaxPrior(-1.2, 0.3),
            "E_BV_young": UniformJaxPrior(0.0, 0.5),
            "tau_gyr": UniformJaxPrior(0.2, 8.0),
            "tage_fraction": UniformJaxPrior(0.02, 0.98),
        },
    )


def configure_jax_environment(platform: str, precision: str) -> None:
    """Set process-wide JAX knobs before importing JAX-heavy modules."""

    if platform != "auto":
        if platform == "metal":
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ["JAX_PLATFORMS"] = "METAL"
        else:
            platform_name = {"cuda": "gpu"}.get(platform, platform)
            os.environ["JAX_PLATFORM_NAME"] = platform_name
            os.environ["JAX_PLATFORMS"] = platform_name
    if precision != "auto":
        enable_x64 = precision == "float64"
        os.environ["COMPOSED_JAX_ENABLE_X64"] = "1" if enable_x64 else "0"
        os.environ["JAX_ENABLE_X64"] = "True" if enable_x64 else "False"


def make_jax_filter_set(JaxFilterSet):
    names = []
    waves = []
    transmissions = []
    for name, center, width in FILTER_SPECS:
        wave, transmission = gaussian_filter_curve(center, width, n=220)
        names.append(name)
        waves.append(wave)
        transmissions.append(transmission)
    return JaxFilterSet.from_curves(names, waves, transmissions)


def gaussian_filter_curve(center_a: float, sigma_a: float, n: int = 220) -> tuple[np.ndarray, np.ndarray]:
    wave = np.linspace(center_a - 3.5 * sigma_a, center_a + 3.5 * sigma_a, int(n))
    transmission = np.exp(-0.5 * ((wave - center_a) / sigma_a) ** 2)
    return wave, transmission


def cigale_spectrum_to_lsun_per_a(sed) -> tuple[np.ndarray, np.ndarray]:
    """Convert CIGALE's nm/W/nm rest spectrum to Angstrom/Lsun/Angstrom."""

    wave_nm = np.asarray(sed.wavelength_grid, dtype=float)
    luminosity_w_per_nm = np.asarray(sed.luminosity, dtype=float)
    wave_a = wave_nm * 10.0
    luminosity_lsun_per_a = luminosity_w_per_nm / (LSUN_W * 10.0)
    return wave_a, luminosity_lsun_per_a


def photometry_from_rest_spectrum(wave_rest_a: np.ndarray, luminosity_lsun_per_a: np.ndarray, z: float) -> np.ndarray:
    """Redshift a rest spectrum and integrate maggies in FILTER_SPECS."""

    wave_obs_a = np.asarray(wave_rest_a, dtype=float) * (1.0 + float(z))
    luminosity = np.asarray(luminosity_lsun_per_a, dtype=float)
    d_l_cm = flat_lcdm_luminosity_distance_mpc(float(z)) * MPC_CM
    flux_lambda = luminosity * LSUN_CGS / (4.0 * np.pi * d_l_cm**2 * (1.0 + float(z)))

    fluxes = []
    for _, center, width in FILTER_SPECS:
        filter_wave, filter_trans = gaussian_filter_curve(center, width, n=220)
        flux_on_filter_grid = np.interp(filter_wave, wave_obs_a, flux_lambda, left=0.0, right=0.0)
        numerator = trapezoid(flux_on_filter_grid * filter_wave * filter_trans, filter_wave)
        denominator = trapezoid((C_A_PER_S / filter_wave) * filter_trans, filter_wave)
        fluxes.append((numerator / denominator) / AB_FNU_CGS)
    return np.asarray(fluxes, dtype=float)


def posterior_predictive_fluxes(model, samples: np.ndarray, max_draws: int = 400) -> np.ndarray:
    """Evaluate photometry for posterior samples, thinning only for plotting."""

    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()
    chosen = np.asarray(samples, dtype=float)
    if chosen.shape[0] > max_draws:
        indices = np.linspace(0, chosen.shape[0] - 1, max_draws).astype(int)
        chosen = chosen[indices]
    predict_many = jax.jit(jax.vmap(model.predict_photometry))
    return np.asarray(predict_many(jnp.asarray(chosen)))


def summarize_samples(samples: np.ndarray, names: tuple[str, ...], truth: np.ndarray) -> dict[str, dict[str, float]]:
    quantiles = np.quantile(samples, [0.16, 0.5, 0.84], axis=0)
    return {
        name: {
            "truth": float(truth[i]),
            "q16": float(quantiles[0, i]),
            "median": float(quantiles[1, i]),
            "q84": float(quantiles[2, i]),
            "median_minus_truth": float(quantiles[1, i] - truth[i]),
        }
        for i, name in enumerate(names)
    }


def samples_to_reported_parameters(samples: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert internal NUTS samples to scientist-facing physical coordinates."""

    samples = np.asarray(samples, dtype=float)
    names = tuple(str(name) for name in names)
    if "tage_fraction" not in names:
        return samples, names
    z_index = names.index("z")
    tage_fraction_index = names.index("tage_fraction")
    reported = samples.copy()
    reported[:, tage_fraction_index] = samples[:, tage_fraction_index] * flat_lcdm_age_gyr_numpy(samples[:, z_index])
    reported_names = tuple("tage_gyr" if name == "tage_fraction" else name for name in names)
    return reported, reported_names


def theta_to_reported_parameters(theta: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert one internal theta vector to physical reporting coordinates."""

    reported, reported_names = samples_to_reported_parameters(np.asarray(theta, dtype=float)[None, :], names)
    return reported[0], reported_names


def make_audit_plots(output_dir: Path) -> None:
    mock_path = output_dir / MOCK_FILE
    fit_path = output_dir / FIT_FILE
    if not mock_path.exists():
        raise FileNotFoundError(f"Missing mock file: {mock_path}")

    mock = np.load(mock_path, allow_pickle=True)
    plot_mock_spectrum(output_dir, mock)

    if fit_path.exists():
        fit = np.load(fit_path, allow_pickle=True)
        plot_photometry_fit(output_dir, mock, fit)
        plot_posterior_summary(output_dir, fit)
    else:
        print("Fit file not present; only mock spectrum plot was written.")


def plot_mock_spectrum(output_dir: Path, mock) -> None:
    wave_a = mock["rest_wave_a"]
    lum = mock["rest_luminosity_lsun_per_a"]
    good = np.isfinite(lum) & (lum > 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(wave_a[good], lum[good], color="black", lw=1.2)
    ax.axvline(912.0, color="0.7", ls=":", lw=1)
    ax.axvline(1216.0, color="0.7", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel(r"CIGALE $L_\lambda$ [L$_\odot$ A$^{-1}$ per formed M$_\odot$]")
    ax.set_title("Mock generator spectrum: CIGALE BC03 + nebular + dust")
    fig.savefig(output_dir / "cigale_mock_rest_spectrum.png", dpi=180)
    plt.close(fig)


def plot_photometry_fit(output_dir: Path, mock, fit) -> None:
    names = [str(x) for x in mock["filter_names"]]
    centers = mock["filter_centers_a"]
    observed = mock["observed_flux_maggies"]
    sigma = mock["sigma_maggies"]
    truth_model = fit["model_flux_at_truth_maggies"]
    median = fit["posterior_flux_median_maggies"]
    q16 = fit["posterior_flux_q16_maggies"]
    q84 = fit["posterior_flux_q84_maggies"]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.errorbar(centers, observed, yerr=sigma, fmt="o", color="black", label="CIGALE mock data")
    ax.plot(centers, truth_model, "s--", color="tab:blue", label="JAX model at CIGALE truth coordinates")
    ax.plot(centers, median, "o-", color="tab:orange", label="JAX/Cue posterior median")
    ax.fill_between(centers, q16, q84, color="tab:orange", alpha=0.2, label="posterior 16-84% model band")
    for x, name in zip(centers, names):
        ax.text(x, max(observed.max(), median.max()) * 1.08, name, rotation=45, ha="right", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Observed filter center [Angstrom]")
    ax.set_ylabel("Flux [maggies]")
    ax.set_title("CIGALE mock photometry fit with JAX-CIGALE + Cue")
    ax.legend(fontsize=9)
    fig.savefig(output_dir / "jaxcigale_cue_fit_photometry.png", dpi=180)
    plt.close(fig)


def plot_posterior_summary(output_dir: Path, fit) -> None:
    samples = fit["samples"]
    truth = fit["true_theta"]
    names = [str(x) for x in fit["theta_names"]]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.8 * n), constrained_layout=True)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row == col:
                ax.hist(samples[:, col], bins=35, color="0.25", alpha=0.8)
                ax.axvline(truth[col], color="tab:red", lw=1.5)
            elif row > col:
                ax.plot(samples[:, col], samples[:, row], ".", ms=2, alpha=0.25, color="0.15")
                ax.axvline(truth[col], color="tab:red", lw=1)
                ax.axhline(truth[row], color="tab:red", lw=1)
            else:
                ax.axis("off")
                continue
            if row == n - 1:
                ax.set_xlabel(names[col])
            if col == 0 and row > 0:
                ax.set_ylabel(names[row])
    fig.suptitle("NUTS posterior samples; red lines are CIGALE generator coordinates")
    fig.savefig(output_dir / "jaxcigale_cue_nuts_corner.png", dpi=180)
    plt.close(fig)


def flat_lcdm_luminosity_distance_mpc(z: float, omega_m: float = 0.3075, h: float = 0.6774) -> float:
    zz = np.linspace(0.0, float(z), 512)
    e_z = np.sqrt(omega_m * (1.0 + zz) ** 3 + (1.0 - omega_m))
    integral = trapezoid(1.0 / e_z, zz)
    return (1.0 + float(z)) * (C_KM_PER_S / (100.0 * h)) * integral


def maggies_to_ab(flux_maggies: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux_maggies, dtype=float)
    return np.where(flux > 0.0, -2.5 * np.log10(flux), np.nan)


def output_path_to_np(path: Path):
    return np.load(path, allow_pickle=True)


def check_photometry_vector(name: str, flux: np.ndarray) -> None:
    flux = np.asarray(flux, dtype=float)
    if flux.shape != (len(FILTER_SPECS),):
        raise ValueError(f"{name} shape {flux.shape} does not match number of filters {len(FILTER_SPECS)}.")
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{name} contains NaN or inf.")
    if np.any(flux <= 0.0):
        raise ValueError(f"{name} contains non-positive broadband fluxes.")


def check_sigma_vector(sigma: np.ndarray) -> None:
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (len(FILTER_SPECS),):
        raise ValueError("sigma shape does not match number of filters.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma must be finite and strictly positive.")


def trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


if __name__ == "__main__":
    main()
