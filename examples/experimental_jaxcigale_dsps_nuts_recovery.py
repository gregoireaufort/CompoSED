"""DSPS + JAX-CIGALE mock photometry recovery with NumPyro NUTS.

This is an experimental science sanity check, not a polished public API demo.
It asks one concrete question:

    If the differentiable composed JAX-CIGALE graph generated the data, can
    NUTS recover redshift, mass, metallicity, and dust from noisy broadband
    photometry when the delayed-SFH shape is fixed?

The calculation is deliberately written top-to-bottom. The important units are:

    - rest-frame wavelength grid: Angstrom
    - DSPS stellar output after conversion: Lsun / Angstrom / formed Msun
    - observed spectrum before filter integration: cgs f_lambda
      [erg s^-1 cm^-2 Angstrom^-1] / formed Msun
    - broadband photometry: maggies, then multiplied by 10**log10_mass

By default the observed flux is the noiseless generator value with finite
error bars. Pass ``--noise-realization`` to add one random Gaussian noise
draw. The deterministic default is useful for checking whether the posterior
geometry is centered on the generating parameter before studying stochastic
noise scatter.

The delayed SFH parameters are fixed on purpose. This keeps the first
validation focused on units, redshifting, mass normalization, dust, filter
integration, and the NUTS machinery. Letting SFH shape float from eight
broadbands is a real science problem, not a plumbing test.

Run locally from the repository root, for example:

    DSPS_CONTINUUM_SSP_FILE="/path/to/fsps_continuum_ssp_data.h5" \
    python examples/experimental_jaxcigale_dsps_nuts_recovery.py

The script saves an ``.npz`` file containing truth, observed photometry,
posterior samples, and posterior summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from composed.experimental.jaxcigale.ssp_data import (
    default_continuum_ssp_path,
    require_continuum_ssp_path,
)


def configure_jax_environment(platform: str, precision: str) -> None:
    """Set JAX process-wide device/precision environment before imports."""

    if platform != "auto":
        if platform == "metal":
            # Apple's official jax-metal backend registers the platform as
            # uppercase METAL. JAX_PLATFORM_NAME is lower-cased by parts of
            # JAX 0.4.x, so use JAX_PLATFORMS only for this backend.
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


def import_jaxcigale_after_device_config() -> None:
    """Import JAX-CIGALE only after JAX device/precision env vars are set."""

    global GaussianPhotometricData
    global JaxFilterSet
    global JaxParameterSpace
    global UniformJaxPrior
    global build_jax_sed_model
    global calzetti_attenuation_module
    global delayed_sfh_cosmic_time_module
    global dsps_stellar_module
    global madau_igm_module
    global no_nebular_module
    global redshift_module
    global require_jax
    global run_numpyro_nuts

    from composed.experimental.jaxcigale import (
        GaussianPhotometricData,
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        calzetti_attenuation_module,
        delayed_sfh_cosmic_time_module,
        dsps_stellar_module,
        madau_igm_module,
        no_nebular_module,
        redshift_module,
        run_numpyro_nuts,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssp-file",
        type=Path,
        default=None,
        help="Continuum-only DSPS SSP template HDF5 file. Defaults to DSPS_CONTINUUM_SSP_FILE or a known local candidate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimental_jaxcigale_dsps_nuts_recovery"),
        help="Directory where posterior samples and summaries are written.",
    )
    parser.add_argument("--warmup", type=int, default=250, help="NumPyro NUTS warmup steps.")
    parser.add_argument("--samples", type=int, default=500, help="NumPyro NUTS posterior samples.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for mock noise and NUTS.")
    parser.add_argument("--relative-error", type=float, default=0.02, help="Fractional mock photometric uncertainty.")
    parser.add_argument("--noise-floor", type=float, default=5.0e-14, help="Absolute mock photometric uncertainty floor in maggies.")
    parser.add_argument("--noise-realization", action="store_true", help="Add one Gaussian noise draw to the mock photometry.")
    parser.add_argument(
        "--jax-platform",
        choices=("auto", "cpu", "cuda", "gpu", "mps", "metal"),
        default="auto",
        help="Requested JAX platform. Must be set before JAX imports.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float64", "float32"),
        default="auto",
        help="JAX precision. Auto uses float64 except on MPS/Metal, where float32 is required.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable NumPyro progress bar.")
    return parser.parse_args()


def resolve_ssp_file(cli_path: Path | None) -> Path:
    """Find the continuum-only DSPS SSP file without hiding where it came from."""

    if cli_path is not None:
        return require_continuum_ssp_path(cli_path.expanduser())
    return require_continuum_ssp_path(default_continuum_ssp_path())


def make_broadband_filters() -> JaxFilterSet:
    """Make simple ugrizYJH-like Gaussian filters for a controlled mock test."""

    # These are not survey-official filters. They are intentionally smooth,
    # fixed, and easy to audit for a first differentiable recovery test.
    centers_a = np.array([3600.0, 4800.0, 6200.0, 7600.0, 9000.0, 10200.0, 12500.0, 16500.0])
    widths_a = np.array([320.0, 520.0, 560.0, 650.0, 720.0, 850.0, 1100.0, 1400.0])
    names = ("u_like", "g_like", "r_like", "i_like", "z_like", "Y_like", "J_like", "H_like")

    filter_waves = []
    filter_transmissions = []
    for center, width in zip(centers_a, widths_a):
        wave = np.linspace(center - 3.5 * width, center + 3.5 * width, 220)
        transmission = np.exp(-0.5 * ((wave - center) / width) ** 2)
        filter_waves.append(wave)
        filter_transmissions.append(transmission)
    return JaxFilterSet.from_curves(names, filter_waves, filter_transmissions)


def make_parameter_space() -> JaxParameterSpace:
    """Define the physical coordinates and broad priors used by NUTS."""

    return JaxParameterSpace(
        names=("log10_mass", "z", "logzsol", "dust2"),
        priors={
            "log10_mass": UniformJaxPrior(8.0, 12.0),
            "z": UniformJaxPrior(0.05, 2.5),
            "logzsol": UniformJaxPrior(-1.5, 0.3),
            "dust2": UniformJaxPrior(0.0, 1.5),
        },
    )


def fixed_sfh_parameters() -> dict[str, float]:
    """Delayed-SFH nuisance parameters held fixed in this first recovery test."""

    return {
        "tau_gyr": 2.2,
        "tage_gyr": 4.6,
    }


def build_model(
    ssp_data,
    filters: JaxFilterSet,
    parameter_space: JaxParameterSpace,
    fixed_parameters: dict[str, float],
):
    """Compile the fixed differentiable SED graph used for generation and fitting."""

    # Wavelength coverage must include rest UV through NIR. Redshifting moves
    # this to the observed filters before photometry integration.
    rest_wave_a = np.geomspace(700.0, 30000.0, 1600)
    modules = [
        delayed_sfh_cosmic_time_module(n_time=160),
        dsps_stellar_module(ssp_data),
        no_nebular_module(),
        calzetti_attenuation_module(
            av_parameter="dust2",
            slope_parameter=None,
            bump_amplitude_parameter=None,
        ),
        madau_igm_module(),
        redshift_module(),
    ]
    return build_jax_sed_model(
        modules,
        rest_wave_a,
        filters,
        parameter_space,
        fixed_parameters=fixed_parameters,
    )


def make_mock_data(
    model,
    parameter_space: JaxParameterSpace,
    seed: int,
    relative_error: float,
    noise_floor: float,
    noise_realization: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one noisy mock SED from the same graph we later fit."""

    jax, jnp = require_jax()
    truth = parameter_space.from_dict(
        {
            "log10_mass": 10.35,
            "z": 0.72,
            "logzsol": -0.35,
            "dust2": 0.35,
        }
    )
    noiseless = np.asarray(jax.jit(model.predict_photometry)(jnp.asarray(truth)))

    # A deliberately boring broadband noise model: fractional relative error
    # plus a tiny absolute floor so that very faint UV bands do not get zero
    # uncertainty. Everything remains in maggies.
    sigma = float(relative_error) * np.abs(noiseless) + float(noise_floor)
    rng = np.random.default_rng(seed)
    observed = noiseless + rng.normal(0.0, sigma) if noise_realization else noiseless.copy()
    return truth, noiseless, observed, sigma


def summarize_samples(samples: np.ndarray, names: tuple[str, ...], truth: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute compact posterior summaries in physical coordinates."""

    out: dict[str, dict[str, float]] = {}
    quantiles = np.quantile(samples, [0.16, 0.5, 0.84], axis=0)
    for i, name in enumerate(names):
        out[name] = {
            "truth": float(truth[i]),
            "q16": float(quantiles[0, i]),
            "median": float(quantiles[1, i]),
            "q84": float(quantiles[2, i]),
            "median_minus_truth": float(quantiles[1, i] - truth[i]),
        }
    return out


def main() -> None:
    args = parse_args()
    configure_jax_environment(args.jax_platform, args.precision)

    import_jaxcigale_after_device_config()
    jax, jnp = require_jax()

    from dsps import load_ssp_templates

    # DSPS may touch JAX global precision during import. Re-apply the composed
    # choice after importing DSPS so --precision float32 remains meaningful on
    # MPS/Metal.
    jax, jnp = require_jax()

    ssp_file = resolve_ssp_file(args.ssp_file)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    print("Requested JAX platform:", args.jax_platform)
    print("Requested precision:", args.precision)
    jax_default_dtype = np.asarray(jnp.asarray(1.0)).dtype
    print("JAX default float dtype:", jax_default_dtype)
    print("DSPS SSP file:", ssp_file)
    print("Output directory:", output_dir)

    filters = make_broadband_filters()
    parameter_space = make_parameter_space()
    fixed_parameters = fixed_sfh_parameters()
    ssp_data = load_ssp_templates(fn=str(ssp_file))
    t0 = time.perf_counter()
    model = build_model(ssp_data, filters, parameter_space, fixed_parameters)
    setup_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    truth, noiseless, observed, sigma = make_mock_data(
        model,
        parameter_space,
        seed=args.seed,
        relative_error=args.relative_error,
        noise_floor=args.noise_floor,
        noise_realization=args.noise_realization,
    )
    compile_predict_seconds = time.perf_counter() - t0
    data = GaussianPhotometricData(observed, sigma)

    # Start near but not exactly at the truth. This is a diagnostic run, so the
    # first question is whether the local posterior is navigable and recovers
    # the generator, not whether a bad optimizer can find the mode.
    initial_theta = parameter_space.from_dict(
        {
            "log10_mass": 10.1,
            "z": 0.65,
            "logzsol": -0.55,
            "dust2": 0.25,
        }
    )

    print("\nBands and mock photometry:")
    for band, f0, fobs, sig in zip(filters.names, noiseless, observed, sigma):
        print(f"  {band:7s} noiseless={f0:.6e} observed={fobs:.6e} sigma={sig:.3e} maggies")

    print("\nTruth:")
    for name, value in parameter_space.to_dict(truth).items():
        print(f"  {name:10s} {value: .5f}")
    print("\nFixed SFH nuisance parameters:")
    for name, value in fixed_parameters.items():
        print(f"  {name:10s} {value: .5f}")

    t0 = time.perf_counter()
    nuts_result = run_numpyro_nuts(
        model,
        data,
        initial_theta=initial_theta,
        num_warmup=args.warmup,
        num_samples=args.samples,
        rng_seed=args.seed + 1,
        progress_bar=not args.no_progress,
        transform_bounds=True,
    )
    nuts_seconds = time.perf_counter() - t0

    samples = nuts_result.samples
    summary = summarize_samples(samples, nuts_result.theta_names, truth)
    print("\nPosterior summaries:")
    for name, item in summary.items():
        print(
            f"  {name:10s} truth={item['truth']: .5f} "
            f"median={item['median']: .5f} "
            f"[{item['q16']: .5f}, {item['q84']: .5f}] "
            f"delta={item['median_minus_truth']: .5f}"
        )
    print("\nTimings:")
    print(f"  setup_seconds            {setup_seconds:.3f}")
    print(f"  first_jit_predict_seconds {compile_predict_seconds:.3f}")
    print(f"  nuts_seconds             {nuts_seconds:.3f}")
    print(f"  nuts_samples_per_second  {args.samples / max(nuts_seconds, 1e-12):.3f}")

    npz_path = output_dir / "dsps_nuts_recovery_samples.npz"
    json_path = output_dir / "dsps_nuts_recovery_summary.json"
    np.savez(
        npz_path,
        samples=samples,
        log_prob=nuts_result.log_prob,
        truth=truth,
        noiseless_flux_maggies=noiseless,
        observed_flux_maggies=observed,
        sigma_maggies=sigma,
        band_names=np.asarray(filters.names),
        theta_names=np.asarray(nuts_result.theta_names),
        initial_theta=initial_theta,
    )
    json_path.write_text(
        json.dumps(
            {
                "ssp_file": str(ssp_file),
                "jax_backend": jax.default_backend(),
                "jax_default_float_dtype": str(jax_default_dtype),
                "theta_names": list(nuts_result.theta_names),
                "band_names": list(filters.names),
                "fixed_parameters": fixed_parameters,
                "warmup": args.warmup,
                "samples": args.samples,
                "relative_error": args.relative_error,
                "noise_floor": args.noise_floor,
                "noise_realization": args.noise_realization,
                "timings": {
                    "setup_seconds": setup_seconds,
                    "first_jit_predict_seconds": compile_predict_seconds,
                    "nuts_seconds": nuts_seconds,
                    "nuts_samples_per_second": args.samples / max(nuts_seconds, 1.0e-12),
                },
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved posterior samples: {npz_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
