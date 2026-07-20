"""Validate the stable photometric MAF against a known one-dimensional posterior.

The toy forward model is deliberately transparent:

    noiseless flux = amplitude
    measured flux ~ Normal(amplitude, sigma)
    amplitude ~ Uniform(0, 1)

The depth ``sigma`` is drawn independently for every simulated object and is
passed to the MAF through CompoSED's default ``snr_logsigma`` context. Because
the depth distribution is independent of amplitude, the exact posterior for a
known measured ``(flux, sigma)`` is a Gaussian truncated to ``[0, 1]``. This
script compares learned posterior means and widths against that numerical
reference, runs rank/coverage diagnostics, saves the trained checkpoint, and
fails if broad documented tolerances are exceeded.

This is a neural-posterior validation, not an FSPS or CIGALE physics test.
Optional dependencies: torch, nflows, and matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from composed import (
    Gaussian,
    MAF,
    ParameterSpace,
    PhotometricContext,
    Problem,
    SEDDataset,
    Simulate,
    UniformPrior,
    collect_run_provenance,
    fit,
    save_npz_with_provenance,
    write_provenance,
)
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.filters import FilterSet
from composed.units import MassNormalization
from composed._numerics import trapezoid


BAND_NAMES = ("toy_flux",)
FILTERS = FilterSet([object()], names=BAND_NAMES)


class IdentityFluxBackend(SEDBackend):
    """Return one absolute maggy whose value is the fitted amplitude."""

    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        flux = np.asarray([float(params["amplitude"])])
        return ModelPhotometry(filters.names, flux, flux_unit="maggies")


def random_depth(noiseless_flux, rng=None):
    """Draw an object-specific Gaussian sigma, independently of amplitude."""

    if rng is None:
        rng = np.random.default_rng()
    sigma = rng.uniform(0.03, 0.20)
    return np.full_like(noiseless_flux, sigma, dtype=float)


def truncated_gaussian_moments(flux, sigma, n_grid=4001):
    """Numerically integrate p(amplitude | flux, sigma) on the prior support."""

    flux = np.asarray(flux, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    amplitude_grid = np.linspace(0.0, 1.0, int(n_grid))
    density = np.exp(-0.5 * ((amplitude_grid[None, :] - flux[:, None]) / sigma[:, None]) ** 2)

    normalization = trapezoid(density, amplitude_grid, axis=1)
    mean = trapezoid(density * amplitude_grid[None, :], amplitude_grid, axis=1) / normalization
    second_moment = (
        trapezoid(density * amplitude_grid[None, :] ** 2, amplitude_grid, axis=1) / normalization
    )
    std = np.sqrt(np.maximum(second_moment - mean**2, 0.0))
    return mean, std


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=30_000)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-mean-mae", type=float, default=0.04)
    parser.add_argument("--max-std-mae", type=float, default=0.04)
    parser.add_argument("--coverage-tolerance", type=float, default=0.12)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validate_maf_photometric_sbi"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    parameters = ParameterSpace(
        names=["amplitude"],
        priors={"amplitude": UniformPrior(0.0, 1.0)},
    )
    problem = Problem(
        backend=IdentityFluxBackend(),
        parameters=parameters,
        data=SEDDataset(BAND_NAMES, flux=[0.5], sigma=[0.1], flux_unit="maggies"),
        likelihood=Gaussian(),
        filters=FILTERS,
    )

    result = fit(
        problem,
        method=MAF(
            hidden_features=64,
            num_transforms=4,
            num_blocks=2,
            learning_rate=1.0e-3,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.15,
            patience=20,
            num_samples=64,
            inference_batch_size=256,
            device=args.device,
        ),
        training=Simulate(
            n=args.n_train,
            noise_fn=random_depth,
            infer=["amplitude"],
            context=PhotometricContext("snr_logsigma", flux_unit="maggies"),
        ),
        seed=args.seed,
    )
    posterior = result.inference_state

    theta_true = parameters.sample_prior(args.n_test, rng=rng)
    measured_flux, measured_sigma = problem.simulate_with_uncertainty(
        theta_true,
        noise_fn=random_depth,
        rng=rng,
    )
    posterior_samples = posterior.sample(
        measured_flux,
        sigma=measured_sigma,
        input_units="native",
        num_samples=args.num_samples,
        batch_size=256,
        seed=args.seed + 1,
    )

    exact_mean, exact_std = truncated_gaussian_moments(measured_flux, measured_sigma)
    learned_mean = np.mean(posterior_samples[:, :, 0], axis=1)
    learned_std = np.std(posterior_samples[:, :, 0], axis=1)
    mean_mae = float(np.mean(np.abs(learned_mean - exact_mean)))
    std_mae = float(np.mean(np.abs(learned_std - exact_std)))
    outside_prior = int(np.count_nonzero((posterior_samples < 0.0) | (posterior_samples > 1.0)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = posterior.diagnostics(
        posterior_samples,
        theta_true,
        output_dir=args.output_dir / "diagnostics",
        make_plots=True,
    )
    levels = diagnostics["coverage"]["levels"]
    coverage = diagnostics["coverage"]["mean_coverage"]
    level_index = int(np.argmin(np.abs(levels - 0.68)))
    nominal_coverage = float(levels[level_index])
    empirical_coverage = float(coverage[level_index])

    metrics = {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "epochs_ran": posterior.history.get("epochs_ran", [None])[-1],
        "device": str(posterior.estimator.device),
        "posterior_mean_mae_vs_exact": mean_mae,
        "posterior_std_mae_vs_exact": std_mae,
        "coverage_level": nominal_coverage,
        "empirical_coverage": empirical_coverage,
        "samples_outside_prior": outside_prior,
        "thresholds": {
            "max_mean_mae": args.max_mean_mae,
            "max_std_mae": args.max_std_mae,
            "coverage_tolerance": args.coverage_tolerance,
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    provenance = collect_run_provenance(
        seed=args.seed,
        command_args=vars(args),
        extra={"validation": "bounded_gaussian_photometric_maf", "metrics": metrics},
        repo_root=Path(__file__).resolve().parents[1],
    )
    save_npz_with_provenance(
        args.output_dir / "validation_arrays.npz",
        provenance=provenance,
        compressed=True,
        theta_true=theta_true,
        measured_flux=measured_flux,
        measured_sigma=measured_sigma,
        posterior_samples=posterior_samples,
        exact_mean=exact_mean,
        exact_std=exact_std,
        learned_mean=learned_mean,
        learned_std=learned_std,
    )
    write_provenance(provenance, args.output_dir / "run.provenance.json")
    posterior.save(args.output_dir / "checkpoint", overwrite=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if outside_prior:
        raise AssertionError(f"MAF returned {outside_prior} sample values outside Uniform(0, 1).")
    if mean_mae > args.max_mean_mae:
        raise AssertionError(f"Posterior mean MAE {mean_mae:.4f} exceeds {args.max_mean_mae:.4f}.")
    if std_mae > args.max_std_mae:
        raise AssertionError(f"Posterior std MAE {std_mae:.4f} exceeds {args.max_std_mae:.4f}.")
    if abs(empirical_coverage - nominal_coverage) > args.coverage_tolerance:
        raise AssertionError(
            f"Coverage {empirical_coverage:.3f} differs from nominal {nominal_coverage:.3f} "
            f"by more than {args.coverage_tolerance:.3f}."
        )


if __name__ == "__main__":
    main()
