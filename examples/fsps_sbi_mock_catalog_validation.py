"""FSPS mock-catalog validation using the public CompoSED simulation path.

This is a real-backend SBI validation rung.  It deliberately stays inside the
"CompoSED user" workflow:

1. declare scalar priors in a ``ParameterSpace``;
2. declare an FSPS backend;
3. wrap FSPS with a small scalar-SFH adapter that translates ``tau_gyr`` and
   ``age_fraction`` into the tabular SFH arrays expected by FSPS;
4. bind backend, priors, filters, units, and active bands into a ``Problem``;
5. call ``simulate_sbi_training_set`` to draw theta and noisy model fluxes;
7. convert the generated active-band flux vectors to AB magnitudes for the
   neural estimator;
8. train a conditional diffusion joint sampler and run generic SBI diagnostics.

The key point is that FSPS photometry, active-band masking, and explicit mass
normalization all go through ``Problem.simulate`` and the bound photometric likelihood. This
script does not manually call ``backend.predict_photometry`` and does not
manually multiply by ``10**log10_mass``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/composed_mplconfig")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from composed import Diffusion, Gaussian, Problem, SBITrainingSet, Simulate
from composed.backends.fsps import FSPSBackend
from composed.data import SEDDataset
from composed.filters import FilterSet
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.provenance import require_provenance, save_npz_with_provenance
from composed.sbi import simulate_sbi_training_set, train_sbi
from inftools.diagnostics import run_sbi_diagnostics


CACHE_SCHEMA = "fsps_sbi_mock_catalog.native_composed.v1"

FILTER_NAMES = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]

# These are the scalar parameters sampled by CompoSED's ParameterSpace.  The
# first four are the labels we conditionally sample with diffusion.  The last
# two are nuisance SFH parameters that are marginalized over by the training set.
PARAMETER_NAMES = ["zred", "log10_mass", "dust2", "logzsol", "tau_gyr", "age_fraction"]
INFERRED_THETA_NAMES = ["zred", "log10_mass", "dust2", "logzsol"]

PARAMETER_PRIORS = {
    "zred": UniformPrior(0.05, 1.50),
    "log10_mass": UniformPrior(8.0, 11.5),
    "dust2": UniformPrior(0.0, 0.8),
    "logzsol": UniformPrior(-1.0, 0.2),
    "tau_gyr": UniformPrior(0.3, 5.0),
    "age_fraction": UniformPrior(0.15, 0.95),
}

# We train the neural estimator on AB magnitudes, but the native CompoSED
# simulator generates flux-like vectors.  These magnitude errors are converted
# to a local fractional flux sigma before simulation.
MAG_SIGMA = np.array([0.12, 0.08, 0.07, 0.07, 0.08])
FRACTIONAL_FLUX_SIGMA = (np.log(10.0) / 2.5) * MAG_SIGMA


class DelayedTauFSPSBackend:
    """Scalar-parameter adapter in front of ``FSPSBackend``.

    FSPS itself requires ``tabular_time_gyr`` and ``tabular_sfr_msun_per_yr``.
    A normal CompoSED user does not want array-valued priors, so this adapter
    exposes scalar delayed-tau SFH parameters and delegates the actual
    photometry call to ``FSPSBackend``.
    """

    def __init__(self, base_backend: FSPSBackend, n_time: int = 64) -> None:
        self.base_backend = base_backend
        self.n_time = int(n_time)

    @property
    def mass_normalization(self):
        return self.base_backend.mass_normalization

    def predict_photometry(self, params, filters):
        params = dict(params)
        zred = float(params["zred"])
        tau_gyr = float(params.pop("tau_gyr"))
        age_fraction = float(params.pop("age_fraction"))
        time_gyr, sfr = delayed_exponential_sfh(
            zred=zred,
            tau_gyr=tau_gyr,
            age_fraction=age_fraction,
            n_time=self.n_time,
        )
        params["tabular_time_gyr"] = time_gyr
        params["tabular_sfr_msun_per_yr"] = sfr
        return self.base_backend.predict_photometry(params, filters)


def check_runtime_requirements() -> None:
    """Fail early with a useful message if the real FSPS path is unavailable."""

    missing = []
    for module_name in ["fsps", "sedpy", "astropy", "torch"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise RuntimeError(f"Missing required packages for this validation: {', '.join(missing)}.")
    if not os.environ.get("SPS_HOME"):
        raise RuntimeError(
            "SPS_HOME is not set. Set SPS_HOME to the FSPS grid directory before "
            "generating the mock catalog, e.g. `export SPS_HOME=/path/to/fsps`."
        )


def build_parameter_space() -> ParameterSpace:
    """Return the scalar prior space used by the Problem simulator."""

    return ParameterSpace(names=PARAMETER_NAMES, priors=PARAMETER_PRIORS)


def build_problem() -> Problem:
    """Build the complete CompoSED model used to generate the mock catalog."""

    from sedpy.observate import load_filters

    filters = FilterSet(load_filters(FILTER_NAMES), names=FILTER_NAMES)
    backend = DelayedTauFSPSBackend(FSPSBackend(sp_kwargs={"add_igm_absorption": True}))

    # The dataset defines band order and active masks.  Its flux/sigma values
    # are placeholders for simulation; no observed object is being fitted here.
    dataset = SEDDataset(
        FILTER_NAMES,
        flux=np.ones(len(FILTER_NAMES)),
        sigma=np.ones(len(FILTER_NAMES)),
        metadata={"filters": filters},
    )
    return Problem(
        backend=backend,
        data=dataset,
        parameters=build_parameter_space(),
        likelihood=Gaussian(),
        filters=filters,
    )


def delayed_exponential_sfh(
    *,
    zred: float,
    tau_gyr: float,
    age_fraction: float,
    n_time: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scalar SFH parameters into FSPS tabular SFH arrays.

    ``time_gyr`` is age since star-formation onset.  The final age is a fitted
    fraction of the Universe age at ``zred`` so FSPSBackend's age validation
    stays physically meaningful.
    """

    from astropy.cosmology import Planck18

    age_universe_gyr = float(Planck18.age(float(zred)).to("Gyr").value)
    final_age_gyr = np.clip(float(age_fraction), 0.05, 0.98) * age_universe_gyr
    time_gyr = np.linspace(0.01, final_age_gyr, int(n_time))
    tau_gyr = max(float(tau_gyr), 0.05)
    sfr_msun_per_yr = time_gyr * np.exp(-time_gyr / tau_gyr)
    return time_gyr, np.maximum(sfr_msun_per_yr, 0.0)


def fsps_flux_noise_sigma(flux_maggies: np.ndarray) -> np.ndarray:
    """Gaussian sigma in flux units, equivalent to a small AB-mag error."""

    flux_maggies = np.asarray(flux_maggies, dtype=float)
    if flux_maggies.shape != FRACTIONAL_FLUX_SIGMA.shape:
        raise ValueError(f"Expected flux shape {FRACTIONAL_FLUX_SIGMA.shape}; got {flux_maggies.shape}.")
    return np.abs(flux_maggies) * FRACTIONAL_FLUX_SIGMA


def flux_to_abmag_features(flux_maggies: np.ndarray) -> np.ndarray:
    """Convert positive active-band maggies to AB magnitudes."""

    flux_maggies = np.asarray(flux_maggies, dtype=float)
    if np.any(flux_maggies <= 0.0) or not np.all(np.isfinite(flux_maggies)):
        raise FloatingPointError("Simulated flux contains non-positive or non-finite values; cannot convert to AB mag.")
    return -2.5 * np.log10(flux_maggies)


def generate_fsps_mock_catalog(
    *,
    n_train: int,
    n_test: int,
    seed: int,
    max_retries: int = 100,
    simulation_batch_size: int = 1,
    n_workers: int = 1,
    executor: str = "serial",
    mp_context: str | None = None,
) -> dict[str, np.ndarray | dict]:
    """Generate train/test mock photometry through a declared Problem."""

    check_runtime_requirements()
    problem = build_problem()
    n_total = int(n_train) + int(n_test)

    t0 = time.perf_counter()
    training = simulate_sbi_training_set(
        problem,
        Simulate(
            n=n_total,
            noise_fn=fsps_flux_noise_sigma,
            infer=INFERRED_THETA_NAMES,
            feature_transform=flux_to_abmag_features,
            max_retries=max_retries,
            batch_size=simulation_batch_size,
            n_workers=n_workers,
            executor=executor,
            mp_context=mp_context,
        ),
        rng=seed,
    )
    generation_seconds = time.perf_counter() - t0
    theta_all = training.theta_full
    theta_inferred = training.theta
    flux_all = training.x_native
    mag_all = training.x
    sim_metadata = training.metadata["simulate_training_set"]

    return {
        "theta_full_train": theta_all[:n_train],
        "theta_full_test": theta_all[n_train:],
        "theta_train": theta_inferred[:n_train],
        "theta_test": theta_inferred[n_train:],
        "x_flux_train": flux_all[:n_train],
        "x_flux_test": flux_all[n_train:],
        "x_train": mag_all[:n_train],
        "x_test": mag_all[n_train:],
        "metadata": {
            "schema": CACHE_SCHEMA,
            "filter_names": FILTER_NAMES,
            "parameter_names": PARAMETER_NAMES,
            "theta_names": INFERRED_THETA_NAMES,
            "x_names": FILTER_NAMES,
            "mag_sigma_approx": MAG_SIGMA.tolist(),
            "fractional_flux_sigma": FRACTIONAL_FLUX_SIGMA.tolist(),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "seed": int(seed),
            "fsps_generation_seconds": float(generation_seconds),
            "simulation_batch_size": int(simulation_batch_size),
            "n_workers": int(n_workers),
            "executor": executor,
            "mp_context": mp_context,
            "simulate_training_set_metadata": sim_metadata,
            "sps_home": os.environ.get("SPS_HOME", ""),
            "mass_normalization": "handled by Problem likelihood; backend declares PER_SOLAR_MASS",
            "sp_kwargs": {"add_igm_absorption": True},
            "simulator_units": "maggies",
            "sbi_feature_units": "AB magnitudes",
        },
    }


def save_catalog(cache_path: Path, catalog: dict[str, np.ndarray | dict], *, seed: int, command_args: dict) -> None:
    """Save the generated catalog with a provenance sidecar."""

    metadata = json.dumps(catalog["metadata"], indent=2)
    sps_home = os.environ.get("SPS_HOME")
    provenance_paths = {"SPS_HOME": sps_home} if sps_home else None
    save_npz_with_provenance(
        cache_path,
        provenance_paths=provenance_paths,
        seed=seed,
        command_args=command_args,
        extra={"validation": "fsps_sbi_mock_catalog", "metadata": catalog["metadata"]},
        theta_full_train=catalog["theta_full_train"],
        theta_full_test=catalog["theta_full_test"],
        theta_train=catalog["theta_train"],
        theta_test=catalog["theta_test"],
        x_flux_train=catalog["x_flux_train"],
        x_flux_test=catalog["x_flux_test"],
        x_train=catalog["x_train"],
        x_test=catalog["x_test"],
        metadata=np.array(metadata),
    )


def load_catalog(cache_path: Path) -> dict[str, np.ndarray | dict]:
    """Load a provenance-backed generated catalog."""

    require_provenance(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("schema") != CACHE_SCHEMA:
            raise ValueError(
                f"Cached catalog {cache_path} has schema {metadata.get('schema')!r}; "
                f"expected {CACHE_SCHEMA!r}."
            )
        return {
            "theta_full_train": np.asarray(data["theta_full_train"], dtype=float),
            "theta_full_test": np.asarray(data["theta_full_test"], dtype=float),
            "theta_train": np.asarray(data["theta_train"], dtype=float),
            "theta_test": np.asarray(data["theta_test"], dtype=float),
            "x_flux_train": np.asarray(data["x_flux_train"], dtype=float),
            "x_flux_test": np.asarray(data["x_flux_test"], dtype=float),
            "x_train": np.asarray(data["x_train"], dtype=float),
            "x_test": np.asarray(data["x_test"], dtype=float),
            "metadata": metadata,
        }


def train_diffusion_and_run_diagnostics(
    catalog: dict[str, np.ndarray | dict],
    *,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_samples: int,
    steps: int,
    seed: int,
) -> dict:
    """Train a joint diffusion sampler and diagnose ``theta | AB magnitudes``."""

    theta_train = np.asarray(catalog["theta_train"], dtype=float)
    x_train = np.asarray(catalog["x_train"], dtype=float)
    theta_test = np.asarray(catalog["theta_test"], dtype=float)
    x_test = np.asarray(catalog["x_test"], dtype=float)

    training_set = SBITrainingSet.from_arrays(
        theta_train,
        x_train,
        theta_names=INFERRED_THETA_NAMES,
        x_names=FILTER_NAMES,
        source="cached_composed_fsps_mock_catalog",
        metadata=catalog["metadata"],
    )
    t0 = time.perf_counter()
    trained = train_sbi(
        training_set,
        Diffusion(
            model="mlp",
            hidden_features=96,
            model_config={"num_blocks": 3, "emb_dim": 64, "time_hidden": 96},
            sigma_min=0.02,
            sigma_max=2.0,
            learning_rate=2e-3,
            device="auto",
            epochs=epochs,
            batch_size=batch_size,
        ),
        seed=seed,
    )
    training_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    joint_samples = trained.sample_joint(
        x_test,
        num_samples=num_samples,
        steps=steps,
        sampler="edm_euler",
        batch_size=min(64, x_test.shape[0]),
    )
    sampling_seconds = time.perf_counter() - t0
    n_bands = len(FILTER_NAMES)
    samples = joint_samples[:, :, n_bands:]
    known_clamp_error = float(np.max(np.abs(joint_samples[:, :, :n_bands] - x_test[:, None, :])))

    diagnostics = run_sbi_diagnostics(
        posterior_samples=samples,
        theta_true=theta_test,
        x_test=x_test,
        theta_names=INFERRED_THETA_NAMES,
        output_dir=output_dir / "diagnostics",
        make_plots=True,
    )

    summary = {
        "estimator": "ConditionalDiffusionEstimator",
        "device": str(trained.estimator.device),
        "final_train_loss": float(trained.history["train_loss"][-1]),
        "training_seconds": float(training_seconds),
        "sampling_seconds": float(sampling_seconds),
        "samples_per_second": float(theta_test.shape[0] * num_samples / max(sampling_seconds, 1e-12)),
        "known_clamp_max_abs_error": known_clamp_error,
        "n_train": int(theta_train.shape[0]),
        "n_test": int(theta_test.shape[0]),
        "num_samples": int(num_samples),
        "steps": int(steps),
        "epochs": int(epochs),
        "mean_coverage": {
            f"{level:.2f}": float(value)
            for level, value in zip(diagnostics["coverage"]["levels"], diagnostics["coverage"]["mean_coverage"])
        },
        "median_abs_error": {
            name: float(value)
            for name, value in zip(
                INFERRED_THETA_NAMES,
                np.median(np.abs(diagnostics["summary"]["median"] - theta_test), axis=0),
            )
        },
        "metadata": catalog["metadata"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    save_npz_with_provenance(
        output_dir / "posterior_samples.npz",
        provenance_paths={"mock_catalog": output_dir / "fsps_mock_catalog.npz"},
        seed=seed,
        command_args={
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "num_samples": int(num_samples),
            "steps": int(steps),
        },
        extra={"validation": "fsps_diffusion_sbi_mock_catalog_posterior_samples", "summary": summary},
        samples=samples,
        joint_samples=joint_samples,
        theta_test=theta_test,
        x_test=x_test,
        theta_names=np.array(INFERRED_THETA_NAMES),
        feature_names=np.array(FILTER_NAMES + INFERRED_THETA_NAMES),
    )
    return summary


def run_validation(
    *,
    output_dir: str | Path = "outputs/fsps_sbi_mock_catalog_validation",
    n_train: int = 48,
    n_test: int = 16,
    epochs: int = 50,
    batch_size: int = 128,
    num_samples: int = 256,
    steps: int = 32,
    seed: int = 20260706,
    regenerate: bool = False,
    max_retries: int = 100,
    simulation_batch_size: int = 1,
    n_workers: int = 1,
    executor: str = "serial",
    mp_context: str | None = None,
) -> dict:
    """Run the full FSPS mock-catalog SBI validation."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = output_path / "fsps_mock_catalog.npz"

    if cache_path.exists() and not regenerate:
        try:
            catalog = load_catalog(cache_path)
        except ValueError as exc:
            print(f"Cached catalog is stale ({exc}); regenerating.", flush=True)
            regenerate = True
    if not cache_path.exists() or regenerate:
        catalog = generate_fsps_mock_catalog(
            n_train=n_train,
            n_test=n_test,
            seed=seed,
            max_retries=max_retries,
            simulation_batch_size=simulation_batch_size,
            n_workers=n_workers,
            executor=executor,
            mp_context=mp_context,
        )
        save_catalog(
            cache_path,
            catalog,
            seed=seed,
            command_args={
                "n_train": int(n_train),
                "n_test": int(n_test),
                "seed": int(seed),
                "max_retries": int(max_retries),
                "simulation_batch_size": int(simulation_batch_size),
                "n_workers": int(n_workers),
                "executor": executor,
                "mp_context": mp_context,
            },
        )

    return train_diffusion_and_run_diagnostics(
        catalog,
        output_dir=output_path,
        epochs=epochs,
        batch_size=batch_size,
        num_samples=num_samples,
        steps=steps,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/fsps_sbi_mock_catalog_validation")
    parser.add_argument("--n-train", type=int, default=48)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--simulation-batch-size", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--executor", choices=["serial", "thread", "process"], default="serial")
    parser.add_argument("--mp-context", default=None)
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_validation(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        steps=args.steps,
        seed=args.seed,
        regenerate=args.regenerate,
        max_retries=args.max_retries,
        simulation_batch_size=args.simulation_batch_size,
        n_workers=args.n_workers,
        executor=args.executor,
        mp_context=args.mp_context,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
