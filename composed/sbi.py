"""Scientist-facing helpers for photometric SBI experiments.

The lower-level CompoSED pieces are deliberately explicit: datasets,
likelihoods, parameter spaces, backends, and neural estimators are separate
objects.  This module wires those pieces together for the common notebook
workflow:

1. choose filters;
2. choose a backend;
3. choose priors;
4. choose a noise model;
5. simulate a noised training set;
6. train a conditional diffusion or MAF model;
7. condition on catalog photometry and sample parameters;
8. run diagnostics.

The functions here do not add new physics. Problem-driven forward modelling,
masks, active bands, parameter mapping, and mass normalization go through
``Problem.simulate``. Pre-existing paired arrays are declared independently
with ``SBITrainingSet`` and never acquire a fictitious backend or prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from composed.data import SEDDataset
from composed.filters import FilterSet
from composed.parameters import ParameterSpace
from inftools.diagnostics import run_sbi_diagnostics
from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata
from inftools.sbi import MAFPosteriorEstimator, simulate_training_set, train_maf_posterior_from_dataset


ObservationTransform = str | Callable[[np.ndarray], np.ndarray]
PhotometryTransform = ObservationTransform


@dataclass
class SBITrainingSet:
    """Paired physical parameters and observations used to train SBI.

    This object has no backend or likelihood requirement.  It can describe
    simulations generated through a :class:`composed.Problem`, a presampled
    forward model, a numerical simulation, or an empirical labeled catalog.
    Rows of ``theta`` and ``x`` must refer to the same realization.

    ``x_native`` records the observation before the optional feature transform;
    for external feature tables it can simply equal ``x``. ``theta_full`` is
    useful for simulator-generated data where nuisance parameters were sampled
    but only a subset of parameters is used as neural labels.
    """

    theta: np.ndarray
    x: np.ndarray
    theta_names: Sequence[str]
    x_names: Sequence[str]
    source: str
    theta_full: np.ndarray | None = None
    full_parameter_names: Sequence[str] | None = None
    x_native: np.ndarray | None = None
    feature_transform: ObservationTransform = "features"
    observation_group: str = "observations"
    observation_groups: Mapping[str, Sequence[str]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        x = np.asarray(self.x, dtype=float)
        if theta.ndim != 2 or x.ndim != 2:
            raise ValueError("SBITrainingSet theta and x must both be two-dimensional arrays.")
        if theta.shape[0] != x.shape[0]:
            raise ValueError("SBITrainingSet theta and x must contain the same number of rows.")
        if theta.shape[0] == 0:
            raise ValueError("SBITrainingSet requires at least one training row.")
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(x)):
            raise ValueError("SBITrainingSet theta and x must be finite.")

        theta_names = tuple(str(name) for name in self.theta_names)
        x_names = tuple(str(name) for name in self.x_names)
        if len(theta_names) != theta.shape[1] or len(set(theta_names)) != len(theta_names):
            raise ValueError("theta_names must be unique and match theta columns.")
        if len(x_names) != x.shape[1] or len(set(x_names)) != len(x_names):
            raise ValueError("x_names must be unique and match x columns.")

        theta_full = theta if self.theta_full is None else np.asarray(self.theta_full, dtype=float)
        if theta_full.ndim != 2 or theta_full.shape[0] != theta.shape[0] or not np.all(np.isfinite(theta_full)):
            raise ValueError("theta_full must be finite with shape (n_training, n_full_parameter).")
        full_names = theta_names if self.full_parameter_names is None else tuple(
            str(name) for name in self.full_parameter_names
        )
        if len(full_names) != theta_full.shape[1] or len(set(full_names)) != len(full_names):
            raise ValueError("full_parameter_names must be unique and match theta_full columns.")

        x_native = x if self.x_native is None else np.asarray(self.x_native, dtype=float)
        if x_native.shape != x.shape or not np.all(np.isfinite(x_native)):
            raise ValueError("x_native must be finite and have the same shape as x.")
        source = str(self.source).strip()
        if not source:
            raise ValueError("SBITrainingSet source must describe where the paired data came from.")
        observation_group = str(self.observation_group).strip()
        if not observation_group:
            raise ValueError("observation_group must be non-empty.")
        if self.observation_groups is None:
            observation_groups = {observation_group: x_names}
        else:
            observation_groups = {
                str(group): tuple(str(name) for name in names)
                for group, names in self.observation_groups.items()
            }
            if "parameters" in observation_groups:
                raise ValueError("'parameters' is reserved for SBI target labels.")
            flattened = tuple(name for names in observation_groups.values() for name in names)
            if flattened != x_names:
                raise ValueError(
                    "observation_groups must list every x_name exactly once and in x column order."
                )

        self.theta = theta
        self.x = x
        self.theta_names = theta_names
        self.x_names = x_names
        self.source = source
        self.theta_full = theta_full
        self.full_parameter_names = full_names
        self.x_native = x_native
        self.observation_group = observation_group
        self.observation_groups = observation_groups
        self.metadata = dict(self.metadata)

    @classmethod
    def from_arrays(
        cls,
        theta: np.ndarray,
        x: np.ndarray,
        *,
        theta_names: Sequence[str],
        x_names: Sequence[str],
        source: str,
        metadata: Mapping[str, Any] | None = None,
        finite: str = "raise",
        observation_groups: Mapping[str, Sequence[str]] | None = None,
    ) -> "SBITrainingSet":
        """Build a standalone SBI dataset without declaring a CompoSED Problem."""

        theta = np.asarray(theta, dtype=float)
        x = np.asarray(x, dtype=float)
        if theta.ndim != 2 or x.ndim != 2 or theta.shape[0] != x.shape[0]:
            raise ValueError("theta and x must be paired two-dimensional arrays.")
        row_is_finite = np.all(np.isfinite(theta), axis=1) & np.all(np.isfinite(x), axis=1)
        if finite == "raise" and not np.all(row_is_finite):
            raise ValueError("Pre-existing SBI arrays contain NaN or inf rows; use finite='drop' explicitly.")
        if finite == "drop":
            theta = theta[row_is_finite]
            x = x[row_is_finite]
        elif finite != "raise":
            raise ValueError("finite must be 'raise' or 'drop'.")
        dataset_metadata = dict(metadata or {})
        dataset_metadata["input_rows"] = int(row_is_finite.size)
        dataset_metadata["dropped_nonfinite_rows"] = int(np.sum(~row_is_finite))

        return cls(
            theta=theta,
            x=x,
            theta_names=theta_names,
            x_names=x_names,
            source=source,
            feature_transform="features",
            observation_group="observations",
            observation_groups=observation_groups,
            metadata=dataset_metadata,
        )

    @property
    def feature_transform_name(self) -> str:
        """Human-readable name of the photometry feature transform."""

        return _transform_name(self.feature_transform)

    @property
    def feature_metadata(self) -> FeatureMetadata:
        return FeatureMetadata.from_groups({**self.observation_groups, "parameters": self.theta_names})

    @property
    def joint_features(self) -> np.ndarray:
        """Return ``[photometry_features, inferred_parameters]``."""

        return np.column_stack([self.x, self.theta])

    # Compatibility names retained for the first photometric SBI notebooks.
    @property
    def x_flux(self) -> np.ndarray:
        return self.x_native

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(self.x_names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.full_parameter_names)


PhotometricTrainingSet = SBITrainingSet


@dataclass(frozen=True)
class Simulate:
    """Training simulations drawn from the declared Problem prior and simulator."""

    n: int
    noise_fn: Callable[[np.ndarray], np.ndarray]
    infer: Sequence[str] | None = None
    feature_transform: ObservationTransform = "flux"
    max_retries: int = 100
    batch_size: int = 1
    n_workers: int = 1
    executor: str = "serial"
    mp_context: str | None = None

    def __post_init__(self) -> None:
        if int(self.n) <= 0:
            raise ValueError("Simulate.n must be positive.")
        if not callable(self.noise_fn):
            raise TypeError("Simulate.noise_fn must be callable.")


@dataclass(frozen=True)
class MAF:
    """Conditional MAF inference configuration."""

    hidden_features: int = 128
    num_transforms: int = 5
    num_blocks: int = 2
    learning_rate: float = 1e-3
    device: str | None = "auto"
    standardize: bool = True
    max_grad_norm: float | None = None
    restore_best: bool = True
    epochs: int = 100
    batch_size: int = 256
    validation_split: float = 0.0
    num_samples: int = 512
    verbose: bool = False


@dataclass(frozen=True)
class Diffusion:
    """Masked conditional diffusion inference configuration."""

    model: str = "mlp"
    hidden_features: int = 128
    model_config: Mapping[str, Any] | None = None
    sigma_min: float = 0.02
    sigma_max: float = 2.0
    learning_rate: float = 1e-3
    device: str | None = "auto"
    standardize: bool = True
    epochs: int = 100
    batch_size: int = 256
    validation_split: float = 0.0
    mask_config: Mapping[str, Any] | None = None
    num_samples: int = 512
    steps: int = 64
    sampler: str = "edm_euler"
    sample_batch_size: int | None = None
    verbose: bool = False


@dataclass
class TrainedDiffusionSBI:
    """Trained conditional diffusion model for parameters given observations."""

    estimator: ConditionalDiffusionEstimator
    training_set: PhotometricTrainingSet
    history: dict[str, list[float]]
    mask_config: Mapping[str, Any]

    @property
    def band_names(self) -> tuple[str, ...]:
        return self.training_set.band_names

    @property
    def theta_names(self) -> tuple[str, ...]:
        return self.training_set.theta_names

    def sample(
        self,
        photometry: np.ndarray,
        *,
        input_units: str = "features",
        num_samples: int = 512,
        steps: int = 64,
        sampler: str = "edm_euler",
        batch_size: int | None = None,
        **sampler_kwargs,
    ) -> np.ndarray:
        """Sample inferred parameters conditional on catalog photometry.

        Parameters
        ----------
        photometry:
            One object ``(n_bands,)`` or a catalog ``(n_objects, n_bands)``.
        input_units:
            ``"features"`` means the array is already in the training feature
            units.  ``"flux"`` means native active-band fluxes and applies the
            same transform used during training.
        """

        joint = self.sample_joint(
            photometry,
            input_units=input_units,
            num_samples=num_samples,
            steps=steps,
            sampler=sampler,
            batch_size=batch_size,
            **sampler_kwargs,
        )
        n_bands = len(self.band_names)
        return joint[:, :, n_bands:]

    def sample_joint(
        self,
        photometry: np.ndarray,
        *,
        input_units: str = "features",
        num_samples: int = 512,
        steps: int = 64,
        sampler: str = "edm_euler",
        batch_size: int | None = None,
        **sampler_kwargs,
    ) -> np.ndarray:
        """Sample full joint vectors while clamping observed photometry."""

        x = _as_2d(photometry, expected_cols=len(self.band_names), name="photometry")
        if input_units in {"flux", "native"}:
            x = transform_photometry(x, self.training_set.feature_transform)
        elif input_units != "features":
            raise ValueError("input_units must be 'features' or 'native'.")

        n_objects = x.shape[0]
        n_bands = len(self.band_names)
        n_theta = len(self.theta_names)
        known = np.full((n_objects, n_bands + n_theta), np.nan, dtype=float)
        known[:, :n_bands] = x
        mask = np.zeros_like(known, dtype=bool)
        mask[:, :n_bands] = True
        return self.estimator.sample(
            known,
            mask,
            num_samples=num_samples,
            steps=steps,
            sampler=sampler,
            batch_size=batch_size,
            **sampler_kwargs,
        )

    def diagnostics(
        self,
        samples: np.ndarray,
        theta_true: np.ndarray,
        *,
        x_test: np.ndarray | None = None,
        output_dir: str | Path | None = None,
        make_plots: bool = True,
    ) -> dict[str, Any]:
        """Run generic SBI diagnostics on parameter posterior samples."""

        return run_sbi_diagnostics(
            posterior_samples=samples,
            theta_true=theta_true,
            x_test=x_test,
            theta_names=self.theta_names,
            output_dir=output_dir,
            make_plots=make_plots,
        )


@dataclass
class TrainedMAFSBI:
    """Trained MAF posterior estimator for parameters given observations."""

    estimator: MAFPosteriorEstimator
    training_set: PhotometricTrainingSet
    history: dict[str, list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def band_names(self) -> tuple[str, ...]:
        return self.training_set.band_names

    @property
    def theta_names(self) -> tuple[str, ...]:
        return self.training_set.theta_names

    def sample(
        self,
        photometry: np.ndarray,
        *,
        input_units: str = "features",
        num_samples: int = 512,
    ) -> np.ndarray:
        """Sample inferred parameters conditional on catalog photometry."""

        x = _as_2d(photometry, expected_cols=len(self.band_names), name="photometry")
        if input_units in {"flux", "native"}:
            x = transform_photometry(x, self.training_set.feature_transform)
        elif input_units != "features":
            raise ValueError("input_units must be 'features' or 'native'.")
        samples = np.asarray(self.estimator.sample(x, num_samples=num_samples), dtype=float)
        if samples.ndim == 2 and x.shape[0] == 1:
            samples = samples[None, :, :]
        return samples

    def diagnostics(
        self,
        samples: np.ndarray,
        theta_true: np.ndarray,
        *,
        x_test: np.ndarray | None = None,
        output_dir: str | Path | None = None,
        make_plots: bool = True,
    ) -> dict[str, Any]:
        """Run generic SBI diagnostics on MAF posterior samples."""

        return run_sbi_diagnostics(
            posterior_samples=samples,
            theta_true=theta_true,
            x_test=x_test,
            theta_names=self.theta_names,
            output_dir=output_dir,
            make_plots=make_plots,
        )


# Compatibility names used by the first photometric SBI notebooks.
DiffusionPhotometricSBIResult = TrainedDiffusionSBI
MAFPhotometricSBIResult = TrainedMAFSBI


def simulate_sbi_training_set(
    problem,
    simulation: Simulate,
    *,
    rng: np.random.Generator | int | None = None,
) -> SBITrainingSet:
    """Generate SBI pairs from one declared CompoSED Problem.

    The observed feature vector is exactly the active photometric vector used
    by the problem likelihood.  Backend parameter mapping, flux units, masks,
    and mass normalization therefore follow the same path as deterministic
    inference.  Upper-limit encodings are not guessed: censored observations
    require a future explicit SBI observation encoder.
    """

    from composed.problem import Problem

    if not isinstance(problem, Problem):
        raise TypeError("simulate_sbi_training_set requires a composed.Problem.")
    if not isinstance(simulation, Simulate):
        raise TypeError("simulation must be composed.Simulate(...).")
    if not isinstance(problem.data, SEDDataset):
        raise NotImplementedError("Problem-driven SBI currently supports photometric SEDDataset observations.")
    if np.any(problem.data.active_upper_limit_mask):
        raise NotImplementedError(
            "Problem-driven SBI does not yet infer an upper-limit feature encoding. "
            "Use detections or construct an explicit standalone SBITrainingSet with censoring features."
        )

    generator = np.random.default_rng(rng)
    theta_full, x_native, sim_metadata = simulate_training_set(
        problem.parameters,
        problem,
        n=int(simulation.n),
        noise_fn=simulation.noise_fn,
        rng=generator,
        max_retries=int(simulation.max_retries),
        return_metadata=True,
        batch_size=int(simulation.batch_size),
        n_workers=int(simulation.n_workers),
        executor=str(simulation.executor),
        mp_context=simulation.mp_context,
    )
    inferred_names, theta = _select_inferred_parameters(
        theta_full,
        problem.parameters.names,
        simulation.infer,
    )
    x_features = transform_photometry(x_native, simulation.feature_transform)
    transform_name = _transform_name(simulation.feature_transform)
    return SBITrainingSet(
        theta=theta,
        x=x_features,
        theta_names=inferred_names,
        x_names=problem.data.active_band_names,
        source="composed.problem.simulate",
        theta_full=theta_full,
        full_parameter_names=problem.parameters.names,
        x_native=x_native,
        feature_transform=simulation.feature_transform,
        observation_group="photometry",
        metadata={
            "problem": problem.specification(),
            "simulator": "Problem.simulate",
            "active_band_names": problem.data.active_band_names,
            "flux_unit": problem.data.flux_unit,
            "feature_transform": transform_name,
            "simulate_training_set": sim_metadata,
        },
    )


def simulate_photometric_training_set(
    *,
    backend: object,
    filters: FilterSet | Sequence[object],
    parameter_space: ParameterSpace,
    noise_fn,
    n: int,
    infer: Sequence[str] | None = None,
    mask: Sequence[bool] | None = None,
    feature_transform: PhotometryTransform = "flux",
    rng: np.random.Generator | int | None = None,
    max_retries: int = 100,
    simulation_batch_size: int = 1,
    n_workers: int = 1,
    executor: str = "serial",
    mp_context: str | None = None,
) -> PhotometricTrainingSet:
    """Compatibility wrapper for the original backend-based SBI helper.

    New analyses should declare a :class:`composed.Problem` and call
    :func:`simulate_sbi_training_set`.  This wrapper constructs that Problem
    explicitly so legacy notebooks retain the same masks and mass scaling.
    """

    filter_set = _coerce_filter_set(filters)
    band_names = tuple(filter_set.names)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (len(filter_set),):
            raise ValueError(f"mask must have shape {(len(filter_set),)}; got {mask_arr.shape}.")
    else:
        mask_arr = None

    dataset = SEDDataset(
        band_names=band_names,
        flux=np.ones(len(filter_set), dtype=float),
        sigma=np.ones(len(filter_set), dtype=float),
        mask=mask_arr,
        metadata={"filters": filter_set},
    )
    from composed.problem import Gaussian, Problem

    problem = Problem(
        backend=backend,
        parameters=parameter_space,
        data=dataset,
        likelihood=Gaussian(),
        filters=filter_set,
        metadata={"compatibility_helper": "simulate_photometric_training_set"},
    )
    return simulate_sbi_training_set(
        problem,
        Simulate(
            n=int(n),
            noise_fn=noise_fn,
            infer=infer,
            feature_transform=feature_transform,
            max_retries=max_retries,
            batch_size=simulation_batch_size,
            n_workers=n_workers,
            executor=executor,
            mp_context=mp_context,
        ),
        rng=rng,
    )


def train_diffusion_photometric_sbi(
    *,
    backend: object,
    filters: FilterSet | Sequence[object],
    parameter_space: ParameterSpace,
    noise_fn,
    n_train: int,
    infer: Sequence[str] | None = None,
    mask: Sequence[bool] | None = None,
    feature_transform: PhotometryTransform = "flux",
    rng: np.random.Generator | int | None = None,
    max_retries: int = 100,
    simulation_batch_size: int = 1,
    n_workers: int = 1,
    executor: str = "serial",
    mp_context: str | None = None,
    model: str = "mlp",
    hidden_features: int = 128,
    model_config: Mapping[str, Any] | None = None,
    sigma_min: float = 0.02,
    sigma_max: float = 2.0,
    learning_rate: float = 1e-3,
    device: str | None = "auto",
    epochs: int = 100,
    batch_size: int = 256,
    mask_config: Mapping[str, Any] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> TrainedDiffusionSBI:
    """Compatibility helper that simulates a photometric diffusion dataset."""

    training_set = simulate_photometric_training_set(
        backend=backend,
        filters=filters,
        parameter_space=parameter_space,
        noise_fn=noise_fn,
        n=n_train,
        infer=infer,
        mask=mask,
        feature_transform=feature_transform,
        rng=rng,
        max_retries=max_retries,
        simulation_batch_size=simulation_batch_size,
        n_workers=n_workers,
        executor=executor,
        mp_context=mp_context,
    )
    return train_sbi(
        training_set,
        Diffusion(
            model=model,
            hidden_features=hidden_features,
            model_config=model_config,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            learning_rate=learning_rate,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            mask_config=mask_config,
            verbose=verbose,
        ),
        seed=seed,
    )


def train_maf_photometric_sbi(
    training_set: SBITrainingSet,
    *,
    hidden_features: int = 128,
    num_transforms: int = 5,
    num_blocks: int = 2,
    learning_rate: float = 1e-3,
    device: str | None = "auto",
    standardize: bool = True,
    max_grad_norm: float | None = None,
    restore_best: bool = True,
    epochs: int = 100,
    batch_size: int = 256,
    validation_split: float = 0.0,
    seed: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> TrainedMAFSBI:
    """Compatibility wrapper around :func:`train_sbi` for a MAF."""

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported MAF training option(s): {unknown}.")
    return train_sbi(
        training_set,
        MAF(
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_blocks=num_blocks,
            learning_rate=learning_rate,
            device=device,
            standardize=standardize,
            max_grad_norm=max_grad_norm,
            restore_best=restore_best,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        ),
        seed=seed,
    )


def train_sbi(
    training_set: SBITrainingSet,
    method: MAF | Diffusion,
    *,
    seed: int | None = None,
) -> TrainedMAFSBI | TrainedDiffusionSBI:
    """Train SBI from a declared paired dataset.

    This is the standalone route for presampled forward models, simulations,
    and empirical labeled catalogs.  It deliberately takes no ``Problem``:
    the supplied rows, feature names, source, and metadata are the complete
    training-data declaration.
    """

    if not isinstance(training_set, SBITrainingSet):
        raise TypeError("train_sbi requires an SBITrainingSet.")

    if isinstance(method, MAF):
        return _train_maf(training_set, method, seed=seed)
    if isinstance(method, Diffusion):
        return _train_diffusion(training_set, method, seed=seed)
    raise TypeError("method must be composed.MAF(...) or composed.Diffusion(...).")


def fit_sbi_problem(
    problem,
    method: MAF | Diffusion,
    simulation: Simulate,
    *,
    seed: int | None = None,
):
    """Train from a Problem simulator and infer that Problem's observed SED."""

    from composed.problem import Problem
    from composed.results import InferenceResult

    if not isinstance(problem, Problem):
        raise TypeError("fit_sbi_problem requires a composed.Problem.")
    if not isinstance(method, (MAF, Diffusion)):
        raise TypeError("method must be composed.MAF(...) or composed.Diffusion(...).")
    if not isinstance(simulation, Simulate):
        raise TypeError(
            "Problem-based SBI requires training=Simulate(...). "
            "For existing paired arrays use train_sbi(SBITrainingSet.from_arrays(...), method)."
        )

    training_set = simulate_sbi_training_set(problem, simulation, rng=seed)
    _validate_continuous_sbi_targets(problem.parameters, training_set.theta_names)
    trained = train_sbi(training_set, method, seed=seed)
    samples = _sample_problem_posterior(problem, trained, method)

    return InferenceResult(
        samples=samples,
        logp=None,
        weights=np.ones(samples.shape[0], dtype=float),
        parameter_names=training_set.theta_names,
        sampler_name="maf" if isinstance(method, MAF) else "diffusion",
        metadata={
            "problem": problem.specification(),
            "training_source": training_set.source,
            "training_rows": int(training_set.theta.shape[0]),
            "observation_names": training_set.x_names,
            "feature_transform": training_set.feature_transform_name,
            "history": trained.history,
            "logp_available": False,
            "map_available": False,
            "seed": seed,
        },
        inference_state=trained,
    )


def _train_maf(training_set: SBITrainingSet, method: MAF, *, seed: int | None) -> TrainedMAFSBI:
    estimator, metadata = train_maf_posterior_from_dataset(
        training_set.theta,
        training_set.x,
        theta_names=training_set.theta_names,
        x_names=training_set.x_names,
        source=training_set.source,
        finite="raise",
        shuffle=False,
        return_metadata=True,
        hidden_features=method.hidden_features,
        num_transforms=method.num_transforms,
        num_blocks=method.num_blocks,
        learning_rate=method.learning_rate,
        device=method.device,
        standardize=method.standardize,
        max_grad_norm=method.max_grad_norm,
        restore_best=method.restore_best,
        epochs=method.epochs,
        batch_size=method.batch_size,
        validation_split=method.validation_split,
        seed=seed,
        verbose=method.verbose,
    )
    history = dict(getattr(estimator, "history", {}))
    return TrainedMAFSBI(
        estimator=estimator,
        training_set=training_set,
        history=history,
        metadata={**metadata, "training_source": training_set.source},
    )


def _train_diffusion(
    training_set: SBITrainingSet,
    method: Diffusion,
    *,
    seed: int | None,
) -> TrainedDiffusionSBI:
    estimator = ConditionalDiffusionEstimator(
        training_set.feature_metadata,
        model=method.model,
        hidden_features=method.hidden_features,
        model_config=method.model_config,
        sigma_min=method.sigma_min,
        sigma_max=method.sigma_max,
        learning_rate=method.learning_rate,
        device=method.device,
        standardize=method.standardize,
    )
    fit_mask_config = dict(
        method.mask_config
        or default_diffusion_mask(observation_groups=tuple(training_set.observation_groups))
    )
    history = estimator.fit(
        training_set.joint_features,
        mask_config=fit_mask_config,
        epochs=method.epochs,
        batch_size=method.batch_size,
        validation_split=method.validation_split,
        seed=seed,
        clamp_known_in_xt=True,
        loss_on_unknown_only=True,
        verbose=method.verbose,
    )
    return TrainedDiffusionSBI(
        estimator=estimator,
        training_set=training_set,
        history=history,
        mask_config=fit_mask_config,
    )


def _sample_problem_posterior(problem, trained, method: MAF | Diffusion) -> np.ndarray:
    observed = np.asarray(problem.data.active_flux, dtype=float)
    requested = int(method.num_samples)
    if requested <= 0:
        raise ValueError("SBI num_samples must be positive.")

    accepted = []
    n_accepted = 0
    for _ in range(12):
        draw_n = max(requested - n_accepted, min(requested, 256))
        if isinstance(method, MAF):
            cube = trained.sample(observed, input_units="native", num_samples=draw_n)
        else:
            cube = trained.sample(
                observed,
                input_units="native",
                num_samples=draw_n,
                steps=method.steps,
                sampler=method.sampler,
                batch_size=method.sample_batch_size,
            )
        draws = np.asarray(cube[0], dtype=float)
        valid = _samples_within_declared_priors(draws, problem.parameters, trained.theta_names)
        if np.any(valid):
            accepted.append(draws[valid])
            n_accepted += int(np.sum(valid))
        if n_accepted >= requested:
            return np.concatenate(accepted, axis=0)[:requested]
    raise RuntimeError(
        f"SBI generated only {n_accepted}/{requested} samples inside the declared prior support. "
        "Inspect training coverage and neural diagnostics."
    )


def _validate_continuous_sbi_targets(parameter_space: ParameterSpace, theta_names: Sequence[str]) -> None:
    unsupported = []
    for name in theta_names:
        prior_name = type(parameter_space.priors[name]).__name__
        if prior_name in {"DeltaPrior", "ChoicePrior", "IntegerUniformPrior"}:
            unsupported.append(f"{name} ({prior_name})")
    if unsupported:
        raise ValueError(
            "MAF and diffusion currently require continuous inferred parameters; unsupported: "
            + ", ".join(unsupported)
        )


def _samples_within_declared_priors(
    samples: np.ndarray,
    parameter_space: ParameterSpace,
    theta_names: Sequence[str],
) -> np.ndarray:
    valid = np.all(np.isfinite(samples), axis=1)
    for j, name in enumerate(theta_names):
        prior = parameter_space.priors[name]
        valid &= np.asarray([np.isfinite(prior.logpdf(float(value))) for value in samples[:, j]], dtype=bool)
    return valid


def default_photometric_diffusion_mask() -> dict[str, Any]:
    """Mask curriculum for learning ``parameters | photometry`` and the joint."""

    return default_diffusion_mask(observation_group="photometry")


def default_diffusion_mask(
    *,
    observation_group: str = "observations",
    observation_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Mask curriculum for inverse, forward, and mixed joint conditionals."""

    groups = (str(observation_group),) if observation_groups is None else tuple(
        str(group) for group in observation_groups
    )
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("observation groups must be non-empty and unique.")
    inverse = {group: 0.0 for group in groups}
    forward = {group: 1.0 for group in groups}
    mixed = {group: [0.2, 0.6] for group in groups}
    inverse["parameters"] = 1.0
    forward["parameters"] = 0.0
    mixed["parameters"] = [0.5, 1.0]

    return {
        "curriculum": [
            {"weight": 3.0, "unknown_fraction": inverse},
            {"weight": 1.0, "unknown_fraction": forward},
            {"weight": 1.0, "unknown_fraction": mixed},
        ]
    }


def transform_photometry(x_flux: np.ndarray, transform: PhotometryTransform = "flux") -> np.ndarray:
    """Convert native flux-like simulator output into neural features."""

    x_flux = np.asarray(x_flux, dtype=float)
    if callable(transform):
        out = np.asarray(transform(x_flux), dtype=float)
    elif transform in {"flux", "features", "identity"}:
        out = x_flux
    elif transform == "abmag":
        if np.any(x_flux <= 0.0):
            raise ValueError("AB magnitude features require strictly positive fluxes.")
        out = -2.5 * np.log10(x_flux)
    elif transform == "log10_flux":
        if np.any(x_flux <= 0.0):
            raise ValueError("log10_flux features require strictly positive fluxes.")
        out = np.log10(x_flux)
    else:
        raise ValueError(
            "feature_transform must be 'features', 'flux', 'abmag', 'log10_flux', or a callable."
        )

    if out.shape != x_flux.shape:
        raise ValueError(f"Feature transform changed shape from {x_flux.shape} to {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError("Photometry features contain NaN or inf values.")
    return out


def _coerce_filter_set(filters: FilterSet | Sequence[object]) -> FilterSet:
    if isinstance(filters, FilterSet):
        return filters
    return FilterSet(filters)


def _transform_name(transform: PhotometryTransform) -> str:
    if callable(transform):
        return getattr(transform, "__name__", "callable")
    return str(transform)


def _select_inferred_parameters(
    theta_full: np.ndarray,
    parameter_names: Sequence[str],
    infer: Sequence[str] | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    parameter_names = tuple(str(name) for name in parameter_names)
    inferred_names = parameter_names if infer is None else tuple(str(name) for name in infer)
    if len(set(inferred_names)) != len(inferred_names):
        raise ValueError("infer parameter names must be unique.")
    name_to_index = {name: i for i, name in enumerate(parameter_names)}
    missing = [name for name in inferred_names if name not in name_to_index]
    if missing:
        raise ValueError(f"infer contains unknown parameter(s): {', '.join(missing)}")
    indices = [name_to_index[name] for name in inferred_names]
    return inferred_names, np.asarray(theta_full, dtype=float)[:, indices]


def _as_2d(values: np.ndarray, *, expected_cols: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != int(expected_cols):
        raise ValueError(f"{name} must have shape ({expected_cols},) or (n, {expected_cols}); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return arr
