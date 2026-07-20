"""Scientist-facing helpers for photometric SBI experiments.

The lower-level CompoSED pieces are deliberately explicit: datasets,
likelihoods, parameter spaces, backends, and neural estimators are separate
objects.  This module wires those pieces together for the common notebook
workflow:

1. choose filters;
2. choose a backend;
3. choose priors;
4. choose a noise model;
5. simulate a noised training set and retain its exact uncertainty;
6. train the stable conditional MAF posterior estimator;
7. condition on catalog photometry and sample parameters;
8. run diagnostics.

The functions here do not add new physics. Problem-driven forward modelling,
masks, active bands, parameter mapping, and mass normalization go through
``Problem.simulate``. Pre-existing paired arrays are declared independently
with ``SBITrainingSet`` and never acquire a fictitious backend or prior.
Conditional diffusion remains available below as an experimental compatibility
path, but it is not exported by the stable :mod:`composed` API.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from composed.data import SEDDataset
from composed.filters import FilterSet
from composed.parameters import ParameterSpace
from composed.priors import LogUniformPrior, NormalPrior, StudentTPrior, UniformPrior
from composed.provenance import save_npz_with_provenance
from inftools.diagnostics import run_sbi_diagnostics
from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata
from inftools.sbi import (
    MAFPosteriorEstimator,
    Standardizer,
    simulate_training_set,
    train_maf_posterior_from_dataset,
)


ObservationTransform = str | Callable[[np.ndarray], np.ndarray]
PhotometryTransform = ObservationTransform


@dataclass(frozen=True)
class PhotometricContext:
    """Ordered photometric inputs supplied to a neural posterior estimator.

    The production default is ``snr_logsigma``. For every active band it stores
    the measured signal-to-noise ratio followed by
    ``log10(sigma / reference_flux)``. Together those two channels retain the
    measured flux and its uncertainty, accept negative noisy fluxes, and avoid
    the singularity of AB magnitudes at non-positive flux.
    """

    mode: str = "snr_logsigma"
    reference_flux: float = 1.0
    flux_unit: str | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        aliases = {"snr+logsigma": "snr_logsigma", "flux+sigma": "flux_sigma"}
        mode = aliases.get(mode, mode)
        if mode not in {"snr_logsigma", "flux_sigma", "abmag_magerr", "flux"}:
            raise ValueError(
                "PhotometricContext.mode must be 'snr_logsigma', 'flux_sigma', "
                "'abmag_magerr', or 'flux'."
            )
        reference = float(self.reference_flux)
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("PhotometricContext.reference_flux must be positive and finite.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "reference_flux", reference)
        if self.flux_unit is not None:
            object.__setattr__(self, "flux_unit", str(self.flux_unit))

    @property
    def conditions_on_sigma(self) -> bool:
        return self.mode != "flux"

    def encode(self, flux: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Encode matching flux and sigma arrays without changing row order."""

        flux = np.asarray(flux, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        if flux.shape != sigma.shape or flux.ndim not in {1, 2}:
            raise ValueError("Photometric context flux and sigma must have matching 1D or 2D shapes.")
        if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(sigma)):
            raise ValueError("Photometric context flux and sigma must be finite.")
        if np.any(sigma < 0.0):
            raise ValueError("Photometric context sigma must be non-negative.")

        if self.mode == "flux":
            return flux.copy()
        if np.any(sigma <= 0.0):
            raise ValueError(f"Photometric context mode {self.mode!r} requires strictly positive sigma.")
        if self.mode == "snr_logsigma":
            return np.concatenate(
                [flux / sigma, np.log10(sigma / self.reference_flux)],
                axis=-1,
            )
        if self.mode == "flux_sigma":
            return np.concatenate([flux, sigma], axis=-1)
        if np.any(flux <= 0.0):
            raise ValueError("abmag_magerr context requires strictly positive measured flux.")
        magnitude = -2.5 * np.log10(flux / self.reference_flux)
        magnitude_error = (2.5 / np.log(10.0)) * sigma / flux
        return np.concatenate([magnitude, magnitude_error], axis=-1)

    def feature_names(self, band_names: Sequence[str]) -> tuple[str, ...]:
        bands = tuple(str(name) for name in band_names)
        if self.mode == "flux":
            return bands
        if self.mode == "snr_logsigma":
            return tuple(f"snr:{name}" for name in bands) + tuple(f"log10_sigma:{name}" for name in bands)
        if self.mode == "flux_sigma":
            return tuple(f"flux:{name}" for name in bands) + tuple(f"sigma:{name}" for name in bands)
        return tuple(f"abmag:{name}" for name in bands) + tuple(f"magerr:{name}" for name in bands)

    def observation_groups(self, band_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
        names = self.feature_names(band_names)
        n_band = len(tuple(band_names))
        if self.mode == "flux":
            return {"photometry": names}
        return {"photometry": names[:n_band], "uncertainty": names[n_band:]}

    def specification(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "reference_flux": self.reference_flux,
            "flux_unit": self.flux_unit,
            "conditions_on_sigma": self.conditions_on_sigma,
        }

    @classmethod
    def from_specification(cls, specification: Mapping[str, object]) -> "PhotometricContext":
        return cls(
            mode=str(specification["mode"]),
            reference_flux=float(specification.get("reference_flux", 1.0)),
            flux_unit=specification.get("flux_unit"),
        )


@dataclass(frozen=True)
class PriorSupportTransform:
    """Map continuous parameters to an unconstrained neural target space."""

    names: Sequence[str]
    kinds: Sequence[str]
    lower: Sequence[float]
    upper: Sequence[float]
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        kinds = tuple(str(kind) for kind in self.kinds)
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        if not (len(names) == len(kinds) == len(lower) == len(upper)):
            raise ValueError("PriorSupportTransform fields must have equal lengths.")
        if len(set(names)) != len(names):
            raise ValueError("PriorSupportTransform names must be unique.")
        if any(kind not in {"identity", "uniform", "log_uniform"} for kind in kinds):
            raise ValueError("Unknown prior-support transform kind.")
        for kind, low, high in zip(kinds, lower, upper):
            if kind != "identity" and (not np.isfinite(low) or not np.isfinite(high) or high <= low):
                raise ValueError("Bounded prior transforms require finite high > low.")
            if kind == "log_uniform" and low <= 0.0:
                raise ValueError("Log-uniform prior transforms require a positive lower bound.")
        epsilon = float(self.epsilon)
        if not 0.0 < epsilon < 0.5:
            raise ValueError("PriorSupportTransform.epsilon must lie in (0, 0.5).")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "epsilon", epsilon)

    @classmethod
    def identity(cls, names: Sequence[str]) -> "PriorSupportTransform":
        names = tuple(str(name) for name in names)
        nan = (float("nan"),) * len(names)
        return cls(names, ("identity",) * len(names), nan, nan)

    @classmethod
    def from_parameter_space(
        cls,
        parameter_space: ParameterSpace,
        names: Sequence[str],
    ) -> "PriorSupportTransform":
        kinds = []
        lower = []
        upper = []
        for name in tuple(str(value) for value in names):
            if name not in parameter_space.priors:
                raise KeyError(f"ParameterSpace has no prior for inferred parameter {name!r}.")
            prior = parameter_space.priors[name]
            if isinstance(prior, UniformPrior):
                kinds.append("uniform")
                lower.append(float(prior.low))
                upper.append(float(prior.high))
            elif isinstance(prior, LogUniformPrior):
                kinds.append("log_uniform")
                lower.append(float(prior.low))
                upper.append(float(prior.high))
            elif isinstance(prior, (NormalPrior, StudentTPrior)):
                kinds.append("identity")
                lower.append(float("nan"))
                upper.append(float("nan"))
            else:
                raise TypeError(
                    f"MAF target {name!r} uses unsupported prior {type(prior).__name__}; "
                    "stable MAF supports UniformPrior, LogUniformPrior, NormalPrior, and StudentTPrior."
                )
        return cls(tuple(names), tuple(kinds), tuple(lower), tuple(upper))

    def transform(self, theta: np.ndarray) -> np.ndarray:
        values, leading_shape = self._coerce(theta)
        out = values.copy()
        for j, kind in enumerate(self.kinds):
            if kind == "identity":
                continue
            if kind == "log_uniform":
                if np.any(values[:, j] <= 0.0):
                    raise ValueError(f"Parameter {self.names[j]!r} must be positive for a log-uniform transform.")
                coordinate = (np.log(values[:, j]) - np.log(self.lower[j])) / (
                    np.log(self.upper[j]) - np.log(self.lower[j])
                )
            else:
                coordinate = (values[:, j] - self.lower[j]) / (self.upper[j] - self.lower[j])
            if np.any(coordinate < -self.epsilon) or np.any(coordinate > 1.0 + self.epsilon):
                raise ValueError(f"Parameter {self.names[j]!r} lies outside its declared prior support.")
            coordinate = np.clip(coordinate, self.epsilon, 1.0 - self.epsilon)
            out[:, j] = np.log(coordinate) - np.log1p(-coordinate)
        return out.reshape((*leading_shape, len(self.names)))

    def inverse(self, unconstrained: np.ndarray) -> np.ndarray:
        values, leading_shape = self._coerce(unconstrained)
        out = values.copy()
        for j, kind in enumerate(self.kinds):
            if kind == "identity":
                continue
            coordinate = _stable_sigmoid(values[:, j])
            if kind == "log_uniform":
                log_value = np.log(self.lower[j]) + coordinate * (
                    np.log(self.upper[j]) - np.log(self.lower[j])
                )
                out[:, j] = np.clip(np.exp(log_value), self.lower[j], self.upper[j])
            else:
                out[:, j] = np.clip(
                    self.lower[j] + coordinate * (self.upper[j] - self.lower[j]),
                    self.lower[j],
                    self.upper[j],
                )
        return out.reshape((*leading_shape, len(self.names)))

    def log_abs_det_forward(self, theta: np.ndarray) -> np.ndarray | float:
        values, leading_shape = self._coerce(theta)
        total = np.zeros(values.shape[0], dtype=float)
        for j, kind in enumerate(self.kinds):
            if kind == "identity":
                continue
            if kind == "log_uniform":
                coordinate = (np.log(values[:, j]) - np.log(self.lower[j])) / (
                    np.log(self.upper[j]) - np.log(self.lower[j])
                )
                coordinate = np.clip(coordinate, self.epsilon, 1.0 - self.epsilon)
                total += (
                    -np.log(values[:, j])
                    - np.log(np.log(self.upper[j]) - np.log(self.lower[j]))
                    - np.log(coordinate)
                    - np.log1p(-coordinate)
                )
            else:
                coordinate = (values[:, j] - self.lower[j]) / (self.upper[j] - self.lower[j])
                coordinate = np.clip(coordinate, self.epsilon, 1.0 - self.epsilon)
                total += (
                    -np.log(self.upper[j] - self.lower[j])
                    - np.log(coordinate)
                    - np.log1p(-coordinate)
                )
        reshaped = total.reshape(leading_shape)
        return float(reshaped) if leading_shape == () else reshaped

    def specification(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "kinds": list(self.kinds),
            "lower": [value if np.isfinite(value) else None for value in self.lower],
            "upper": [value if np.isfinite(value) else None for value in self.upper],
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_specification(cls, specification: Mapping[str, object]) -> "PriorSupportTransform":
        return cls(
            names=specification["names"],
            kinds=specification["kinds"],
            lower=[float("nan") if value is None else value for value in specification["lower"]],
            upper=[float("nan") if value is None else value for value in specification["upper"]],
            epsilon=float(specification.get("epsilon", 1.0e-6)),
        )

    def _coerce(self, values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        array = np.asarray(values, dtype=float)
        if array.ndim < 1 or array.shape[-1] != len(self.names):
            raise ValueError(f"Expected parameter array with final dimension {len(self.names)}; got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Parameter transform received NaN or inf values.")
        leading_shape = array.shape[:-1]
        return array.reshape((-1, len(self.names))), leading_shape


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0))),
        np.exp(np.clip(values, -700.0, 700.0)) / (1.0 + np.exp(np.clip(values, -700.0, 700.0))),
    )


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
    sigma_native: np.ndarray | None = None
    native_names: Sequence[str] | None = None
    feature_transform: ObservationTransform = "features"
    context: PhotometricContext | None = None
    theta_transform: PriorSupportTransform | None = None
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
        if x_native.ndim != 2 or x_native.shape[0] != x.shape[0] or not np.all(np.isfinite(x_native)):
            raise ValueError("x_native must be finite and have the same number of rows as x.")
        native_names = x_names if self.native_names is None else tuple(str(name) for name in self.native_names)
        if len(native_names) != x_native.shape[1] or len(set(native_names)) != len(native_names):
            raise ValueError("native_names must be unique and match x_native columns.")
        sigma_native = None if self.sigma_native is None else np.asarray(self.sigma_native, dtype=float)
        if sigma_native is not None:
            if sigma_native.shape != x_native.shape:
                raise ValueError("sigma_native must have the same shape as x_native.")
            if not np.all(np.isfinite(sigma_native)) or np.any(sigma_native < 0.0):
                raise ValueError("sigma_native must be finite and non-negative.")
        if self.context is not None and not isinstance(self.context, PhotometricContext):
            raise TypeError("context must be a PhotometricContext or None.")
        theta_transform = self.theta_transform or PriorSupportTransform.identity(theta_names)
        if tuple(theta_transform.names) != theta_names:
            raise ValueError("theta_transform names must exactly match theta_names order.")
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
        self.sigma_native = sigma_native
        self.native_names = native_names
        self.theta_transform = theta_transform
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
        parameter_space: ParameterSpace | None = None,
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

        theta_transform = (
            PriorSupportTransform.identity(tuple(theta_names))
            if parameter_space is None
            else PriorSupportTransform.from_parameter_space(parameter_space, theta_names)
        )
        return cls(
            theta=theta,
            x=x,
            theta_names=theta_names,
            x_names=x_names,
            source=source,
            feature_transform="features",
            native_names=x_names,
            theta_transform=theta_transform,
            observation_group="observations",
            observation_groups=observation_groups,
            metadata=dataset_metadata,
        )

    @classmethod
    def from_photometry(
        cls,
        theta: np.ndarray,
        flux: np.ndarray,
        sigma: np.ndarray,
        *,
        theta_names: Sequence[str],
        band_names: Sequence[str],
        source: str,
        context: PhotometricContext | str = "snr_logsigma",
        flux_unit: str = "maggies",
        parameter_space: ParameterSpace | None = None,
        metadata: Mapping[str, Any] | None = None,
        finite: str = "raise",
    ) -> "SBITrainingSet":
        """Build paired SBI data from measured fluxes and their uncertainties."""

        theta = np.asarray(theta, dtype=float)
        flux = np.asarray(flux, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        if theta.ndim != 2 or flux.ndim != 2 or sigma.shape != flux.shape:
            raise ValueError("theta, flux, and sigma must be paired 2D arrays; flux and sigma shapes must match.")
        if theta.shape[0] != flux.shape[0]:
            raise ValueError("theta, flux, and sigma must contain the same number of rows.")
        bands = tuple(str(name) for name in band_names)
        if len(bands) != flux.shape[1] or len(set(bands)) != len(bands):
            raise ValueError("band_names must be unique and match the photometry columns.")
        theta_names = tuple(str(name) for name in theta_names)
        if len(theta_names) != theta.shape[1] or len(set(theta_names)) != len(theta_names):
            raise ValueError("theta_names must be unique and match theta columns.")
        row_is_finite = (
            np.all(np.isfinite(theta), axis=1)
            & np.all(np.isfinite(flux), axis=1)
            & np.all(np.isfinite(sigma), axis=1)
        )
        if finite == "raise" and not np.all(row_is_finite):
            raise ValueError("Pre-existing SBI photometry contains NaN or inf rows; use finite='drop' explicitly.")
        if finite == "drop":
            theta = theta[row_is_finite]
            flux = flux[row_is_finite]
            sigma = sigma[row_is_finite]
        elif finite != "raise":
            raise ValueError("finite must be 'raise' or 'drop'.")
        if theta.shape[0] == 0:
            raise ValueError("SBITrainingSet requires at least one finite photometric row.")

        context_encoder = _coerce_photometric_context(context, flux_unit=flux_unit)
        features = context_encoder.encode(flux, sigma)
        feature_names = context_encoder.feature_names(bands)
        theta_transform = (
            PriorSupportTransform.identity(theta_names)
            if parameter_space is None
            else PriorSupportTransform.from_parameter_space(parameter_space, theta_names)
        )
        dataset_metadata = dict(metadata or {})
        dataset_metadata.update(
            {
                "input_rows": int(row_is_finite.size),
                "dropped_nonfinite_rows": int(np.sum(~row_is_finite)),
                "active_band_names": bands,
                "flux_unit": str(flux_unit),
                "photometric_context": context_encoder.specification(),
            }
        )
        return cls(
            theta=theta,
            x=features,
            theta_names=theta_names,
            x_names=feature_names,
            source=source,
            theta_full=theta,
            full_parameter_names=theta_names,
            x_native=flux,
            sigma_native=sigma,
            native_names=bands,
            feature_transform="features",
            context=context_encoder,
            theta_transform=theta_transform,
            observation_group="photometry",
            observation_groups=context_encoder.observation_groups(bands),
            metadata=dataset_metadata,
        )

    @property
    def feature_transform_name(self) -> str:
        """Human-readable name of the photometry feature transform."""

        if self.context is not None and self.context.mode != "flux":
            return self.context.mode
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
    def x_sigma(self) -> np.ndarray | None:
        return self.sigma_native

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(self.native_names)

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
    context: PhotometricContext | str = "snr_logsigma"
    feature_transform: ObservationTransform | None = None
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
        context_mode = self.context.mode if isinstance(self.context, PhotometricContext) else str(self.context)
        if self.feature_transform is not None and str(context_mode).lower() != "flux":
            raise ValueError(
                "feature_transform is a legacy flux-only option. Set context='flux' explicitly, "
                "or use the stable uncertainty-conditioned context without feature_transform."
            )


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
    validation_split: float = 0.1
    patience: int | None = 20
    min_delta: float = 0.0
    num_samples: int = 512
    inference_batch_size: int | None = 8192
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
    training_set: PhotometricTrainingSet | None
    history: dict[str, list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)
    target_transform: PriorSupportTransform | None = None
    schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_transform is None:
            if self.training_set is None:
                raise ValueError("Loaded MAF state requires an explicit target_transform.")
            self.target_transform = self.training_set.theta_transform
        if self.training_set is not None:
            generated_schema = _maf_schema_from_training_set(self.training_set)
            if self.schema and dict(self.schema) != generated_schema:
                raise ValueError("Provided MAF schema does not match the training set.")
            self.schema = generated_schema
        else:
            self.schema = dict(self.schema)
            _validate_maf_schema(self.schema)
        if self.estimator.theta_dim != len(self.theta_names) or self.estimator.x_dim != len(self.x_names):
            raise ValueError("MAF estimator dimensions do not match the saved scientific schema.")
        if tuple(self.target_transform.names) != self.theta_names:
            raise ValueError("MAF target transform names do not match the saved parameter order.")
        self.metadata = dict(self.metadata)
        self.history = {str(key): list(value) for key, value in self.history.items()}

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(self.schema["band_names"])

    @property
    def theta_names(self) -> tuple[str, ...]:
        return tuple(self.schema["theta_names"])

    @property
    def x_names(self) -> tuple[str, ...]:
        return tuple(self.schema["x_names"])

    @property
    def context(self) -> PhotometricContext | None:
        specification = self.schema.get("photometric_context")
        return None if specification is None else PhotometricContext.from_specification(specification)

    @property
    def feature_transform(self) -> ObservationTransform:
        if self.training_set is not None:
            return self.training_set.feature_transform
        return str(self.schema.get("feature_transform", "features"))

    def sample(
        self,
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None = None,
        input_units: str = "features",
        num_samples: int = 512,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample physical parameters from one SED or a photometric catalog.

        Passing an :class:`SEDDataset` is the least ambiguous route: active
        fluxes, sigmas, masks, units, and band order are checked against the
        training schema. Array input in native units requires ``sigma=`` when
        the trained context conditions on uncertainty.
        """

        x = self._context_features(photometry, sigma=sigma, input_units=input_units)
        n_object = x.shape[0]
        if batch_size is None:
            batch_size = n_object
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("MAF inference batch_size must be positive.")
        if seed is not None:
            self.estimator.torch.manual_seed(int(seed))
            if self.estimator.torch.cuda.is_available():
                self.estimator.torch.cuda.manual_seed_all(int(seed))
        cubes = []
        for start in range(0, n_object, batch_size):
            chunk = x[start : start + batch_size]
            samples = np.asarray(self.estimator.sample(chunk, num_samples=num_samples), dtype=float)
            if samples.ndim == 2 and chunk.shape[0] == 1:
                samples = samples[None, :, :]
            if samples.shape != (chunk.shape[0], int(num_samples), len(self.theta_names)):
                raise ValueError(
                    "MAF estimator returned sample shape "
                    f"{samples.shape}; expected {(chunk.shape[0], int(num_samples), len(self.theta_names))}."
                )
            cubes.append(samples)
        unconstrained = np.concatenate(cubes, axis=0)
        return self.target_transform.inverse(unconstrained)

    def log_prob(
        self,
        theta: np.ndarray,
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None = None,
        input_units: str = "features",
    ) -> np.ndarray | float:
        """Evaluate the learned posterior density in physical parameter units."""

        x = self._context_features(photometry, sigma=sigma, input_units=input_units)
        transformed = self.target_transform.transform(theta)
        logp = self.estimator.log_prob(transformed, x)
        return logp + self.target_transform.log_abs_det_forward(theta)

    def summarize_catalog(
        self,
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None = None,
        input_units: str = "features",
        num_samples: int = 128,
        batch_size: int = 8192,
        quantiles: Sequence[float] = (0.16, 0.5, 0.84),
        seed: int | None = None,
        dtype=np.float32,
    ) -> "MAFCatalogSummary":
        """Sample and summarize a large catalog without retaining its sample cube."""

        x = self._context_features(photometry, sigma=sigma, input_units=input_units)
        levels = np.asarray(quantiles, dtype=float)
        if levels.ndim != 1 or levels.size == 0 or np.any((levels < 0.0) | (levels > 1.0)):
            raise ValueError("quantiles must be non-empty one-dimensional values in [0, 1].")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("MAF catalog summary batch_size must be positive.")

        n_object = x.shape[0]
        n_parameter = len(self.theta_names)
        quantile_values = np.empty((n_object, levels.size, n_parameter), dtype=dtype)
        means = np.empty((n_object, n_parameter), dtype=dtype)
        stds = np.empty((n_object, n_parameter), dtype=dtype)
        n_chunk = (n_object + batch_size - 1) // batch_size
        chunk_seeds = None if seed is None else np.random.SeedSequence(int(seed)).generate_state(n_chunk)

        for chunk_index, start in enumerate(range(0, n_object, batch_size)):
            stop = min(start + batch_size, n_object)
            chunk_seed = None if chunk_seeds is None else int(chunk_seeds[chunk_index])
            draws = self.sample(
                x[start:stop],
                input_units="features",
                num_samples=int(num_samples),
                batch_size=stop - start,
                seed=chunk_seed,
            )
            quantile_values[start:stop] = np.transpose(np.quantile(draws, levels, axis=1), (1, 0, 2))
            means[start:stop] = np.mean(draws, axis=1)
            stds[start:stop] = np.std(draws, axis=1)

        return MAFCatalogSummary(
            quantile_levels=levels,
            quantile_values=quantile_values,
            mean=means,
            std=stds,
            theta_names=self.theta_names,
            metadata={
                "num_samples_per_object": int(num_samples),
                "batch_size": batch_size,
                "seed": seed,
                "context_schema": dict(self.schema),
            },
        )

    def _context_features(
        self,
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None,
        input_units: str,
    ) -> np.ndarray:
        if isinstance(photometry, SEDDataset):
            if np.any(photometry.active_upper_limit_mask):
                raise NotImplementedError("Stable MAF inference does not yet encode censored upper-limit bands.")
            flux, data_sigma, _, names = photometry.active_arrays()
            if tuple(names) != self.band_names:
                raise ValueError(
                    f"SEDDataset active bands {tuple(names)} do not match trained bands {self.band_names}."
                )
            context = self.context
            if context is not None and context.flux_unit is not None:
                if str(photometry.flux_unit) != str(context.flux_unit):
                    raise ValueError(
                        f"SEDDataset flux unit {photometry.flux_unit!r} does not match trained context unit "
                        f"{context.flux_unit!r}."
                    )
            photometry = flux
            sigma = data_sigma
            input_units = "native"

        if input_units == "features":
            return _as_2d(photometry, expected_cols=len(self.x_names), name="context features")
        if input_units not in {"flux", "native"}:
            raise ValueError("input_units must be 'features' or 'native'.")
        flux = _as_2d(photometry, expected_cols=len(self.band_names), name="photometry")
        context = self.context
        if context is None:
            if sigma is not None:
                raise ValueError("This precomputed-feature MAF has no native uncertainty encoding schema.")
            if not bool(self.schema.get("native_input_supported", False)):
                raise ValueError("This loaded MAF accepts pre-encoded context features only.")
            return transform_photometry(flux, self.feature_transform)
        if sigma is None:
            if context.conditions_on_sigma:
                raise ValueError(
                    f"MAF context {context.mode!r} requires sigma for every native photometric input."
                )
            sigma_arr = np.zeros_like(flux)
        else:
            sigma_arr = _as_2d(sigma, expected_cols=len(self.band_names), name="sigma")
            if sigma_arr.shape[0] == 1 and flux.shape[0] > 1:
                sigma_arr = np.repeat(sigma_arr, flux.shape[0], axis=0)
            if sigma_arr.shape != flux.shape:
                raise ValueError("sigma must have one row or the same row count as photometry.")
        if context.mode == "flux" and self.feature_transform not in {"features", "flux", "identity"}:
            if self.training_set is None:
                raise ValueError("A custom callable feature transform cannot be reconstructed from a checkpoint.")
            return transform_photometry(flux, self.feature_transform)
        return context.encode(flux, sigma_arr)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save an auditable MAF checkpoint directory without training rows."""

        path = Path(path)
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"MAF checkpoint path exists and is not a directory: {path}")
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"MAF checkpoint path already exists and is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if self.estimator.theta_standardizer is None or self.estimator.x_standardizer is None:
            raise RuntimeError("Cannot save an unfitted MAF estimator.")

        arrays_path = path / "standardizers.npz"
        np.savez_compressed(
            arrays_path,
            theta_mean=self.estimator.theta_standardizer.mean,
            theta_std=self.estimator.theta_standardizer.std,
            x_mean=self.estimator.x_standardizer.mean,
            x_std=self.estimator.x_standardizer.std,
        )
        weights = {name: tensor.detach().cpu() for name, tensor in self.estimator.flow.state_dict().items()}
        self.estimator.torch.save(weights, path / "weights.pt")

        manifest = {
            "format": "composed.maf.v1",
            "composed_version": _distribution_version("composed"),
            "torch_version": str(getattr(self.estimator.torch, "__version__", "unknown")),
            "nflows_version": _distribution_version("nflows"),
            "estimator": self.estimator.configuration(),
            "schema": dict(self.schema),
            "target_transform": self.target_transform.specification(),
            "history": self.history,
            "metadata": self.metadata,
        }
        with (path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = "auto") -> "TrainedMAFSBI":
        """Load a checkpoint on the requested available torch device."""

        path = Path(path)
        with (path / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != "composed.maf.v1":
            raise ValueError(f"Unsupported MAF checkpoint format {manifest.get('format')!r}.")
        config = dict(manifest["estimator"])
        config["device"] = device
        estimator = MAFPosteriorEstimator(**config)
        with np.load(path / "standardizers.npz", allow_pickle=False) as arrays:
            estimator.theta_standardizer = Standardizer(arrays["theta_mean"], arrays["theta_std"])
            estimator.x_standardizer = Standardizer(arrays["x_mean"], arrays["x_std"])
        try:
            state = estimator.torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        except TypeError:  # Older supported torch releases lack weights_only.
            state = estimator.torch.load(path / "weights.pt", map_location="cpu")
        estimator.flow.load_state_dict(state)
        estimator.flow.eval()
        estimator.history = {str(key): list(value) for key, value in manifest.get("history", {}).items()}
        return cls(
            estimator=estimator,
            training_set=None,
            history=estimator.history,
            metadata=dict(manifest.get("metadata", {})),
            target_transform=PriorSupportTransform.from_specification(manifest["target_transform"]),
            schema=dict(manifest["schema"]),
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
        """Run generic SBI diagnostics on MAF posterior samples."""

        return run_sbi_diagnostics(
            posterior_samples=samples,
            theta_true=theta_true,
            x_test=x_test,
            theta_names=self.theta_names,
            output_dir=output_dir,
            make_plots=make_plots,
        )


@dataclass
class MAFCatalogSummary:
    """Posterior moments and quantiles for an object-first photometric catalog."""

    quantile_levels: np.ndarray
    quantile_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    theta_names: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        levels = np.asarray(self.quantile_levels, dtype=float)
        values = np.asarray(self.quantile_values)
        mean = np.asarray(self.mean)
        std = np.asarray(self.std)
        names = tuple(str(name) for name in self.theta_names)
        if levels.ndim != 1 or values.ndim != 3:
            raise ValueError("MAFCatalogSummary requires 1D levels and 3D quantile values.")
        if values.shape[1] != levels.size or values.shape[2] != len(names):
            raise ValueError("MAFCatalogSummary quantile axes do not match levels and theta_names.")
        if mean.shape != (values.shape[0], values.shape[2]) or std.shape != mean.shape:
            raise ValueError("MAFCatalogSummary mean/std shapes must match object and parameter axes.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("MAFCatalogSummary arrays must be finite.")
        self.quantile_levels = levels
        self.quantile_values = values
        self.mean = mean
        self.std = std
        self.theta_names = names
        self.metadata = dict(self.metadata)

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.5)

    def quantile(self, level: float) -> np.ndarray:
        match = np.flatnonzero(np.isclose(self.quantile_levels, float(level), rtol=0.0, atol=1e-12))
        if match.size != 1:
            raise KeyError(f"Quantile level {level} is not stored in this catalog summary.")
        return self.quantile_values[:, int(match[0]), :]

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Save compact summary arrays with a provenance sidecar."""

        return save_npz_with_provenance(
            path,
            compressed=True,
            extra={"theta_names": self.theta_names, "metadata": self.metadata},
            quantile_levels=self.quantile_levels,
            quantile_values=self.quantile_values,
            mean=self.mean,
            std=self.std,
            theta_names=np.asarray(self.theta_names, dtype=str),
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
    theta_full, x_native, sigma_native, sim_metadata = simulate_training_set(
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
        return_sigma=True,
    )
    inferred_names, theta = _select_inferred_parameters(
        theta_full,
        problem.parameters.names,
        simulation.infer,
    )
    context = _coerce_photometric_context(simulation.context, flux_unit=problem.data.flux_unit)
    if context.mode == "flux" and simulation.feature_transform is not None:
        x_features = transform_photometry(x_native, simulation.feature_transform)
        x_names = tuple(problem.data.active_band_names)
        observation_groups = {"photometry": x_names}
        transform_name = _transform_name(simulation.feature_transform)
    else:
        x_features = context.encode(x_native, sigma_native)
        x_names = context.feature_names(problem.data.active_band_names)
        observation_groups = context.observation_groups(problem.data.active_band_names)
        transform_name = context.mode
    theta_transform = PriorSupportTransform.from_parameter_space(problem.parameters, inferred_names)
    return SBITrainingSet(
        theta=theta,
        x=x_features,
        theta_names=inferred_names,
        x_names=x_names,
        source="composed.problem.simulate",
        theta_full=theta_full,
        full_parameter_names=problem.parameters.names,
        x_native=x_native,
        sigma_native=sigma_native,
        native_names=problem.data.active_band_names,
        feature_transform=simulation.feature_transform or "features",
        context=context,
        theta_transform=theta_transform,
        observation_group="photometry",
        observation_groups=observation_groups,
        metadata={
            "problem": problem.specification(),
            "simulator": "Problem.simulate",
            "noise_model": _transform_name(simulation.noise_fn),
            "requested_training_rows": int(simulation.n),
            "active_band_names": problem.data.active_band_names,
            "flux_unit": problem.data.flux_unit,
            "feature_transform": transform_name,
            "photometric_context": context.specification(),
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
    context: PhotometricContext | str = "flux",
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
            context=context,
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
    validation_split: float = 0.1,
    patience: int | None = 20,
    min_delta: float = 0.0,
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
            patience=patience,
            min_delta=min_delta,
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

    inferred_names = problem.parameters.names if simulation.infer is None else tuple(simulation.infer)
    _validate_continuous_sbi_targets(problem.parameters, inferred_names)
    if isinstance(method, Diffusion):
        context_mode = simulation.context.mode if isinstance(simulation.context, PhotometricContext) else str(simulation.context)
        if str(context_mode).lower() != "flux":
            raise ValueError(
                "Experimental diffusion currently requires Simulate(context='flux'). "
                "The stable uncertainty-conditioned context is implemented for MAF."
            )

    training_set = simulate_sbi_training_set(problem, simulation, rng=seed)
    trained = train_sbi(training_set, method, seed=seed)
    samples = _sample_problem_posterior(problem, trained, method, seed=seed)

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
            "photometric_context": (
                None if training_set.context is None else training_set.context.specification()
            ),
            "target_transform": training_set.theta_transform.specification(),
            "device": str(getattr(trained.estimator, "device", "unknown")),
            "inference_batch_size": (
                method.inference_batch_size if isinstance(method, MAF) else method.sample_batch_size
            ),
            "history": trained.history,
            "logp_available": False,
            "map_available": False,
            "seed": seed,
        },
        inference_state=trained,
    )


def _train_maf(training_set: SBITrainingSet, method: MAF, *, seed: int | None) -> TrainedMAFSBI:
    target_transform = training_set.theta_transform or PriorSupportTransform.identity(training_set.theta_names)
    unconstrained_theta = target_transform.transform(training_set.theta)
    estimator, metadata = train_maf_posterior_from_dataset(
        unconstrained_theta,
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
        patience=method.patience,
        min_delta=method.min_delta,
        seed=seed,
        verbose=method.verbose,
    )
    history = dict(getattr(estimator, "history", {}))
    return TrainedMAFSBI(
        estimator=estimator,
        training_set=training_set,
        history=history,
        metadata={
            **metadata,
            "training_source": training_set.source,
            "training_set_metadata": _checkpoint_training_metadata(training_set.metadata),
        },
        target_transform=target_transform,
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


def _sample_problem_posterior(
    problem,
    trained,
    method: MAF | Diffusion,
    *,
    seed: int | None = None,
) -> np.ndarray:
    requested = int(method.num_samples)
    if requested <= 0:
        raise ValueError("SBI num_samples must be positive.")

    if isinstance(method, MAF):
        cube = trained.sample(
            problem.data,
            num_samples=requested,
            batch_size=method.inference_batch_size,
            seed=seed,
        )
        draws = np.asarray(cube[0], dtype=float)
        valid = _samples_within_declared_priors(draws, problem.parameters, trained.theta_names)
        if not np.all(valid):
            raise FloatingPointError(
                "Bounded MAF target transform produced samples outside declared prior support."
            )
        return draws

    observed = np.asarray(problem.data.active_flux, dtype=float)
    accepted = []
    n_accepted = 0
    for _ in range(12):
        draw_n = max(requested - n_accepted, min(requested, 256))
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


def _coerce_photometric_context(
    context: PhotometricContext | str,
    *,
    flux_unit: str,
) -> PhotometricContext:
    if isinstance(context, PhotometricContext):
        if context.flux_unit is None:
            return replace(context, flux_unit=str(flux_unit))
        if str(context.flux_unit) != str(flux_unit):
            raise ValueError(
                f"PhotometricContext flux unit {context.flux_unit!r} does not match dataset unit {flux_unit!r}."
            )
        return context
    return PhotometricContext(mode=str(context), flux_unit=str(flux_unit))


def _maf_schema_from_training_set(training_set: SBITrainingSet) -> dict[str, object]:
    feature_transform = training_set.feature_transform
    transform_is_named = isinstance(feature_transform, str)
    context = training_set.context
    native_supported = context is not None and (context.mode != "flux" or transform_is_named)
    return {
        "theta_names": list(training_set.theta_names),
        "x_names": list(training_set.x_names),
        "band_names": list(training_set.band_names),
        "photometric_context": None if context is None else context.specification(),
        "feature_transform": _transform_name(feature_transform),
        "native_input_supported": bool(native_supported),
        "training_source": training_set.source,
    }


def _validate_maf_schema(schema: Mapping[str, object]) -> None:
    required = {"theta_names", "x_names", "band_names", "photometric_context", "native_input_supported"}
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError("MAF checkpoint schema is missing: " + ", ".join(missing))
    for key in ("theta_names", "x_names", "band_names"):
        names = tuple(str(name) for name in schema[key])
        if not names or len(set(names)) != len(names):
            raise ValueError(f"MAF checkpoint {key} must be non-empty and unique.")
    context = schema.get("photometric_context")
    if context is not None:
        PhotometricContext.from_specification(context)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _checkpoint_training_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep scientific training metadata without serializing failed theta rows."""

    checkpoint_metadata = dict(metadata)
    simulation = checkpoint_metadata.get("simulate_training_set")
    if isinstance(simulation, Mapping):
        simulation = dict(simulation)
        failures = list(simulation.pop("failures", []))
        simulation["failure_count"] = len(failures)
        if failures:
            simulation["failure_examples"] = [
                {"error": str(failure.get("error", "unknown simulation failure"))}
                for failure in failures[:5]
            ]
        checkpoint_metadata["simulate_training_set"] = simulation
    return checkpoint_metadata


def _json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else repr(scalar)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


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
