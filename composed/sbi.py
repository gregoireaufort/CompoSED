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
6. train a stable conditional MAF or MDN posterior estimator;
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
from composed.priors import DeltaPrior, LogUniformPrior, NormalPrior, StudentTPrior, UniformPrior
from composed.provenance import save_npz_with_provenance
from inftools.diagnostics import run_sbi_diagnostics
from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata
from inftools.sbi import (
    MAFPosteriorEstimator,
    MDNPosteriorEstimator,
    Standardizer,
    simulate_training_set,
    train_maf_posterior_from_dataset,
    train_mdn_posterior_from_dataset,
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
                    f"Neural SBI target {name!r} uses unsupported prior {type(prior).__name__}; "
                    "MAF and diffusion support UniformPrior, LogUniformPrior, "
                    "NormalPrior, and StudentTPrior."
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


def _photometric_state_valid_rows(
    theta: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    measurement_mask: np.ndarray,
    upper_limit: np.ndarray,
    upper_limit_mask: np.ndarray,
    *,
    context: PhotometricContext,
) -> np.ndarray:
    """Identify rows whose declared photometric observations are usable."""

    detected = measurement_mask & ~upper_limit_mask
    sigma_ok = np.isfinite(sigma)
    if context.conditions_on_sigma:
        sigma_ok &= sigma > 0.0
    else:
        sigma_ok &= sigma >= 0.0
    flux_ok = np.isfinite(flux)
    if context.mode == "abmag_magerr":
        flux_ok &= flux > 0.0

    limit_ok = np.isfinite(upper_limit)
    if context.mode == "abmag_magerr":
        limit_ok &= upper_limit > 0.0
    declared_limit = measurement_mask & np.isfinite(upper_limit)

    return (
        np.all(np.isfinite(theta), axis=1)
        & np.any(measurement_mask, axis=1)
        & np.all(~upper_limit_mask | measurement_mask, axis=1)
        & np.all(~measurement_mask | sigma_ok, axis=1)
        & np.all(~detected | flux_ok, axis=1)
        & np.all(~upper_limit_mask | limit_ok, axis=1)
        & np.all(~declared_limit | limit_ok, axis=1)
    )


def _photometric_state_schema(
    context: PhotometricContext,
    band_names: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    bands = tuple(str(name) for name in band_names)
    base_names = context.feature_names(bands)
    base_groups = context.observation_groups(bands)
    state_groups = {
        "availability": tuple(f"available:{name}" for name in bands),
        "censoring": tuple(f"censored:{name}" for name in bands),
        "limit_known": tuple(f"limit_known:{name}" for name in bands),
        "upper_limit": tuple(f"limit_value:{name}" for name in bands),
    }
    names = base_names + tuple(
        name for group_names in state_groups.values() for name in group_names
    )
    return names, {**base_groups, **state_groups}


def _encode_photometric_state(
    flux: np.ndarray,
    sigma: np.ndarray,
    measurement_mask: np.ndarray,
    upper_limit: np.ndarray,
    upper_limit_mask: np.ndarray,
    *,
    context: PhotometricContext,
    band_names: Sequence[str],
    feature_transform: ObservationTransform = "features",
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Encode values, uncertainty, availability, and censoring without leakage.

    Flux values hidden by a non-detection are never passed to the neural
    estimator. Their photometric feature is set to a neutral zero, while the
    one-sigma uncertainty, upper-limit depth, and explicit state flags retain
    the information that is actually observed.
    """

    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    measurement_mask = np.asarray(measurement_mask, dtype=bool)
    upper_limit = np.asarray(upper_limit, dtype=float)
    upper_limit_mask = np.asarray(upper_limit_mask, dtype=bool)
    bands = tuple(str(name) for name in band_names)
    expected_shape = flux.shape
    if flux.ndim != 2 or any(
        array.shape != expected_shape
        for array in (sigma, measurement_mask, upper_limit, upper_limit_mask)
    ):
        raise ValueError("Photometric state arrays must have one matching two-dimensional shape.")
    if flux.shape[1] != len(bands):
        raise ValueError("Photometric state columns must match band_names.")

    dummy_theta = np.zeros((flux.shape[0], 1), dtype=float)
    valid = _photometric_state_valid_rows(
        dummy_theta,
        flux,
        sigma,
        measurement_mask,
        upper_limit,
        upper_limit_mask,
        context=context,
    )
    if not np.all(valid):
        bad = np.flatnonzero(~valid)
        raise ValueError(
            "Invalid photometric observation state in row(s) "
            + ", ".join(str(int(index)) for index in bad[:8])
            + (", ..." if bad.size > 8 else "")
            + ". Detections require finite flux and valid sigma; censored bands "
            "require a finite upper limit and valid sigma. Sigma must be positive "
            "for uncertainty-conditioned contexts."
        )

    detected = measurement_mask & ~upper_limit_mask
    limit_known = measurement_mask & np.isfinite(upper_limit)

    if context.mode == "flux":
        safe_flux = np.zeros_like(flux)
        safe_flux[detected] = flux[detected]
        if feature_transform not in {"features", "flux", "identity"} and not np.all(detected):
            raise ValueError(
                "Legacy nonlinear flux feature transforms cannot encode missing or censored bands. "
                "Use PhotometricContext('snr_logsigma') for state-aware diffusion."
            )
        base = transform_photometry(safe_flux, feature_transform)
    elif context.mode == "snr_logsigma":
        signal_to_noise = np.zeros_like(flux)
        signal_to_noise[detected] = flux[detected] / sigma[detected]
        log_sigma = np.zeros_like(sigma)
        log_sigma[measurement_mask] = np.log10(
            sigma[measurement_mask] / context.reference_flux
        )
        base = np.column_stack([signal_to_noise, log_sigma])
    elif context.mode == "flux_sigma":
        measured_flux = np.zeros_like(flux)
        measured_flux[detected] = flux[detected]
        measured_sigma = np.zeros_like(sigma)
        measured_sigma[measurement_mask] = sigma[measurement_mask]
        base = np.column_stack([measured_flux, measured_sigma])
    else:
        magnitude = np.zeros_like(flux)
        magnitude_error = np.zeros_like(sigma)
        magnitude[detected] = -2.5 * np.log10(
            flux[detected] / context.reference_flux
        )
        magnitude_error[detected] = (
            (2.5 / np.log(10.0)) * sigma[detected] / flux[detected]
        )
        magnitude_error[upper_limit_mask] = (
            (2.5 / np.log(10.0))
            * sigma[upper_limit_mask]
            / upper_limit[upper_limit_mask]
        )
        base = np.column_stack([magnitude, magnitude_error])

    limit_feature = np.zeros_like(upper_limit)
    if context.mode == "snr_logsigma":
        limit_feature[limit_known] = (
            upper_limit[limit_known] / sigma[limit_known]
        )
    elif context.mode == "abmag_magerr":
        limit_feature[limit_known] = -2.5 * np.log10(
            upper_limit[limit_known] / context.reference_flux
        )
    else:
        limit_feature[limit_known] = upper_limit[limit_known]

    state = np.column_stack(
        [
            measurement_mask.astype(float),
            upper_limit_mask.astype(float),
            limit_known.astype(float),
            limit_feature,
        ]
    )
    features = np.column_stack([base, state])
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("Encoded diffusion observation features contain NaN or inf.")

    names, groups = _photometric_state_schema(context, bands)
    return features, names, groups


@dataclass
class SBITrainingSet:
    """Paired physical parameters and observations used to train SBI.

    This object has no backend or likelihood requirement.  It can describe
    simulations generated through a :class:`composed.Problem`, a presampled
    forward model, a numerical simulation, or an empirical labeled catalog.
    Rows of ``theta`` and ``x`` must refer to the same realization.

    ``x_native`` records the observation before the optional feature transform;
    for external feature tables it can simply equal ``x``. State-aware
    photometry additionally stores row-wise measurement availability,
    censoring flags, and upper-limit depths. These are observed data, not the
    random condition mask used by diffusion training. ``theta_full`` is useful
    for simulator-generated data where nuisance parameters were sampled but
    only a subset of parameters is used as neural labels.
    """

    theta: np.ndarray
    x: np.ndarray
    theta_names: Sequence[str]
    x_names: Sequence[str]
    source: str
    condition_names: Sequence[str] = ()
    condition_values: np.ndarray | None = None
    theta_full: np.ndarray | None = None
    full_parameter_names: Sequence[str] | None = None
    x_native: np.ndarray | None = None
    sigma_native: np.ndarray | None = None
    measurement_mask_native: np.ndarray | None = None
    upper_limit_native: np.ndarray | None = None
    upper_limit_mask_native: np.ndarray | None = None
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

        condition_names = tuple(str(name) for name in self.condition_names)
        if len(set(condition_names)) != len(condition_names):
            raise ValueError("condition_names must be unique.")
        overlap = sorted(set(condition_names) & set(theta_names))
        if overlap:
            raise ValueError(
                "SBI target and condition names must be disjoint; overlap: "
                + ", ".join(overlap)
            )
        if condition_names:
            if self.condition_values is None:
                raise ValueError(
                    "condition_values are required when condition_names are declared."
                )
            condition_values = np.asarray(self.condition_values, dtype=float)
            expected_shape = (theta.shape[0], len(condition_names))
            if condition_values.shape != expected_shape:
                raise ValueError(
                    f"condition_values has shape {condition_values.shape}; "
                    f"expected {expected_shape}."
                )
            if not np.all(np.isfinite(condition_values)):
                raise ValueError("condition_values must be finite.")
            condition_features = tuple(
                f"condition:{name}" for name in condition_names
            )
            if x_names[-len(condition_names) :] != condition_features:
                raise ValueError(
                    "Condition columns must be the final SBI context columns in "
                    "condition_names order."
                )
            if not np.allclose(
                x[:, -len(condition_names) :],
                condition_values,
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "Final SBI context columns must equal condition_values exactly."
                )
        else:
            if self.condition_values is not None:
                supplied = np.asarray(self.condition_values, dtype=float)
                if supplied.shape != (theta.shape[0], 0):
                    raise ValueError(
                        "condition_values must be omitted when condition_names is empty."
                    )
            condition_values = np.empty((theta.shape[0], 0), dtype=float)
            condition_features = ()

        theta_full = theta if self.theta_full is None else np.asarray(self.theta_full, dtype=float)
        if theta_full.ndim != 2 or theta_full.shape[0] != theta.shape[0] or not np.all(np.isfinite(theta_full)):
            raise ValueError("theta_full must be finite with shape (n_training, n_full_parameter).")
        full_names = theta_names if self.full_parameter_names is None else tuple(
            str(name) for name in self.full_parameter_names
        )
        if len(full_names) != theta_full.shape[1] or len(set(full_names)) != len(full_names):
            raise ValueError("full_parameter_names must be unique and match theta_full columns.")

        x_native = x if self.x_native is None else np.asarray(self.x_native, dtype=float)
        if x_native.ndim != 2 or x_native.shape[0] != x.shape[0]:
            raise ValueError("x_native must be two-dimensional and have the same number of rows as x.")
        native_names = x_names if self.native_names is None else tuple(str(name) for name in self.native_names)
        if len(native_names) != x_native.shape[1] or len(set(native_names)) != len(native_names):
            raise ValueError("native_names must be unique and match x_native columns.")
        sigma_native = None if self.sigma_native is None else np.asarray(self.sigma_native, dtype=float)
        if sigma_native is not None:
            if sigma_native.shape != x_native.shape:
                raise ValueError("sigma_native must have the same shape as x_native.")
        if self.context is not None and not isinstance(self.context, PhotometricContext):
            raise TypeError("context must be a PhotometricContext or None.")

        state_was_declared = any(
            value is not None
            for value in (
                self.measurement_mask_native,
                self.upper_limit_native,
                self.upper_limit_mask_native,
            )
        )
        if state_was_declared:
            if self.context is None:
                raise ValueError("State-aware photometry requires an explicit PhotometricContext.")
            if sigma_native is None:
                raise ValueError("State-aware photometry requires sigma_native.")
            measurement_mask = (
                np.ones(x_native.shape, dtype=bool)
                if self.measurement_mask_native is None
                else np.asarray(self.measurement_mask_native, dtype=bool)
            )
            upper_limit = (
                np.full(x_native.shape, np.nan, dtype=float)
                if self.upper_limit_native is None
                else np.asarray(self.upper_limit_native, dtype=float)
            )
            upper_limit_mask = (
                np.zeros(x_native.shape, dtype=bool)
                if self.upper_limit_mask_native is None
                else np.asarray(self.upper_limit_mask_native, dtype=bool)
            )
            if any(
                array.shape != x_native.shape
                for array in (measurement_mask, upper_limit, upper_limit_mask)
            ):
                raise ValueError(
                    "measurement_mask_native, upper_limit_native, and "
                    "upper_limit_mask_native must match x_native."
                )
            valid_rows = _photometric_state_valid_rows(
                theta,
                x_native,
                sigma_native,
                measurement_mask,
                upper_limit,
                upper_limit_mask,
                context=self.context,
            )
            if not np.all(valid_rows):
                raise ValueError(
                    "State-aware native photometry is invalid. Every row needs at least one "
                    "available band; detections require finite flux and positive sigma; "
                    "censored bands require a finite upper limit and positive sigma."
                )
        else:
            if not np.all(np.isfinite(x_native)):
                raise ValueError(
                    "x_native must be finite unless explicit measurement and censoring masks are supplied."
                )
            if sigma_native is not None:
                if not np.all(np.isfinite(sigma_native)) or np.any(sigma_native < 0.0):
                    raise ValueError("sigma_native must be finite and non-negative.")
            measurement_mask = None
            upper_limit = None
            upper_limit_mask = None
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
            n_condition = len(condition_names)
            observation_names = (
                x_names if n_condition == 0 else x_names[:-n_condition]
            )
            observation_groups = {observation_group: observation_names}
            if condition_names:
                observation_groups["conditions"] = condition_features
        else:
            observation_groups = {
                str(group): tuple(str(name) for name in names)
                for group, names in self.observation_groups.items()
            }
            if "parameters" in observation_groups:
                raise ValueError("'parameters' is reserved for SBI target labels.")
            if condition_names and observation_groups.get("conditions") != condition_features:
                raise ValueError(
                    "observation_groups['conditions'] must match condition_names order."
                )
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
        self.condition_names = condition_names
        self.condition_values = condition_values
        self.theta_full = theta_full
        self.full_parameter_names = full_names
        self.x_native = x_native
        self.sigma_native = sigma_native
        self.measurement_mask_native = measurement_mask
        self.upper_limit_native = upper_limit
        self.upper_limit_mask_native = upper_limit_mask
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
        measurement_mask: np.ndarray | None = None,
        upper_limit: np.ndarray | None = None,
        upper_limit_mask: np.ndarray | None = None,
        conditions: np.ndarray | None = None,
        condition_names: Sequence[str] = (),
    ) -> "SBITrainingSet":
        """Build paired SBI data from measured fluxes and their uncertainties.

        ``measurement_mask`` describes data availability and is part of the
        learned observation. It is distinct from the random diffusion
        conditioning mask used during score training. ``upper_limit_mask``
        marks censored measurements; their corresponding ``flux`` value may be
        NaN because only ``upper_limit`` and ``sigma`` are observed.
        """

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
        condition_names = tuple(str(name) for name in condition_names)
        if len(set(condition_names)) != len(condition_names):
            raise ValueError("condition_names must be unique.")
        if condition_names:
            conditions = np.asarray(conditions, dtype=float)
            expected_shape = (theta.shape[0], len(condition_names))
            if conditions.shape != expected_shape:
                raise ValueError(
                    f"conditions has shape {conditions.shape}; expected {expected_shape}."
                )
        elif conditions is not None:
            supplied = np.asarray(conditions, dtype=float)
            if supplied.shape != (theta.shape[0], 0):
                raise ValueError(
                    "condition_names are required when conditions are supplied."
                )
            conditions = supplied
        else:
            conditions = np.empty((theta.shape[0], 0), dtype=float)
        context_encoder = _coerce_photometric_context(context, flux_unit=flux_unit)
        measurement_mask = (
            np.ones(flux.shape, dtype=bool)
            if measurement_mask is None
            else np.asarray(measurement_mask, dtype=bool)
        )
        upper_limit = (
            np.full(flux.shape, np.nan, dtype=float)
            if upper_limit is None
            else np.asarray(upper_limit, dtype=float)
        )
        upper_limit_mask = (
            np.zeros(flux.shape, dtype=bool)
            if upper_limit_mask is None
            else np.asarray(upper_limit_mask, dtype=bool)
        )
        if any(
            array.shape != flux.shape
            for array in (measurement_mask, upper_limit, upper_limit_mask)
        ):
            raise ValueError(
                "measurement_mask, upper_limit, and upper_limit_mask must match the flux shape."
            )
        row_is_finite = _photometric_state_valid_rows(
            theta,
            flux,
            sigma,
            measurement_mask,
            upper_limit,
            upper_limit_mask,
            context=context_encoder,
        )
        if condition_names:
            row_is_finite &= np.all(np.isfinite(conditions), axis=1)
        if finite == "raise" and not np.all(row_is_finite):
            raise ValueError(
                "Pre-existing SBI photometry contains invalid observation rows; use finite='drop' "
                "explicitly to remove them."
            )
        if finite == "drop":
            theta = theta[row_is_finite]
            flux = flux[row_is_finite]
            sigma = sigma[row_is_finite]
            measurement_mask = measurement_mask[row_is_finite]
            upper_limit = upper_limit[row_is_finite]
            upper_limit_mask = upper_limit_mask[row_is_finite]
            conditions = conditions[row_is_finite]
        elif finite != "raise":
            raise ValueError("finite must be 'raise' or 'drop'.")
        if theta.shape[0] == 0:
            raise ValueError("SBITrainingSet requires at least one finite photometric row.")

        features, _, _ = _encode_photometric_state(
            flux,
            sigma,
            measurement_mask,
            upper_limit,
            upper_limit_mask,
            context=context_encoder,
            band_names=bands,
        )
        features = features[:, : len(context_encoder.feature_names(bands))]
        photometric_feature_names = context_encoder.feature_names(bands)
        condition_feature_names = tuple(
            f"condition:{name}" for name in condition_names
        )
        if condition_names:
            features = np.column_stack([features, conditions])
        feature_names = photometric_feature_names + condition_feature_names
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
                "state_aware_photometry": True,
            }
        )
        return cls(
            theta=theta,
            x=features,
            theta_names=theta_names,
            x_names=feature_names,
            source=source,
            condition_names=condition_names,
            condition_values=conditions,
            theta_full=theta,
            full_parameter_names=theta_names,
            x_native=flux,
            sigma_native=sigma,
            measurement_mask_native=measurement_mask,
            upper_limit_native=upper_limit,
            upper_limit_mask_native=upper_limit_mask,
            native_names=bands,
            feature_transform="features",
            context=context_encoder,
            theta_transform=theta_transform,
            observation_group="photometry",
            observation_groups={
                **context_encoder.observation_groups(bands),
                **(
                    {"conditions": condition_feature_names}
                    if condition_names
                    else {}
                ),
            },
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

    @property
    def has_photometric_state(self) -> bool:
        """Whether row-wise availability and censoring were declared."""

        return self.measurement_mask_native is not None

    @property
    def diffusion_observation_features(self) -> np.ndarray:
        """Observation vector learned by the masked diffusion estimator.

        Generic paired arrays use ``x`` directly. Photometric datasets append
        four explicit state channels per band: availability, censoring,
        whether a depth is known, and the depth value in context-consistent
        units.
        """

        if not self.has_photometric_state:
            return self.x
        features, _, _ = _encode_photometric_state(
            self.x_native,
            self.sigma_native,
            self.measurement_mask_native,
            self.upper_limit_native,
            self.upper_limit_mask_native,
            context=self.context,
            band_names=self.native_names,
            feature_transform=self.feature_transform,
        )
        if self.condition_names:
            features = np.column_stack([features, self.condition_values])
        return features

    @property
    def diffusion_observation_groups(self) -> dict[str, tuple[str, ...]]:
        if not self.has_photometric_state:
            return dict(self.observation_groups)
        _, groups = _photometric_state_schema(self.context, self.native_names)
        if self.condition_names:
            groups["conditions"] = tuple(
                f"condition:{name}" for name in self.condition_names
            )
        return groups

    @property
    def diffusion_feature_metadata(self) -> FeatureMetadata:
        """Feature schema for neural joint training."""

        return FeatureMetadata.from_groups(
            {**self.diffusion_observation_groups, "parameters": self.theta_names}
        )

    @property
    def diffusion_joint_features(self) -> np.ndarray:
        """Return observations plus prior-unconstrained inferred parameters."""

        theta_neural = self.theta_transform.transform(self.theta)
        return np.column_stack([self.diffusion_observation_features, theta_neural])

    @property
    def diffusion_observation_size(self) -> int:
        return int(self.diffusion_observation_features.shape[1])

    def encode_diffusion_observation(
        self,
        flux: np.ndarray,
        sigma: np.ndarray,
        measurement_mask: np.ndarray,
        upper_limit: np.ndarray,
        upper_limit_mask: np.ndarray,
        conditions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the exact diffusion observation encoding used for training."""

        if not self.has_photometric_state:
            raise ValueError("This training set has no native photometric state schema.")
        features, names, _ = _encode_photometric_state(
            flux,
            sigma,
            measurement_mask,
            upper_limit,
            upper_limit_mask,
            context=self.context,
            band_names=self.native_names,
            feature_transform=self.feature_transform,
        )
        if self.condition_names:
            condition_values = _condition_matrix(
                conditions,
                condition_names=self.condition_names,
                n_object=features.shape[0],
            )
            features = np.column_stack([features, condition_values])
            names = tuple(names) + tuple(
                f"condition:{name}" for name in self.condition_names
            )
        expected_names = self.diffusion_feature_metadata.names[: self.diffusion_observation_size]
        if tuple(names) != tuple(expected_names):
            raise RuntimeError("Diffusion observation encoding no longer matches its training schema.")
        return features

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
    condition_on: Sequence[str] | None = None
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
        if int(self.max_retries) < 0:
            raise ValueError("Simulate.max_retries must be non-negative.")
        if self.condition_on is not None:
            condition_names = tuple(str(name) for name in self.condition_on)
            if len(set(condition_names)) != len(condition_names):
                raise ValueError("Simulate.condition_on names must be unique.")
            object.__setattr__(self, "condition_on", condition_names)
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
class MDN:
    """Conditional Gaussian-mixture posterior inference configuration."""

    n_components: int = 8
    hidden_features: int = 128
    num_blocks: int = 3
    min_scale: float = 1.0e-3
    learning_rate: float = 1.0e-3
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
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None = None,
        conditions=None,
        measurement_mask: np.ndarray | None = None,
        upper_limit: np.ndarray | None = None,
        upper_limit_mask: np.ndarray | None = None,
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
            An :class:`SEDDataset`, one object, or a photometric catalog.
        input_units:
            ``"features"`` means an array already contains the complete
            diffusion observation vector, including state channels.
            ``"native"`` means flux-like values in the training flux unit and
            requires ``sigma`` for state-aware photometry. Passing an
            ``SEDDataset`` supplies units, masks, errors, and limits directly.
        """

        joint = self.sample_joint(
            photometry,
            sigma=sigma,
            conditions=conditions,
            measurement_mask=measurement_mask,
            upper_limit=upper_limit,
            upper_limit_mask=upper_limit_mask,
            input_units=input_units,
            num_samples=num_samples,
            steps=steps,
            sampler=sampler,
            batch_size=batch_size,
            **sampler_kwargs,
        )
        return joint[:, :, self.training_set.diffusion_observation_size :]

    def sample_joint(
        self,
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None = None,
        conditions=None,
        measurement_mask: np.ndarray | None = None,
        upper_limit: np.ndarray | None = None,
        upper_limit_mask: np.ndarray | None = None,
        input_units: str = "features",
        num_samples: int = 512,
        steps: int = 64,
        sampler: str = "edm_euler",
        batch_size: int | None = None,
        **sampler_kwargs,
    ) -> np.ndarray:
        """Sample observation features and physical parameters.

        The estimator itself operates on prior-unconstrained parameter
        coordinates. Only the returned parameter columns are converted back to
        their declared physical units.
        """

        x = self._observation_features(
            photometry,
            sigma=sigma,
            conditions=conditions,
            measurement_mask=measurement_mask,
            upper_limit=upper_limit,
            upper_limit_mask=upper_limit_mask,
            input_units=input_units,
        )
        n_objects = x.shape[0]
        n_observation = self.training_set.diffusion_observation_size
        n_theta = len(self.theta_names)
        known = np.full((n_objects, n_observation + n_theta), np.nan, dtype=float)
        known[:, :n_observation] = x
        mask = np.zeros_like(known, dtype=bool)
        mask[:, :n_observation] = True
        neural_joint = self.estimator.sample(
            known,
            mask,
            num_samples=num_samples,
            steps=steps,
            sampler=sampler,
            batch_size=batch_size,
            **sampler_kwargs,
        )
        physical_joint = np.asarray(neural_joint, dtype=float).copy()
        physical_joint[:, :, n_observation:] = self.training_set.theta_transform.inverse(
            physical_joint[:, :, n_observation:]
        )
        return physical_joint

    def _observation_features(
        self,
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None,
        conditions,
        measurement_mask: np.ndarray | None,
        upper_limit: np.ndarray | None,
        upper_limit_mask: np.ndarray | None,
        input_units: str,
    ) -> np.ndarray:
        """Build the exact observation vector used during diffusion training."""

        if isinstance(photometry, SEDDataset):
            if any(
                value is not None
                for value in (sigma, measurement_mask, upper_limit, upper_limit_mask)
            ):
                raise ValueError(
                    "Do not pass sigma or state arrays alongside an SEDDataset; "
                    "they are read from the dataset."
                )
            if not self.training_set.has_photometric_state:
                raise ValueError(
                    "This diffusion model was trained on generic precomputed features, "
                    "not native SEDDataset photometry."
                )
            training_bands = self.band_names
            data_index = {name: index for index, name in enumerate(photometry.band_names)}
            missing = [name for name in training_bands if name not in data_index]
            if missing:
                raise ValueError(
                    "SEDDataset is missing trained band(s): " + ", ".join(missing)
                )
            indices = np.asarray([data_index[name] for name in training_bands], dtype=int)
            context = self.training_set.context
            if context.flux_unit is not None and str(photometry.flux_unit) != str(context.flux_unit):
                raise ValueError(
                    f"SEDDataset flux unit {photometry.flux_unit!r} does not match trained "
                    f"context unit {context.flux_unit!r}."
                )
            flux = photometry.flux[indices][None, :]
            sigma = photometry.sigma[indices][None, :]
            measurement_mask = photometry.active_mask[indices][None, :]
            upper_limit_mask = (
                photometry.upper_limit_mask[indices] & photometry.active_mask[indices]
            )[None, :]
            upper_values = np.full((1, len(training_bands)), np.nan, dtype=float)
            upper_values[upper_limit_mask] = photometry.upper_limit[indices][
                upper_limit_mask[0]
            ]
            return self.training_set.encode_diffusion_observation(
                flux,
                sigma,
                measurement_mask,
                upper_values,
                upper_limit_mask,
                conditions=conditions,
            )

        if input_units == "features":
            if any(
                value is not None
                for value in (sigma, measurement_mask, upper_limit, upper_limit_mask)
            ):
                raise ValueError(
                    "State arrays are only accepted with input_units='native'."
                )
            array = np.asarray(photometry, dtype=float)
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                raise ValueError(
                    "diffusion observation features must be a finite one- or "
                    "two-dimensional array."
                )
            n_condition = len(self.training_set.condition_names)
            full_size = self.training_set.diffusion_observation_size
            base_size = full_size - n_condition
            if array.shape[1] == full_size:
                if conditions is not None and n_condition:
                    supplied = _condition_matrix(
                        conditions,
                        condition_names=self.training_set.condition_names,
                        n_object=array.shape[0],
                    )
                    if not np.allclose(
                        array[:, base_size:],
                        supplied,
                        rtol=0.0,
                        atol=0.0,
                    ):
                        raise ValueError(
                            "Condition values do not match the condition columns "
                            "already present in diffusion features."
                        )
                return array
            if n_condition and array.shape[1] == base_size:
                supplied = _condition_matrix(
                    conditions,
                    condition_names=self.training_set.condition_names,
                    n_object=array.shape[0],
                )
                return np.column_stack([array, supplied])
            raise ValueError(
                "diffusion observation features have shape "
                f"{array.shape}; expected (*, {full_size})"
                + (
                    f" or (*, {base_size}) with explicit conditions."
                    if n_condition
                    else "."
                )
            )
        if input_units not in {"flux", "native"}:
            raise ValueError("input_units must be 'features' or 'native'.")

        if not self.training_set.has_photometric_state:
            x = _as_2d(photometry, expected_cols=len(self.band_names), name="photometry")
            base = transform_photometry(x, self.training_set.feature_transform)
            condition_values = _condition_matrix(
                conditions,
                condition_names=self.training_set.condition_names,
                n_object=base.shape[0],
            )
            return (
                base
                if condition_values.shape[1] == 0
                else np.column_stack([base, condition_values])
            )

        flux = _as_2d_allow_nonfinite(
            photometry,
            expected_cols=len(self.band_names),
            name="photometry",
        )
        sigma_arr = _broadcast_native_state(
            sigma,
            shape=flux.shape,
            name="sigma",
            dtype=float,
            required=True,
        )
        measurement_arr = _broadcast_native_state(
            measurement_mask,
            shape=flux.shape,
            name="measurement_mask",
            dtype=bool,
            default=True,
        )
        upper_arr = _broadcast_native_state(
            upper_limit,
            shape=flux.shape,
            name="upper_limit",
            dtype=float,
            default=np.nan,
        )
        upper_mask_arr = _broadcast_native_state(
            upper_limit_mask,
            shape=flux.shape,
            name="upper_limit_mask",
            dtype=bool,
            default=False,
        )
        return self.training_set.encode_diffusion_observation(
            flux,
            sigma_arr,
            measurement_arr,
            upper_arr,
            upper_mask_arr,
            conditions=conditions,
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
    """Trained fixed-context neural posterior for parameters given observations."""

    estimator: MAFPosteriorEstimator
    training_set: PhotometricTrainingSet | None
    history: dict[str, list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)
    target_transform: PriorSupportTransform | None = None
    schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_transform is None:
            if self.training_set is None:
                raise ValueError(
                    "Loaded fixed-context SBI state requires an explicit target_transform."
                )
            self.target_transform = self.training_set.theta_transform
        if self.training_set is not None:
            generated_schema = _maf_schema_from_training_set(self.training_set)
            if self.schema and dict(self.schema) != generated_schema:
                raise ValueError(
                    "Provided fixed-context SBI schema does not match the training set."
                )
            self.schema = generated_schema
        else:
            self.schema = dict(self.schema)
            _validate_maf_schema(self.schema)
        if self.estimator.theta_dim != len(self.theta_names) or self.estimator.x_dim != len(self.x_names):
            raise ValueError(
                "Neural posterior dimensions do not match the saved scientific schema."
            )
        if tuple(self.target_transform.names) != self.theta_names:
            raise ValueError(
                "Neural-posterior target-transform names do not match the saved parameter order."
            )
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
    def condition_names(self) -> tuple[str, ...]:
        return tuple(self.schema.get("condition_names", ()))

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
        conditions=None,
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

        x = self._context_features(
            photometry,
            sigma=sigma,
            conditions=conditions,
            input_units=input_units,
        )
        n_object = x.shape[0]
        if batch_size is None:
            batch_size = n_object
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("SBI inference batch_size must be positive.")
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
                    "Neural posterior returned sample shape "
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
        conditions=None,
        input_units: str = "features",
    ) -> np.ndarray | float:
        """Evaluate the learned posterior density in physical parameter units."""

        x = self._context_features(
            photometry,
            sigma=sigma,
            conditions=conditions,
            input_units=input_units,
        )
        transformed = self.target_transform.transform(theta)
        logp = self.estimator.log_prob(transformed, x)
        return logp + self.target_transform.log_abs_det_forward(theta)

    def summarize_catalog(
        self,
        photometry: np.ndarray | SEDDataset,
        *,
        sigma: np.ndarray | None = None,
        conditions=None,
        input_units: str = "features",
        num_samples: int = 128,
        batch_size: int = 8192,
        quantiles: Sequence[float] = (0.16, 0.5, 0.84),
        seed: int | None = None,
        dtype=np.float32,
    ) -> "MAFCatalogSummary":
        """Sample and summarize a large catalog without retaining its sample cube."""

        x = self._context_features(
            photometry,
            sigma=sigma,
            conditions=conditions,
            input_units=input_units,
        )
        levels = np.asarray(quantiles, dtype=float)
        if levels.ndim != 1 or levels.size == 0 or np.any((levels < 0.0) | (levels > 1.0)):
            raise ValueError("quantiles must be non-empty one-dimensional values in [0, 1].")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("SBI catalog summary batch_size must be positive.")

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
        conditions,
        input_units: str,
    ) -> np.ndarray:
        if isinstance(photometry, SEDDataset):
            if np.any(photometry.active_upper_limit_mask):
                raise NotImplementedError(
                    "Stable fixed-context SBI does not yet encode censored upper-limit bands."
                )
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
            array = np.asarray(photometry, dtype=float)
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                raise ValueError(
                    "context features must be a finite one- or two-dimensional array."
                )
            n_condition = len(self.condition_names)
            full_size = len(self.x_names)
            base_size = full_size - n_condition
            if array.shape[1] == full_size:
                if conditions is not None and n_condition:
                    supplied = _condition_matrix(
                        conditions,
                        condition_names=self.condition_names,
                        n_object=array.shape[0],
                    )
                    if not np.allclose(
                        array[:, base_size:],
                        supplied,
                        rtol=0.0,
                        atol=0.0,
                    ):
                        raise ValueError(
                            "Condition values do not match the condition columns "
                            "already present in context features."
                        )
                return array
            if n_condition and array.shape[1] == base_size:
                supplied = _condition_matrix(
                    conditions,
                    condition_names=self.condition_names,
                    n_object=array.shape[0],
                )
                return np.column_stack([array, supplied])
            raise ValueError(
                "context features have shape "
                f"{array.shape}; expected (*, {full_size})"
                + (
                    f" or (*, {base_size}) with explicit conditions."
                    if n_condition
                    else "."
                )
            )
        if input_units not in {"flux", "native"}:
            raise ValueError("input_units must be 'features' or 'native'.")
        flux = _as_2d(photometry, expected_cols=len(self.band_names), name="photometry")
        context = self.context
        if context is None:
            if sigma is not None:
                raise ValueError(
                    "This precomputed-feature neural posterior has no native uncertainty encoding schema."
                )
            if not bool(self.schema.get("native_input_supported", False)):
                raise ValueError(
                    "This loaded neural posterior accepts pre-encoded context features only."
                )
            base = transform_photometry(flux, self.feature_transform)
        elif sigma is None:
            if context.conditions_on_sigma:
                raise ValueError(
                    f"SBI context {context.mode!r} requires sigma for every native photometric input."
                )
            sigma_arr = np.zeros_like(flux)
        else:
            sigma_arr = _as_2d(sigma, expected_cols=len(self.band_names), name="sigma")
            if sigma_arr.shape[0] == 1 and flux.shape[0] > 1:
                sigma_arr = np.repeat(sigma_arr, flux.shape[0], axis=0)
            if sigma_arr.shape != flux.shape:
                raise ValueError("sigma must have one row or the same row count as photometry.")
        if context is None:
            pass
        elif context.mode == "flux" and self.feature_transform not in {"features", "flux", "identity"}:
            if self.training_set is None:
                raise ValueError("A custom callable feature transform cannot be reconstructed from a checkpoint.")
            base = transform_photometry(flux, self.feature_transform)
        else:
            base = context.encode(flux, sigma_arr)
        condition_values = _condition_matrix(
            conditions,
            condition_names=self.condition_names,
            n_object=base.shape[0],
        )
        return (
            base
            if condition_values.shape[1] == 0
            else np.column_stack([base, condition_values])
        )

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
        """Run generic SBI diagnostics on posterior samples."""

        return run_sbi_diagnostics(
            posterior_samples=samples,
            theta_true=theta_true,
            x_test=x_test,
            theta_names=self.theta_names,
            output_dir=output_dir,
            make_plots=make_plots,
        )


@dataclass
class TrainedMDNSBI(TrainedMAFSBI):
    """Trained conditional Gaussian-mixture posterior estimator."""

    estimator: MDNPosteriorEstimator

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save an auditable MDN checkpoint directory without training rows."""

        path = Path(path)
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"MDN checkpoint path exists and is not a directory: {path}")
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"MDN checkpoint path already exists and is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if self.estimator.theta_standardizer is None or self.estimator.x_standardizer is None:
            raise RuntimeError("Cannot save an unfitted MDN estimator.")

        np.savez_compressed(
            path / "standardizers.npz",
            theta_mean=self.estimator.theta_standardizer.mean,
            theta_std=self.estimator.theta_standardizer.std,
            x_mean=self.estimator.x_standardizer.mean,
            x_std=self.estimator.x_standardizer.std,
        )
        weights = {
            name: tensor.detach().cpu()
            for name, tensor in self.estimator.network.state_dict().items()
        }
        self.estimator.torch.save(weights, path / "weights.pt")

        manifest = {
            "format": "composed.mdn.v1",
            "composed_version": _distribution_version("composed"),
            "torch_version": str(getattr(self.estimator.torch, "__version__", "unknown")),
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
    def load(cls, path: str | Path, *, device: str | None = "auto") -> "TrainedMDNSBI":
        """Load a normalized MDN checkpoint on the requested torch device."""

        path = Path(path)
        with (path / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != "composed.mdn.v1":
            raise ValueError(f"Unsupported MDN checkpoint format {manifest.get('format')!r}.")
        config = dict(manifest["estimator"])
        config["device"] = device
        estimator = MDNPosteriorEstimator(**config)
        with np.load(path / "standardizers.npz", allow_pickle=False) as arrays:
            estimator.theta_standardizer = Standardizer(arrays["theta_mean"], arrays["theta_std"])
            estimator.x_standardizer = Standardizer(arrays["x_mean"], arrays["x_std"])
        try:
            state = estimator.torch.load(
                path / "weights.pt", map_location="cpu", weights_only=True
            )
        except TypeError:
            state = estimator.torch.load(path / "weights.pt", map_location="cpu")
        estimator.network.load_state_dict(state)
        estimator.network.eval()
        estimator.history = {
            str(key): list(value)
            for key, value in manifest.get("history", {}).items()
        }
        return cls(
            estimator=estimator,
            training_set=None,
            history=estimator.history,
            metadata=dict(manifest.get("metadata", {})),
            target_transform=PriorSupportTransform.from_specification(
                manifest["target_transform"]
            ),
            schema=dict(manifest["schema"]),
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
MDNPhotometricSBIResult = TrainedMDNSBI


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
    inference. Bands declared as upper limits in the observed Problem define
    survey thresholds for training simulations. A simulated measurement at or
    below its threshold is represented by the threshold and a censoring flag,
    never by its latent noisy flux.
    """

    from composed.problem import Problem

    if not isinstance(problem, Problem):
        raise TypeError("simulate_sbi_training_set requires a composed.Problem.")
    if not isinstance(simulation, Simulate):
        raise TypeError("simulation must be composed.Simulate(...).")
    if not isinstance(problem.data, SEDDataset):
        raise NotImplementedError("Problem-driven SBI currently supports photometric SEDDataset observations.")

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
    condition_names = _ordered_parameter_subset(
        problem.parameters.names,
        simulation.condition_on,
        label="condition_on",
    )
    inferred_request = (
        tuple(
            name
            for name in problem.parameters.names
            if name not in set(condition_names)
        )
        if simulation.infer is None
        else tuple(str(name) for name in simulation.infer)
    )
    overlap = sorted(set(inferred_request) & set(condition_names))
    if overlap:
        raise ValueError(
            "Parameters cannot be both inferred and conditioned: "
            + ", ".join(overlap)
        )
    if not inferred_request:
        raise ValueError("SBI requires at least one inferred parameter.")
    inferred_names, theta = _select_inferred_parameters(
        theta_full,
        problem.parameters.names,
        inferred_request,
    )
    parameter_index = {
        name: index for index, name in enumerate(problem.parameters.names)
    }
    condition_values = np.asarray(
        theta_full[:, [parameter_index[name] for name in condition_names]],
        dtype=float,
    )
    context = _coerce_photometric_context(simulation.context, flux_unit=problem.data.flux_unit)
    band_names = tuple(problem.data.active_band_names)
    measurement_mask = np.ones(x_native.shape, dtype=bool)
    candidate_limit = np.asarray(problem.data.active_upper_limit_mask, dtype=bool)
    limit_values = np.where(
        candidate_limit,
        np.asarray(problem.data.active_upper_limit, dtype=float),
        np.nan,
    )
    upper_limit = np.broadcast_to(limit_values, x_native.shape).copy()
    upper_limit_mask = np.broadcast_to(candidate_limit, x_native.shape) & (
        x_native <= upper_limit
    )
    observed_flux = np.asarray(x_native, dtype=float).copy()
    observed_flux[upper_limit_mask] = np.nan

    feature_transform = simulation.feature_transform or "features"
    encoded, _, _ = _encode_photometric_state(
        observed_flux,
        sigma_native,
        measurement_mask,
        upper_limit,
        upper_limit_mask,
        context=context,
        band_names=band_names,
        feature_transform=feature_transform,
    )
    photometric_names = context.feature_names(band_names)
    x_features = encoded[:, : len(photometric_names)]
    condition_features = tuple(f"condition:{name}" for name in condition_names)
    if condition_names:
        x_features = np.column_stack([x_features, condition_values])
    x_names = photometric_names + condition_features
    observation_groups = context.observation_groups(band_names)
    if condition_names:
        observation_groups["conditions"] = condition_features
    transform_name = (
        _transform_name(simulation.feature_transform)
        if context.mode == "flux" and simulation.feature_transform is not None
        else context.mode
    )
    theta_transform = PriorSupportTransform.from_parameter_space(problem.parameters, inferred_names)
    return SBITrainingSet(
        theta=theta,
        x=x_features,
        theta_names=inferred_names,
        x_names=x_names,
        source="composed.problem.simulate",
        condition_names=condition_names,
        condition_values=condition_values,
        theta_full=theta_full,
        full_parameter_names=problem.parameters.names,
        x_native=observed_flux,
        sigma_native=sigma_native,
        measurement_mask_native=measurement_mask,
        upper_limit_native=upper_limit,
        upper_limit_mask_native=upper_limit_mask,
        native_names=band_names,
        feature_transform=feature_transform,
        context=context,
        theta_transform=theta_transform,
        observation_group="photometry",
        observation_groups=observation_groups,
        metadata={
            "problem": problem.specification(),
            "simulator": "Problem.simulate",
            "noise_model": _transform_name(simulation.noise_fn),
            "requested_training_rows": int(simulation.n),
            "active_band_names": band_names,
            "flux_unit": problem.data.flux_unit,
            "feature_transform": transform_name,
            "photometric_context": context.specification(),
            "conditioned_parameter_names": condition_names,
            "observation_state": {
                "availability": "1 for every active simulated band",
                "censoring_rule": "measured_flux <= upper_limit",
                "upper_limit_candidate_bands": tuple(
                    name for name, is_candidate in zip(band_names, candidate_limit) if is_candidate
                ),
                "censored_fraction_by_band": tuple(
                    float(value) for value in np.mean(upper_limit_mask, axis=0)
                ),
            },
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


def train_mdn_photometric_sbi(
    training_set: SBITrainingSet,
    *,
    n_components: int = 8,
    hidden_features: int = 128,
    num_blocks: int = 3,
    min_scale: float = 1.0e-3,
    learning_rate: float = 1.0e-3,
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
) -> TrainedMDNSBI:
    """Compatibility wrapper around :func:`train_sbi` for an MDN."""

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported MDN training option(s): {unknown}.")
    return train_sbi(
        training_set,
        MDN(
            n_components=n_components,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            min_scale=min_scale,
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
    method: MAF | MDN | Diffusion,
    *,
    seed: int | None = None,
) -> TrainedMAFSBI | TrainedMDNSBI | TrainedDiffusionSBI:
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
    if isinstance(method, MDN):
        return _train_mdn(training_set, method, seed=seed)
    if isinstance(method, Diffusion):
        return _train_diffusion(training_set, method, seed=seed)
    raise TypeError(
        "method must be composed.MAF(...), composed.MDN(...), or composed.Diffusion(...)."
    )


def fit_sbi_problem(
    problem,
    method: MAF | MDN | Diffusion,
    simulation: Simulate,
    *,
    conditions: Mapping[str, float] | None = None,
    seed: int | None = None,
):
    """Train from a Problem simulator and infer that Problem's observed SED."""

    from composed.problem import Problem
    from composed.results import InferenceResult

    if not isinstance(problem, Problem):
        raise TypeError("fit_sbi_problem requires a composed.Problem.")
    if not isinstance(method, (MAF, MDN, Diffusion)):
        raise TypeError(
            "method must be composed.MAF(...), composed.MDN(...), or composed.Diffusion(...)."
        )
    if not isinstance(simulation, Simulate):
        raise TypeError(
            "Problem-based SBI requires training=Simulate(...). "
            "For existing paired arrays use train_sbi(SBITrainingSet.from_arrays(...), method)."
        )
    if not isinstance(problem.data, SEDDataset):
        raise NotImplementedError(
            "Problem-driven SBI currently supports photometric SEDDataset observations."
        )

    supplied_conditions = {} if conditions is None else dict(conditions)
    condition_names = _ordered_parameter_subset(
        problem.parameters.names,
        tuple(supplied_conditions),
        label="conditions",
    )
    declared_condition_names = _ordered_parameter_subset(
        problem.parameters.names,
        simulation.condition_on,
        label="condition_on",
    )
    if declared_condition_names and not supplied_conditions:
        raise ValueError(
            "Problem-based SBI requires fit(..., conditions=...) values for every "
            "Simulate.condition_on parameter."
        )
    if declared_condition_names and declared_condition_names != condition_names:
        raise ValueError(
            "Simulate.condition_on must match the names supplied to fit(..., "
            "conditions=...)."
        )

    canonical_conditions = {}
    for name in condition_names:
        value = float(supplied_conditions[name])
        if not np.isfinite(value):
            raise ValueError(f"Conditioned parameter {name!r} must be finite.")
        if not np.isfinite(problem.parameters.priors[name].logpdf(value)):
            raise ValueError(
                f"Conditioned value {value:.8g} for {name!r} lies outside its "
                "declared prior support."
            )
        canonical_conditions[name] = value

    inferred_names = (
        tuple(
            name
            for name in problem.parameters.names
            if name not in set(condition_names)
        )
        if simulation.infer is None
        else tuple(str(name) for name in simulation.infer)
    )
    if len(set(inferred_names)) != len(inferred_names):
        raise ValueError("infer parameter names must be unique.")
    unknown_inferred = sorted(
        set(inferred_names) - set(problem.parameters.names)
    )
    if unknown_inferred:
        raise ValueError(
            "infer contains unknown parameter(s): "
            + ", ".join(unknown_inferred)
        )
    overlap = sorted(set(inferred_names) & set(condition_names))
    if overlap:
        raise ValueError(
            "Parameters cannot be both inferred and conditioned: "
            + ", ".join(overlap)
        )
    if not inferred_names:
        raise ValueError("SBI requires at least one inferred parameter.")
    _validate_continuous_sbi_targets(problem.parameters, inferred_names)
    if isinstance(method, (MAF, MDN)) and np.any(problem.data.active_upper_limit_mask):
        raise NotImplementedError(
            "MAF does not yet encode censored upper limits, and neither does MDN. "
            "Use Diffusion or a detections-only Problem."
        )

    effective_simulation = replace(
        simulation,
        infer=inferred_names,
        condition_on=condition_names,
    )
    training_set = simulate_sbi_training_set(
        problem,
        effective_simulation,
        rng=seed,
    )
    trained = train_sbi(training_set, method, seed=seed)
    samples = _sample_problem_posterior(
        problem,
        trained,
        method,
        conditions=canonical_conditions,
        seed=seed,
    )
    samples, reported_names, fixed_names, marginalized_names = _reported_problem_sbi_samples(
        samples,
        problem=problem,
        inferred_names=training_set.theta_names,
        conditions=canonical_conditions,
    )
    if isinstance(method, Diffusion):
        observation_names = training_set.diffusion_feature_metadata.names[
            : training_set.diffusion_observation_size
        ]
        observation_groups = training_set.diffusion_observation_groups
    else:
        observation_names = training_set.x_names
        observation_groups = training_set.observation_groups

    return InferenceResult(
        samples=samples,
        logp=None,
        weights=np.ones(samples.shape[0], dtype=float),
        parameter_names=reported_names,
        sampler_name=(
            "maf"
            if isinstance(method, MAF)
            else "mdn"
            if isinstance(method, MDN)
            else "diffusion"
        ),
        metadata={
            "problem": problem.specification(),
            "training_source": training_set.source,
            "training_rows": int(training_set.theta.shape[0]),
            "observation_names": observation_names,
            "observation_groups": observation_groups,
            "feature_transform": training_set.feature_transform_name,
            "photometric_context": (
                None if training_set.context is None else training_set.context.specification()
            ),
            "target_transform": training_set.theta_transform.specification(),
            "conditions": canonical_conditions,
            "conditioned_parameter_names": condition_names,
            "inferred_parameter_names": training_set.theta_names,
            "fixed_parameter_names": fixed_names,
            "marginalized_parameter_names": marginalized_names,
            "device": str(getattr(trained.estimator, "device", "unknown")),
            "inference_batch_size": (
                method.inference_batch_size
                if isinstance(method, (MAF, MDN))
                else method.sample_batch_size
            ),
            "history": trained.history,
            "logp_available": False,
            "map_available": False,
            "seed": seed,
        },
        inference_state=trained,
    )


def _reported_problem_sbi_samples(
    samples: np.ndarray,
    *,
    problem,
    inferred_names: Sequence[str],
    conditions: Mapping[str, float],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Restore deterministic columns without fabricating marginalized values."""

    samples = np.asarray(samples, dtype=float)
    inferred_names = tuple(str(name) for name in inferred_names)
    if samples.ndim != 2 or samples.shape[1] != len(inferred_names):
        raise ValueError(
            "SBI posterior samples do not match the trained inferred-parameter order."
        )
    inferred_index = {name: index for index, name in enumerate(inferred_names)}
    condition_values = {str(name): float(value) for name, value in conditions.items()}
    fixed_values = {
        name: float(prior.value)
        for name, prior in problem.parameters.priors.items()
        if isinstance(prior, DeltaPrior)
        and name not in inferred_index
        and name not in condition_values
    }

    reported_names = tuple(
        name
        for name in problem.parameters.names
        if name in inferred_index or name in condition_values or name in fixed_values
    )
    marginalized_names = tuple(
        name for name in problem.parameters.names if name not in set(reported_names)
    )
    reported = np.empty((samples.shape[0], len(reported_names)), dtype=float)
    for column, name in enumerate(reported_names):
        if name in inferred_index:
            reported[:, column] = samples[:, inferred_index[name]]
        elif name in condition_values:
            reported[:, column] = condition_values[name]
        else:
            reported[:, column] = fixed_values[name]
    return reported, reported_names, tuple(fixed_values), marginalized_names


def _train_maf(training_set: SBITrainingSet, method: MAF, *, seed: int | None) -> TrainedMAFSBI:
    if training_set.has_photometric_state:
        has_missing = not np.all(training_set.measurement_mask_native)
        has_limit_depth = np.any(np.isfinite(training_set.upper_limit_native))
        if has_missing or has_limit_depth:
            raise NotImplementedError(
                "Stable MAF does not yet consume availability or upper-limit state channels. "
                "Use Diffusion for censored or row-masked photometry."
            )
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


def _train_mdn(training_set: SBITrainingSet, method: MDN, *, seed: int | None) -> TrainedMDNSBI:
    if training_set.has_photometric_state:
        has_missing = not np.all(training_set.measurement_mask_native)
        has_limit_depth = np.any(np.isfinite(training_set.upper_limit_native))
        if has_missing or has_limit_depth:
            raise NotImplementedError(
                "Stable MDN does not yet consume availability or upper-limit state channels. "
                "Use Diffusion for censored or row-masked photometry."
            )
    target_transform = (
        training_set.theta_transform
        or PriorSupportTransform.identity(training_set.theta_names)
    )
    unconstrained_theta = target_transform.transform(training_set.theta)
    estimator, metadata = train_mdn_posterior_from_dataset(
        unconstrained_theta,
        training_set.x,
        theta_names=training_set.theta_names,
        x_names=training_set.x_names,
        source=training_set.source,
        finite="raise",
        shuffle=False,
        return_metadata=True,
        n_components=method.n_components,
        hidden_features=method.hidden_features,
        num_blocks=method.num_blocks,
        min_scale=method.min_scale,
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
    return TrainedMDNSBI(
        estimator=estimator,
        training_set=training_set,
        history=history,
        metadata={
            **metadata,
            "training_source": training_set.source,
            "training_set_metadata": _checkpoint_training_metadata(
                training_set.metadata
            ),
        },
        target_transform=target_transform,
    )


def _train_diffusion(
    training_set: SBITrainingSet,
    method: Diffusion,
    *,
    seed: int | None,
) -> TrainedDiffusionSBI:
    feature_metadata = training_set.diffusion_feature_metadata
    joint_features = training_set.diffusion_joint_features
    estimator = ConditionalDiffusionEstimator(
        feature_metadata,
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
        or default_diffusion_mask(
            observation_groups=tuple(training_set.diffusion_observation_groups)
        )
    )
    if training_set.has_photometric_state and "tie_groups" not in fit_mask_config:
        fit_mask_config["tie_groups"] = tuple(
            training_set.diffusion_observation_groups
        )
    history = estimator.fit(
        joint_features,
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
    method: MAF | MDN | Diffusion,
    *,
    conditions: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    requested = int(method.num_samples)
    if requested <= 0:
        raise ValueError("SBI num_samples must be positive.")

    if isinstance(method, (MAF, MDN)):
        cube = trained.sample(
            problem.data,
            conditions=conditions,
            num_samples=requested,
            batch_size=method.inference_batch_size,
            seed=seed,
        )
        draws = np.asarray(cube[0], dtype=float)
        valid = _samples_within_declared_priors(draws, problem.parameters, trained.theta_names)
        if not np.all(valid):
            raise FloatingPointError(
                "Bounded neural target transform produced samples outside declared prior support."
            )
        return draws

    cube = trained.sample(
        problem.data,
        conditions=conditions,
        num_samples=requested,
        steps=method.steps,
        sampler=method.sampler,
        batch_size=method.sample_batch_size,
    )
    draws = np.asarray(cube[0], dtype=float)
    valid = _samples_within_declared_priors(
        draws, problem.parameters, trained.theta_names
    )
    if not np.all(valid):
        raise FloatingPointError(
            "Diffusion prior-support transform produced non-finite or out-of-support "
            "physical samples."
        )
    return draws


def _validate_continuous_sbi_targets(parameter_space: ParameterSpace, theta_names: Sequence[str]) -> None:
    unsupported = []
    for name in theta_names:
        prior_name = type(parameter_space.priors[name]).__name__
        if prior_name in {"DeltaPrior", "ChoicePrior", "IntegerUniformPrior"}:
            unsupported.append(f"{name} ({prior_name})")
    if unsupported:
        raise ValueError(
            "MAF, MDN, and diffusion currently require continuous inferred parameters; unsupported: "
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
        "condition_names": list(training_set.condition_names),
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
    condition_names = tuple(str(name) for name in schema.get("condition_names", ()))
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("MAF checkpoint condition_names must be unique.")
    if condition_names:
        expected = tuple(f"condition:{name}" for name in condition_names)
        x_names = tuple(str(name) for name in schema["x_names"])
        if x_names[-len(expected) :] != expected:
            raise ValueError(
                "MAF checkpoint condition columns do not match condition_names."
            )
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


def _ordered_parameter_subset(
    parameter_names: Sequence[str],
    selected: Sequence[str] | None,
    *,
    label: str,
) -> tuple[str, ...]:
    """Validate names and return them in canonical model-parameter order."""

    parameter_names = tuple(str(name) for name in parameter_names)
    if selected is None:
        return ()
    requested = tuple(str(name) for name in selected)
    if len(set(requested)) != len(requested):
        raise ValueError(f"{label} parameter names must be unique.")
    unknown = sorted(set(requested) - set(parameter_names))
    if unknown:
        raise ValueError(
            f"{label} contains unknown parameter(s): " + ", ".join(unknown)
        )
    requested_set = set(requested)
    return tuple(name for name in parameter_names if name in requested_set)


def _condition_matrix(
    values,
    *,
    condition_names: Sequence[str],
    n_object: int,
) -> np.ndarray:
    """Return finite condition values with shape ``(n_object, n_condition)``."""

    names = tuple(str(name) for name in condition_names)
    n_object = int(n_object)
    if not names:
        if isinstance(values, Mapping) and not values:
            values = None
        if values is not None:
            supplied = np.asarray(values)
            if supplied.size:
                raise ValueError(
                    "This SBI estimator was not trained with condition variables."
                )
        return np.empty((n_object, 0), dtype=float)
    if values is None:
        raise ValueError(
            "This SBI estimator requires condition values for: "
            + ", ".join(names)
        )

    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        unknown = sorted(set(values) - set(names))
        if missing:
            raise ValueError(
                "Missing SBI condition value(s): " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "Unknown SBI condition value(s): " + ", ".join(unknown)
            )
        columns = []
        for name in names:
            column = np.asarray(values[name], dtype=float)
            if column.ndim == 0:
                column = np.full(n_object, float(column), dtype=float)
            elif column.shape == (1,):
                column = np.full(n_object, float(column[0]), dtype=float)
            elif column.shape != (n_object,):
                raise ValueError(
                    f"Condition {name!r} must be scalar or have shape "
                    f"{(n_object,)}; got {column.shape}."
                )
            columns.append(column)
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 0 and len(names) == 1:
            matrix = np.full((n_object, 1), float(matrix), dtype=float)
        elif matrix.ndim == 1:
            if matrix.shape != (len(names),):
                raise ValueError(
                    f"Condition vector must have shape {(len(names),)}; "
                    f"got {matrix.shape}."
                )
            matrix = np.broadcast_to(matrix[None, :], (n_object, len(names))).copy()
        elif matrix.shape == (1, len(names)):
            matrix = np.broadcast_to(matrix, (n_object, len(names))).copy()
        elif matrix.shape != (n_object, len(names)):
            raise ValueError(
                "Condition matrix must have shape "
                f"{(n_object, len(names))}; got {matrix.shape}."
            )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("SBI condition values must be finite.")
    return np.asarray(matrix, dtype=float)


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


def _as_2d_allow_nonfinite(
    values: np.ndarray,
    *,
    expected_cols: int,
    name: str,
) -> np.ndarray:
    """Coerce native observations whose missing/censored entries may be NaN."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != int(expected_cols):
        raise ValueError(
            f"{name} must have shape ({expected_cols},) or "
            f"(n, {expected_cols}); got {arr.shape}."
        )
    return arr


def _broadcast_native_state(
    values,
    *,
    shape: tuple[int, int],
    name: str,
    dtype,
    required: bool = False,
    default=None,
) -> np.ndarray:
    """Broadcast one native state row across a catalog when requested."""

    if values is None:
        if required:
            raise ValueError(f"State-aware native diffusion input requires {name}.")
        return np.full(shape, default, dtype=dtype)
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape == (1, shape[1]) and shape[0] > 1:
        array = np.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, {(shape[1],)}, or one row "
            f"{(1, shape[1])}; got {array.shape}."
        )
    return np.asarray(array, dtype=dtype)
