"""Named, auditable star-formation-history models for production backends.

The classes in this module translate scalar fit parameters into a canonical
tabular history: time since star-formation onset in Gyr and SFR in solar masses
per year. They do not call FSPS or CIGALE and they do not own priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Mapping, Sequence

import numpy as np

from composed._numerics import trapezoid
from composed.errors import ModelDomainError
from composed.transforms.sfh import normalize_sfh_to_formed_mass


AgeKind = Literal["gyr", "fraction_of_universe"]


@dataclass(frozen=True)
class SFHHistory:
    """Validated tabular SFH on an increasing time-since-onset grid."""

    time_gyr: np.ndarray
    sfr_msun_per_yr: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time_gyr, dtype=float)
        sfr = np.asarray(self.sfr_msun_per_yr, dtype=float)
        _validate_history_arrays(time, sfr)
        object.__setattr__(self, "time_gyr", time)
        object.__setattr__(self, "sfr_msun_per_yr", sfr)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def formed_mass_msun(self) -> float:
        """Mass formed by integrating the tabulated SFR."""

        return _trapz(self.sfr_msun_per_yr, self.time_gyr) * 1.0e9


class SFHModel:
    """Base contract for a named SFH parameterization.

    Every named model produces the same canonical history for every backend:
    increasing time since star-formation onset in Gyr and SFR in Msun/yr.
    Backend-specific code is responsible only for projecting that history onto
    the population-synthesis engine's native time representation.
    """

    name: ClassVar[str] = "sfh"
    supported_backends: ClassVar[tuple[str, ...]] = ()

    @property
    def required_parameters(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def requires_redshift(self) -> bool:
        return False

    def evaluate(
        self,
        params: Mapping[str, object],
        *,
        redshift: float | None = None,
        cosmology=None,
    ) -> SFHHistory:
        raise NotImplementedError

    def specification(self) -> dict[str, object]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "required_parameters": self.required_parameters,
            "supported_backends": self.supported_backends,
        }


@dataclass(frozen=True)
class _AgeParameterizedSFH(SFHModel):
    age: str = "tage_gyr"
    age_kind: AgeKind = "gyr"
    n_time: int = 256

    def __post_init__(self) -> None:
        if not str(self.age):
            raise ValueError("SFH age parameter name must be non-empty.")
        if self.age_kind not in {"gyr", "fraction_of_universe"}:
            raise ValueError("age_kind must be 'gyr' or 'fraction_of_universe'.")
        if int(self.n_time) < 2:
            raise ValueError("n_time must be at least 2.")

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (str(self.age),)

    @property
    def requires_redshift(self) -> bool:
        return self.age_kind == "fraction_of_universe"

    def _age_gyr(self, params, *, redshift, cosmology) -> float:
        return _resolve_age_gyr(
            params,
            parameter=self.age,
            age_kind=self.age_kind,
            redshift=redshift,
            cosmology=cosmology,
        )

    def _time_grid(self, age_gyr: float) -> np.ndarray:
        return np.linspace(0.0, age_gyr, int(self.n_time), dtype=float)

    def specification(self) -> dict[str, object]:
        return {
            **super().specification(),
            "age_parameter": self.age,
            "age_kind": self.age_kind,
            "n_time": int(self.n_time),
        }


@dataclass(frozen=True)
class ConstantSFH(_AgeParameterizedSFH):
    """Constant SFR from formation onset to the fitted galaxy age."""

    name: ClassVar[str] = "constant"
    supported_backends: ClassVar[tuple[str, ...]] = ("fsps", "cigale")

    def evaluate(self, params, *, redshift=None, cosmology=None) -> SFHHistory:
        age_gyr = self._age_gyr(params, redshift=redshift, cosmology=cosmology)
        time = self._time_grid(age_gyr)
        return _normalized_history(
            time,
            np.ones_like(time),
            {"model": self.name, "age_gyr": age_gyr},
        )


@dataclass(frozen=True)
class _TauSFH(_AgeParameterizedSFH):
    tau: str = "tau_gyr"
    mode: ClassVar[str] = "tau"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not str(self.tau):
            raise ValueError("SFH tau parameter name must be non-empty.")

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (str(self.age), str(self.tau))

    def _tau_gyr(self, params: Mapping[str, object]) -> float:
        tau = _finite_parameter(params, self.tau)
        if tau <= 0.0:
            raise ModelDomainError(f"SFH parameter {self.tau!r} must be positive.")
        return tau

    def evaluate(self, params, *, redshift=None, cosmology=None) -> SFHHistory:
        age_gyr = self._age_gyr(params, redshift=redshift, cosmology=cosmology)
        tau_gyr = self._tau_gyr(params)
        time = self._time_grid(age_gyr)
        if self.mode == "delayed_tau":
            raw_sfr = time * np.exp(-time / tau_gyr)
        elif self.mode == "exponential":
            raw_sfr = np.exp(-time / tau_gyr)
        else:  # pragma: no cover - fixed by public subclasses.
            raise ValueError(f"Unsupported parametric SFH mode {self.mode!r}.")
        return _normalized_history(
            time,
            raw_sfr,
            {"model": self.name, "age_gyr": age_gyr, "tau_gyr": tau_gyr},
        )

    def specification(self) -> dict[str, object]:
        return {**super().specification(), "tau_parameter": self.tau}


@dataclass(frozen=True)
class DelayedTauSFH(_TauSFH):
    """Delayed-tau history, ``SFR(t) proportional to t exp(-t/tau)``."""

    name: ClassVar[str] = "delayed_tau"
    mode: ClassVar[str] = "delayed_tau"
    supported_backends: ClassVar[tuple[str, ...]] = ("fsps", "cigale")


@dataclass(frozen=True)
class ExponentialSFH(_TauSFH):
    """Exponentially declining history, ``SFR(t) proportional to exp(-t/tau)``."""

    name: ClassVar[str] = "exponential"
    mode: ClassVar[str] = "exponential"
    supported_backends: ClassVar[tuple[str, ...]] = ("fsps", "cigale")


@dataclass(frozen=True)
class ContinuitySFH(SFHModel):
    """Piecewise SFH parameterized by adjacent log10 SFR ratios.

    ``lookback_edges_gyr`` starts at zero and lists fixed recent-bin edges. The
    fitted galaxy age is appended as the oldest edge. Parameters
    ``logsfr_ratio_0`` ... are ordered recent-to-old and mean
    ``log10(SFR_recent_bin / SFR_next_older_bin)``.
    """

    age: str = "tage_gyr"
    age_kind: AgeKind = "gyr"
    lookback_edges_gyr: Sequence[float] = (0.0, 0.03, 0.1, 0.3, 1.0)
    ratio_prefix: str = "logsfr_ratio"
    samples_per_bin: int = 8
    boundary_epsilon_gyr: float = 1.0e-5
    name: ClassVar[str] = "continuity"
    supported_backends: ClassVar[tuple[str, ...]] = ("fsps", "cigale")

    def __post_init__(self) -> None:
        if not str(self.age):
            raise ValueError("SFH age parameter name must be non-empty.")
        if self.age_kind not in {"gyr", "fraction_of_universe"}:
            raise ValueError("age_kind must be 'gyr' or 'fraction_of_universe'.")
        edges = tuple(float(value) for value in self.lookback_edges_gyr)
        if len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError("lookback_edges_gyr must be finite and strictly increasing.")
        if not np.isclose(edges[0], 0.0):
            raise ValueError("lookback_edges_gyr must start at 0.0 Gyr.")
        if not str(self.ratio_prefix):
            raise ValueError("ratio_prefix must be non-empty.")
        if int(self.samples_per_bin) < 2:
            raise ValueError("samples_per_bin must be at least 2.")
        if not np.isfinite(self.boundary_epsilon_gyr) or self.boundary_epsilon_gyr <= 0.0:
            raise ValueError("boundary_epsilon_gyr must be finite and positive.")
        object.__setattr__(self, "lookback_edges_gyr", edges)

    @property
    def ratio_names(self) -> tuple[str, ...]:
        return tuple(f"{self.ratio_prefix}_{i}" for i in range(len(self.lookback_edges_gyr) - 1))

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (str(self.age), *self.ratio_names)

    @property
    def requires_redshift(self) -> bool:
        return self.age_kind == "fraction_of_universe"

    def evaluate(self, params, *, redshift=None, cosmology=None) -> SFHHistory:
        age_gyr = _resolve_age_gyr(
            params,
            parameter=self.age,
            age_kind=self.age_kind,
            redshift=redshift,
            cosmology=cosmology,
        )
        fixed_edges = np.asarray(self.lookback_edges_gyr, dtype=float)
        if age_gyr <= fixed_edges[-1]:
            raise ModelDomainError(
                f"Continuity SFH age {age_gyr:.6g} Gyr must exceed the oldest fixed "
                f"lookback edge {fixed_edges[-1]:.6g} Gyr."
            )
        lookback_edges = np.concatenate([fixed_edges, [age_gyr]])
        ratios = np.asarray([_finite_parameter(params, name) for name in self.ratio_names], dtype=float)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sfr_recent_to_old = np.ones(lookback_edges.size - 1, dtype=float)
            for i, ratio in enumerate(ratios):
                sfr_recent_to_old[i + 1] = sfr_recent_to_old[i] / (10.0**ratio)
        if not np.all(np.isfinite(sfr_recent_to_old)) or np.any(sfr_recent_to_old <= 0.0):
            raise ModelDomainError("Continuity SFH ratios produced a non-finite or non-positive SFR.")

        time_parts = []
        sfr_parts = []
        n_bins = sfr_recent_to_old.size
        for recent_bin in range(n_bins - 1, -1, -1):
            start = age_gyr - lookback_edges[recent_bin + 1]
            stop = age_gyr - lookback_edges[recent_bin]
            is_most_recent = recent_bin == 0
            if is_most_recent:
                times = np.linspace(start, stop, int(self.samples_per_bin), endpoint=True)
            else:
                # FSPS linearly interpolates tabular SFHs. Put the final point
                # just below the boundary so adjacent constant bins do not
                # become broad artificial ramps. A float64 nextafter step is
                # lost when python-fsps copies times into its lower-precision
                # work arrays, so retain a small but explicit physical width.
                bin_width = stop - start
                epsilon = min(float(self.boundary_epsilon_gyr), 0.01 * bin_width)
                just_below_stop = stop - epsilon
                times = np.linspace(start, just_below_stop, int(self.samples_per_bin), endpoint=True)
            time_parts.append(times)
            sfr_parts.append(np.full(times.shape, sfr_recent_to_old[recent_bin], dtype=float))

        time = np.concatenate(time_parts)
        raw_sfr = np.concatenate(sfr_parts)
        return _normalized_history(
            time,
            raw_sfr,
            {
                "model": self.name,
                "age_gyr": age_gyr,
                "lookback_edges_gyr": lookback_edges.tolist(),
                "bin_sfr_recent_to_old": sfr_recent_to_old.tolist(),
                "ratio_definition": "log10(SFR_recent/SFR_next_older)",
            },
        )

    def specification(self) -> dict[str, object]:
        return {
            **super().specification(),
            "age_parameter": self.age,
            "age_kind": self.age_kind,
            "lookback_edges_gyr": self.lookback_edges_gyr,
            "ratio_prefix": self.ratio_prefix,
            "samples_per_bin": int(self.samples_per_bin),
            "boundary_epsilon_gyr": float(self.boundary_epsilon_gyr),
        }


@dataclass(frozen=True)
class TabularSFH(SFHModel):
    """Read an arbitrary tabular SFH from two named backend parameters."""

    time: str = "tabular_time_gyr"
    sfr: str = "tabular_sfr_msun_per_yr"
    name: ClassVar[str] = "tabular"
    supported_backends: ClassVar[tuple[str, ...]] = ("fsps", "cigale")

    def __post_init__(self) -> None:
        if not str(self.time) or not str(self.sfr):
            raise ValueError("Tabular SFH parameter names must be non-empty.")
        if self.time == self.sfr:
            raise ValueError("Tabular SFH time and SFR parameter names must differ.")

    @property
    def required_parameters(self) -> tuple[str, ...]:
        return (str(self.time), str(self.sfr))

    def evaluate(self, params, *, redshift=None, cosmology=None) -> SFHHistory:
        del redshift, cosmology
        if self.time not in params or self.sfr not in params:
            raise KeyError(f"TabularSFH requires parameters {self.time!r} and {self.sfr!r}.")
        history = SFHHistory(
            time_gyr=np.asarray(params[self.time], dtype=float),
            sfr_msun_per_yr=np.asarray(params[self.sfr], dtype=float),
            metadata={"model": self.name},
        )
        return SFHHistory(
            history.time_gyr,
            history.sfr_msun_per_yr,
            {**history.metadata, "input_formed_mass_msun": history.formed_mass_msun},
        )

    def specification(self) -> dict[str, object]:
        return {
            **super().specification(),
            "time_parameter": self.time,
            "sfr_parameter": self.sfr,
        }


_SFH_MODELS = {
    "constant": ConstantSFH,
    "exponential": ExponentialSFH,
    "delayed_tau": DelayedTauSFH,
    "continuity": ContinuitySFH,
    "tabular": TabularSFH,
}
_SFH_ALIASES = {
    "delayed": "delayed_tau",
    "delayed-tau": "delayed_tau",
    "exp": "exponential",
}


def available_sfh_models(backend: str | None = None) -> tuple[str, ...]:
    """List stable named SFHs, optionally filtered by backend support."""

    if backend is None:
        return tuple(_SFH_MODELS)
    backend_name = str(backend).strip().lower()
    if backend_name not in {"fsps", "cigale"}:
        raise ValueError("backend must be 'fsps', 'cigale', or None.")
    return tuple(name for name, cls in _SFH_MODELS.items() if backend_name in cls.supported_backends)


def make_sfh(name: str, **kwargs) -> SFHModel:
    """Construct one named SFH model."""

    key = str(name).strip().lower()
    key = _SFH_ALIASES.get(key, key)
    try:
        model_class = _SFH_MODELS[key]
    except KeyError as exc:
        choices = ", ".join(_SFH_MODELS)
        raise ValueError(f"Unknown SFH model {name!r}; choose one of: {choices}.") from exc
    return model_class(**kwargs)


def coerce_sfh_model(sfh: SFHModel | str | None, *, backend: str) -> SFHModel | None:
    """Resolve a user SFH declaration and enforce backend compatibility."""

    if sfh is None:
        return None
    model = make_sfh(sfh) if isinstance(sfh, str) else sfh
    if not isinstance(model, SFHModel):
        raise TypeError("sfh must be a named string or composed.sfh.SFHModel instance.")
    backend_name = str(backend).strip().lower()
    if backend_name not in model.supported_backends:
        supported = ", ".join(model.supported_backends) or "none"
        available = ", ".join(available_sfh_models(backend_name))
        raise ValueError(
            f"SFH model {model.name!r} does not support backend {backend_name!r}; "
            f"supported backends: {supported}. Available {backend_name} models: {available}."
        )
    return model


def _normalized_history(time: np.ndarray, raw_sfr: np.ndarray, metadata: Mapping[str, object]) -> SFHHistory:
    _validate_history_arrays(np.asarray(time, dtype=float), np.asarray(raw_sfr, dtype=float))
    normalized = normalize_sfh_to_formed_mass(time, raw_sfr)
    history = SFHHistory(time, normalized, metadata)
    if not np.isclose(history.formed_mass_msun, 1.0, rtol=1e-10, atol=1e-10):
        raise FloatingPointError("Normalized SFH does not integrate to one solar mass formed.")
    return history


def _resolve_age_gyr(
    params: Mapping[str, object],
    *,
    parameter: str,
    age_kind: AgeKind,
    redshift: float | None,
    cosmology,
) -> float:
    raw_age = _finite_parameter(params, parameter)
    if age_kind == "fraction_of_universe":
        if not 0.0 < raw_age <= 1.0:
            raise ModelDomainError(f"SFH age fraction {parameter!r} must lie in (0, 1].")
        universe_age = _universe_age_gyr(redshift, cosmology)
        age_gyr = raw_age * universe_age
    else:
        age_gyr = raw_age
        universe_age = None if redshift is None else _universe_age_gyr(redshift, cosmology)
    if not np.isfinite(age_gyr) or age_gyr <= 0.0:
        raise ModelDomainError(f"SFH age parameter {parameter!r} must produce a positive finite age.")
    if universe_age is not None and age_gyr > universe_age + 1.0e-10:
        raise ModelDomainError(
            f"SFH age {age_gyr:.6g} Gyr exceeds the Universe age "
            f"{universe_age:.6g} Gyr at z={float(redshift):.6g}."
        )
    return float(age_gyr)


def _universe_age_gyr(redshift: float | None, cosmology) -> float:
    if redshift is None:
        raise ValueError("This SFH age convention requires a redshift.")
    z = float(redshift)
    if not np.isfinite(z) or z < 0.0:
        raise ModelDomainError("SFH redshift must be finite and non-negative.")
    if cosmology is None:
        from astropy.cosmology import Planck18

        cosmology = Planck18
    age = cosmology.age(z)
    if hasattr(age, "to"):
        age = age.to("Gyr")
    value = float(getattr(age, "value", age))
    if not np.isfinite(value) or value <= 0.0:
        raise FloatingPointError(f"Cosmology returned invalid Universe age {value!r} at z={z}.")
    return value


def _finite_parameter(params: Mapping[str, object], name: str) -> float:
    if name not in params:
        raise KeyError(f"Missing SFH parameter {name!r}.")
    value = np.asarray(params[name])
    if value.ndim != 0:
        raise ValueError(f"SFH parameter {name!r} must be scalar; got shape {value.shape}.")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ModelDomainError(f"SFH parameter {name!r} must be finite.")
    return scalar


def _validate_history_arrays(time: np.ndarray, sfr: np.ndarray) -> None:
    if time.ndim != 1 or sfr.ndim != 1 or time.shape != sfr.shape:
        raise ValueError("SFH time and SFR must be matching one-dimensional arrays.")
    if time.size < 2:
        raise ValueError("SFH history must contain at least two time points.")
    if not np.all(np.isfinite(time)) or np.any(time < 0.0) or np.any(np.diff(time) <= 0.0):
        raise ModelDomainError("SFH time must be finite, non-negative, and strictly increasing.")
    if not np.all(np.isfinite(sfr)) or np.any(sfr < 0.0):
        raise ModelDomainError("SFH SFR must be finite and non-negative.")
    formed_mass = _trapz(sfr, time) * 1.0e9
    if not np.isfinite(formed_mass) or formed_mass <= 0.0:
        raise ModelDomainError("SFH must form a positive finite stellar mass.")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(trapezoid(y, x))


__all__ = [
    "ConstantSFH",
    "ContinuitySFH",
    "DelayedTauSFH",
    "ExponentialSFH",
    "SFHHistory",
    "SFHModel",
    "TabularSFH",
    "available_sfh_models",
    "make_sfh",
]
