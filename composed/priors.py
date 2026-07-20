from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


class Prior:
    """Scalar prior interface used by ``ParameterSpace``."""

    def logpdf(self, x: float) -> float:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, size=None):
        raise NotImplementedError


@dataclass(frozen=True)
class UniformPrior(Prior):
    """Uniform prior on the closed interval ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError("UniformPrior requires high > low.")

    def logpdf(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return -np.inf
        if x < self.low or x > self.high:
            return -np.inf
        return float(-np.log(self.high - self.low))

    def sample(self, rng: np.random.Generator, size=None):
        return rng.uniform(self.low, self.high, size=size)


@dataclass(frozen=True)
class NormalPrior(Prior):
    """Gaussian prior with mean ``mu`` and standard deviation ``sigma``."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not self.sigma > 0:
            raise ValueError("NormalPrior requires sigma > 0.")

    def logpdf(self, x: float) -> float:
        if not np.isfinite(x):
            return -np.inf
        z = (float(x) - self.mu) / self.sigma
        return float(-0.5 * z**2 - np.log(self.sigma) - 0.5 * np.log(2.0 * np.pi))

    def sample(self, rng: np.random.Generator, size=None):
        return rng.normal(self.mu, self.sigma, size=size)


@dataclass(frozen=True)
class StudentTPrior(Prior):
    """Student-t prior with degrees of freedom, location, and scale.

    The standard continuity-SFH prior is represented by
    ``StudentTPrior(df=2, loc=0, scale=0.3)`` on adjacent log10 SFR ratios.
    """

    df: float
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.df) or self.df <= 0.0:
            raise ValueError("StudentTPrior requires finite df > 0.")
        if not np.isfinite(self.loc):
            raise ValueError("StudentTPrior requires a finite location.")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("StudentTPrior requires finite scale > 0.")

    def logpdf(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return -np.inf
        z = (x - self.loc) / self.scale
        log_normalization = (
            math.lgamma(0.5 * (self.df + 1.0))
            - math.lgamma(0.5 * self.df)
            - 0.5 * np.log(self.df * np.pi)
            - np.log(self.scale)
        )
        return float(log_normalization - 0.5 * (self.df + 1.0) * np.log1p(z**2 / self.df))

    def sample(self, rng: np.random.Generator, size=None):
        return self.loc + self.scale * rng.standard_t(self.df, size=size)


@dataclass(frozen=True)
class LogUniformPrior(Prior):
    """Log-uniform prior on the closed positive interval ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low > 0 or not self.high > self.low:
            raise ValueError("LogUniformPrior requires 0 < low < high.")

    def logpdf(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return -np.inf
        if x <= 0 or x < self.low or x > self.high:
            return -np.inf
        return float(-np.log(np.log(self.high) - np.log(self.low)) - np.log(x))

    def sample(self, rng: np.random.Generator, size=None):
        return np.exp(rng.uniform(np.log(self.low), np.log(self.high), size=size))


@dataclass(frozen=True)
class IntegerUniformPrior(Prior):
    """Discrete uniform prior on the inclusive integer interval ``[low, high]``."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if not int(self.high) >= int(self.low):
            raise ValueError("IntegerUniformPrior requires high >= low.")
        object.__setattr__(self, "low", int(self.low))
        object.__setattr__(self, "high", int(self.high))

    def logpdf(self, x: float) -> float:
        if not np.isfinite(x):
            return -np.inf
        rounded = int(round(float(x)))
        if not np.isclose(float(x), rounded, rtol=0.0, atol=1e-12):
            return -np.inf
        if rounded < self.low or rounded > self.high:
            return -np.inf
        return float(-np.log(self.high - self.low + 1))

    def sample(self, rng: np.random.Generator, size=None):
        return rng.integers(self.low, self.high + 1, size=size).astype(float)


@dataclass(frozen=True)
class ChoicePrior(Prior):
    """Discrete uniform prior over a finite set of numeric values."""

    values: Sequence[float]
    rtol: float = 1e-12
    atol: float = 1e-12

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("ChoicePrior requires a non-empty one-dimensional value list.")
        if not np.all(np.isfinite(values)):
            raise ValueError("ChoicePrior values must be finite.")
        if np.unique(values).size != values.size:
            raise ValueError("ChoicePrior values must be unique.")
        object.__setattr__(self, "values", tuple(float(v) for v in values))

    def logpdf(self, x: float) -> float:
        if not np.isfinite(x):
            return -np.inf
        values = np.asarray(self.values, dtype=float)
        match = np.isclose(float(x), values, rtol=self.rtol, atol=self.atol)
        return float(-np.log(values.size)) if np.any(match) else -np.inf

    def sample(self, rng: np.random.Generator, size=None):
        values = np.asarray(self.values, dtype=float)
        return rng.choice(values, size=size)


@dataclass(frozen=True)
class DeltaPrior(Prior):
    """Point-mass prior at ``value`` with configurable numerical tolerance."""

    value: float
    rtol: float = 1e-12
    atol: float = 1e-12

    def logpdf(self, x: float) -> float:
        if not np.isfinite(x):
            return -np.inf
        return 0.0 if np.isclose(float(x), self.value, rtol=self.rtol, atol=self.atol) else -np.inf

    def sample(self, rng: np.random.Generator, size=None):
        del rng
        if size is None:
            return float(self.value)
        return np.full(size, self.value, dtype=float)
