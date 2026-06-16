from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from composed.priors import Prior


@dataclass(frozen=True)
class ParameterSpace:
    """Ordered vector/dictionary bridge for model parameters.

    ``names`` is the canonical parameter order used for theta vectors. The
    mapping in ``priors`` is keyed by those names. The class is intentionally
    deterministic: ``to_dict`` and ``from_dict`` always follow ``names``.
    """

    names: Sequence[str]
    priors: Mapping[str, Prior]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        if len(set(names)) != len(names):
            raise ValueError("ParameterSpace names must be unique.")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "priors", dict(self.priors))

    @property
    def ndim(self) -> int:
        return len(self.names)

    def sample_prior(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` samples in the canonical parameter order."""

        if int(n) < 0:
            raise ValueError("n must be non-negative.")
        if rng is None:
            rng = np.random.default_rng()
        samples = np.empty((int(n), self.ndim), dtype=float)
        for j, name in enumerate(self.names):
            try:
                prior = self.priors[name]
            except KeyError as exc:
                raise KeyError(f"Missing prior for parameter {name!r}.") from exc
            samples[:, j] = prior.sample(rng, size=int(n))
        return samples

    def to_dict(self, theta: Sequence[float]) -> dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError(f"Expected theta shape {(self.ndim,)}, got {theta.shape}.")
        return {name: float(value) for name, value in zip(self.names, theta)}

    def from_dict(self, params: Mapping[str, float]) -> np.ndarray:
        missing = [name for name in self.names if name not in params]
        if missing:
            raise KeyError(f"Missing parameter(s): {', '.join(missing)}")
        return np.asarray([params[name] for name in self.names], dtype=float)

    def log_prior(self, theta: Sequence[float]) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError(f"Expected theta shape {(self.ndim,)}, got {theta.shape}.")
        total = 0.0
        for name, value in zip(self.names, theta):
            prior = self.priors.get(name)
            if prior is None:
                continue
            logp = prior.logpdf(float(value))
            if not np.isfinite(logp):
                return -np.inf
            total += logp
        return float(total)
