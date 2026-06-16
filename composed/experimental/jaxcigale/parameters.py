from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from composed.experimental.jaxcigale.dependencies import require_jax


@dataclass(frozen=True)
class UniformJaxPrior:
    """Continuous uniform prior on ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not float(self.high) > float(self.low):
            raise ValueError("UniformJaxPrior requires high > low.")

    def logpdf(self, x):
        _, jnp = require_jax()
        x = jnp.asarray(x)
        inside = (x >= self.low) & (x <= self.high) & jnp.isfinite(x)
        return jnp.where(inside, -jnp.log(self.high - self.low), -jnp.inf)

    def sample(self, rng: np.random.Generator, size=None):
        return rng.uniform(self.low, self.high, size=size)

    @property
    def bounds(self) -> tuple[float, float]:
        return (float(self.low), float(self.high))


@dataclass(frozen=True)
class NormalJaxPrior:
    """Gaussian prior with mean ``mu`` and standard deviation ``sigma``."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not float(self.sigma) > 0.0:
            raise ValueError("NormalJaxPrior requires sigma > 0.")

    def logpdf(self, x):
        _, jnp = require_jax()
        x = jnp.asarray(x)
        z = (x - self.mu) / self.sigma
        return jnp.where(
            jnp.isfinite(x),
            -0.5 * z**2 - jnp.log(self.sigma) - 0.5 * jnp.log(2.0 * jnp.pi),
            -jnp.inf,
        )

    def sample(self, rng: np.random.Generator, size=None):
        return rng.normal(self.mu, self.sigma, size=size)


@dataclass(frozen=True)
class LogUniformJaxPrior:
    """Log-uniform prior on the positive interval ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not float(self.low) > 0.0 or not float(self.high) > float(self.low):
            raise ValueError("LogUniformJaxPrior requires 0 < low < high.")

    def logpdf(self, x):
        _, jnp = require_jax()
        x = jnp.asarray(x)
        inside = (x >= self.low) & (x <= self.high) & jnp.isfinite(x)
        norm = jnp.log(jnp.log(self.high) - jnp.log(self.low))
        return jnp.where(inside, -norm - jnp.log(x), -jnp.inf)

    def sample(self, rng: np.random.Generator, size=None):
        return np.exp(rng.uniform(np.log(self.low), np.log(self.high), size=size))

    @property
    def bounds(self) -> tuple[float, float]:
        return (float(self.low), float(self.high))


@dataclass(frozen=True)
class JaxParameterSpace:
    """Deterministic vector/dictionary bridge with JAX log-priors."""

    names: Sequence[str]
    priors: Mapping[str, object]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        if len(names) == 0:
            raise ValueError("JaxParameterSpace requires at least one parameter.")
        if len(set(names)) != len(names):
            raise ValueError("JaxParameterSpace names must be unique.")
        missing = [name for name in names if name not in self.priors]
        if missing:
            raise ValueError(f"Missing prior(s) for: {', '.join(missing)}")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "priors", dict(self.priors))

    @property
    def ndim(self) -> int:
        return len(self.names)

    def params_from_theta(self, theta) -> dict[str, object]:
        """Return a static-key dictionary whose values may be JAX tracers."""

        return {name: theta[i] for i, name in enumerate(self.names)}

    def from_dict(self, params: Mapping[str, float]) -> np.ndarray:
        return np.asarray([params[name] for name in self.names], dtype=float)

    def to_dict(self, theta: Sequence[float]) -> dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError(f"theta shape {theta.shape} does not match {(self.ndim,)}.")
        return {name: float(theta[i]) for i, name in enumerate(self.names)}

    def log_prior(self, theta):
        _, jnp = require_jax()
        theta = jnp.asarray(theta)
        total = jnp.asarray(0.0, dtype=theta.dtype)
        for i, name in enumerate(self.names):
            total = total + self.priors[name].logpdf(theta[i])
        return total

    def sample_prior(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        samples = np.empty((int(n), self.ndim), dtype=float)
        for i, name in enumerate(self.names):
            samples[:, i] = self.priors[name].sample(rng, size=int(n))
        return samples

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        out = []
        for name in self.names:
            prior = self.priors[name]
            out.append(getattr(prior, "bounds", (None, None)))
        return out


def jax_parameter_space_from_composed(space) -> JaxParameterSpace:
    """Convert the simple continuous composed priors to JAX priors."""

    from composed.priors import LogUniformPrior, NormalPrior, UniformPrior

    priors = {}
    for name in space.names:
        prior = space.priors[name]
        if isinstance(prior, UniformPrior):
            priors[name] = UniformJaxPrior(prior.low, prior.high)
        elif isinstance(prior, NormalPrior):
            priors[name] = NormalJaxPrior(prior.mu, prior.sigma)
        elif isinstance(prior, LogUniformPrior):
            priors[name] = LogUniformJaxPrior(prior.low, prior.high)
        else:
            raise TypeError(
                f"Cannot convert prior for {name!r} to JAX. "
                "Use UniformPrior, NormalPrior, or LogUniformPrior for differentiable fits."
            )
    return JaxParameterSpace(names=space.names, priors=priors)
