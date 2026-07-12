from __future__ import annotations

from dataclasses import dataclass
import inspect
from itertools import product
from typing import Mapping

import numpy as np

from .core import Array, Posterior, SamplingResult


@dataclass(frozen=True)
class DiscreteGrid:
    """Finite Cartesian grid implied by discrete priors in a ParameterSpace.

    The grid stores only genuinely finite-valued parameters, such as CIGALE
    module choices. Continuous parameters are intentionally absent from
    ``points`` so the scientific meaning stays clear: every row is one model
    family / grid choice, not a full theta vector.
    """

    names: tuple[str, ...]
    indices: tuple[int, ...]
    values: tuple[np.ndarray, ...]
    points: np.ndarray

    @property
    def size(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class ParameterBlocks:
    """Index split of a ParameterSpace into continuous, discrete, and fixed axes."""

    continuous_names: tuple[str, ...]
    continuous_indices: tuple[int, ...]
    discrete_names: tuple[str, ...]
    discrete_indices: tuple[int, ...]
    fixed_names: tuple[str, ...]
    fixed_indices: tuple[int, ...]
    fixed_values: tuple[float, ...]


def split_parameter_space(parameter_space) -> ParameterBlocks:
    """Classify ParameterSpace axes by how samplers should move them.

    ``ChoicePrior`` and ``IntegerUniformPrior`` become finite discrete axes.
    ``DeltaPrior`` becomes fixed and is never perturbed by a continuous sampler.
    Everything else is treated as continuous.
    """

    continuous_names: list[str] = []
    continuous_indices: list[int] = []
    discrete_names: list[str] = []
    discrete_indices: list[int] = []
    fixed_names: list[str] = []
    fixed_indices: list[int] = []
    fixed_values: list[float] = []

    for index, name in enumerate(parameter_space.names):
        prior = parameter_space.priors.get(name)
        fixed_value = _fixed_prior_value(prior)
        if fixed_value is not None:
            fixed_names.append(name)
            fixed_indices.append(index)
            fixed_values.append(float(fixed_value))
            continue

        values = finite_prior_values(prior)
        if values is not None:
            discrete_names.append(name)
            discrete_indices.append(index)
            continue

        continuous_names.append(name)
        continuous_indices.append(index)

    return ParameterBlocks(
        continuous_names=tuple(continuous_names),
        continuous_indices=tuple(continuous_indices),
        discrete_names=tuple(discrete_names),
        discrete_indices=tuple(discrete_indices),
        fixed_names=tuple(fixed_names),
        fixed_indices=tuple(fixed_indices),
        fixed_values=tuple(fixed_values),
    )


def finite_prior_values(prior) -> np.ndarray | None:
    """Return finite support values for discrete priors, otherwise ``None``."""

    if prior is None:
        return None

    class_name = type(prior).__name__
    if class_name == "ChoicePrior":
        return np.asarray(prior.values, dtype=float)
    if class_name == "IntegerUniformPrior":
        return np.arange(int(prior.low), int(prior.high) + 1, dtype=float)
    return None


def enumerate_discrete_grid(parameter_space, max_size: int | None = 1_000_000) -> DiscreteGrid:
    """Enumerate the Cartesian product of finite-valued parameters.

    This recovers the usual CIGALE-style grid for parameters specified as
    explicit lists. The returned ``points`` array has shape
    ``(n_grid, n_discrete)`` and follows ``grid.names``.
    """

    names: list[str] = []
    indices: list[int] = []
    values_by_axis: list[np.ndarray] = []

    for index, name in enumerate(parameter_space.names):
        values = finite_prior_values(parameter_space.priors.get(name))
        if values is None:
            continue
        names.append(name)
        indices.append(index)
        values_by_axis.append(values)

    if not values_by_axis:
        points = np.empty((1, 0), dtype=float)
        return DiscreteGrid(names=(), indices=(), values=(), points=points)

    size = int(np.prod([axis.size for axis in values_by_axis], dtype=np.int64))
    if max_size is not None and size > int(max_size):
        raise ValueError(
            f"Discrete grid has {size} points, larger than max_size={max_size}. "
            "Use sample_discrete_grid for a uniform random subset."
        )

    points = np.asarray(list(product(*values_by_axis)), dtype=float)
    return DiscreteGrid(
        names=tuple(names),
        indices=tuple(indices),
        values=tuple(np.asarray(v, dtype=float) for v in values_by_axis),
        points=points,
    )


def sample_discrete_grid(parameter_space, n: int, rng: np.random.Generator | None = None) -> DiscreteGrid:
    """Draw ``n`` configurations uniformly from the finite Cartesian grid."""

    if int(n) < 0:
        raise ValueError("n must be non-negative.")
    if rng is None:
        rng = np.random.default_rng()

    full_grid = enumerate_discrete_grid(parameter_space, max_size=None)
    if full_grid.points.shape[1] == 0:
        points = np.empty((int(n), 0), dtype=float)
        return DiscreteGrid(full_grid.names, full_grid.indices, full_grid.values, points)

    points = np.empty((int(n), len(full_grid.values)), dtype=float)
    for axis, values in enumerate(full_grid.values):
        points[:, axis] = rng.choice(values, size=int(n))
    return DiscreteGrid(full_grid.names, full_grid.indices, full_grid.values, points)


def run_grid_sampler(
    posterior: Posterior,
    parameter_space,
    max_size: int | None = 1_000_000,
) -> SamplingResult:
    """Evaluate the posterior on the full finite CIGALE-style grid.

    This is the deliberately plain "official CIGALE" mode: every non-fixed
    parameter must have finite support, every Cartesian grid point is evaluated
    once, and posterior weights are just normalized exponentiated log
    probabilities. There is no tempering, no adaptation, and no MCMC move.
    """

    blocks = split_parameter_space(parameter_space)
    if blocks.continuous_indices:
        names = ", ".join(blocks.continuous_names)
        raise ValueError(
            "run_grid_sampler only supports finite-valued or fixed parameters. "
            f"Continuous parameter(s) found: {names}."
        )

    grid = enumerate_discrete_grid(parameter_space, max_size=max_size)
    samples = np.empty((grid.size, len(parameter_space.names)), dtype=float)
    logp = np.empty(grid.size, dtype=float)
    for i, discrete_values in enumerate(grid.points):
        samples[i] = full_theta_from_blocks(parameter_space, np.empty(0), discrete_values)
        logp[i] = float(posterior.log_prob_fn(samples[i]))

    if not np.any(np.isfinite(logp)):
        raise RuntimeError("All grid points have non-finite log probability.")

    weights = _probabilities_from_log_weights(logp)
    best = int(np.nanargmax(logp))
    mean = np.average(samples, axis=0, weights=weights)
    diff = samples - mean
    cov = (weights[:, None] * diff).T @ diff
    return SamplingResult(
        samples=samples,
        logp=logp,
        map_estimate=samples[best],
        cov=cov,
        meta={
            "weights_norm": weights,
            "grid": grid,
            "parameter_blocks": blocks,
        },
    )


def full_theta_from_blocks(
    parameter_space,
    continuous_values: Array,
    discrete_values: Array | None = None,
) -> np.ndarray:
    """Build a full theta vector from continuous, discrete, and fixed blocks."""

    blocks = split_parameter_space(parameter_space)
    theta = np.empty(len(parameter_space.names), dtype=float)

    continuous_values = np.asarray(continuous_values, dtype=float)
    if continuous_values.shape != (len(blocks.continuous_indices),):
        raise ValueError(
            "continuous_values has shape "
            f"{continuous_values.shape}, expected {(len(blocks.continuous_indices),)}."
        )
    theta[list(blocks.continuous_indices)] = continuous_values

    if blocks.discrete_indices:
        if discrete_values is None:
            raise ValueError("discrete_values are required for a ParameterSpace with discrete axes.")
        discrete_values = np.asarray(discrete_values, dtype=float)
        if discrete_values.shape != (len(blocks.discrete_indices),):
            raise ValueError(
                "discrete_values has shape "
                f"{discrete_values.shape}, expected {(len(blocks.discrete_indices),)}."
            )
        theta[list(blocks.discrete_indices)] = discrete_values
    elif discrete_values is not None and np.asarray(discrete_values).size:
        raise ValueError("discrete_values were supplied but the ParameterSpace has no discrete axes.")

    if blocks.fixed_indices:
        theta[list(blocks.fixed_indices)] = np.asarray(blocks.fixed_values, dtype=float)

    return theta


def conditional_continuous_posterior(
    posterior: Posterior,
    parameter_space,
    discrete_values: Array | None = None,
) -> Posterior:
    """Create a Posterior over only the continuous block at fixed grid values."""

    blocks = split_parameter_space(parameter_space)
    discrete_values = np.asarray(discrete_values if discrete_values is not None else [], dtype=float)

    def log_prob_continuous(continuous_values: Array) -> float:
        theta = full_theta_from_blocks(parameter_space, continuous_values, discrete_values)
        return float(posterior.log_prob_fn(theta))

    return Posterior(
        log_prob_fn=log_prob_continuous,
        dim=len(blocks.continuous_indices),
        theta_names=blocks.continuous_names,
        extra={
            "full_posterior": posterior,
            "parameter_space": parameter_space,
            "discrete_names": blocks.discrete_names,
            "discrete_values": discrete_values,
        },
    )


def run_mixed_gibbs(
    posterior: Posterior,
    parameter_space,
    x0: Array,
    nsteps: int,
    continuous_sampler=None,
    continuous_sampler_kwargs: Mapping | None = None,
    rng: np.random.Generator | None = None,
    discrete_candidates: int | None = None,
    discrete_probability_floor: float | str | None = None,
    discrete_floor_failure_probability: float = 1e-5,
    discrete_floor_max_mass: float = 0.05,
    max_exact_grid_size: int = 50_000,
) -> SamplingResult:
    """Gibbs sampler for mixed continuous/discrete CIGALE-like posteriors.

    Each iteration does two transparent scientific operations:

    1. update the finite CIGALE grid block by categorical resampling at the
       current continuous parameter values;
    2. update the continuous block with a user-supplied sampler conditional on
       the selected grid point.

    The discrete block is updated by SIR over the Cartesian grid. If
    ``discrete_candidates`` is ``None`` the candidate set is the full finite
    grid. If it is an integer, that many grid points are sampled uniformly.
    The default full-grid update with no probability floor is an exact
    categorical conditional. Setting ``discrete_candidates`` or
    ``discrete_probability_floor`` requests an explicitly approximate SIR
    update; this choice is recorded in the result metadata.
    """

    if int(nsteps) <= 0:
        raise ValueError("nsteps must be positive.")
    if rng is None:
        rng = np.random.default_rng()
    if continuous_sampler_kwargs is None:
        continuous_sampler_kwargs = {}

    blocks = split_parameter_space(parameter_space)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (len(parameter_space.names),):
        raise ValueError(f"Expected x0 shape {(len(parameter_space.names),)}, got {x0.shape}.")

    continuous_state = x0[list(blocks.continuous_indices)] if blocks.continuous_indices else np.empty(0)
    discrete_state = x0[list(blocks.discrete_indices)] if blocks.discrete_indices else np.empty(0)

    if continuous_sampler is None and blocks.continuous_indices:
        from .mcmc import run_rw_metropolis

        continuous_sampler = run_rw_metropolis
        continuous_sampler_kwargs = {
            "nsteps": 20,
            "burnin": 10,
            "thin": 1,
            **dict(continuous_sampler_kwargs),
        }

    samples = np.empty((int(nsteps), len(parameter_space.names)), dtype=float)
    logp = np.empty(int(nsteps), dtype=float)
    discrete_log_normalizers = np.empty(int(nsteps), dtype=float)
    discrete_probabilities: list[np.ndarray] = []
    discrete_candidate_points: list[np.ndarray] = []
    inner_results: list[object] = []

    for step in range(int(nsteps)):
        if blocks.discrete_indices:
            discrete_state, discrete_info = _sample_discrete_conditional(
                posterior=posterior,
                parameter_space=parameter_space,
                continuous_values=continuous_state,
                rng=rng,
                discrete_candidates=discrete_candidates,
                discrete_probability_floor=discrete_probability_floor,
                discrete_floor_failure_probability=discrete_floor_failure_probability,
                discrete_floor_max_mass=discrete_floor_max_mass,
                max_exact_grid_size=max_exact_grid_size,
            )
            log_norm, probabilities, candidate_points = discrete_info
            discrete_probabilities.append(probabilities)
            discrete_candidate_points.append(candidate_points)
        else:
            log_norm = np.nan

        if blocks.continuous_indices:
            conditional = conditional_continuous_posterior(posterior, parameter_space, discrete_state)
            result = _call_continuous_sampler(
                continuous_sampler,
                conditional,
                continuous_state,
                rng=rng,
                kwargs=continuous_sampler_kwargs,
            )
            continuous_state = _choose_continuous_state(result, rng)
            inner_results.append(result)

        theta = full_theta_from_blocks(parameter_space, continuous_state, discrete_state)
        samples[step] = theta
        logp[step] = float(posterior.log_prob_fn(theta))
        discrete_log_normalizers[step] = log_norm

    best = int(np.nanargmax(logp))
    cov = np.cov(samples.T) if samples.shape[0] > 1 else None
    return SamplingResult(
        samples=samples,
        logp=logp,
        map_estimate=samples[best],
        cov=cov,
        meta={
            "parameter_blocks": blocks,
            "discrete_log_normalizers": discrete_log_normalizers,
            "discrete_probabilities": discrete_probabilities,
            "discrete_candidate_points": discrete_candidate_points,
            "inner_results": inner_results,
            "discrete_candidates": discrete_candidates,
            "discrete_probability_floor": discrete_probability_floor,
            "discrete_floor_failure_probability": discrete_floor_failure_probability,
            "discrete_floor_max_mass": discrete_floor_max_mass,
            "max_exact_grid_size": max_exact_grid_size,
            "approximate_discrete_kernel": (
                discrete_candidates is not None or discrete_probability_floor is not None
            ),
        },
    )


def run_thresholded_mixed_gibbs(*args, discrete_probability_floor="survival", **kwargs) -> SamplingResult:
    """Run the pragmatic thresholded-SIR variant of mixed Gibbs.

    This named entry point preserves low-probability discrete states by
    modifying the categorical conditional. It is useful as an exploration
    heuristic but is not an exact Gibbs kernel.
    """

    return run_mixed_gibbs(
        *args,
        discrete_probability_floor=discrete_probability_floor,
        **kwargs,
    )


def _sample_discrete_conditional(
    posterior: Posterior,
    parameter_space,
    continuous_values: np.ndarray,
    rng: np.random.Generator,
    discrete_candidates: int | None,
    discrete_probability_floor: float | str | None,
    discrete_floor_failure_probability: float,
    discrete_floor_max_mass: float,
    max_exact_grid_size: int,
) -> tuple[np.ndarray, tuple[float, np.ndarray, np.ndarray]]:
    if discrete_candidates is None:
        grid = enumerate_discrete_grid(parameter_space, max_size=max_exact_grid_size)
    else:
        grid = sample_discrete_grid(parameter_space, int(discrete_candidates), rng=rng)

    log_weights = np.empty(grid.points.shape[0], dtype=float)
    for i, discrete_values in enumerate(grid.points):
        theta = full_theta_from_blocks(parameter_space, continuous_values, discrete_values)
        log_weights[i] = posterior.log_prob_fn(theta)

    if not np.any(np.isfinite(log_weights)):
        raise RuntimeError("All discrete Gibbs candidates have non-finite log probability.")

    log_norm = _logsumexp(log_weights)
    probabilities = _probabilities_from_log_weights(
        log_weights,
        floor=discrete_probability_floor,
        failure_probability=discrete_floor_failure_probability,
        max_floor_mass=discrete_floor_max_mass,
    )
    choice = int(rng.choice(np.arange(grid.points.shape[0]), p=probabilities))
    return np.asarray(grid.points[choice], dtype=float), (
        float(log_norm),
        np.asarray(probabilities, dtype=float),
        np.asarray(grid.points, dtype=float),
    )


def _choose_continuous_state(result, rng: np.random.Generator) -> np.ndarray:
    samples = np.asarray(result.samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("continuous_sampler must return samples with shape (n, ndim).")

    weights = getattr(result, "weights", None)
    if weights is None and hasattr(result, "meta"):
        weights = result.meta.get("weights_norm")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
        return np.asarray(samples[int(rng.choice(np.arange(samples.shape[0]), p=weights))], dtype=float)

    return np.asarray(samples[-1], dtype=float)


def _call_continuous_sampler(sampler, posterior: Posterior, x0: np.ndarray, rng: np.random.Generator, kwargs):
    if getattr(sampler, "__name__", "") == "run_pocomc":
        raise ValueError(
            "run_mixed_gibbs does not support run_pocomc as the conditional continuous sampler. "
            "pocoMC manages its own prior/bounds object, which does not map cleanly onto a "
            "fixed-discrete-block Gibbs update."
        )
    call_kwargs = dict(kwargs)
    signature = inspect.signature(sampler)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs or "rng" in signature.parameters:
        call_kwargs.setdefault("rng", rng)
    elif "seed" in signature.parameters:
        call_kwargs.setdefault("seed", int(rng.integers(0, np.iinfo(np.uint32).max)))
    return sampler(posterior, x0, **call_kwargs)


def _fixed_prior_value(prior) -> float | None:
    if prior is None:
        return None
    if type(prior).__name__ == "DeltaPrior":
        return float(prior.value)
    return None


def _logsumexp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    max_value = np.max(finite[np.isfinite(finite)])
    return float(max_value + np.log(np.sum(np.exp(finite - max_value))))


def _probabilities_from_log_weights(
    log_weights: np.ndarray,
    floor: float | str | None = None,
    failure_probability: float = 1e-5,
    max_floor_mass: float = 0.05,
) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("Cannot normalize probabilities: all log weights are non-finite.")

    log_norm = _logsumexp(log_weights)
    probabilities = np.zeros_like(log_weights, dtype=float)
    probabilities[finite] = np.exp(log_weights[finite] - log_norm)
    probabilities = probabilities / np.sum(probabilities)

    epsilon = _probability_floor_value(
        n_candidates=probabilities.size,
        floor=floor,
        failure_probability=failure_probability,
        max_floor_mass=max_floor_mass,
    )
    if epsilon > 0.0:
        probabilities = (1.0 - probabilities.size * epsilon) * probabilities + epsilon
        probabilities = probabilities / np.sum(probabilities)
    return probabilities


def _probability_floor_value(
    n_candidates: int,
    floor: float | str | None,
    failure_probability: float,
    max_floor_mass: float,
) -> float:
    if floor is None or floor == 0:
        return 0.0
    if int(n_candidates) <= 0:
        raise ValueError("n_candidates must be positive.")
    if isinstance(floor, str):
        if floor != "survival":
            raise ValueError("discrete_probability_floor must be None, a float, or 'survival'.")
        if not 0.0 < failure_probability < 1.0:
            raise ValueError("failure_probability must lie in (0, 1).")
        raw = 1.0 - failure_probability ** (1.0 / float(n_candidates))
    else:
        raw = float(floor)
    if raw < 0.0:
        raise ValueError("discrete probability floor must be non-negative.")
    if not 0.0 <= max_floor_mass < 1.0:
        raise ValueError("max_floor_mass must lie in [0, 1).")
    return float(min(raw, max_floor_mass / float(n_candidates)))
