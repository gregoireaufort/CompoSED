from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from composed.experimental.jaxcigale.core import JaxSedModel
from composed.experimental.jaxcigale.dependencies import require_jax, require_numpyro


@dataclass(frozen=True)
class NutsResult:
    """Small NumPy-facing result for experimental JAX-CIGALE NUTS runs."""

    samples: np.ndarray
    log_prob: np.ndarray
    theta_names: tuple[str, ...]
    extra_fields: dict[str, object]


@dataclass(frozen=True)
class MapInitializationResult:
    """Audit record for QMC + Nadam MAP initialization.

    ``initial_theta`` is the scientist-facing parameter vector used to start
    NUTS. ``initial_raw`` is the corresponding unconstrained coordinate.  The
    log densities are the transformed NUTS log density,

    ``log p(theta | data) + log |d theta / d raw|``,

    because that is the density optimized in the coordinate system sampled by
    NUTS.
    """

    initial_theta: np.ndarray
    initial_raw: np.ndarray
    initial_log_density: float
    candidate_theta: np.ndarray
    candidate_log_density: np.ndarray
    optimized_theta: np.ndarray
    optimized_log_density: np.ndarray
    settings: dict[str, object]


def run_numpyro_nuts(
    model: JaxSedModel,
    data,
    initial_theta,
    num_warmup: int = 500,
    num_samples: int = 1000,
    rng_seed: int = 1,
    progress_bar: bool = True,
    transform_bounds: bool = True,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    dense_mass: bool = False,
    extra_fields: tuple[str, ...] = ("num_steps", "accept_prob", "adapt_state.step_size", "diverging"),
    num_chains: int = 1,
    chain_method: str = "sequential",
    init_strategy: str = "provided",
    init_num_candidates: int = 2048,
    init_num_starts: int = 12,
    init_optimizer_steps: int = 300,
    init_learning_rate: float = 0.03,
    init_batch_size: int = 512,
    init_rng_seed: int | None = None,
) -> NutsResult:
    """Run NumPyro NUTS on the vector-valued JAX-CIGALE log posterior.

    Parameters with finite prior bounds are sampled in an unconstrained
    coordinate system by default. The returned samples are always in the
    scientist-facing parameter coordinates used by ``JaxSedModel``.

    ``init_strategy='provided'`` preserves the historical behavior and starts
    from ``initial_theta``.  ``init_strategy='qmc_nadam'`` does a deterministic
    initialization pass before NUTS:

    1. draw Sobol/QMC points over finite prior bounds;
    2. evaluate the posterior in batches;
    3. polish the best candidates with Nadam in the NUTS coordinate system;
    4. start every chain at the single best local MAP.

    This is meant to avoid obvious line-identification traps in spectra. It is
    an initializer, not a replacement for posterior sampling.
    """

    jax, jnp = require_jax()
    _, MCMC, NUTS = require_numpyro()
    initial_theta = jnp.asarray(initial_theta)

    def potential_fn_raw(raw_theta):
        if transform_bounds:
            theta, log_abs_det = _unconstrained_to_theta_and_log_abs_det(model, raw_theta)
        else:
            theta = raw_theta
            log_abs_det = jnp.asarray(0.0)
        return -model.log_prob(theta, data) - log_abs_det

    def theta_from_raw(raw_theta):
        if transform_bounds:
            return _unconstrained_to_theta_and_log_abs_det(model, raw_theta)[0]
        return raw_theta

    map_initialization = None
    if init_strategy == "provided":
        initial_raw = _theta_to_unconstrained(model, initial_theta) if transform_bounds else initial_theta
    elif init_strategy == "qmc_nadam":
        map_initialization = find_map_initial_position(
            model,
            data,
            initial_theta=np.asarray(initial_theta, dtype=float),
            transform_bounds=transform_bounds,
            num_candidates=init_num_candidates,
            num_starts=init_num_starts,
            optimizer_steps=init_optimizer_steps,
            learning_rate=init_learning_rate,
            batch_size=init_batch_size,
            rng_seed=rng_seed + 7919 if init_rng_seed is None else init_rng_seed,
        )
        initial_raw = jnp.asarray(map_initialization.initial_raw)
    else:
        raise ValueError("init_strategy must be 'provided' or 'qmc_nadam'.")

    if int(num_chains) < 1:
        raise ValueError("num_chains must be at least one.")
    init_params = initial_raw
    if int(num_chains) > 1:
        init_params = jnp.repeat(initial_raw[jnp.newaxis, :], int(num_chains), axis=0)

    kernel = NUTS(
        potential_fn=potential_fn_raw,
        target_accept_prob=float(target_accept_prob),
        max_tree_depth=int(max_tree_depth),
        dense_mass=bool(dense_mass),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        chain_method=str(chain_method),
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(int(rng_seed)), init_params=init_params, extra_fields=extra_fields)
    raw_samples = mcmc.get_samples()
    raw_array = raw_samples["theta"] if isinstance(raw_samples, dict) else raw_samples
    samples = np.asarray(jax.vmap(theta_from_raw)(raw_array))
    log_prob = np.asarray(jax.vmap(lambda theta: model.log_prob(theta, data))(jnp.asarray(samples)))
    extra = {
        "numpyro": mcmc.get_extra_fields(),
        "unconstrained_samples": np.asarray(raw_array),
        "transform_bounds": transform_bounds,
        "init_strategy": init_strategy,
        "initial_theta_used": np.asarray(theta_from_raw(initial_raw)),
        "initial_raw_used": np.asarray(initial_raw),
        "num_chains": int(num_chains),
        "chain_method": str(chain_method),
    }
    if map_initialization is not None:
        extra["map_initialization"] = {
            "initial_theta": map_initialization.initial_theta,
            "initial_raw": map_initialization.initial_raw,
            "initial_log_density": map_initialization.initial_log_density,
            "candidate_theta": map_initialization.candidate_theta,
            "candidate_log_density": map_initialization.candidate_log_density,
            "optimized_theta": map_initialization.optimized_theta,
            "optimized_log_density": map_initialization.optimized_log_density,
            "settings": map_initialization.settings,
        }
    return NutsResult(
        samples=samples,
        log_prob=log_prob,
        theta_names=tuple(model.parameter_space.names),
        extra_fields=extra,
    )


def find_map_initial_position(
    model: JaxSedModel,
    data,
    initial_theta=None,
    *,
    transform_bounds: bool = True,
    num_candidates: int = 2048,
    num_starts: int = 12,
    optimizer_steps: int = 300,
    learning_rate: float = 0.03,
    batch_size: int = 512,
    rng_seed: int = 1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
    raw_clip: float = 20.0,
) -> MapInitializationResult:
    """Find a robust single starting point for NUTS.

    The scientific data flow is intentionally explicit:

    - candidate ``theta`` vectors are drawn over the prior space;
    - candidates are transformed to the unconstrained NUTS coordinates;
    - the transformed posterior density is optimized by Nadam;
    - the best optimized point is returned as the NUTS initializer.

    ``initial_theta`` is included among the candidates when supplied, so a good
    hand-written start cannot be made worse by the QMC stage.
    """

    if int(num_candidates) < 1:
        raise ValueError("num_candidates must be at least one.")
    if int(num_starts) < 1:
        raise ValueError("num_starts must be at least one.")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least one.")
    if int(optimizer_steps) < 0:
        raise ValueError("optimizer_steps must be non-negative.")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive.")

    jax, jnp = require_jax()
    theta_candidates, qmc_method = _draw_initialization_candidates(
        model,
        int(num_candidates),
        rng_seed=int(rng_seed),
    )
    if initial_theta is not None:
        theta_candidates = np.vstack([np.asarray(initial_theta, dtype=float)[np.newaxis, :], theta_candidates])

    raw_candidates = np.asarray(jax.vmap(lambda theta: _theta_to_raw_for_strategy(model, theta, transform_bounds))(
        jnp.asarray(theta_candidates)
    ))
    candidate_log_density = _evaluate_raw_log_density_in_batches(
        model,
        data,
        raw_candidates,
        transform_bounds=transform_bounds,
        batch_size=int(batch_size),
    )

    finite = np.isfinite(candidate_log_density)
    if not np.any(finite):
        raise RuntimeError("QMC initialization found no finite posterior candidates.")
    order = np.argsort(np.where(finite, candidate_log_density, -np.inf))[::-1]
    n_keep = min(int(num_starts), order.size)
    start_raw = raw_candidates[order[:n_keep]]
    start_log_density = candidate_log_density[order[:n_keep]]

    optimized_raw, optimized_log_density = _nadam_optimize_raw_log_density(
        model,
        data,
        start_raw,
        transform_bounds=transform_bounds,
        optimizer_steps=int(optimizer_steps),
        learning_rate=float(learning_rate),
        beta1=float(beta1),
        beta2=float(beta2),
        epsilon=float(epsilon),
        raw_clip=float(raw_clip),
    )
    optimized_theta = np.asarray(jax.vmap(lambda raw: _theta_from_raw_for_strategy(model, raw, transform_bounds))(
        jnp.asarray(optimized_raw)
    ))
    best = int(np.nanargmax(optimized_log_density))
    return MapInitializationResult(
        initial_theta=np.asarray(optimized_theta[best], dtype=float),
        initial_raw=np.asarray(optimized_raw[best], dtype=float),
        initial_log_density=float(optimized_log_density[best]),
        candidate_theta=np.asarray(theta_candidates[order[:n_keep]], dtype=float),
        candidate_log_density=np.asarray(start_log_density, dtype=float),
        optimized_theta=np.asarray(optimized_theta, dtype=float),
        optimized_log_density=np.asarray(optimized_log_density, dtype=float),
        settings={
            "num_candidates": int(num_candidates),
            "num_starts": int(num_starts),
            "optimizer_steps": int(optimizer_steps),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "rng_seed": int(rng_seed),
            "qmc_method": qmc_method,
            "transform_bounds": bool(transform_bounds),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "epsilon": float(epsilon),
            "raw_clip": float(raw_clip),
        },
    )


def _draw_initialization_candidates(model: JaxSedModel, n: int, *, rng_seed: int) -> tuple[np.ndarray, str]:
    """Draw candidate points in scientist-facing parameter coordinates."""

    bounds = model.parameter_space.bounds
    finite_bounds = all(
        low is not None
        and high is not None
        and np.isfinite(low)
        and np.isfinite(high)
        and float(high) > float(low)
        for low, high in bounds
    )
    rng = np.random.default_rng(int(rng_seed))
    if finite_bounds:
        ndim = model.parameter_space.ndim
        try:
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=ndim, scramble=True, seed=int(rng_seed))
            m_power = int(np.ceil(np.log2(max(n, 2))))
            unit = sampler.random_base2(m_power)[:n]
            method = "sobol"
        except Exception:
            unit = rng.random((n, ndim))
            method = "uniform_random_fallback"
        eps = np.finfo(float).eps
        unit = np.clip(unit, eps, 1.0 - eps)
        low = np.asarray([pair[0] for pair in bounds], dtype=float)
        high = np.asarray([pair[1] for pair in bounds], dtype=float)
        return low + unit * (high - low), method
    return model.parameter_space.sample_prior(n, rng=rng), "parameter_space_sample_prior"


def _raw_log_density(model: JaxSedModel, data, raw_theta, *, transform_bounds: bool):
    """Transformed log posterior optimized and sampled by NUTS."""

    _, jnp = require_jax()
    if transform_bounds:
        theta, log_abs_det = _unconstrained_to_theta_and_log_abs_det(model, raw_theta)
    else:
        theta = raw_theta
        log_abs_det = jnp.asarray(0.0)
    return model.log_prob(theta, data) + log_abs_det


def _theta_to_raw_for_strategy(model: JaxSedModel, theta, transform_bounds: bool):
    return _theta_to_unconstrained(model, theta) if transform_bounds else theta


def _theta_from_raw_for_strategy(model: JaxSedModel, raw_theta, transform_bounds: bool):
    return _unconstrained_to_theta_and_log_abs_det(model, raw_theta)[0] if transform_bounds else raw_theta


def _evaluate_raw_log_density_in_batches(
    model: JaxSedModel,
    data,
    raw_candidates: np.ndarray,
    *,
    transform_bounds: bool,
    batch_size: int,
) -> np.ndarray:
    """Evaluate candidate starts without hiding shape or finite checks."""

    jax, jnp = require_jax()
    one = lambda raw: _raw_log_density(model, data, raw, transform_bounds=transform_bounds)
    batched = jax.jit(jax.vmap(one))
    values = []
    for start in range(0, raw_candidates.shape[0], int(batch_size)):
        batch = jnp.asarray(raw_candidates[start : start + int(batch_size)])
        values.append(np.asarray(batched(batch)))
    return np.concatenate(values)


def _nadam_optimize_raw_log_density(
    model: JaxSedModel,
    data,
    start_raw: np.ndarray,
    *,
    transform_bounds: bool,
    optimizer_steps: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    raw_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a small vectorized Nadam maximization in NUTS coordinates."""

    jax, jnp = require_jax()
    raw = jnp.asarray(start_raw)
    m = jnp.zeros_like(raw)
    v = jnp.zeros_like(raw)

    def one_value(raw_theta):
        value = _raw_log_density(model, data, raw_theta, transform_bounds=transform_bounds)
        # Keep the replacement finite in float32 as well as float64.  This
        # matters on accelerator backends where exploratory runs often use
        # float32 and literal 1e300 would become an infinity.
        return jnp.where(jnp.isfinite(value), value, -jnp.asarray(1.0e30, dtype=value.dtype))

    value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(one_value)))
    values, grads = value_and_grad(raw)
    grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
    best_raw = raw
    best_values = values

    for step in range(1, int(optimizer_steps) + 1):
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * grads * grads
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        grad_hat = grads / (1.0 - beta1**step)
        nesterov_m = beta1 * m_hat + (1.0 - beta1) * grad_hat
        raw = raw + learning_rate * nesterov_m / (jnp.sqrt(v_hat) + epsilon)
        raw = jnp.clip(raw, -raw_clip, raw_clip)

        values, grads = value_and_grad(raw)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        better = values > best_values
        best_values = jnp.where(better, values, best_values)
        best_raw = jnp.where(better[:, jnp.newaxis], raw, best_raw)

    return np.asarray(best_raw), np.asarray(best_values)


def _theta_to_unconstrained(model: JaxSedModel, theta):
    """Map physical parameters to NUTS coordinates using prior bounds."""

    _, jnp = require_jax()
    theta = jnp.asarray(theta)
    raw_values = []
    eps = jnp.asarray(1.0e-8)
    for i, (low, high) in enumerate(model.parameter_space.bounds):
        x = theta[i]
        if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
            frac = jnp.clip((x - low) / (high - low), eps, 1.0 - eps)
            raw_values.append(jnp.log(frac) - jnp.log1p(-frac))
        elif low is not None and np.isfinite(low):
            raw_values.append(jnp.log(jnp.maximum(x - low, eps)))
        elif high is not None and np.isfinite(high):
            raw_values.append(jnp.log(jnp.maximum(high - x, eps)))
        else:
            raw_values.append(x)
    return jnp.asarray(raw_values)


def _unconstrained_to_theta_and_log_abs_det(model: JaxSedModel, raw_theta):
    """Map NUTS coordinates back to physical parameters and Jacobian."""

    jax, jnp = require_jax()
    raw_theta = jnp.asarray(raw_theta)
    theta_values = []
    log_abs_det = jnp.asarray(0.0)
    for i, (low, high) in enumerate(model.parameter_space.bounds):
        u = raw_theta[i]
        if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
            width = high - low
            sigmoid_u = jax.nn.sigmoid(u)
            theta_values.append(low + width * sigmoid_u)
            log_abs_det = log_abs_det + jnp.log(width) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u)
        elif low is not None and np.isfinite(low):
            theta_values.append(low + jnp.exp(u))
            log_abs_det = log_abs_det + u
        elif high is not None and np.isfinite(high):
            theta_values.append(high - jnp.exp(u))
            log_abs_det = log_abs_det + u
        else:
            theta_values.append(u)
    return jnp.asarray(theta_values), log_abs_det
