# inftools/mcmc.py

from __future__ import annotations
from typing import Optional
import numpy as np
import emcee

from .core import Posterior, SamplingResult, Array


def run_rw_metropolis(
    posterior: Posterior,
    x0: Array,
    nsteps: int = 5000,
    proposal_cov: Optional[Array] = None,
    burnin: int = 0,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> SamplingResult:
    """
    Simple random-walk Metropolis-Hastings with Gaussian proposal.

    proposal_cov:
        Proposal covariance matrix. If None, uses 0.1 * I.
    """
    if rng is None:
        rng = np.random.default_rng()

    posterior, x, reduction = _prepare_continuous_mcmc_target(posterior, x0)
    dim = posterior.dim
    assert x.shape[0] == dim

    if proposal_cov is None:
        proposal_cov = 0.1 * np.eye(dim)
    proposal_cov = np.asarray(proposal_cov, dtype=float)
    if reduction is not None and proposal_cov.shape == (reduction["full_dim"], reduction["full_dim"]):
        idx = np.asarray(reduction["free_indices"], dtype=int)
        proposal_cov = proposal_cov[np.ix_(idx, idx)]
    if proposal_cov.shape != (dim, dim):
        raise ValueError(f"proposal_cov has shape {proposal_cov.shape}; expected {(dim, dim)}.")

    chol = np.linalg.cholesky(proposal_cov)

    samples = np.zeros((nsteps, dim))
    logp = np.zeros(nsteps)

    current_lp = posterior.log_prob_fn(x)
    accept = 0

    for t in range(nsteps):
        step = chol @ rng.normal(size=dim)
        x_prop = x + step
        lp_prop = posterior.log_prob_fn(x_prop)

        if np.isfinite(lp_prop):
            log_alpha = lp_prop - current_lp
            alpha = 1.0 if log_alpha >= 0.0 else np.exp(log_alpha)
        else:
            alpha = 0.0

        if rng.uniform() < alpha:
            x = x_prop
            current_lp = lp_prop
            accept += 1

        samples[t] = x
        logp[t] = current_lp

    acc_rate = accept / nsteps

    # Burn-in / thinning
    idx = np.arange(nsteps)
    mask = idx >= burnin
    idx = idx[mask][::thin]

    samples_thin = samples[idx]
    logp_thin = logp[idx]

    cov = np.cov(samples_thin.T) if samples_thin.shape[0] > 1 else None

    map_est = samples_thin[np.argmax(logp_thin)] if samples_thin.size > 0 else None

    result = SamplingResult(
        samples=samples_thin,
        logp=logp_thin,
        map_estimate=map_est,
        cov=cov,
        meta={"accept_rate": acc_rate, "proposal_cov": proposal_cov},
    )
    return _expand_fixed_parameter_result(result, reduction)


def run_emcee(
    posterior: Posterior,
    x0: Array,
    nwalkers: int = 32,
    nsteps: int = 1000,
    pool=None,
    burnin: int = 0,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    progress: bool = True,
) -> SamplingResult:
    """
    Run emcee ensemble sampler given a Posterior.

    x0:
        Initial point (used to initialize all walkers in a small Gaussian ball).
    """
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    if rng is None:
        rng = np.random.default_rng(seed)
    posterior, x0, reduction = _prepare_continuous_mcmc_target(posterior, x0)
    assert x0.shape[0] == posterior.dim

    pos = x0 + 1e-4 * rng.normal(size=(nwalkers, posterior.dim))

    sampler = emcee.EnsembleSampler(
        nwalkers,
        posterior.dim,
        posterior.log_prob_fn,
        pool=pool,
    )
    # emcee internally still uses NumPy's legacy global RandomState in common
    # releases.  Seed it only inside this call and restore the caller's global
    # state afterwards, so same-seed runs are reproducible without leaking
    # stochastic state into the rest of a notebook.
    emcee_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
    numpy_random_state = np.random.get_state()
    try:
        np.random.seed(emcee_seed)
        sampler.run_mcmc(pos, nsteps, progress=progress)
    finally:
        np.random.set_state(numpy_random_state)

    chain = sampler.get_chain()
    logp_chain = sampler.get_log_prob()

    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    logp = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)

    cov = np.cov(samples.T) if samples.shape[0] > 1 else None
    map_est = samples[np.argmax(logp)] if samples.size > 0 else None

    result = SamplingResult(
        samples=samples,
        logp=logp,
        map_estimate=map_est,
        cov=cov,
        meta={
            "raw_chain": chain,
            "raw_logp": logp_chain,
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "seed": seed,
            "emcee_seed": emcee_seed,
        },
    )
    return _expand_fixed_parameter_result(result, reduction)


def _prepare_continuous_mcmc_target(posterior: Posterior, x0: Array):
    """Remove fixed axes and reject finite-choice axes before continuous MCMC."""

    x0 = np.asarray(x0, dtype=float)
    parameter_space = posterior.extra.get("parameter_space") if posterior.extra else None
    if parameter_space is None or posterior.dim != len(parameter_space.names):
        return posterior, x0, None

    fixed_indices = []
    fixed_values = []
    discrete_names = []
    free_indices = []
    for index, name in enumerate(parameter_space.names):
        prior = parameter_space.priors[name]
        prior_name = type(prior).__name__
        if prior_name == "DeltaPrior":
            fixed_indices.append(index)
            fixed_values.append(float(prior.value))
        elif prior_name in {"ChoicePrior", "IntegerUniformPrior"}:
            discrete_names.append(str(name))
        else:
            free_indices.append(index)

    if discrete_names:
        raise ValueError(
            "Continuous MCMC cannot propose discrete parameter(s): "
            f"{', '.join(discrete_names)}. Use run_grid_sampler, run_mixed_gibbs, or run_mixed_tamis."
        )
    if not fixed_indices:
        return posterior, x0, None
    if not free_indices:
        raise ValueError("Continuous MCMC has no free parameters after removing DeltaPrior axes.")
    if x0.shape != (len(parameter_space.names),):
        raise ValueError(
            f"x0 has shape {x0.shape}; expected full ParameterSpace shape {(len(parameter_space.names),)}."
        )

    fixed_indices_arr = np.asarray(fixed_indices, dtype=int)
    fixed_values_arr = np.asarray(fixed_values, dtype=float)
    free_indices_arr = np.asarray(free_indices, dtype=int)

    def to_full(theta_free):
        theta_full = np.empty(len(parameter_space.names), dtype=float)
        theta_full[fixed_indices_arr] = fixed_values_arr
        theta_full[free_indices_arr] = np.asarray(theta_free, dtype=float)
        return theta_full

    reduced = Posterior(
        log_prob_fn=lambda theta: posterior.log_prob_fn(to_full(theta)),
        log_likelihood_fn=(
            None if posterior.log_likelihood_fn is None else lambda theta: posterior.log_likelihood_fn(to_full(theta))
        ),
        log_prior_fn=None if posterior.log_prior_fn is None else lambda theta: posterior.log_prior_fn(to_full(theta)),
        dim=len(free_indices),
        theta_names=[parameter_space.names[i] for i in free_indices],
        extra={"full_posterior": posterior},
    )
    reduction = {
        "full_dim": len(parameter_space.names),
        "free_indices": tuple(free_indices),
        "fixed_indices": tuple(fixed_indices),
        "fixed_values": fixed_values_arr,
        "parameter_names": tuple(parameter_space.names),
    }
    return reduced, x0[free_indices_arr], reduction


def _expand_fixed_parameter_result(result: SamplingResult, reduction):
    if reduction is None:
        return result
    free = np.asarray(reduction["free_indices"], dtype=int)
    fixed = np.asarray(reduction["fixed_indices"], dtype=int)
    full_dim = int(reduction["full_dim"])

    samples = np.empty((result.samples.shape[0], full_dim), dtype=float)
    samples[:, free] = result.samples
    samples[:, fixed] = np.asarray(reduction["fixed_values"], dtype=float)[None, :]
    map_estimate = None
    if result.map_estimate is not None:
        map_estimate = np.empty(full_dim, dtype=float)
        map_estimate[free] = result.map_estimate
        map_estimate[fixed] = reduction["fixed_values"]

    covariance = None
    if result.cov is not None:
        covariance = np.zeros((full_dim, full_dim), dtype=float)
        reduced_cov = np.atleast_2d(np.asarray(result.cov, dtype=float))
        covariance[np.ix_(free, free)] = reduced_cov

    meta = dict(result.meta)
    if "raw_chain" in meta:
        raw = np.asarray(meta["raw_chain"], dtype=float)
        expanded = np.empty(raw.shape[:-1] + (full_dim,), dtype=float)
        expanded[..., free] = raw
        expanded[..., fixed] = np.asarray(reduction["fixed_values"], dtype=float)
        meta["raw_chain"] = expanded
    meta["parameter_reduction"] = reduction
    return SamplingResult(
        samples=samples,
        logp=result.logp,
        map_estimate=map_estimate,
        cov=covariance,
        meta=meta,
    )
