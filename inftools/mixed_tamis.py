from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Posterior, SamplingResult
from .grid import (
    _probabilities_from_log_weights,
    _probability_floor_value,
    _logsumexp,
    full_theta_from_blocks,
    run_grid_sampler,
    split_parameter_space,
)
from .transforms import box_logit_transform_from_parameter_space


@dataclass(frozen=True)
class MixedTamisProposal:
    """Proposal state for a mixed continuous/discrete TAMIS iteration."""

    means: np.ndarray
    covariances: np.ndarray
    proportions: np.ndarray
    discrete_values: tuple[np.ndarray, ...]
    discrete_probabilities: tuple[np.ndarray, ...]


def run_mixed_tamis(
    posterior: Posterior,
    parameter_space,
    x0=None,
    n_comp: int = 4,
    T_max: int = 10,
    n_per_iter: int = 500,
    init_span: float = 1.0,
    var0=None,
    alpha: float = 200.0,
    discrete_probability_floor: float | str | None = "survival",
    discrete_floor_failure_probability: float = 1e-5,
    discrete_floor_max_mass: float = 0.05,
    covariance_jitter: float = 1e-6,
    continuous_transform=None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> SamplingResult:
    """TAMIS-style adaptive importance sampler with a mixed proposal.

    The proposal factorizes as

    ``q_t(theta_cont, d) = GMM_t(theta_cont) * prod_j Categorical_t(d_j)``.

    This is the direct descendant of the old MC-CIGALE idea: discrete CIGALE
    choices are proposal variables, not Gaussian coordinates. Tempered weights
    adapt both the continuous Gaussian mixture and the categorical proposal.
    The final weights use AMIS recycling over all proposal rounds.

    If ``continuous_transform`` is ``"auto"`` or ``True``, bounded continuous
    parameters are sampled in unconstrained box-logit coordinates while results
    are returned in the original physical parameter space. In that case
    ``init_span`` and ``var0`` live in transformed coordinates.
    """

    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    if rng is None:
        rng = np.random.default_rng(seed)
    if int(T_max) <= 0:
        raise ValueError("T_max must be positive.")
    if int(n_per_iter) <= 0:
        raise ValueError("n_per_iter must be positive.")
    if int(n_comp) <= 0:
        raise ValueError("n_comp must be positive.")

    blocks = split_parameter_space(parameter_space)
    if not blocks.continuous_indices:
        return run_grid_sampler(posterior, parameter_space)
    transform = _resolve_continuous_transform(parameter_space, blocks, continuous_transform)

    if x0 is None:
        x0 = parameter_space.sample_prior(1, rng=rng)[0]
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (len(parameter_space.names),):
        raise ValueError(f"Expected x0 shape {(len(parameter_space.names),)}, got {x0.shape}.")

    proposal = _initial_mixed_proposal(
        parameter_space=parameter_space,
        x0=x0,
        n_comp=int(n_comp),
        init_span=float(init_span),
        var0=var0,
        transform=transform,
        rng=rng,
    )

    proposal_history: list[MixedTamisProposal] = []
    sample_history: list[np.ndarray] = []
    log_target_history: list[np.ndarray] = []
    log_proposal_history: list[np.ndarray] = []
    weight_history: list[np.ndarray] = []
    tempered_weight_history: list[np.ndarray] = []
    beta_history: list[float] = []
    ess_history: list[float] = []
    tempered_ess_history: list[float] = []

    for _ in range(int(T_max)):
        proposal_history.append(proposal)
        samples = _sample_mixed_proposal(proposal, parameter_space, int(n_per_iter), rng, transform=transform)
        log_target = np.asarray([posterior.log_prob_fn(theta) for theta in samples], dtype=float)
        log_proposal = _mixed_proposal_logpdf(samples, proposal, parameter_space, transform=transform)
        log_weight = log_target - log_proposal

        weights = _probabilities_from_log_weights(log_weight)
        ess = _effective_sample_size(weights)
        beta = _adapt_beta(log_weight, target_ess=min(float(alpha), float(n_per_iter)))
        tempered_weights = _probabilities_from_log_weights(beta * log_weight)
        tempered_ess = _effective_sample_size(tempered_weights)

        sample_history.append(samples)
        log_target_history.append(log_target)
        log_proposal_history.append(log_proposal)
        weight_history.append(weights)
        tempered_weight_history.append(tempered_weights)
        beta_history.append(beta)
        ess_history.append(ess)
        tempered_ess_history.append(tempered_ess)

        proposal = _update_mixed_proposal(
            samples=samples,
            parameter_space=parameter_space,
            old=proposal,
            weights=tempered_weights,
            transform=transform,
            discrete_probability_floor=discrete_probability_floor,
            discrete_floor_failure_probability=discrete_floor_failure_probability,
            discrete_floor_max_mass=discrete_floor_max_mass,
            covariance_jitter=float(covariance_jitter),
        )

    all_samples = np.vstack(sample_history)
    all_log_target = np.concatenate(log_target_history)
    final_log_weights = _amis_log_weights(
        samples=all_samples,
        log_target=all_log_target,
        proposals=proposal_history,
        parameter_space=parameter_space,
        n_per_iter=int(n_per_iter),
        transform=transform,
    )
    final_weights = _probabilities_from_log_weights(final_log_weights)

    best = int(np.nanargmax(all_log_target))
    mean = np.average(all_samples, axis=0, weights=final_weights)
    diff = all_samples - mean
    cov = (final_weights[:, None] * diff).T @ diff
    return SamplingResult(
        samples=all_samples,
        logp=all_log_target,
        map_estimate=all_samples[best],
        cov=cov,
        meta={
            "weights_norm": final_weights,
            "final_log_weights": final_log_weights,
            "proposal_history": proposal_history,
            "log_proposal_history": log_proposal_history,
            "weight_history": weight_history,
            "tempered_weight_history": tempered_weight_history,
            "betas": np.asarray(beta_history, dtype=float),
            "ESS": np.asarray(ess_history, dtype=float),
            "tempered_ESS": np.asarray(tempered_ess_history, dtype=float),
            "parameter_blocks": blocks,
            "continuous_transform": type(transform).__name__ if transform is not None else None,
            "continuous_transform_names": getattr(transform, "names", None),
            "discrete_probability_floor": discrete_probability_floor,
            "discrete_floor_failure_probability": discrete_floor_failure_probability,
            "discrete_floor_max_mass": discrete_floor_max_mass,
        },
    )


def _resolve_continuous_transform(parameter_space, blocks, continuous_transform):
    if continuous_transform is None or continuous_transform is False:
        return None
    if continuous_transform is True or continuous_transform == "auto":
        return box_logit_transform_from_parameter_space(parameter_space, names=blocks.continuous_names)
    return continuous_transform


def _initial_mixed_proposal(parameter_space, x0, n_comp, init_span, var0, transform, rng) -> MixedTamisProposal:
    blocks = split_parameter_space(parameter_space)
    continuous_x0 = x0[list(blocks.continuous_indices)]
    if transform is not None:
        continuous_x0 = transform.theta_to_y(continuous_x0)
    dim_cont = len(blocks.continuous_indices)
    means = continuous_x0[None, :] + rng.uniform(-init_span, init_span, size=(n_comp, dim_cont))

    if var0 is None:
        var_vec = np.full(dim_cont, init_span**2, dtype=float)
    else:
        var_vec = np.broadcast_to(np.asarray(var0, dtype=float), (dim_cont,)).copy()
    var_vec = np.maximum(var_vec, 1e-12)
    covariances = np.asarray([np.diag(var_vec) for _ in range(n_comp)], dtype=float)
    proportions = np.full(n_comp, 1.0 / n_comp, dtype=float)

    discrete_values: list[np.ndarray] = []
    discrete_probabilities: list[np.ndarray] = []
    for name in blocks.discrete_names:
        prior = parameter_space.priors[name]
        if type(prior).__name__ == "ChoicePrior":
            values = np.asarray(prior.values, dtype=float)
        elif type(prior).__name__ == "IntegerUniformPrior":
            values = np.arange(int(prior.low), int(prior.high) + 1, dtype=float)
        else:  # pragma: no cover - split_parameter_space should prevent this
            raise TypeError(f"Unsupported discrete prior for {name!r}: {type(prior).__name__}")
        discrete_values.append(values)
        discrete_probabilities.append(np.full(values.size, 1.0 / values.size, dtype=float))

    return MixedTamisProposal(
        means=means,
        covariances=covariances,
        proportions=proportions,
        discrete_values=tuple(discrete_values),
        discrete_probabilities=tuple(discrete_probabilities),
    )


def _sample_mixed_proposal(proposal, parameter_space, n, rng, transform=None) -> np.ndarray:
    blocks = split_parameter_space(parameter_space)
    dim_cont = len(blocks.continuous_indices)
    component = rng.choice(np.arange(proposal.proportions.size), size=n, p=proposal.proportions)
    continuous_work = np.empty((n, dim_cont), dtype=float)
    for k in range(proposal.proportions.size):
        mask = component == k
        if np.any(mask):
            continuous_work[mask] = rng.multivariate_normal(
                mean=proposal.means[k],
                cov=proposal.covariances[k],
                size=int(np.sum(mask)),
            )
    continuous = transform.y_to_theta(continuous_work) if transform is not None else continuous_work

    discrete = np.empty((n, len(blocks.discrete_indices)), dtype=float)
    for j, values in enumerate(proposal.discrete_values):
        discrete[:, j] = rng.choice(values, size=n, p=proposal.discrete_probabilities[j])

    return np.asarray(
        [full_theta_from_blocks(parameter_space, continuous[i], discrete[i]) for i in range(n)],
        dtype=float,
    )


def _mixed_proposal_logpdf(samples: np.ndarray, proposal: MixedTamisProposal, parameter_space, transform=None) -> np.ndarray:
    blocks = split_parameter_space(parameter_space)
    samples = np.asarray(samples, dtype=float)
    continuous = samples[:, list(blocks.continuous_indices)]
    if transform is not None:
        continuous_work = transform.theta_to_y(continuous)
        log_cont = _gaussian_mixture_logpdf(continuous_work, proposal) - transform.log_abs_det_jac_each(
            continuous_work
        )
    else:
        log_cont = _gaussian_mixture_logpdf(continuous, proposal)
    log_disc = np.zeros(samples.shape[0], dtype=float)
    for axis, full_index in enumerate(blocks.discrete_indices):
        values = proposal.discrete_values[axis]
        probs = proposal.discrete_probabilities[axis]
        observed = samples[:, full_index]
        axis_logp = np.full(samples.shape[0], -np.inf, dtype=float)
        for k, value in enumerate(values):
            axis_logp[np.isclose(observed, value, rtol=1e-12, atol=1e-12)] = np.log(probs[k])
        log_disc += axis_logp
    return log_cont + log_disc


def _gaussian_mixture_logpdf(continuous: np.ndarray, proposal: MixedTamisProposal) -> np.ndarray:
    terms = []
    for k in range(proposal.proportions.size):
        terms.append(
            np.log(proposal.proportions[k])
            + _multivariate_normal_logpdf(continuous, proposal.means[k], proposal.covariances[k])
        )
    terms = np.asarray(terms, dtype=float)
    return np.asarray([_logsumexp(terms[:, i]) for i in range(continuous.shape[0])], dtype=float)


def _multivariate_normal_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    dim = mean.size
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + 1e-8 * np.eye(dim)
        sign, logdet = np.linalg.slogdet(cov)
    diff = x - mean[None, :]
    solve = np.linalg.solve(cov, diff.T).T
    quad = np.sum(diff * solve, axis=1)
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)


def _update_mixed_proposal(
    samples,
    parameter_space,
    old,
    weights,
    transform,
    discrete_probability_floor,
    discrete_floor_failure_probability,
    discrete_floor_max_mass,
    covariance_jitter,
) -> MixedTamisProposal:
    blocks = split_parameter_space(parameter_space)
    continuous = samples[:, list(blocks.continuous_indices)]
    if transform is not None:
        continuous = transform.theta_to_y(continuous)
    responsibilities = _mixture_responsibilities(continuous, old)
    weighted_resp = weights[:, None] * responsibilities
    component_mass = np.sum(weighted_resp, axis=0)
    component_mass = np.maximum(component_mass, 1e-12)
    proportions = component_mass / np.sum(component_mass)

    means = np.empty_like(old.means)
    covariances = np.empty_like(old.covariances)
    for k in range(old.proportions.size):
        wk = weighted_resp[:, k]
        total = np.sum(wk)
        if total <= 1e-12:
            means[k] = old.means[k]
            covariances[k] = old.covariances[k]
            continue
        means[k] = np.sum(wk[:, None] * continuous, axis=0) / total
        diff = continuous - means[k][None, :]
        cov = (wk[:, None] * diff).T @ diff / total
        covariances[k] = cov + covariance_jitter * np.eye(cov.shape[0])

    discrete_probabilities: list[np.ndarray] = []
    for axis, full_index in enumerate(blocks.discrete_indices):
        values = old.discrete_values[axis]
        probs = np.empty(values.size, dtype=float)
        for k, value in enumerate(values):
            probs[k] = np.sum(weights[np.isclose(samples[:, full_index], value, rtol=1e-12, atol=1e-12)])
        probs = _smooth_probabilities(
            probs,
            floor=discrete_probability_floor,
            failure_probability=discrete_floor_failure_probability,
            max_floor_mass=discrete_floor_max_mass,
        )
        discrete_probabilities.append(probs)

    return MixedTamisProposal(
        means=means,
        covariances=covariances,
        proportions=proportions,
        discrete_values=old.discrete_values,
        discrete_probabilities=tuple(discrete_probabilities),
    )


def _mixture_responsibilities(continuous: np.ndarray, proposal: MixedTamisProposal) -> np.ndarray:
    terms = []
    for k in range(proposal.proportions.size):
        terms.append(
            np.log(proposal.proportions[k])
            + _multivariate_normal_logpdf(continuous, proposal.means[k], proposal.covariances[k])
        )
    terms = np.asarray(terms, dtype=float).T
    denom = np.asarray([_logsumexp(row) for row in terms], dtype=float)
    return np.exp(terms - denom[:, None])


def _adapt_beta(log_weight: np.ndarray, target_ess: float) -> float:
    log_weight = np.asarray(log_weight, dtype=float)
    finite = np.isfinite(log_weight)
    if not np.any(finite):
        return 1.0
    n = np.sum(finite)
    target_ess = min(max(float(target_ess), 1.0), float(n))
    if _effective_sample_size(_probabilities_from_log_weights(log_weight)) >= target_ess:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        ess = _effective_sample_size(_probabilities_from_log_weights(mid * log_weight))
        if ess > target_ess:
            low = mid
        else:
            high = mid
    return float(high)


def _amis_log_weights(samples, log_target, proposals, parameter_space, n_per_iter, transform=None):
    log_q_terms = []
    for proposal in proposals:
        log_q_terms.append(
            np.log(float(n_per_iter)) + _mixed_proposal_logpdf(samples, proposal, parameter_space, transform=transform)
        )
    log_q_terms = np.asarray(log_q_terms, dtype=float)
    log_denom = np.asarray([_logsumexp(log_q_terms[:, i]) for i in range(samples.shape[0])], dtype=float)
    log_denom -= np.log(float(n_per_iter * len(proposals)))
    return np.asarray(log_target, dtype=float) - log_denom


def _smooth_probabilities(probs, floor, failure_probability, max_floor_mass):
    probs = np.asarray(probs, dtype=float)
    if np.sum(probs) <= 0.0:
        probs = np.full(probs.size, 1.0 / probs.size, dtype=float)
    else:
        probs = probs / np.sum(probs)
    epsilon = _probability_floor_value(
        n_candidates=probs.size,
        floor=floor,
        failure_probability=failure_probability,
        max_floor_mass=max_floor_mass,
    )
    if epsilon > 0.0:
        probs = (1.0 - probs.size * epsilon) * probs + epsilon
        probs = probs / np.sum(probs)
    return probs


def _effective_sample_size(weights) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return float(1.0 / np.sum(weights**2))
