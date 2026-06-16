import numpy as np
import pytest

from inftools.core import Posterior, SamplingResult
from inftools.grid import (
    conditional_continuous_posterior,
    enumerate_discrete_grid,
    full_theta_from_blocks,
    run_grid_sampler,
    run_mixed_gibbs,
    sample_discrete_grid,
    split_parameter_space,
)
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, DeltaPrior, IntegerUniformPrior, UniformPrior


def test_enumerate_discrete_grid_from_choice_and_integer_priors():
    space = ParameterSpace(
        names=("log10_mass", "metallicity", "imf", "fixed"),
        priors={
            "log10_mass": UniformPrior(8.0, 12.0),
            "metallicity": ChoicePrior([0.004, 0.008, 0.02]),
            "imf": IntegerUniformPrior(0, 1),
            "fixed": DeltaPrior(3.0),
        },
    )

    grid = enumerate_discrete_grid(space)

    assert grid.names == ("metallicity", "imf")
    assert grid.indices == (1, 2)
    assert grid.points.shape == (6, 2)
    assert set(grid.points[:, 0]) == {0.004, 0.008, 0.02}
    assert set(grid.points[:, 1]) == {0.0, 1.0}


def test_sample_discrete_grid_draws_only_allowed_values_without_full_enumeration():
    space = ParameterSpace(
        names=("x", "choice", "integer"),
        priors={
            "x": UniformPrior(-1.0, 1.0),
            "choice": ChoicePrior([10.0, 20.0]),
            "integer": IntegerUniformPrior(2, 4),
        },
    )

    grid = sample_discrete_grid(space, 128, rng=np.random.default_rng(2))

    assert grid.points.shape == (128, 2)
    assert set(np.unique(grid.points[:, 0])).issubset({10.0, 20.0})
    assert set(np.unique(grid.points[:, 1])).issubset({2.0, 3.0, 4.0})


def test_split_parameter_space_treats_delta_as_fixed_not_continuous():
    space = ParameterSpace(
        names=("x", "template", "fixed"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([0.0, 1.0]),
            "fixed": DeltaPrior(42.0),
        },
    )

    blocks = split_parameter_space(space)

    assert blocks.continuous_names == ("x",)
    assert blocks.discrete_names == ("template",)
    assert blocks.fixed_names == ("fixed",)
    theta = full_theta_from_blocks(space, continuous_values=[1.5], discrete_values=[1.0])
    assert np.allclose(theta, [1.5, 1.0, 42.0])


def test_conditional_continuous_posterior_uses_full_theta_order():
    space = ParameterSpace(
        names=("x", "template", "fixed"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([-1.0, 1.0]),
            "fixed": DeltaPrior(2.0),
        },
    )

    seen = []

    def log_prob(theta):
        seen.append(np.asarray(theta, dtype=float))
        return -0.5 * np.sum(np.asarray(theta) ** 2)

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    conditional = conditional_continuous_posterior(posterior, space, discrete_values=[1.0])

    assert conditional.theta_names == ("x",)
    value = conditional.log_prob_fn([3.0])

    assert np.isfinite(value)
    assert np.allclose(seen[-1], [3.0, 1.0, 2.0])


def test_discrete_grid_size_guard_raises_clear_error():
    space = ParameterSpace(
        names=("a", "b"),
        priors={
            "a": IntegerUniformPrior(0, 9),
            "b": IntegerUniformPrior(0, 9),
        },
    )

    with pytest.raises(ValueError, match="Discrete grid has 100 points"):
        enumerate_discrete_grid(space, max_size=50)


def test_run_grid_sampler_evaluates_full_cigale_style_grid():
    space = ParameterSpace(
        names=("metallicity", "imf", "fixed"),
        priors={
            "metallicity": ChoicePrior([0.008, 0.02]),
            "imf": IntegerUniformPrior(0, 1),
            "fixed": DeltaPrior(5.0),
        },
    )

    def log_prob(theta):
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        metallicity, imf, fixed = theta
        return prior - (10.0 if metallicity == 0.008 else 0.0) - (0.5 if imf == 0 else 0.0) - abs(fixed - 5.0)

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_grid_sampler(posterior, space)

    assert result.samples.shape == (4, 3)
    assert np.all(result.samples[:, 2] == 5.0)
    assert np.isclose(np.sum(result.meta["weights_norm"]), 1.0)
    assert np.allclose(result.map_estimate, [0.02, 1.0, 5.0])


def test_run_grid_sampler_rejects_continuous_parameters():
    space = ParameterSpace(
        names=("x", "template"),
        priors={
            "x": UniformPrior(-1.0, 1.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )
    posterior = Posterior(lambda theta: space.log_prior(theta), dim=space.ndim, theta_names=space.names)

    with pytest.raises(ValueError, match="Continuous parameter"):
        run_grid_sampler(posterior, space)


def test_mixed_gibbs_keeps_discrete_block_on_grid_and_updates_continuous_block():
    space = ParameterSpace(
        names=("x", "template", "fixed"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([-1.0, 1.0]),
            "fixed": DeltaPrior(7.0),
        },
    )

    def log_prob(theta):
        theta = np.asarray(theta, dtype=float)
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        x, template, fixed = theta
        if fixed != 7.0:
            return -np.inf
        template_bonus = 0.0 if template == 1.0 else -100.0
        return prior + template_bonus - 0.5 * ((x - template) / 0.2) ** 2

    def deterministic_continuous_update(posterior, x0, rng=None):
        del x0, rng
        grid = np.linspace(-2.0, 2.0, 401)
        logp = np.asarray([posterior.log_prob_fn([x]) for x in grid])
        best = int(np.argmax(logp))
        sample = np.asarray([[grid[best]]], dtype=float)
        return SamplingResult(samples=sample, logp=np.asarray([logp[best]]), map_estimate=sample[0])

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_mixed_gibbs(
        posterior,
        space,
        x0=np.asarray([0.0, -1.0, 7.0]),
        nsteps=8,
        continuous_sampler=deterministic_continuous_update,
        rng=np.random.default_rng(3),
    )

    assert result.samples.shape == (8, 3)
    assert set(np.unique(result.samples[:, 1])) == {1.0}
    assert np.all(result.samples[:, 2] == 7.0)
    assert np.allclose(result.samples[:, 0], 1.0)
    assert np.all(np.isfinite(result.logp))


def test_mixed_gibbs_discrete_sir_applies_probability_floor():
    space = ParameterSpace(
        names=("template",),
        priors={"template": ChoicePrior([0.0, 1.0, 2.0])},
    )

    def log_prob(theta):
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        return prior if theta[0] == 0.0 else prior - 1000.0

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_mixed_gibbs(
        posterior,
        space,
        x0=np.asarray([0.0]),
        nsteps=1,
        discrete_probability_floor=0.1,
        discrete_floor_max_mass=0.9,
        rng=np.random.default_rng(9),
    )

    probabilities = result.meta["discrete_probabilities"][0]
    assert np.all(probabilities >= 0.1)
    assert np.isclose(np.sum(probabilities), 1.0)


def test_mixed_gibbs_passes_seed_to_samplers_without_rng_argument():
    space = ParameterSpace(
        names=("x", "template"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )

    def log_prob(theta):
        theta = np.asarray(theta, dtype=float)
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        return prior - 0.5 * theta[0] ** 2

    seen_seeds = []

    def seed_only_sampler(posterior, x0, seed=None):
        seen_seeds.append(seed)
        x0 = np.asarray(x0, dtype=float)
        return SamplingResult(samples=x0[None, :], logp=np.asarray([posterior.log_prob_fn(x0)]))

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    run_mixed_gibbs(
        posterior,
        space,
        x0=np.asarray([0.0, 0.0]),
        nsteps=3,
        continuous_sampler=seed_only_sampler,
        rng=np.random.default_rng(5),
    )

    assert len(seen_seeds) == 3
    assert all(isinstance(seed, int) for seed in seen_seeds)


def test_mixed_gibbs_rejects_pocomc_continuous_sampler_by_name():
    space = ParameterSpace(
        names=("x", "template"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )

    def log_prob(theta):
        return space.log_prior(theta)

    def run_pocomc(posterior, bounds=None):
        del posterior, bounds
        raise AssertionError("pocoMC should be rejected before it is called.")

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    with pytest.raises(ValueError, match="does not support run_pocomc"):
        run_mixed_gibbs(
            posterior,
            space,
            x0=np.asarray([0.0, 0.0]),
            nsteps=1,
            continuous_sampler=run_pocomc,
            rng=np.random.default_rng(6),
        )
