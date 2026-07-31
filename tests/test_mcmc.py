import numpy as np
import pytest

from inftools.core import Posterior
from composed.backends.mock import MockBackend
from composed.data import SEDDataset
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, DeltaPrior, UniformPrior
from composed.runners import build_inftools_posterior


def test_run_emcee_seed_reproducible_when_emcee_installed():
    pytest.importorskip("emcee")
    from inftools.mcmc import run_emcee

    posterior = Posterior(
        log_prob_fn=lambda theta: -0.5 * float(theta[0] ** 2),
        dim=1,
        theta_names=["x"],
    )

    result_a = run_emcee(posterior, x0=np.array([0.1]), nwalkers=8, nsteps=12, seed=123, progress=False)
    result_b = run_emcee(posterior, x0=np.array([0.1]), nwalkers=8, nsteps=12, seed=123, progress=False)

    assert np.allclose(result_a.samples, result_b.samples)
    assert np.allclose(result_a.logp, result_b.logp)
    assert np.allclose(result_a.meta["raw_chain"], result_b.meta["raw_chain"])
    assert result_a.meta["diagnostic_chain"].shape == (12, 8, 1)
    assert result_a.meta["acceptance_fraction"].shape == (8,)
    assert 0.0 <= result_a.meta["acceptance_fraction_mean"] <= 1.0


def test_run_emcee_rejects_rng_and_seed_together_when_emcee_installed():
    pytest.importorskip("emcee")
    from inftools.mcmc import run_emcee

    posterior = Posterior(log_prob_fn=lambda theta: -0.5 * float(theta[0] ** 2), dim=1)
    with pytest.raises(ValueError, match="either rng or seed"):
        run_emcee(
            posterior,
            x0=np.array([0.0]),
            nwalkers=8,
            nsteps=2,
            rng=np.random.default_rng(1),
            seed=1,
            progress=False,
        )


def test_random_walk_removes_and_restores_delta_prior_axes():
    from inftools.mcmc import run_rw_metropolis

    space = ParameterSpace(
        names=("x", "fixed"),
        priors={"x": UniformPrior(-3.0, 3.0), "fixed": DeltaPrior(7.0)},
    )
    likelihood = GaussianPhotometricLikelihood(
        MockBackend([1.0], band_names=["g"]),
        SEDDataset(["g"], np.asarray([1.0]), np.asarray([0.2])),
        space,
    )
    posterior = build_inftools_posterior(likelihood)

    result = run_rw_metropolis(
        posterior,
        x0=np.asarray([0.0, 7.0]),
        nsteps=100,
        proposal_cov=np.diag([0.2, 99.0]),
        rng=np.random.default_rng(4),
    )

    assert result.samples.shape == (100, 2)
    assert np.all(result.samples[:, 1] == 7.0)
    assert np.std(result.samples[:, 0]) > 0.0
    assert result.meta["accept_rate"] > 0.0
    assert result.meta["diagnostic_chain"].shape == (100, 1, 2)


def test_continuous_mcmc_rejects_choice_prior_with_sampler_guidance():
    from inftools.mcmc import run_rw_metropolis

    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0, 1.0])})
    likelihood = GaussianPhotometricLikelihood(
        MockBackend([1.0], band_names=["g"]),
        SEDDataset(["g"], np.asarray([1.0]), np.asarray([0.2])),
        space,
    )

    with pytest.raises(ValueError, match="cannot propose discrete parameter"):
        run_rw_metropolis(build_inftools_posterior(likelihood), x0=np.asarray([0.0]), nsteps=2)


def test_emcee_removes_and_restores_delta_prior_axes_when_installed():
    pytest.importorskip("emcee")
    from inftools.mcmc import run_emcee

    space = ParameterSpace(
        names=("x", "fixed"),
        priors={"x": UniformPrior(-3.0, 3.0), "fixed": DeltaPrior(7.0)},
    )
    likelihood = GaussianPhotometricLikelihood(
        MockBackend([1.0], band_names=["g"]),
        SEDDataset(["g"], np.asarray([1.0]), np.asarray([0.2])),
        space,
    )
    result = run_emcee(
        build_inftools_posterior(likelihood),
        x0=np.asarray([0.0, 7.0]),
        nwalkers=8,
        nsteps=10,
        seed=5,
        progress=False,
    )

    assert result.samples.shape[1] == 2
    assert np.all(result.samples[:, 1] == 7.0)
    assert result.meta["raw_chain"].shape[-1] == 2
