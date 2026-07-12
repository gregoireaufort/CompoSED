from types import SimpleNamespace

import numpy as np
import pytest

from inftools.core import Posterior
import inftools.pocomc_adapter as adapter


class FakePocoSampler:
    last_likelihood_values = None

    def __init__(self, prior, likelihood, vectorize, random_state, **kwargs):
        del prior, vectorize, random_state, kwargs
        self.likelihood = likelihood

    def run(self, **kwargs):
        del kwargs
        FakePocoSampler.last_likelihood_values = self.likelihood(np.asarray([[0.0], [1.0]]))

    def posterior(self):
        samples = np.asarray([[0.0], [1.0]])
        weights = np.asarray([0.9, 0.1])
        log_likelihood = np.asarray([-10.0, 0.0])
        log_prior = np.asarray([0.0, 0.0])
        return samples, weights, log_likelihood, log_prior

    def evidence(self):
        return -1.0, 0.1


def test_pocomc_uses_likelihood_only_and_map_uses_posterior_density(monkeypatch):
    monkeypatch.setattr(adapter, "_HAS_POCOMC", True)
    monkeypatch.setattr(adapter, "pc", SimpleNamespace(Sampler=FakePocoSampler))
    posterior = Posterior(
        log_prob_fn=lambda theta: -100.0 + float(theta[0]),
        log_likelihood_fn=lambda theta: float(theta[0]),
        log_prior_fn=lambda theta: -100.0,
        dim=1,
    )

    result = adapter.run_pocomc(posterior, prior=object())

    assert np.allclose(FakePocoSampler.last_likelihood_values, [0.0, 1.0])
    assert np.allclose(result.map_estimate, [1.0])


def test_pocomc_rejects_posterior_without_separate_likelihood(monkeypatch):
    monkeypatch.setattr(adapter, "_HAS_POCOMC", True)
    posterior = Posterior(log_prob_fn=lambda theta: -0.5 * float(theta[0]) ** 2, dim=1)

    with pytest.raises(ValueError, match="log_likelihood_fn"):
        adapter.run_pocomc(posterior, prior=object())
