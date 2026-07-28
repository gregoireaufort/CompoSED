import numpy as np

from inftools.core import Posterior
from inftools.laplace import ExperimentalLaplaceWarning, run_laplace


def test_laplace_warns_that_runner_is_experimental():
    posterior = Posterior(
        log_prob_fn=lambda theta: -0.5 * float(np.sum(np.asarray(theta) ** 2)),
        dim=1,
        theta_names=("x",),
    )

    with np.testing.assert_warns(ExperimentalLaplaceWarning):
        result = run_laplace(posterior, np.asarray([0.5]))

    assert result.samples.shape == (1, 1)
    assert np.all(np.isfinite(result.cov))
