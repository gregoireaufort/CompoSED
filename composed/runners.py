from __future__ import annotations

import numpy as np

from inftools import Posterior


def build_inftools_posterior(likelihood, theta_names=None) -> Posterior:
    names = list(theta_names if theta_names is not None else likelihood.parameter_space.names)
    return Posterior(log_prob_fn=lambda theta: float(likelihood.log_prob(np.asarray(theta, dtype=float))), dim=len(names), theta_names=names)
