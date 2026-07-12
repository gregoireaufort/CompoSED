# inftools/core.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Dict, Any
import numpy as np

Array = np.ndarray
LogProbFn = Callable[[Array], float]


@dataclass
class Posterior:
    """
    Wrapper for a log-posterior function.

    log_prob_fn:
        Function theta -> log p(theta | data), unnormalised is fine.
    dim:
        Dimension of theta.
    theta_names:
        Optional list of parameter names, used in plotting/diagnostics.
    extra:
        Arbitrary metadata (e.g. data, priors, model info).
    """
    log_prob_fn: LogProbFn
    dim: int
    theta_names: Optional[Sequence[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    log_likelihood_fn: Optional[LogProbFn] = None
    log_prior_fn: Optional[LogProbFn] = None


@dataclass
class SamplingResult:
    """
    Standardised output for all samplers.
    """
    samples: Array               # (Nsamples, dim)
    logp: Array                  # (Nsamples,)
    map_estimate: Optional[Array] = None
    cov: Optional[Array] = None
    meta: Dict[str, Any] = field(default_factory=dict)
