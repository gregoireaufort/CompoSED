from __future__ import annotations
from typing import Optional, Dict, Any, Sequence
import numpy as np

from .core import Posterior, SamplingResult, Array

try:
    import pocomc as pc
    from scipy.stats import uniform
    _HAS_POCOMC = True
except ImportError:  # if pocomc or scipy not installed in some env
    pc = None
    uniform = None
    _HAS_POCOMC = False


def _weighted_cov(samples: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted covariance matrix."""
    w = np.asarray(w, dtype=float)
    w = w / np.sum(w)
    mean = np.average(samples, axis=0, weights=w)
    xm = samples - mean
    cov = np.cov(xm.T, aweights=w, bias=False)
    return cov


def run_pocomc(
    posterior: Posterior,
    bounds: Optional[Sequence[tuple]] = None,
    prior: Optional[Any] = None,  # pc.Prior, but typed as Any to avoid hard dep
    random_state: Optional[int] = None,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    run_kwargs: Optional[Dict[str, Any]] = None,
) -> SamplingResult:
    """
    Run pocoMC using a separately declared likelihood and sampler prior.

    Parameters
    ----------
    posterior : Posterior
        Must provide ``log_likelihood_fn`` in addition to ``log_prob_fn``.
    bounds : sequence of (low, high), optional
        If `prior` is not provided, build a uniform prior over these bounds for
        each dimension using scipy.stats.uniform.
    prior : pc.Prior, optional
        A pocomc Prior instance. If given, `bounds` is ignored.
    random_state : int, optional
        Seed for pocoMC's RNG.
    sampler_kwargs : dict, optional
        Extra kwargs passed to pc.Sampler(...).
    run_kwargs : dict, optional
        Extra kwargs passed to sampler.run(...), e.g. {"max_calls": ...}.

    Notes
    -----
    * If you pass `bounds`, this uses a **flat prior** in that box. The
      resulting evidence is then for that uniform prior, which is usually fine
      for toy problems.
    ``posterior.log_prob_fn`` is never passed as the likelihood: doing so would
    apply the prior twice and corrupt both posterior shape and evidence.
    """
    if not _HAS_POCOMC:
        raise ImportError(
            "pocomc (and scipy) must be installed to use run_pocomc.\n"
            "Try: pip install pocomc scipy"
        )

    if sampler_kwargs is None:
        sampler_kwargs = {}
    if run_kwargs is None:
        run_kwargs = {}

    dim = posterior.dim
    if posterior.log_likelihood_fn is None:
        raise ValueError(
            "run_pocomc requires Posterior.log_likelihood_fn so the sampler prior is applied exactly once. "
            "Build the Posterior with composed.runners.build_inftools_posterior or provide the callback explicitly."
        )

    # 1) Build prior if not provided
    if prior is None:
        if bounds is None:
            raise ValueError("run_pocomc: either `prior` or `bounds` must be provided.")
        if len(bounds) != dim:
            raise ValueError("run_pocomc: len(bounds) must equal posterior.dim.")

        dists = []
        for (low, high) in bounds:
            scale = high - low
            dists.append(uniform(loc=low, scale=scale))

        prior = pc.Prior(dists)

    # 2) Vectorized data likelihood. The pc.Prior supplied above is the sole
    # prior used by pocoMC.
    def log_likelihood(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.array([posterior.log_likelihood_fn(theta) for theta in x])

    # 3) Build and run the pocoMC sampler
    sampler = pc.Sampler(
        prior=prior,
        likelihood=log_likelihood,
        vectorize=True,
        random_state=random_state,
        **sampler_kwargs,
    )

    sampler.run(**run_kwargs)

    # 4) Extract posterior samples & evidence
    samples, weights, logl, logp = sampler.posterior()
    # pocoMC's logp here is *log prior*, so log posterior = logl + logp
    logpost = logl + logp

    # Normalise weights
    w = weights / np.sum(weights)
    ess = 1.0 / np.sum(w**2)

    # MAP is maximum posterior density, not maximum importance weight.
    map_idx = np.argmax(logpost)
    theta_map = samples[map_idx]

    cov = _weighted_cov(samples, w)

    # Evidence
    logz, logz_err = sampler.evidence()

    meta = {
        "weights": weights,
        "weights_norm": w,
        "log_likelihood": logl,
        "log_prior": logp,
        "ESS": ess,
        "log_evidence": logz,
        "log_evidence_err": logz_err,
        "sampler": sampler,
    }

    return SamplingResult(
        samples=samples,
        logp=logpost,
        map_estimate=theta_map,
        cov=cov,
        meta=meta,
    )
