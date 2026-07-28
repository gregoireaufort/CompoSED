# inftools/tamis_adapter.py

from __future__ import annotations
from typing import Optional, Dict, Any
import warnings

import numpy as np
from dataclasses import dataclass
from scipy.stats import qmc

from .core import Posterior, SamplingResult, Array

try:
    import TAMIS
    _HAS_TAMIS = True
    _TAMIS_IMPORT_ERROR = None
except ImportError as exc:  # safety if you don't have TAMIS installed in a given env
    TAMIS = None
    _HAS_TAMIS = False
    _TAMIS_IMPORT_ERROR = exc


class ExperimentalExternalTAMISWarning(UserWarning):
    """Warning emitted by the separately packaged historical TAMIS adapter."""


def qmc_sobol_means(n, d, seed=0, eps=1e-6):
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = eng.random(n=n)
    u = np.clip(u, eps, 1.0 - eps)
    return u


def _weighted_cov(samples: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted covariance matrix."""
    w = np.asarray(w, dtype=float)
    w = w / np.sum(w)
    mean = np.average(samples, axis=0, weights=w)
    xm = samples - mean
    cov = np.cov(xm.T, aweights=w, bias=False)
    return cov

@dataclass
class TamisResult:
    samples: np.ndarray
    weights: np.ndarray
    cov: np.ndarray
    meta: dict


class _TamisTarget:
    """
    Adapter that makes a Posterior look like a TAMIS 'target'.

    map_func: optional callable (f, iterable) -> list of floats
              If None, we just do a serial Python loop.
              If not None, you can pass a multiprocessing-based mapper.
    """
    def __init__(self, posterior, map_func=None):
        self.posterior = posterior
        self.dim = posterior.dim
        self._map_func = map_func

    def log_prior(self, sample):
        # Prior is folded inside posterior.log_prob_fn already
        # Return zeros so TAMIS treats all of it as "likelihood"
        sample = np.atleast_2d(sample)
        return np.zeros(len(sample))

    def log_likelihood(self, sample):
        """
        sample: (N, dim)
        returns array of shape (N,) with log posterior values.
        """
        sample = np.atleast_2d(sample)
        if self._map_func is None:
            vals = [self.posterior.log_prob_fn(theta) for theta in sample]
        else:
            # map_func takes (func, iterable)
            vals = self._map_func(self.posterior.log_prob_fn, list(sample))
        return np.asarray(vals)
class PosteriorTarget:
    """
    Minimal 'target' class to satisfy TAMIS' interface:

      - .dim
      - .log_prior(sample)
      - .log_likelihood(sample=...)

    We set log_prior = 0 and log_likelihood = log_posterior, since TAMIS
    only needs the unnormalised log density.
    """
    def __init__(self, log_post_fn, dim: int):
        self._log_post_fn = log_post_fn
        self.dim = dim

    def log_prior(self, sample):
        sample = np.atleast_2d(sample)
        return np.zeros(sample.shape[0])

    def log_likelihood(self, sample=None):
        sample = np.atleast_2d(sample)
        vals = [self._log_post_fn(theta) for theta in sample]
        return np.array(vals)


class _LogProbWork:
    """
    Picklable log-prob wrapper to avoid nested functions (required for multiprocessing).
    If transform is provided, expects inputs in y-space and evaluates:
        log p(theta(y)) + log|det dtheta/dy|
    Otherwise, just calls base_log_prob_fn(x).
    """
    def __init__(self, base_log_prob_fn, transform=None):
        self.base_log_prob_fn = base_log_prob_fn
        self.transform = transform

    def __call__(self, x):
        x = np.asarray(x, float)
        if self.transform is None:
            return self.base_log_prob_fn(x)

        theta = self.transform.y_to_theta(x)
        lp = self.base_log_prob_fn(theta)
        # add Jacobian term for change of variables y -> theta
        lp = lp + self.transform.log_abs_det_jac(x)
        return lp


def run_tamis(
    posterior,
    x0,
    n_comp=5,
    T_max=20,
    n_per_iter=1000,
    init_span=5.0,
    var0=None,
    map_func=None,
    tamis_kwargs=None,
    transform=None,
    init="qmc",
    seed=None,
    qmc_engine="sobol",
    qmc_eps=1e-6,
    jitter_means=1e-3,
    return_space="theta",  # "theta" or "work"
):
    """
    Run TAMIS on a given Posterior, optionally in transformed (unconstrained) space.

    Key point: to make multiprocessing work, we avoid nested closures and use a
    picklable callable (_LogProbWork) for the log-prob passed to pool.map.

    Parameters
    ----------
    posterior : Posterior
        Posterior over theta-space (original parameterization).
        posterior.log_prob_fn must be picklable if you use multiprocessing.
    x0 : array-like
        Reference point (typically MAP in theta space). Only used for some init modes.
    transform : object or None
        Must have methods y_to_theta(y), theta_to_y(theta), log_abs_det_jac(y).
        If provided, TAMIS runs in y-space (R^d), targeting p(theta(y))|J|.
    init : {"qmc","around_x0"}
        How to place initial mixture means in y-space (or theta-space if transform is None).
    var0 : array-like or float or "auto_local"
        Initial diagonal variance in the sampling space (y if transform else theta).
        "auto_local" sets it based on component spread.
    map_func : callable or None
        Parallel map. Must accept (func, iterable) and return list. Example:
            lambda f, xs: pool.map(f, xs)

    return_space : {"theta","work"}
        If "theta", returns samples in theta-space (recommended).
        If "work", returns samples in y-space when transform is not None.

    Returns
    -------
    TamisResult
    """
    import numpy as np

    if not _HAS_TAMIS:
        raise ImportError(
            "The external TAMIS adapter requires a package that provides `import TAMIS`; "
            "it is not distributed by CompoSED or its sampler extra. Use composed.MixedTAMIS "
            "for the self-contained implementation, or install the external TAMIS package "
            "explicitly before using composed.TAMIS."
        ) from _TAMIS_IMPORT_ERROR

    warnings.warn(
        "The external TAMIS adapter is experimental in CompoSED 0.1.1 and "
        "depends on a separately installed `TAMIS` package. Prefer "
        "composed.MixedTAMIS for the self-contained supported implementation.",
        ExperimentalExternalTAMISWarning,
        stacklevel=2,
    )

    rng = np.random.default_rng(seed)
    dim = int(posterior.dim)
    x0 = np.asarray(x0, float)

    # ---------------------------------------
    # Decide "work space" (what TAMIS samples)
    # ---------------------------------------
    if transform is None:
        work_dim = dim
        x0_work = x0
    else:
        work_dim = dim
        x0_work = transform.theta_to_y(x0)

    # ---------------------------------------
    # Build a picklable working log-prob
    # ---------------------------------------
    log_prob_work = _LogProbWork(posterior.log_prob_fn, transform=transform)

    # Wrap into a Posterior-like object TAMIS expects
    # (keep theta_names for diagnostics)
    post_work = Posterior(
        log_prob_fn=log_prob_work,
        dim=work_dim,
        theta_names=getattr(posterior, "theta_names", None),
    )

    # ---------------------------------------
    # Initialize mixture means
    # ---------------------------------------
    if init == "qmc":
        # QMC points in (0,1)^d -> (eps,1-eps)^d -> logit -> y-space
        try:
            from scipy.stats import qmc
        except Exception as e:
            raise ImportError("scipy.stats.qmc is required for init='qmc'") from e

        if qmc_engine.lower() == "sobol":
            engine = qmc.Sobol(d=work_dim, scramble=True, seed=seed)
            u = engine.random(n_comp)
        elif qmc_engine.lower() == "halton":
            engine = qmc.Halton(d=work_dim, scramble=True, seed=seed)
            u = engine.random(n_comp)
        else:
            raise ValueError(f"Unknown qmc_engine={qmc_engine!r} (use 'sobol' or 'halton')")

        u = np.clip(u, qmc_eps, 1.0 - qmc_eps)

        if transform is None:
            # If no transform, we still need some bounded region; use around x0 in theta space
            # mapped from u to a box [x0-init_span, x0+init_span]
            init_means = x0_work + (2.0 * u - 1.0) * init_span
        else:
            # With BoxLogitTransform, y is the logit-coordinates of u
            # (this gives a principled spread over the box in theta space)
            init_means = np.log(u) - np.log1p(-u)

        if jitter_means and jitter_means > 0:
            init_means = init_means + jitter_means * rng.standard_normal(size=init_means.shape)

    elif init == "around_x0":
        init_means = x0_work + rng.uniform(
            low=-init_span, high=init_span, size=(n_comp, work_dim)
        )
    else:
        raise ValueError(f"Unknown init={init!r} (use 'qmc' or 'around_x0')")

    # ---------------------------------------
    # Initial diagonal covariance
    # ---------------------------------------
    if isinstance(var0, str) and var0 == "auto_local":
        # set var from component spread (robust-ish): variance per dim of init_means
        v = np.var(init_means, axis=0)
        v = np.maximum(v, 1e-3)
        var0_vec = v
    elif var0 is None:
        var0_vec = (init_span ** 2) * np.ones(work_dim)
    else:
        var0_vec = np.broadcast_to(np.asarray(var0, float), (work_dim,)).copy()
        var0_vec = np.maximum(var0_vec, 1e-12)

    init_covs = np.array([np.diag(var0_vec)] * n_comp)
    init_weights = np.ones(n_comp) / n_comp

    prior = [init_means, init_covs, init_weights]
    init_theta = TAMIS.theta_params(prior)

    n_sample = [int(n_per_iter)] * int(T_max)

    # ---------------------------------------
    # Target (with optional parallel mapping)
    # ---------------------------------------
    target = _TamisTarget(post_work, map_func=map_func)

    # ---------------------------------------
    # TAMIS kwargs
    # ---------------------------------------
    base_kwargs = dict(
        target=target,
        n_comp=int(n_comp),
        init_theta=init_theta,
        ESS_tol=np.inf,
        alpha=200,
        tau=0.0,
        proposal=TAMIS.Mixture_gaussian,
        n_sample=n_sample,
        EM_solver="homemade_full",
        integer_weights=False,
        recycle=True,
        recycling_iters="auto",
        verbose=0,
    )
    if tamis_kwargs is not None:
        base_kwargs.update(tamis_kwargs)

    tamis_obj = TAMIS.TAMIS(**base_kwargs)
    result = tamis_obj.result(T=int(T_max))

    # ---------------------------------------
    # Collect samples/weights
    # ---------------------------------------
    samples_work = np.asarray(result.total_sample)
    weights_norm = np.asarray(result.final_weights, float)
    weights_norm = weights_norm / np.sum(weights_norm)

    # Return in theta-space if requested
    if (transform is not None) and (return_space == "theta"):
        samples = np.array([transform.y_to_theta(y) for y in samples_work])
    else:
        samples = samples_work

    # Weighted covariance in returned space
    mean = np.average(samples, axis=0, weights=weights_norm)
    diff = samples - mean
    cov = (weights_norm[:, None] * diff).T @ diff

    meta = dict(
        weights_norm=weights_norm,
        tamis_object=result,
        betas=result.betas,
        ESS=result.ESS,
        tmprd_ESS=result.tmprd_ESS,
        return_space=return_space,
        work_samples=samples_work if (transform is not None and return_space == "theta") else None,
    )

    return TamisResult(samples=samples, weights=weights_norm, cov=cov, meta=meta)
