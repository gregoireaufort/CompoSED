# inftools/laplace.py

from __future__ import annotations
from typing import Optional, Sequence
import warnings

import numpy as np
from scipy.optimize import minimize

from .core import Posterior, SamplingResult, Array


class ExperimentalLaplaceWarning(UserWarning):
    """Warning emitted by the experimental finite-difference Laplace runner."""


def finite_difference_hessian(
    f,
    x0: Array,
    eps: Optional[Array] = None,
) -> Array:
    """
    Central finite-difference Hessian of a scalar function f at x0.

    f:  callable(theta) -> scalar
    x0: 1D array
    eps: optional step sizes per dimension.
    """
    x0 = np.asarray(x0, dtype=float)
    ndim = x0.size
    H = np.zeros((ndim, ndim), dtype=float)

    if eps is None:
        eps = 1e-3 * np.maximum(np.abs(x0), 1.0)
    eps = np.asarray(eps, dtype=float)

    f0 = f(x0)

    # Diagonal terms
    for i in range(ndim):
        ei = np.zeros(ndim)
        ei[i] = eps[i]
        f_plus = f(x0 + ei)
        f_minus = f(x0 - ei)
        H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (eps[i] ** 2)

    # Off-diagonal terms
    for i in range(ndim):
        for j in range(i + 1, ndim):
            ei = np.zeros(ndim)
            ej = np.zeros(ndim)
            ei[i] = eps[i]
            ej[j] = eps[j]

            f_pp = f(x0 + ei + ej)
            f_pm = f(x0 + ei - ej)
            f_mp = f(x0 - ei + ej)
            f_mm = f(x0 - ei - ej)

            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps[i] * eps[j])
            H[i, j] = H_ij
            H[j, i] = H_ij

    return H


def run_laplace(
    posterior: Posterior,
    x0: Array,
    bounds: Optional[Sequence[tuple]] = None,
    method: str = "Powell",
) -> SamplingResult:
    """
    Laplace approximation: find MAP and approximate covariance using the
    Hessian of -log posterior at the MAP.

    Returns a SamplingResult with:
      - samples: single row = MAP
      - logp: log posterior at MAP
      - cov: Gaussian covariance (H^{-1})
    """

    warnings.warn(
        "inftools.run_laplace is experimental in CompoSED 0.1.1. Its "
        "finite-difference Hessian is not hardened at prior boundaries; inspect "
        "the optimizer result and covariance before scientific use.",
        ExperimentalLaplaceWarning,
        stacklevel=2,
    )

    def neg_logp(theta: Array) -> float:
        val = posterior.log_prob_fn(theta)
        if not np.isfinite(val):
            return 1e50
        return -val

    # MAP via optimisation
    opt = minimize(neg_logp, x0, method=method, bounds=bounds)
    theta_map = opt.x

    # Hessian at MAP
    H = finite_difference_hessian(neg_logp, theta_map)
    jitter = 1e-10 * np.eye(posterior.dim)
    cov = np.linalg.inv(H + jitter)

    logp_map = posterior.log_prob_fn(theta_map)

    return SamplingResult(
        samples=theta_map[None, :],
        logp=np.array([logp_map]),
        map_estimate=theta_map,
        cov=cov,
        meta={
            "opt_result": opt,
            "optimizer_success": bool(opt.success),
            "optimizer_status": int(opt.status),
            "optimizer_message": str(opt.message),
            "optimizer_nfev": int(opt.nfev),
            "H": H,
        },
    )
