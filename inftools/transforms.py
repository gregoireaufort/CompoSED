import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(u):
    return np.log(u) - np.log1p(-u)

class BoxLogitTransform:
    """
    y in R^d  -> u = sigmoid(y) in (0,1)^d -> theta = low + (high-low)*u
    """
    def __init__(self, low, high, eps=1e-12):
        self.low = np.asarray(low, float)
        self.high = np.asarray(high, float)
        self.eps = float(eps)
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have same shape")
        if np.any(self.high <= self.low):
            raise ValueError("Need high > low elementwise")
        self.scale = self.high - self.low

    def y_to_theta(self, y):
        u = sigmoid(np.asarray(y, float))
        return self.low + self.scale * u

    def theta_to_y(self, theta):
        theta = np.asarray(theta, float)
        u = (theta - self.low) / self.scale
        u = np.clip(u, self.eps, 1.0 - self.eps)
        return logit(u)

    def log_abs_det_jac(self, y):
        values = self.log_abs_det_jac_each(y)
        if np.asarray(y).ndim == 1:
            return float(values[0])
        return values

    def log_abs_det_jac_each(self, y):
        y = np.asarray(y, float)
        y = np.atleast_2d(y)
        u = sigmoid(y)
        u = np.clip(u, self.eps, 1.0 - self.eps)
        return np.sum(np.log(self.scale)[None, :] + np.log(u) + np.log1p(-u), axis=1)


def box_logit_transform_from_parameter_space(parameter_space, names=None):
    """Build a box-logit transform from bounded continuous priors.

    The returned transform maps unconstrained coordinates to the physical
    parameter values for ``names``. Supported priors are currently those with
    explicit finite ``low`` and ``high`` attributes, such as ``UniformPrior``
    and ``LogUniformPrior``. Discrete choices should stay outside this
    transform and be handled as discrete variables.
    """

    if names is None:
        names = tuple(parameter_space.names)
    else:
        names = tuple(str(name) for name in names)

    low = []
    high = []
    for name in names:
        prior = parameter_space.priors.get(name)
        if prior is None or not hasattr(prior, "low") or not hasattr(prior, "high"):
            raise ValueError(
                f"Cannot build a box-logit transform for {name!r}; "
                "the prior must expose finite low/high bounds."
            )
        lo = float(prior.low)
        hi = float(prior.high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Parameter {name!r} has invalid finite bounds: ({lo}, {hi}).")
        low.append(lo)
        high.append(hi)

    transform = BoxLogitTransform(low=np.asarray(low, dtype=float), high=np.asarray(high, dtype=float))
    transform.names = names
    return transform
