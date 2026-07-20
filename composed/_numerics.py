"""Small NumPy compatibility helpers used by stable scientific calculations."""

from __future__ import annotations

import numpy as np


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Integrate with the NumPy 1.x or 2.x trapezoid spelling."""

    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(y, x=x, dx=dx, axis=axis)
