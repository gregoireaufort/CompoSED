import importlib.util

import numpy as np
import pytest
from scipy import interpolate


RV_B_SMC_BAR = 2.74


def reference_rv_a_from_mixture_rv(rv, f_a):
    """BEAST/Gordon16 equation for the Milky-Way component R(V)_A."""

    if f_a <= 0.0:
        return 3.1
    return 1.0 / (1.0 / (rv * f_a) - (1.0 - f_a) / (f_a * RV_B_SMC_BAR))


def reference_fitzpatrick99_a_over_av(wave_a, rv):
    """Independent NumPy/SciPy copy of BEAST's Fitzpatrick99 implementation."""

    wave_a = np.asarray(wave_a, dtype=float)
    x = np.clip(1.0e4 / wave_a, 0.3, 10.0)

    c2 = -0.824 + 4.717 / rv
    c1 = 2.030 - 3.007 * c2
    c3 = 3.23
    c4 = 0.41
    x0 = 4.596
    gamma = 0.99

    k = np.zeros_like(x)
    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / np.array([2700.0, 2600.0])

    uv = x >= xcutuv
    k[uv] = c1 + c2 * x[uv] + c3 * x[uv] ** 2 / ((x[uv] ** 2 - x0**2) ** 2 + gamma**2 * x[uv] ** 2)
    yspluv = c1 + c2 * xspluv + c3 * xspluv**2 / ((xspluv**2 - x0**2) ** 2 + gamma**2 * xspluv**2)

    fuv = x >= 5.9
    k[fuv] += c4 * (0.5392 * (x[fuv] - 5.9) ** 2 + 0.05644 * (x[fuv] - 5.9) ** 3)
    k[uv] += rv
    yspluv += rv

    optical = x < xcutuv
    xsplopir = np.zeros(7)
    xsplopir[1:7] = 10000.0 / np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])
    ysplopir = np.zeros(7)
    ysplopir[0:3] = np.array([0.0, 0.26469, 0.82925]) * rv / 3.1
    ysplopir[3:7] = np.array(
        [
            np.poly1d([2.13572e-04, 1.00270, -4.22809e-01])(rv),
            np.poly1d([-7.35778e-05, 1.00216, -5.13540e-02])(rv),
            np.poly1d([-3.32598e-05, 1.00184, 7.00127e-01])(rv),
            np.poly1d([1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1])(rv),
        ]
    )
    spline = interpolate.splrep(np.hstack([xsplopir, xspluv]), np.hstack([ysplopir, yspluv]), k=3)
    k[optical] = interpolate.splev(x[optical], spline)
    return k / rv


def reference_gordon03_smcbar_a_over_av(wave_a):
    """Independent NumPy/SciPy copy of BEAST's Gordon03 SMC-Bar curve."""

    wave_a = np.asarray(wave_a, dtype=float)
    x = np.clip(1.0e4 / wave_a, 0.3, 10.0)

    rv = RV_B_SMC_BAR
    c1 = -4.959 / rv
    c2 = 2.264 / rv
    c3 = 0.389 / rv
    c4 = 0.461 / rv
    x0 = 4.6
    gamma = 1.0

    k = np.zeros_like(x)
    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / np.array([2700.0, 2600.0])

    uv = x >= xcutuv
    k[uv] = 1.0 + c1 + c2 * x[uv] + c3 * x[uv] ** 2 / ((x[uv] ** 2 - x0**2) ** 2 + gamma**2 * x[uv] ** 2)
    yspluv = 1.0 + c1 + c2 * xspluv + c3 * xspluv**2 / ((xspluv**2 - x0**2) ** 2 + gamma**2 * xspluv**2)

    fuv = x >= 5.9
    k[fuv] += c4 * (0.5392 * (x[fuv] - 5.9) ** 2 + 0.05644 * (x[fuv] - 5.9) ** 3)

    optical = x < xcutuv
    xsplopir = np.zeros(9)
    xsplopir[1:10] = 1.0 / np.array([2.198, 1.65, 1.25, 0.81, 0.65, 0.55, 0.44, 0.37])
    ysplopir = np.array([0.0, 0.11, 0.169, 0.25, 0.567, 0.801, 1.00, 1.374, 1.672])
    spline = interpolate.splrep(np.hstack([xsplopir, xspluv]), np.hstack([ysplopir, yspluv]), k=3)
    k[optical] = interpolate.splev(x[optical], spline)
    return k


def reference_gordon16_a_over_av(wave_a, rv, f_a):
    """Reference BEAST/Gordon16 mixture as A(lambda)/A(V)."""

    rv_a = reference_rv_a_from_mixture_rv(rv, f_a)
    component_a = reference_fitzpatrick99_a_over_av(wave_a, rv_a)
    component_b = reference_gordon03_smcbar_a_over_av(wave_a)
    return f_a * component_a + (1.0 - f_a) * component_b


@pytest.mark.jaxcigale
def test_gordon16_jax_curve_matches_independent_beast_reference_grid():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.modules import _gordon16_rvfa_a_over_av

    _, jnp = require_jax()

    wave_a = np.geomspace(1000.0, 30000.0, 256)
    parameter_grid = [
        (2.75, 0.55),
        (2.95, 0.65),
        (3.10, 0.75),
        (3.25, 0.90),
    ]

    for rv, f_a in parameter_grid:
        rv_a = reference_rv_a_from_mixture_rv(rv, f_a)
        assert 2.0 <= rv_a <= 6.0
        expected = reference_gordon16_a_over_av(wave_a, rv=rv, f_a=f_a)
        got = np.asarray(_gordon16_rvfa_a_over_av(jnp.asarray(wave_a), rv=jnp.asarray(rv), f_a=jnp.asarray(f_a)))
        assert np.all(np.isfinite(got))
        assert np.allclose(got, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.jaxcigale
def test_gordon16_transmission_matches_reference_and_is_monotonic_in_av():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.modules import _gordon16_rvfa_a_over_av

    _, jnp = require_jax()

    wave_a = np.asarray([1500.0, 2175.0, 4400.0, 5500.0, 9000.0, 22000.0])
    rv = 3.1
    f_a = 0.7
    reference_curve = reference_gordon16_a_over_av(wave_a, rv=rv, f_a=f_a)
    jax_curve = np.asarray(_gordon16_rvfa_a_over_av(jnp.asarray(wave_a), rv=jnp.asarray(rv), f_a=jnp.asarray(f_a)))
    assert np.allclose(jax_curve, reference_curve, rtol=3e-6, atol=3e-6)

    transmission_av_0p2 = 10.0 ** (-0.4 * 0.2 * jax_curve)
    transmission_av_0p8 = 10.0 ** (-0.4 * 0.8 * jax_curve)
    assert np.all((transmission_av_0p2 > 0.0) & (transmission_av_0p2 <= 1.0))
    assert np.all((transmission_av_0p8 > 0.0) & (transmission_av_0p8 <= transmission_av_0p2))

    v_band = int(np.argmin(np.abs(wave_a - 5500.0)))
    assert np.isclose(jax_curve[v_band], 1.0, rtol=3e-2)
    assert np.isclose(transmission_av_0p8[v_band], 10.0 ** (-0.4 * 0.8), rtol=3e-2)


@pytest.mark.jaxcigale
def test_gordon16_curve_gradients_are_finite_in_physical_domain():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.modules import _gordon16_rvfa_a_over_av

    jax, jnp = require_jax()

    wave_a = jnp.asarray(np.geomspace(1100.0, 25000.0, 128))

    def scalar_metric(theta):
        rv, f_a = theta
        curve = _gordon16_rvfa_a_over_av(wave_a, rv=rv, f_a=f_a)
        return jnp.mean(jnp.log(curve + 1.0e-12))

    grad = jax.grad(scalar_metric)(jnp.asarray([3.05, 0.72]))
    assert np.all(np.isfinite(np.asarray(grad)))
