"""Small differentiable JAX-CIGALE photometry demo.

This uses the analytic stellar module so the plumbing can run without DSPS SSP
data. For science runs, replace ``analytic_stellar_module`` with
``dsps_stellar_module(ssp_data)``.
"""

from __future__ import annotations

import numpy as np

from composed.experimental.jaxcigale import (
    GaussianPhotometricData,
    JaxFilterSet,
    JaxParameterSpace,
    UniformJaxPrior,
    analytic_stellar_module,
    build_jax_sed_model,
    calzetti_attenuation_module,
    delayed_sfh_module,
    madau_igm_module,
    no_nebular_module,
    redshift_module,
)
from composed.experimental.jaxcigale.dependencies import require_jax


def make_tophat_filters() -> JaxFilterSet:
    wave = np.linspace(3000.0, 12000.0, 400)
    curves = []
    names = ["g_like", "r_like", "i_like"]
    for center in [4800.0, 6200.0, 7600.0]:
        curves.append(np.exp(-0.5 * ((wave - center) / 450.0) ** 2))
    return JaxFilterSet.from_curves(names, [wave, wave, wave], curves)


def main() -> None:
    jax, jnp = require_jax()

    wave_rest_a = np.linspace(900.0, 25000.0, 1200)
    age_grid_gyr = np.linspace(0.02, 10.0, 128)
    filters = make_tophat_filters()

    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol", "dust2", "dust_slope", "uv_bump"],
        priors={
            "log10_mass": UniformJaxPrior(8.0, 12.0),
            "z": UniformJaxPrior(0.0, 3.0),
            "tau_gyr": UniformJaxPrior(0.2, 8.0),
            "tage_gyr": UniformJaxPrior(0.2, 10.0),
            "logzsol": UniformJaxPrior(-1.0, 0.3),
            "dust2": UniformJaxPrior(0.0, 2.0),
            "dust_slope": UniformJaxPrior(-0.7, 0.3),
            "uv_bump": UniformJaxPrior(0.0, 2.0),
        },
    )

    modules = [
        delayed_sfh_module(age_grid_gyr),
        analytic_stellar_module(),
        no_nebular_module(),
        calzetti_attenuation_module(),
        madau_igm_module(),
        redshift_module(),
    ]
    model = build_jax_sed_model(modules, wave_rest_a, filters, space)

    theta = space.from_dict(
        {
            "log10_mass": 10.0,
            "z": 0.5,
            "tau_gyr": 2.0,
            "tage_gyr": 4.0,
            "logzsol": -0.3,
            "dust2": 0.4,
            "dust_slope": -0.2,
            "uv_bump": 0.5,
        }
    )
    theta = jnp.asarray(theta)
    phot = jax.jit(model.predict_photometry)(theta)
    sigma = 0.05 * np.asarray(phot)
    data = GaussianPhotometricData(np.asarray(phot), sigma)
    logp = jax.jit(lambda x: model.log_prob(x, data))(theta)

    print("bands:", filters.names)
    print("photometry [maggies]:", np.asarray(phot))
    print("self log posterior:", float(logp))


if __name__ == "__main__":
    main()
