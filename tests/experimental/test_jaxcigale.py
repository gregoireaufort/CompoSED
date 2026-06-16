import importlib.util
import math

import numpy as np
import pytest


def trapz_numpy(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def test_jaxcigale_imports_without_jax_dependency():
    import composed.experimental.jaxcigale as jaxcigale

    assert hasattr(jaxcigale, "build_jax_sed_model")
    assert hasattr(jaxcigale, "JaxParameterSpace")
    assert hasattr(jaxcigale, "find_map_initial_position")


def test_missing_jax_error_is_helpful_when_jax_unavailable():
    if importlib.util.find_spec("jax") is not None:
        pytest.skip("JAX is installed in this environment.")

    from composed.experimental.jaxcigale.dependencies import require_jax

    with pytest.raises(ImportError, match="requires optional JAX"):
        require_jax()


def test_filter_set_padding_and_numpy_reference():
    from composed.experimental.jaxcigale.photometry import JaxFilterSet, integrate_maggies_numpy

    wave_model = np.linspace(3000.0, 9000.0, 256)
    flat_fnu_cgs = 3631.0e-23
    c_a_per_s = 2.99792458e18
    flat_flam = flat_fnu_cgs * c_a_per_s / wave_model**2

    filters = JaxFilterSet.from_curves(
        ["a", "b"],
        [np.linspace(4000.0, 5000.0, 20), np.linspace(5500.0, 7500.0, 31)],
        [np.ones(20), np.linspace(0.2, 1.0, 31)],
    )
    maggies = integrate_maggies_numpy(wave_model, flat_flam, filters)
    assert filters.wavelength_a.shape == (2, 31)
    assert np.allclose(maggies, np.ones(2), rtol=5e-4)


def test_jax_photometric_data_requires_active_band():
    from composed.experimental.jaxcigale import GaussianPhotometricData

    with pytest.raises(ValueError, match="at least one active band"):
        GaussianPhotometricData(
            flux_maggies=np.array([1.0, 2.0]),
            sigma_maggies=np.array([0.1, 0.2]),
            mask=np.array([False, False]),
        )


def test_jax_spectral_data_requires_active_pixel():
    from composed.experimental.jaxcigale import GaussianSpectralData

    with pytest.raises(ValueError, match="at least one active spectral pixel"):
        GaussianSpectralData(
            wavelength_obs_a=np.array([4000.0, 5000.0]),
            flux_lambda_cgs=np.array([1.0, 2.0]),
            sigma_lambda_cgs=np.array([0.1, 0.2]),
            mask=np.array([False, False]),
        )


@pytest.mark.jaxcigale
def test_bin_average_constant_spectrum_requires_edge_coverage():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import bin_average_spectrum

    pixel_edges = np.array([0.5, 1.5, 2.5, 3.5])
    with pytest.raises(ValueError, match="cover all spectral pixel edges"):
        bin_average_spectrum(np.array([1.0, 2.0, 3.0]), np.ones(3), pixel_edges)

    wave = np.array([0.5, 1.0, 2.0, 3.0, 3.5])
    out = bin_average_spectrum(wave, np.ones_like(wave), pixel_edges)
    assert np.allclose(np.asarray(out), np.ones(3))


def test_restricted_cigale_bridge_builds_static_module_specs_without_jax():
    from composed.experimental.jaxcigale.bridge import restricted_cigale_modules_to_jaxcigale

    specs = restricted_cigale_modules_to_jaxcigale(
        ["sfhdelayed", "bc03", "nebular", "dustatt_modified_starburst", "redshifting"],
        age_grid_gyr=np.linspace(0.02, 10.0, 16),
        ssp_data=None,
    )
    assert [spec.name for spec in specs] == [
        "delayed_sfh",
        "analytic_stellar",
        "no_nebular",
        "modified_starburst_attenuation",
        "madau_igm",
        "redshift",
    ]


@pytest.mark.jaxcigale
def test_build_jax_sed_model_auto_adds_dust_emission_after_attenuation():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        calzetti_attenuation_module,
        madau_igm_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.geomspace(1.0e3, 3.0e6, 512)
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(4000.0, 6000.0, 16)], [np.ones(16)])
    space = JaxParameterSpace(names=["A_V"], priors={"A_V": UniformJaxPrior(0.0, 2.0)})
    model = build_jax_sed_model(
        [
            calzetti_attenuation_module(av_parameter="A_V", slope_parameter=None, bump_amplitude_parameter=None),
            madau_igm_module(),
            redshift_module(),
        ],
        wave_rest,
        filters,
        space,
        fixed_parameters={"z": 0.5, "log10_mass": 0.0},
    )

    assert [module.name for module in model.modules] == [
        "calzetti_attenuation",
        "modified_blackbody_dust",
        "madau_igm",
        "redshift",
    ]

    stellar = jnp.ones_like(jnp.asarray(wave_rest))
    pre_dust = model.initial_state()._replace(
        stellar_lum_lsun_per_a=stellar,
        total_lum_lsun_per_a=stellar,
    )
    params = model.params_from_theta(jnp.asarray([1.0]))
    attenuated = model.modules[0].apply(params, pre_dust)
    reradiated = model.modules[1].apply(params, attenuated)

    dust_lum = np.asarray(reradiated.dust_lum_lsun_per_a)
    assert float(attenuated.absorbed_lum_lsun) > 0.0
    assert np.all(np.isfinite(dust_lum))
    assert np.all(dust_lum >= 0.0)
    assert np.isclose(trapz_numpy(dust_lum, wave_rest), float(attenuated.absorbed_lum_lsun), rtol=5e-3)


@pytest.mark.jaxcigale
def test_build_jax_sed_model_allows_explicit_attenuation_only_diagnostics():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        calzetti_attenuation_module,
        redshift_module,
    )

    wave_rest = np.geomspace(1.0e3, 3.0e4, 64)
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(4000.0, 6000.0, 16)], [np.ones(16)])
    space = JaxParameterSpace(names=["A_V"], priors={"A_V": UniformJaxPrior(0.0, 2.0)})
    model = build_jax_sed_model(
        [
            calzetti_attenuation_module(av_parameter="A_V", slope_parameter=None, bump_amplitude_parameter=None),
            redshift_module(),
        ],
        wave_rest,
        filters,
        space,
        fixed_parameters={"z": 0.5, "log10_mass": 0.0},
        dust_mode="attenuation_only",
    )

    assert [module.name for module in model.modules] == ["calzetti_attenuation", "redshift"]


def test_dsps_default_ssp_is_rejected_for_nebular_cue(tmp_path):
    from composed.experimental.jaxcigale.ssp_data import require_continuum_ssp_path

    default_file = tmp_path / "ssp_data_fsps_v3.2_lgmet_age.h5"
    default_file.write_bytes(b"not a real hdf5 file; filename provenance is enough")

    with pytest.raises(ValueError, match="includes FSPS nebular emission"):
        require_continuum_ssp_path(default_file)


def test_continuum_ssp_filename_is_allowed(tmp_path):
    from composed.experimental.jaxcigale.ssp_data import require_continuum_ssp_path

    continuum_file = tmp_path / "ssp_data_continuum_fsps_v3.2_lgmet_age.h5"
    continuum_file.write_bytes(b"not a real hdf5 file; filename provenance is enough")

    assert require_continuum_ssp_path(continuum_file) == continuum_file


def test_unknown_ssp_filename_requires_explicit_override(tmp_path):
    from composed.experimental.jaxcigale.ssp_data import require_continuum_ssp_path

    unknown_file = tmp_path / "my_ssp_grid.h5"
    unknown_file.write_bytes(b"not a real hdf5 file; filename provenance is enough")

    with pytest.raises(ValueError, match="Cannot verify"):
        require_continuum_ssp_path(unknown_file)
    assert require_continuum_ssp_path(unknown_file, allow_nebular_included=True) == unknown_file


@pytest.mark.jaxcigale
def test_qmc_nadam_initialization_finds_simple_map():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import JaxParameterSpace, UniformJaxPrior, find_map_initial_position
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    class QuadraticPosterior:
        def __init__(self):
            self.parameter_space = JaxParameterSpace(
                names=("x", "y"),
                priors={
                    "x": UniformJaxPrior(-2.0, 2.0),
                    "y": UniformJaxPrior(-2.0, 2.0),
                },
            )

        def log_prob(self, theta, data):
            del data
            target = jnp.asarray([0.35, -0.75])
            scale = jnp.asarray([0.08, 0.12])
            residual = (theta - target) / scale
            return self.parameter_space.log_prior(theta) - 0.5 * jnp.sum(residual**2)

    model = QuadraticPosterior()
    result = find_map_initial_position(
        model,
        data=None,
        initial_theta=np.asarray([1.8, 1.8]),
        num_candidates=128,
        num_starts=8,
        optimizer_steps=120,
        learning_rate=0.04,
        batch_size=32,
        rng_seed=11,
    )

    assert result.initial_theta.shape == (2,)
    assert np.allclose(result.initial_theta, np.asarray([0.35, -0.75]), atol=3e-2)
    assert result.optimized_theta.shape == (8, 2)
    assert np.all(np.isfinite(result.optimized_log_density))
    assert result.initial_log_density >= np.max(result.candidate_log_density)


@pytest.mark.jaxcigale
def test_madau_igm_module_can_use_fitted_factor_parameter():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        delayed_sfh_module,
        madau_igm_module,
        no_nebular_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(700.0, 5000.0, 512)
    age_grid = np.linspace(0.02, 1.0, 64)
    # At z=5 this observed filter samples rest-frame wavelengths below Ly-alpha,
    # so increasing the IGM factor must reduce the predicted flux.
    filter_wave = np.linspace(5500.0, 7500.0, 128)
    filters = JaxFilterSet.from_curves(["observed_uv"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol", "igm_factor"],
        priors={
            "log10_mass": UniformJaxPrior(8.0, 12.0),
            "z": UniformJaxPrior(4.0, 6.0),
            "tau_gyr": UniformJaxPrior(0.1, 3.0),
            "tage_gyr": UniformJaxPrior(0.1, 1.0),
            "logzsol": UniformJaxPrior(-1.0, 0.2),
            "igm_factor": UniformJaxPrior(0.0, 2.0),
        },
    )
    model = build_jax_sed_model(
        [
            delayed_sfh_module(age_grid),
            analytic_stellar_module(),
            no_nebular_module(),
            madau_igm_module(igm_factor_parameter="igm_factor"),
            redshift_module(),
        ],
        wave_rest,
        filters,
        space,
    )

    no_igm = model.predict_photometry(jnp.asarray([10.0, 5.0, 0.5, 0.5, -0.3, 0.0]))[0]
    strong_igm = model.predict_photometry(jnp.asarray([10.0, 5.0, 0.5, 0.5, -0.3, 2.0]))[0]
    assert float(no_igm) > float(strong_igm)


@pytest.mark.jaxcigale
def test_jaxcigale_analytic_chain_jit_and_gradient():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

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

    jax, jnp = require_jax()

    wave_rest = np.linspace(900.0, 20000.0, 512)
    age_grid = np.linspace(0.02, 8.0, 64)
    filter_wave = np.linspace(4000.0, 9000.0, 128)
    filters = JaxFilterSet.from_curves(
        ["blue", "red"],
        [filter_wave, filter_wave],
        [
            np.exp(-0.5 * ((filter_wave - 5000.0) / 400.0) ** 2),
            np.exp(-0.5 * ((filter_wave - 7500.0) / 500.0) ** 2),
        ],
    )
    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol", "dust2", "dust_slope", "uv_bump"],
        priors={name: UniformJaxPrior(-10.0, 20.0) for name in ["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol", "dust2", "dust_slope", "uv_bump"]},
    )
    model = build_jax_sed_model(
        [
            delayed_sfh_module(age_grid),
            analytic_stellar_module(),
            no_nebular_module(),
            calzetti_attenuation_module(),
            madau_igm_module(),
            redshift_module(),
        ],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.5, 2.0, 4.0, -0.3, 0.3, -0.2, 0.5])
    phot = jax.jit(model.predict_photometry)(theta)
    assert phot.shape == (2,)
    assert np.all(np.isfinite(np.asarray(phot)))

    data = GaussianPhotometricData(np.asarray(phot), 0.1 * np.asarray(phot))
    grad = jax.grad(lambda x: model.log_prob(x, data))(theta)
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.jaxcigale
def test_jaxcigale_mass_normalization_applied_once():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        delayed_sfh_module,
        no_nebular_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 256)
    age_grid = np.linspace(0.02, 5.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 128)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"],
        priors={name: UniformJaxPrior(-20.0, 20.0) for name in ["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"]},
    )
    model = build_jax_sed_model(
        [delayed_sfh_module(age_grid), analytic_stellar_module(), no_nebular_module(), redshift_module()],
        wave_rest,
        filters,
        space,
    )
    theta_a = jnp.asarray([9.0, 0.4, 2.0, 3.0, -0.2])
    theta_b = jnp.asarray([10.0, 0.4, 2.0, 3.0, -0.2])
    ratio = model.predict_photometry(theta_b)[0] / model.predict_photometry(theta_a)[0]
    assert np.isclose(float(ratio), 10.0, rtol=1e-6)


@pytest.mark.jaxcigale
def test_jax_redshift_zero_returns_controlled_minus_inf_log_prob():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        GaussianPhotometricData,
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        delayed_sfh_module,
        no_nebular_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.core import observed_flux_from_luminosity
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_obs, flux_obs = observed_flux_from_luminosity(
        jnp.asarray([1000.0, 2000.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray(0.0),
    )
    assert np.all(np.isfinite(np.asarray(wave_obs)))
    assert not np.all(np.isfinite(np.asarray(flux_obs)))

    wave_rest = np.linspace(1000.0, 15000.0, 256)
    age_grid = np.linspace(0.02, 5.0, 32)
    filters = JaxFilterSet.from_curves(["wide"], [np.linspace(4500.0, 8500.0, 128)], [np.ones(128)])
    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"],
        priors={
            "log10_mass": UniformJaxPrior(8.0, 12.0),
            "z": UniformJaxPrior(0.0, 1.0),
            "tau_gyr": UniformJaxPrior(0.1, 5.0),
            "tage_gyr": UniformJaxPrior(0.1, 5.0),
            "logzsol": UniformJaxPrior(-1.0, 0.2),
        },
    )
    model = build_jax_sed_model(
        [delayed_sfh_module(age_grid), analytic_stellar_module(), no_nebular_module(), redshift_module()],
        wave_rest,
        filters,
        space,
    )
    data = GaussianPhotometricData(flux_maggies=np.array([1.0]), sigma_maggies=np.array([0.1]))
    logp = model.log_prob(jnp.asarray([10.0, 0.0, 1.0, 2.0, -0.3]), data)
    assert float(logp) == -np.inf


@pytest.mark.jaxcigale
def test_cosmic_time_sfh_module_outputs_dsps_clock():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        delayed_sfh_cosmic_time_module,
    )
    from composed.experimental.jaxcigale.core import flat_lcdm_age_gyr
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 16)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["z", "tau_gyr", "tage_gyr"],
        priors={name: UniformJaxPrior(0.0, 20.0) for name in ["z", "tau_gyr", "tage_gyr"]},
    )
    model = build_jax_sed_model([delayed_sfh_cosmic_time_module(n_time=64)], wave_rest, filters, space)
    theta = jnp.asarray([1.0, 2.0, 3.0])
    state = model.run_modules(theta)
    time = np.asarray(state.sfh_time_gyr)
    sfr = np.asarray(state.sfr_msun_per_yr)
    t_obs = float(flat_lcdm_age_gyr(jnp.asarray(1.0)))

    assert np.all(np.diff(time) > 0.0)
    assert np.isclose(time[-1], t_obs, rtol=1e-6)
    assert np.isclose(time[0], t_obs - 3.0 + 0.02, rtol=1e-6)
    assert np.isclose(trapz_numpy(sfr, time * 1.0e9), 1.0, rtol=1e-5)


@pytest.mark.jaxcigale
def test_cosmic_time_sfh_rejects_age_older_than_universe():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        delayed_sfh_cosmic_time_module,
    )
    from composed.experimental.jaxcigale.core import flat_lcdm_age_gyr_numpy
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 32)
    filters = JaxFilterSet.from_curves(["wide"], [np.linspace(4500.0, 8500.0, 16)], [np.ones(16)])
    space = JaxParameterSpace(
        names=["z", "tau_gyr", "tage_gyr"],
        priors={name: UniformJaxPrior(0.0, 20.0) for name in ["z", "tau_gyr", "tage_gyr"]},
    )
    model = build_jax_sed_model([delayed_sfh_cosmic_time_module(n_time=32)], wave_rest, filters, space)
    z = 10.0
    impossible_tage = float(flat_lcdm_age_gyr_numpy(z)) + 1.0
    state = model.run_modules(jnp.asarray([z, 0.5, impossible_tage]))

    assert not np.all(np.isfinite(np.asarray(state.sfh_time_gyr)))
    assert not np.all(np.isfinite(np.asarray(state.sfr_msun_per_yr)))


@pytest.mark.jaxcigale
def test_cosmic_time_sfh_module_supports_tage_fraction_table():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        delayed_sfh_cosmic_time_module,
    )
    from composed.experimental.jaxcigale.core import flat_lcdm_age_gyr_numpy
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 16)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["z", "tau_gyr", "tage_fraction"],
        priors={name: UniformJaxPrior(0.0, 20.0) for name in ["z", "tau_gyr", "tage_fraction"]},
    )
    model = build_jax_sed_model(
        [
            delayed_sfh_cosmic_time_module(
                n_time=64,
                tage_parameter="tage_fraction",
                tage_is_fraction_of_universe_age=True,
            )
        ],
        wave_rest,
        filters,
        space,
    )
    z = 1.0
    fraction = 0.5
    state = model.run_modules(jnp.asarray([z, 2.0, fraction]))
    time = np.asarray(state.sfh_time_gyr)
    t_obs = float(flat_lcdm_age_gyr_numpy(z))
    tage = fraction * t_obs

    assert np.all(np.diff(time) > 0.0)
    assert np.isclose(time[-1], t_obs, rtol=5e-4)
    assert np.isclose(time[0], t_obs - tage + 0.02, rtol=5e-4)


@pytest.mark.jaxcigale
def test_constant_and_powerlaw_sfh_modules_normalize_to_one_msun():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        constant_sfh_module,
        powerlaw_sfh_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 16)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    age_grid = np.linspace(0.02, 5.0, 128)

    constant_space = JaxParameterSpace(
        names=["tage_gyr"],
        priors={"tage_gyr": UniformJaxPrior(0.1, 5.0)},
    )
    constant_model = build_jax_sed_model([constant_sfh_module(age_grid)], wave_rest, filters, constant_space)
    constant_state = constant_model.run_modules(jnp.asarray([3.0]))

    powerlaw_space = JaxParameterSpace(
        names=["tage_gyr", "sfh_alpha"],
        priors={
            "tage_gyr": UniformJaxPrior(0.1, 5.0),
            "sfh_alpha": UniformJaxPrior(-2.0, 3.0),
        },
    )
    powerlaw_model = build_jax_sed_model([powerlaw_sfh_module(age_grid)], wave_rest, filters, powerlaw_space)
    powerlaw_state = powerlaw_model.run_modules(jnp.asarray([3.0, 1.2]))

    for state in (constant_state, powerlaw_state):
        time = np.asarray(state.sfh_time_gyr)
        sfr = np.asarray(state.sfr_msun_per_yr)
        assert np.all(np.diff(time) > 0.0)
        assert np.all(np.isfinite(sfr))
        assert np.all(sfr >= 0.0)
        assert np.isclose(trapz_numpy(sfr, time * 1.0e9), 1.0, rtol=1e-5)


@pytest.mark.jaxcigale
def test_cosmic_constant_and_powerlaw_sfh_modules_are_finite_and_normalized():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        constant_sfh_cosmic_time_module,
        powerlaw_sfh_cosmic_time_module,
    )
    from composed.experimental.jaxcigale.core import flat_lcdm_age_gyr
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 16)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])

    constant_space = JaxParameterSpace(
        names=["z", "tage_gyr"],
        priors={"z": UniformJaxPrior(0.0, 5.0), "tage_gyr": UniformJaxPrior(0.1, 5.0)},
    )
    constant_model = build_jax_sed_model(
        [constant_sfh_cosmic_time_module(n_time=64)],
        wave_rest,
        filters,
        constant_space,
    )
    constant_state = constant_model.run_modules(jnp.asarray([1.0, 2.0]))

    powerlaw_space = JaxParameterSpace(
        names=["z", "tage_gyr", "sfh_alpha"],
        priors={
            "z": UniformJaxPrior(0.0, 5.0),
            "tage_gyr": UniformJaxPrior(0.1, 5.0),
            "sfh_alpha": UniformJaxPrior(-2.0, 3.0),
        },
    )
    powerlaw_model = build_jax_sed_model(
        [powerlaw_sfh_cosmic_time_module(n_time=64)],
        wave_rest,
        filters,
        powerlaw_space,
    )
    powerlaw_state = powerlaw_model.run_modules(jnp.asarray([1.0, 2.0, 0.8]))
    t_obs = float(flat_lcdm_age_gyr(jnp.asarray(1.0)))

    for state in (constant_state, powerlaw_state):
        time = np.asarray(state.sfh_time_gyr)
        sfr = np.asarray(state.sfr_msun_per_yr)
        assert np.all(np.diff(time) > 0.0)
        assert np.isclose(time[-1], t_obs, rtol=1e-6)
        assert np.all(np.isfinite(sfr))
        assert np.all(sfr >= 0.0)
        assert np.isclose(trapz_numpy(sfr, time * 1.0e9), 1.0, rtol=1e-5)


@pytest.mark.jaxcigale
def test_redshift_aware_continuity_sfh_hits_age_of_universe_for_fitted_z():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        continuity_sfh_cosmic_time_module,
    )
    from composed.experimental.jaxcigale.core import flat_lcdm_age_gyr_numpy
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 32)
    filters = JaxFilterSet.from_curves(["wide"], [np.linspace(4500.0, 8500.0, 16)], [np.ones(16)])
    lookback_edges = np.asarray([0.0, 0.03, 0.1, 0.3, 1.0])
    logsfr = np.asarray([0.0, 0.2, -0.1, 0.1, -0.2])
    z_table = np.asarray([0.0, 1.0, 2.0])
    age_table = flat_lcdm_age_gyr_numpy(z_table)

    names = ["z"] + [f"logsfr_{i}" for i in range(lookback_edges.size)]
    space = JaxParameterSpace(
        names=names,
        priors={name: UniformJaxPrior(-10.0, 20.0) for name in names},
    )
    model = build_jax_sed_model(
        [
            continuity_sfh_cosmic_time_module(
                lookback_edges,
                age_table_z=z_table,
                age_table_gyr=age_table,
            )
        ],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray(np.concatenate([[1.0], logsfr]))
    state = model.run_modules(theta)

    t_obs = float(age_table[1])
    full_edges = np.concatenate([lookback_edges, [t_obs]])
    widths = full_edges[1:] - full_edges[:-1]
    centers = 0.5 * (full_edges[:-1] + full_edges[1:])
    expected_time = (t_obs - centers)[::-1]
    expected_sfr_recent_to_old = 10.0**logsfr
    expected_sfr_recent_to_old = expected_sfr_recent_to_old / np.sum(expected_sfr_recent_to_old * widths * 1.0e9)

    assert np.allclose(np.asarray(state.sfh_time_gyr), expected_time, rtol=1e-8, atol=1e-10)
    assert np.allclose(np.asarray(state.sfr_msun_per_yr), expected_sfr_recent_to_old[::-1], rtol=1e-8)
    assert np.isclose(np.sum(expected_sfr_recent_to_old * widths * 1.0e9), 1.0, rtol=1e-12)
    assert np.all(np.diff(np.asarray(state.sfh_time_gyr)) > 0.0)


@pytest.mark.jaxcigale
def test_redshift_aware_continuity_sfh_supports_fixed_z_and_clips_old_edges():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        continuity_sfh_cosmic_time_module,
    )
    from composed.experimental.jaxcigale.core import flat_lcdm_age_gyr_numpy
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 32)
    filters = JaxFilterSet.from_curves(["wide"], [np.linspace(4500.0, 8500.0, 16)], [np.ones(16)])
    lookback_edges = np.asarray([0.0, 0.03, 0.1, 0.3, 1.0, 3.0])
    z_table = np.asarray([0.0, 10.0, 20.0])
    age_table = flat_lcdm_age_gyr_numpy(z_table)
    names = [f"logsfr_{i}" for i in range(lookback_edges.size)]
    logsfr = np.linspace(-0.3, 0.4, lookback_edges.size)
    space = JaxParameterSpace(names=names, priors={name: UniformJaxPrior(-5.0, 5.0) for name in names})
    model = build_jax_sed_model(
        [
            continuity_sfh_cosmic_time_module(
                lookback_edges,
                age_table_z=z_table,
                age_table_gyr=age_table,
            )
        ],
        wave_rest,
        filters,
        space,
        fixed_parameters={"z": 10.0},
    )
    state = model.run_modules(jnp.asarray(logsfr))

    time = np.asarray(state.sfh_time_gyr)
    sfr = np.asarray(state.sfr_msun_per_yr)
    t_obs = float(age_table[1])
    assert time.shape == (lookback_edges.size,)
    assert np.all(np.diff(time) > 0.0)
    assert np.all(time > 0.0)
    assert time[-1] < t_obs
    assert np.all(np.isfinite(sfr))
    assert np.all(sfr >= 0.0)


@pytest.mark.jaxcigale
def test_modified_starburst_attenuation_young_old_bookkeeping():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        modified_starburst_attenuation_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(950.0, 9000.0, 128)
    filter_wave = np.linspace(3000.0, 7000.0, 32)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["E_BV_young", "E_BV_old_factor", "powerlaw_slope", "uv_bump_amplitude"],
        priors={
            "E_BV_young": UniformJaxPrior(0.0, 2.0),
            "E_BV_old_factor": UniformJaxPrior(0.0, 1.0),
            "powerlaw_slope": UniformJaxPrior(-1.0, 1.0),
            "uv_bump_amplitude": UniformJaxPrior(0.0, 5.0),
        },
    )
    model = build_jax_sed_model([modified_starburst_attenuation_module()], wave_rest, filters, space)
    initial = model.initial_state()
    wave = initial.wave_rest_a
    young = jnp.ones_like(wave) * 2.0
    old = jnp.ones_like(wave) * 5.0
    total = young + old
    pre_dust = initial._replace(
        stellar_young_lum_lsun_per_a=young,
        stellar_old_lum_lsun_per_a=old,
        stellar_lum_lsun_per_a=total,
        total_lum_lsun_per_a=total,
    )

    module = model.modules[0]
    no_dust = module.apply(model.params_from_theta(jnp.asarray([0.0, 0.44, 0.0, 0.0])), pre_dust)
    young_only_dust = module.apply(model.params_from_theta(jnp.asarray([0.5, 0.0, 0.0, 0.0])), pre_dust)
    both_dust = module.apply(model.params_from_theta(jnp.asarray([0.5, 0.44, 0.0, 0.0])), pre_dust)

    assert np.allclose(np.asarray(no_dust.total_lum_lsun_per_a), np.asarray(total), rtol=1e-7)
    assert float(young_only_dust.absorbed_lum_lsun) > 0.0
    assert float(both_dust.absorbed_lum_lsun) > float(young_only_dust.absorbed_lum_lsun)
    assert np.all(np.asarray(both_dust.total_lum_lsun_per_a) <= np.asarray(total) + 1e-10)

    wave_np = np.asarray(wave)
    uv = int(np.argmin(np.abs(wave_np - 1500.0)))
    optical = int(np.argmin(np.abs(wave_np - 5500.0)))
    transmission = np.asarray(both_dust.total_lum_lsun_per_a / total)
    assert transmission[uv] < transmission[optical]


@pytest.mark.jaxcigale
def test_modified_starburst_attenuates_nebular_lines_with_emission_law():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        modified_starburst_attenuation_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    # H-beta and H-alpha in Angstrom. CCM89 should attenuate H-beta more.
    wave_rest = np.asarray([4861.0, 6563.0])
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(4500.0, 7000.0, 8)], [np.ones(8)])
    space = JaxParameterSpace(
        names=["E_BV_young", "E_BV_old_factor", "E_BV_lines"],
        priors={
            "E_BV_young": UniformJaxPrior(0.0, 2.0),
            "E_BV_old_factor": UniformJaxPrior(0.0, 1.0),
            "E_BV_lines": UniformJaxPrior(0.0, 2.0),
        },
    )
    model = build_jax_sed_model(
        [modified_starburst_attenuation_module(nebular_ebv_parameter="E_BV_lines")],
        wave_rest,
        filters,
        space,
    )
    initial = model.initial_state()
    line_lum = jnp.ones_like(initial.wave_rest_a)
    pre_dust = initial._replace(
        nebular_lum_lsun_per_a=line_lum,
        nebular_line_lum_lsun_per_a=line_lum,
        total_lum_lsun_per_a=line_lum,
    )
    module = model.modules[0]
    no_dust = module.apply(model.params_from_theta(jnp.asarray([0.0, 0.0, 0.0])), pre_dust)
    dust = module.apply(model.params_from_theta(jnp.asarray([0.0, 0.0, 0.5])), pre_dust)

    assert np.allclose(np.asarray(no_dust.nebular_line_lum_lsun_per_a), np.ones(2), rtol=1e-7)
    transmission = np.asarray(dust.nebular_line_lum_lsun_per_a)
    assert 0.0 < transmission[0] < transmission[1] < 1.0
    assert float(dust.absorbed_lum_lsun) > 0.0


@pytest.mark.jaxcigale
def test_gordon16_rvfa_curve_limiting_cases_and_rv_mapping():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.modules import (
        _fitzpatrick99_a_over_av,
        _gordon03_smcbar_a_over_av,
        _gordon16_mixture_rv_from_rv_a,
        _gordon16_rv_a_from_mixture_rv,
        _gordon16_rvfa_a_over_av,
    )

    _, jnp = require_jax()

    wave_a = jnp.asarray([1000.0, 1500.0, 2175.0, 4400.0, 5500.0, 10000.0, 25000.0])
    rv = jnp.asarray(3.1)
    f_a = jnp.asarray(0.7)
    rv_a = _gordon16_rv_a_from_mixture_rv(rv, f_a)
    rv_roundtrip = _gordon16_mixture_rv_from_rv_a(rv_a, f_a)

    assert np.isclose(float(rv_roundtrip), 3.1, rtol=1e-6)

    pure_mw = _gordon16_rvfa_a_over_av(wave_a, rv=rv, f_a=jnp.asarray(1.0))
    pure_f99 = _fitzpatrick99_a_over_av(wave_a, rv=rv)
    pure_smc = _gordon16_rvfa_a_over_av(wave_a, rv=rv, f_a=jnp.asarray(0.0))
    smc_reference = _gordon03_smcbar_a_over_av(wave_a)

    assert np.allclose(np.asarray(pure_mw), np.asarray(pure_f99), rtol=1e-7, atol=1e-7)
    assert np.allclose(np.asarray(pure_smc), np.asarray(smc_reference), rtol=1e-7, atol=1e-7)
    assert np.all(np.isfinite(np.asarray(_gordon16_rvfa_a_over_av(wave_a, rv=rv, f_a=f_a))))
    assert np.isclose(float(_gordon16_rvfa_a_over_av(jnp.asarray([5500.0]), rv=rv, f_a=f_a)[0]), 1.0, rtol=0.03)


@pytest.mark.jaxcigale
def test_gordon16_rvfa_extinction_module_attenuates_young_old_and_nebular():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        gordon16_rvfa_extinction_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.asarray([1500.0, 5500.0])
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(1400.0, 5600.0, 8)], [np.ones(8)])
    space = JaxParameterSpace(
        names=["A_V", "R_V", "f_A", "A_V_old_factor", "A_V_nebular"],
        priors={
            "A_V": UniformJaxPrior(0.0, 5.0),
            "R_V": UniformJaxPrior(2.0, 6.0),
            "f_A": UniformJaxPrior(0.0, 1.0),
            "A_V_old_factor": UniformJaxPrior(0.0, 1.0),
            "A_V_nebular": UniformJaxPrior(0.0, 5.0),
        },
    )
    model = build_jax_sed_model(
        [
            gordon16_rvfa_extinction_module(
                old_av_factor_parameter="A_V_old_factor",
                nebular_av_parameter="A_V_nebular",
            )
        ],
        wave_rest,
        filters,
        space,
    )
    initial = model.initial_state()
    young = jnp.asarray([2.0, 2.0])
    old = jnp.asarray([5.0, 5.0])
    nebular = jnp.asarray([1.0, 1.0])
    total_stellar = young + old
    total = total_stellar + nebular
    pre_dust = initial._replace(
        stellar_young_lum_lsun_per_a=young,
        stellar_old_lum_lsun_per_a=old,
        stellar_lum_lsun_per_a=total_stellar,
        nebular_lum_lsun_per_a=nebular,
        nebular_line_lum_lsun_per_a=nebular,
        total_lum_lsun_per_a=total,
    )

    module = model.modules[0]
    no_dust = module.apply(model.params_from_theta(jnp.asarray([0.0, 3.1, 0.5, 0.0, 0.0])), pre_dust)
    dust = module.apply(model.params_from_theta(jnp.asarray([1.0, 3.1, 0.5, 0.0, 0.5])), pre_dust)

    assert np.allclose(np.asarray(no_dust.total_lum_lsun_per_a), np.asarray(total), rtol=1e-7)
    assert np.all(np.asarray(dust.total_lum_lsun_per_a) < np.asarray(total))
    assert np.allclose(np.asarray(dust.stellar_old_lum_lsun_per_a), np.asarray(old), rtol=1e-7)
    assert np.all(np.asarray(dust.nebular_line_lum_lsun_per_a) < np.asarray(nebular))
    transmission = np.asarray(dust.total_lum_lsun_per_a / total)
    assert transmission[0] < transmission[1]
    assert float(dust.absorbed_lum_lsun) > 0.0


@pytest.mark.jaxcigale
def test_smc_screen_attenuation_has_stronger_uv_than_optical_absorption():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        smc_screen_attenuation_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.asarray([1500.0, 5500.0])
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(1400.0, 5600.0, 8)], [np.ones(8)])
    space = JaxParameterSpace(
        names=["A_V", "A_V_old_factor", "A_V_nebular"],
        priors={
            "A_V": UniformJaxPrior(0.0, 5.0),
            "A_V_old_factor": UniformJaxPrior(0.0, 1.0),
            "A_V_nebular": UniformJaxPrior(0.0, 5.0),
        },
    )
    model = build_jax_sed_model(
        [
            smc_screen_attenuation_module(
                old_av_factor_parameter="A_V_old_factor",
                nebular_av_parameter="A_V_nebular",
            )
        ],
        wave_rest,
        filters,
        space,
    )
    initial = model.initial_state()
    young = jnp.asarray([2.0, 2.0])
    old = jnp.asarray([5.0, 5.0])
    nebular = jnp.asarray([1.0, 1.0])
    total_stellar = young + old
    total = total_stellar + nebular
    pre_dust = initial._replace(
        stellar_young_lum_lsun_per_a=young,
        stellar_old_lum_lsun_per_a=old,
        stellar_lum_lsun_per_a=total_stellar,
        nebular_lum_lsun_per_a=nebular,
        nebular_line_lum_lsun_per_a=nebular,
        total_lum_lsun_per_a=total,
    )

    module = model.modules[0]
    dust = module.apply(model.params_from_theta(jnp.asarray([1.0, 0.0, 0.5])), pre_dust)

    assert np.all(np.asarray(dust.total_lum_lsun_per_a) < np.asarray(total))
    assert np.allclose(np.asarray(dust.stellar_old_lum_lsun_per_a), np.asarray(old), rtol=1e-7)
    transmission = np.asarray(dust.total_lum_lsun_per_a / total)
    assert transmission[0] < transmission[1]
    assert float(dust.absorbed_lum_lsun) > 0.0


@pytest.mark.jaxcigale
def test_two_temperature_dust_conserves_absorbed_luminosity():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        build_jax_sed_model,
        two_temperature_dust_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.geomspace(1.0e4, 3.0e6, 512)
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(1.0e4, 3.0e6, 32)], [np.ones(32)])
    space = JaxParameterSpace(
        names=["dust_cold_temperature", "dust_warm_temperature", "dust_beta", "dust_warm_fraction"],
        priors={
            "dust_cold_temperature": UniformJaxPrior(5.0, 80.0),
            "dust_warm_temperature": UniformJaxPrior(20.0, 200.0),
            "dust_beta": UniformJaxPrior(0.0, 3.0),
            "dust_warm_fraction": UniformJaxPrior(0.0, 1.0),
        },
    )
    model = build_jax_sed_model([two_temperature_dust_module()], wave_rest, filters, space)
    initial = model.initial_state()._replace(absorbed_lum_lsun=jnp.asarray(12.5))
    module = model.modules[0]
    cold_only = module.apply(model.params_from_theta(jnp.asarray([25.0, 80.0, 1.5, 0.0])), initial)
    mixed = module.apply(model.params_from_theta(jnp.asarray([25.0, 80.0, 1.5, 0.4])), initial)

    wave = np.asarray(initial.wave_rest_a)
    mixed_dust = np.asarray(mixed.dust_lum_lsun_per_a)
    cold_dust = np.asarray(cold_only.dust_lum_lsun_per_a)
    assert np.all(np.isfinite(mixed_dust))
    assert np.all(mixed_dust >= 0.0)
    assert np.isclose(trapz_numpy(mixed_dust, wave), 12.5, rtol=1e-5)

    mid_ir = int(np.argmin(np.abs(wave - 3.0e5)))
    far_ir = int(np.argmin(np.abs(wave - 1.5e6)))
    assert mixed_dust[mid_ir] / mixed_dust[far_ir] > cold_dust[mid_ir] / cold_dust[far_ir]


@pytest.mark.jaxcigale
def test_jaxcigale_fixed_parameters_feed_modules():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        delayed_sfh_module,
        no_nebular_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 256)
    age_grid = np.linspace(0.02, 5.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 128)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["log10_mass", "z", "logzsol"],
        priors={name: UniformJaxPrior(-20.0, 20.0) for name in ["log10_mass", "z", "logzsol"]},
    )
    model = build_jax_sed_model(
        [delayed_sfh_module(age_grid), analytic_stellar_module(), no_nebular_module(), redshift_module()],
        wave_rest,
        filters,
        space,
        fixed_parameters={"tau_gyr": 2.0, "tage_gyr": 3.0},
    )
    theta = jnp.asarray([10.0, 0.4, -0.2])
    phot = model.predict_photometry(theta)
    params = model.params_from_theta(theta)
    assert phot.shape == (1,)
    assert np.all(np.isfinite(np.asarray(phot)))
    assert float(params["tau_gyr"]) == 2.0
    assert float(params["tage_gyr"]) == 3.0


@pytest.mark.jaxcigale
def test_photometric_upper_limit_uses_gaussian_cdf():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        GaussianPhotometricData,
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        delayed_sfh_module,
        no_nebular_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 256)
    age_grid = np.linspace(0.02, 5.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 128)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"],
        priors={name: UniformJaxPrior(-20.0, 20.0) for name in ["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"]},
    )
    model = build_jax_sed_model(
        [delayed_sfh_module(age_grid), analytic_stellar_module(), no_nebular_module(), redshift_module()],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.4, 2.0, 3.0, -0.2])
    model_flux = np.asarray(jax.jit(model.predict_photometry)(theta))
    data = GaussianPhotometricData(
        flux_maggies=np.array([np.nan]),
        sigma_maggies=np.array([0.1 * model_flux[0]]),
        upper_limit_maggies=model_flux.copy(),
        upper_limit_mask=np.array([True]),
    )

    logp = float(jax.jit(lambda x: model.log_prob(x, data))(theta))
    log_prior = float(model.parameter_space.log_prior(theta))
    assert np.isclose(logp, log_prior + math.log(0.5), rtol=1e-6)


@pytest.mark.jaxcigale
def test_spectro_photometric_log_prob_combines_both_data_types():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        GaussianPhotometricData,
        GaussianSpectralData,
        GaussianSpectroPhotometricData,
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        delayed_sfh_module,
        no_nebular_module,
        redshift_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 15000.0, 256)
    age_grid = np.linspace(0.02, 5.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 128)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"],
        priors={name: UniformJaxPrior(-20.0, 20.0) for name in ["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"]},
    )
    model = build_jax_sed_model(
        [delayed_sfh_module(age_grid), analytic_stellar_module(), no_nebular_module(), redshift_module()],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.4, 2.0, 3.0, -0.2])
    phot = np.asarray(model.predict_photometry(theta))
    spectral_wave = np.linspace(5000.0, 9000.0, 25)
    spectral_flux = np.asarray(model.predict_spectrum(theta, jnp.asarray(spectral_wave)))
    phot_sigma = 0.1 * np.abs(phot)
    spec_sigma = 0.1 * np.abs(spectral_flux) + 1.0e-40

    data = GaussianSpectroPhotometricData(
        photometry=GaussianPhotometricData(phot, phot_sigma),
        spectrum=GaussianSpectralData(spectral_wave, spectral_flux, spec_sigma, resample_mode="interp"),
    )
    logp = float(jax.jit(lambda x: model.log_prob(x, data))(theta))
    log_prior = float(model.parameter_space.log_prior(theta))
    expected_phot = -0.5 * np.sum(np.log(2.0 * np.pi * phot_sigma**2))
    expected_spec = -0.5 * np.sum(np.log(2.0 * np.pi * spec_sigma**2))
    assert np.isclose(logp, log_prior + expected_phot + expected_spec, rtol=1e-6)


@pytest.mark.jaxcigale
def test_spectral_operator_broadens_then_bins_flux_conservingly():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        bin_average_spectrum,
        gaussian_lsf_smooth_observed,
        pixel_edges_from_centers_numpy,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave = jnp.asarray(np.linspace(5000.0, 5100.0, 401))
    constant_flux = jnp.ones_like(wave) * 3.0
    smoothed_constant = gaussian_lsf_smooth_observed(wave, constant_flux, lsf_fwhm_a=5.0)
    assert np.allclose(np.asarray(smoothed_constant[20:-20]), 3.0, rtol=1e-5)

    coarse_centers = np.linspace(5005.0, 5095.0, 10)
    coarse_edges = pixel_edges_from_centers_numpy(coarse_centers)
    binned_constant = bin_average_spectrum(wave, constant_flux, jnp.asarray(coarse_edges))
    assert np.allclose(np.asarray(binned_constant), 3.0, rtol=1e-6)


@pytest.mark.jaxcigale
def test_spectral_operator_accepts_wavelength_dependent_resolving_power():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import GaussianSpectralData, gaussian_lsf_smooth_observed
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave = np.linspace(6000.0, 7000.0, 80)
    resolving_power = np.linspace(40.0, 300.0, wave.size)
    data = GaussianSpectralData(
        wavelength_obs_a=wave,
        flux_lambda_cgs=np.ones_like(wave),
        sigma_lambda_cgs=np.ones_like(wave),
        resolving_power=resolving_power,
    )
    smoothed = gaussian_lsf_smooth_observed(
        jnp.asarray(wave),
        jnp.ones_like(jnp.asarray(wave)),
        resolving_power=data.resolving_power,
        resolving_power_wavelength_obs_a=jnp.asarray(wave),
    )
    assert np.asarray(data.resolving_power).shape == wave.shape
    assert np.all(np.isfinite(np.asarray(smoothed)))

@pytest.mark.jaxcigale
def test_spectral_likelihood_uses_binning_and_lsf_options():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        GaussianSpectralData,
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        delayed_sfh_module,
        no_nebular_module,
        pixel_edges_from_centers_numpy,
        redshift_module,
    )
    from composed.experimental.jaxcigale.spectroscopy import model_spectrum_on_observed_pixels
    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()

    wave_rest = np.linspace(1000.0, 12000.0, 512)
    age_grid = np.linspace(0.02, 5.0, 32)
    filters = JaxFilterSet.from_curves(["dummy"], [np.linspace(4500.0, 8500.0, 16)], [np.ones(16)])
    space = JaxParameterSpace(
        names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"],
        priors={name: UniformJaxPrior(-20.0, 20.0) for name in ["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"]},
    )
    model = build_jax_sed_model(
        [delayed_sfh_module(age_grid), analytic_stellar_module(), no_nebular_module(), redshift_module()],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.2, 2.0, 3.0, -0.2])
    state = model.run_modules_mass_scaled(theta)
    spectral_wave = np.linspace(3600.0, 9000.0, 60)
    pixel_edges = pixel_edges_from_centers_numpy(spectral_wave)
    spectral_flux = np.asarray(
        model_spectrum_on_observed_pixels(
            state.wave_obs_a,
            state.flux_lambda_cgs,
            jnp.asarray(spectral_wave),
            jnp.asarray(pixel_edges),
            resample_mode="bin",
            resolving_power=700.0,
        )
    )
    sigma = 0.1 * np.maximum(np.abs(spectral_flux), np.nanmax(np.abs(spectral_flux)) * 1.0e-4)
    mask = np.ones_like(spectral_wave, dtype=bool)
    mask[10:15] = False
    data = GaussianSpectralData(
        spectral_wave,
        spectral_flux,
        sigma,
        mask=mask,
        pixel_edges_obs_a=pixel_edges,
        resolving_power=700.0,
    )

    logp = float(jax.jit(lambda x: model.log_prob(x, data))(theta))
    log_prior = float(model.parameter_space.log_prior(theta))
    expected = -0.5 * np.sum(np.log(2.0 * np.pi * sigma[mask] ** 2))
    assert np.isclose(logp, log_prior + expected, rtol=1e-6)


@pytest.mark.jaxcigale
def test_observed_flux_conversion_is_float32_safe():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.core import observed_flux_from_luminosity
    from composed.experimental.jaxcigale.dependencies import require_jax

    jax, jnp = require_jax()
    previous_x64 = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)
        wave = jnp.asarray(np.linspace(900.0, 20000.0, 64), dtype=jnp.float32)
        lum = jnp.ones_like(wave) * jnp.asarray(1.0e10, dtype=jnp.float32)
        _, flux = observed_flux_from_luminosity(wave, lum, jnp.asarray(2.0, dtype=jnp.float32))
        flux_np = np.asarray(flux)
        assert flux_np.dtype == np.float32
        assert np.all(np.isfinite(flux_np))
        assert np.all(flux_np > 0.0)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
