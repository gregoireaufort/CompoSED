import importlib.util

import numpy as np
import pytest


def test_cue_exports_import_without_jax_dependency():
    from composed.experimental import jaxcigale

    assert hasattr(jaxcigale, "cue_nebular_module")
    assert hasattr(jaxcigale, "derive_cue_inputs_from_stellar_spectrum")
    assert hasattr(jaxcigale, "CueJaxPort")


@pytest.mark.jaxcigale
def test_cue_speculator_jax_matches_numpy_for_toy_weights():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.cue_port import CuePCAWeights, CueSpeculatorWeights
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    spec = CueSpeculatorWeights(
        weights=(
            np.asarray([[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]]),
            np.asarray([[0.3, -0.2], [0.5, 0.1], [-0.4, 0.2]]),
        ),
        biases=(np.asarray([0.01, -0.02, 0.03]), np.asarray([0.04, -0.05])),
        alphas=(np.asarray([1.2, 0.8, 1.5]),),
        betas=(np.asarray([0.7, 0.9, 0.6]),),
        parameter_shift=np.asarray([0.5, -0.2]),
        parameter_scale=np.asarray([2.0, 3.0]),
        pca_shift=np.asarray([0.1, -0.1]),
        pca_scale=np.asarray([1.5, 0.5]),
        log_spectrum_shift=np.asarray([1.0, 2.0, 3.0]),
        log_spectrum_scale=np.asarray([0.2, 0.3, 0.4]),
    )
    pca = CuePCAWeights(
        components=np.asarray([[1.0, 0.0, -1.0], [0.2, 0.5, 0.1]]),
        mean=np.asarray([0.1, -0.2, 0.3]),
    )
    theta = np.asarray([[0.2, -0.1], [1.0, 0.5]])
    coeff_np = spec.pca_coefficients_numpy(theta)
    coeff_jax = np.asarray(spec.pca_coefficients_jax(jnp.asarray(theta)))
    log_np = pca.inverse_transform_numpy(coeff_np) * spec.log_spectrum_scale + spec.log_spectrum_shift
    log_jax = np.asarray(pca.inverse_transform_jax(spec.pca_coefficients_jax(jnp.asarray(theta)))) * spec.log_spectrum_scale + spec.log_spectrum_shift

    assert np.allclose(coeff_np, coeff_jax, rtol=1e-10, atol=1e-10)
    assert np.allclose(log_np, log_jax, rtol=1e-10, atol=1e-10)


@pytest.mark.jaxcigale
def test_real_cue_jax_port_matches_public_numpy_if_data_available():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")
    if importlib.util.find_spec("dill") is None or importlib.util.find_spec("sklearn") is None:
        pytest.skip("Cue public pickle readers are not installed.")

    import os
    from pathlib import Path

    data_dir = Path(os.environ.get("CUE_DATA_DIR", "/private/tmp/cue/src/cue/data"))
    if not data_dir.exists():
        pytest.skip("Public Cue data directory is not available.")

    from composed.experimental.jaxcigale.cue_port import CueJaxPort
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()
    port = CueJaxPort.from_public_cue_data_dir(data_dir)
    theta = np.asarray(
        [
            [2.5, 1.0, 0.2, -0.6, 2.0, 0.5, 0.4, 52.0, 100.0, -0.3, -0.134, -0.134],
            [4.0, 2.0, 1.0, 0.0, 1.0, 0.2, 0.1, 51.5, 300.0, -1.0, -0.5, -0.3],
        ]
    )
    _, continuum_np = port.predict_continuum_native_numpy(theta)
    _, continuum_jax = port.predict_continuum_native_jax(jnp.asarray(theta))
    _, lines_np = port.predict_lines_native_numpy(theta)
    _, lines_jax = port.predict_lines_native_jax(jnp.asarray(theta))

    assert np.max(np.abs(np.log10(continuum_np) - np.log10(np.asarray(continuum_jax)))) < 1.0e-8
    assert np.max(np.abs(np.log10(lines_np) - np.log10(np.asarray(lines_jax)))) < 1.0e-8


@pytest.mark.jaxcigale
def test_cue_derives_power_law_shape_from_stellar_spectrum():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.cue import (
        CUE_IONIZING_EDGES_A,
        derive_cue_inputs_from_stellar_spectrum,
        reconstruct_cue_piecewise_lnu,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.photometry import C_A_PER_S

    _, jnp = require_jax()

    wave = np.geomspace(40.0, 911.6, 2048)
    slopes_true = np.asarray([2.5, 1.0, 0.2, -0.6])
    amplitudes = np.asarray([1.0e16, 3.0e17, 8.0e17, 5.0e17])
    lnu = np.zeros_like(wave)
    for i in range(4):
        lo, hi = CUE_IONIZING_EDGES_A[i], CUE_IONIZING_EDGES_A[i + 1]
        mask = (wave >= lo) & (wave <= hi)
        lnu[mask] = amplitudes[i] * wave[mask] ** slopes_true[i]
    l_lambda = lnu * C_A_PER_S / wave**2

    cue = derive_cue_inputs_from_stellar_spectrum(
        jnp.asarray(wave),
        jnp.asarray(l_lambda),
        logu=-2.5,
        logn_h=2.0,
        gas_logoh=-0.3,
        log_no=-0.134,
        log_co=-0.134,
        clip_derived_ionizing_shape=False,
    )

    slopes = np.asarray(cue.ionizing_slopes)
    assert np.allclose(slopes, slopes_true, atol=2e-2)
    assert np.isfinite(float(cue.log_q_h_intrinsic))
    assert np.isfinite(np.asarray(cue.theta12)).all()

    reconstructed = np.asarray(
        reconstruct_cue_piecewise_lnu(wave, cue.ionizing_slopes, cue.segment_log_luminosity_lsun)
    )
    valid = lnu > 0.0
    median_ratio = np.median(reconstructed[valid] / lnu[valid])
    assert np.isclose(median_ratio, 1.0, rtol=0.08)


@pytest.mark.jaxcigale
def test_cue_escape_and_dust_fractions_scale_qh_budget():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.cue import derive_cue_inputs_from_stellar_spectrum
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave = jnp.asarray(np.geomspace(50.0, 2000.0, 1024))
    l_lambda = 1.0e9 * (wave / 900.0) ** -1.2
    base = derive_cue_inputs_from_stellar_spectrum(
        wave,
        l_lambda,
        logu=-2.5,
        logn_h=2.0,
        gas_logoh=-0.3,
        log_no=-0.134,
        log_co=-0.134,
    )
    attenuated = derive_cue_inputs_from_stellar_spectrum(
        wave,
        l_lambda,
        logu=-2.5,
        logn_h=2.0,
        gas_logoh=-0.3,
        log_no=-0.134,
        log_co=-0.134,
        f_esc=0.2,
        f_dust=0.1,
    )
    expected = np.log10(0.7)
    measured = float(attenuated.log_q_h_gas - base.log_q_h_gas)
    assert np.isclose(measured, expected, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(attenuated.gas_photon_fraction), 0.7, rtol=1e-6)


@pytest.mark.jaxcigale
def test_cue_public_package_theta_conversion_uses_logu_and_density():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.cue import (
        cue_logq_from_logu,
        cue_theta12_to_public_package_theta,
        derive_cue_inputs_from_stellar_spectrum,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()
    wave = jnp.asarray(np.geomspace(50.0, 2000.0, 512))
    l_lambda = 1.0e9 * (wave / 900.0) ** -1.0
    cue = derive_cue_inputs_from_stellar_spectrum(
        wave,
        l_lambda,
        logu=-2.25,
        logn_h=2.5,
        gas_logoh=-0.4,
        log_no=-0.1,
        log_co=-0.2,
    )
    public_theta = np.asarray(cue_theta12_to_public_package_theta(cue))
    assert public_theta.shape == (12,)
    assert np.isclose(public_theta[7], float(cue_logq_from_logu(-2.25, 2.5)))
    assert np.isclose(public_theta[8], 10.0**2.5)


@pytest.mark.jaxcigale
def test_cue_module_plugs_into_sed_graph_with_toy_emulator():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        cue_nebular_module,
        delayed_sfh_module,
        redshift_module,
        toy_cue_emulator,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.geomspace(50.0, 20000.0, 768)
    age_grid = np.linspace(0.02, 5.0, 32)
    filter_wave = np.linspace(4500.0, 8500.0, 128)
    filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])
    names = [
        "log10_mass",
        "z",
        "tau_gyr",
        "tage_gyr",
        "logzsol",
        "gas_logu",
        "gas_logn_h",
        "gas_logoh",
        "gas_f_esc",
        "gas_f_dust",
    ]
    space = JaxParameterSpace(names=names, priors={name: UniformJaxPrior(-20.0, 20.0) for name in names})
    model = build_jax_sed_model(
        [
            delayed_sfh_module(age_grid),
            analytic_stellar_module(),
            cue_nebular_module(toy_cue_emulator),
            redshift_module(),
        ],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.4, 2.0, 3.0, -0.2, -2.5, 2.0, -0.3, 0.1, 0.0])
    state = model.run_modules(theta)
    phot = model.predict_photometry(theta)

    assert phot.shape == (1,)
    assert np.all(np.isfinite(np.asarray(phot)))
    assert np.max(np.asarray(state.nebular_lum_lsun_per_a)) > 0.0
    assert np.allclose(
        np.asarray(state.total_lum_lsun_per_a),
        np.asarray(state.stellar_lum_lsun_per_a + state.nebular_lum_lsun_per_a),
    )


@pytest.mark.jaxcigale
def test_cue_module_removes_nonescaped_lyc_from_stellar_spectrum():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        cue_nebular_module,
        delayed_sfh_module,
        toy_cue_emulator,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave_rest = np.geomspace(50.0, 2000.0, 512)
    age_grid = np.linspace(0.02, 3.0, 24)
    filters = JaxFilterSet.from_curves(["wide"], [np.linspace(4500.0, 8500.0, 32)], [np.ones(32)])
    names = [
        "log10_mass",
        "z",
        "tau_gyr",
        "tage_gyr",
        "logzsol",
        "gas_logu",
        "gas_logn_h",
        "gas_logoh",
        "gas_f_esc",
        "gas_f_dust",
    ]
    space = JaxParameterSpace(names=names, priors={name: UniformJaxPrior(-20.0, 20.0) for name in names})
    base_modules = [
        delayed_sfh_module(age_grid),
        analytic_stellar_module(),
    ]
    absorbed_model = build_jax_sed_model(
        [*base_modules, cue_nebular_module(toy_cue_emulator, absorb_lyc=True)],
        wave_rest,
        filters,
        space,
    )
    unabsorbed_model = build_jax_sed_model(
        [*base_modules, cue_nebular_module(toy_cue_emulator, absorb_lyc=False)],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.4, 1.5, 2.0, -0.2, -2.5, 2.0, -0.3, 0.25, 0.10])
    absorbed = absorbed_model.run_modules(theta)
    unabsorbed = unabsorbed_model.run_modules(theta)

    wave = np.asarray(absorbed.wave_rest_a)
    lyc = wave < 911.6
    non_lyc = ~lyc
    ratio = np.asarray(absorbed.stellar_lum_lsun_per_a)[lyc] / np.asarray(unabsorbed.stellar_lum_lsun_per_a)[lyc]

    assert np.allclose(np.nanmedian(ratio), 0.25, rtol=1e-6)
    assert np.allclose(
        np.asarray(absorbed.stellar_lum_lsun_per_a)[non_lyc],
        np.asarray(unabsorbed.stellar_lum_lsun_per_a)[non_lyc],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.jaxcigale
def test_cue_zeroes_numerical_lyc_floors_but_preserves_real_escape():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.cue import zero_numerical_lyc_floor
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    wave = jnp.asarray([500.0, 800.0, 1000.0, 1500.0, 3000.0, 9000.0])
    luminosity = jnp.asarray([1.0e-16, 1.0e-4, 1.0e-3, 1.0, 2.0, 3.0])
    cleaned = np.asarray(zero_numerical_lyc_floor(wave, luminosity, floor_fraction=1.0e-12))

    assert cleaned[0] == 0.0
    assert cleaned[1] > 0.0
    assert np.allclose(cleaned[1:], np.asarray(luminosity)[1:])


@pytest.mark.jaxcigale
def test_cue_module_outputs_exact_zero_for_nebular_lyc_floor_when_fesc_is_zero():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        cue_nebular_module,
        delayed_sfh_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    def emulator_with_lyc_floor(wave_rest_a, theta12, cue_inputs):
        del theta12, cue_inputs
        wave = jnp.asarray(wave_rest_a)
        continuum = jnp.where(wave < 911.6, 1.0e-16, 1.0e-2)
        lines = jnp.zeros_like(wave)
        return continuum, lines

    wave_rest = np.geomspace(50.0, 2000.0, 512)
    age_grid = np.linspace(0.02, 3.0, 24)
    filters = JaxFilterSet.from_curves(["wide"], [np.linspace(4500.0, 8500.0, 32)], [np.ones(32)])
    names = [
        "log10_mass",
        "z",
        "tau_gyr",
        "tage_gyr",
        "logzsol",
        "gas_logu",
        "gas_logn_h",
        "gas_logoh",
        "gas_f_esc",
        "gas_f_dust",
    ]
    space = JaxParameterSpace(names=names, priors={name: UniformJaxPrior(-20.0, 20.0) for name in names})
    model = build_jax_sed_model(
        [
            delayed_sfh_module(age_grid),
            analytic_stellar_module(),
            cue_nebular_module(emulator_with_lyc_floor),
        ],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.4, 1.5, 2.0, -0.2, -2.5, 2.0, -0.3, 0.0, 0.0])
    state = model.run_modules(theta)

    lyc = np.asarray(state.wave_rest_a) < 911.6
    assert np.all(np.asarray(state.nebular_lum_lsun_per_a)[lyc] == 0.0)
    assert np.all(np.asarray(state.total_lum_lsun_per_a)[lyc] == 0.0)


@pytest.mark.jaxcigale
def test_cue_module_can_add_physical_lyc_continuum_after_floor_cleanup():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale import (
        JaxFilterSet,
        JaxParameterSpace,
        UniformJaxPrior,
        analytic_stellar_module,
        build_jax_sed_model,
        cue_nebular_module,
        delayed_sfh_module,
    )
    from composed.experimental.jaxcigale.dependencies import require_jax

    _, jnp = require_jax()

    def emulator_with_lyc_floor(wave_rest_a, theta12, cue_inputs):
        del theta12, cue_inputs
        wave = jnp.asarray(wave_rest_a)
        continuum = jnp.where(wave < 911.6, 1.0e-16, 1.0e-2)
        return continuum, jnp.zeros_like(wave)

    def physical_lyc_continuum(wave_rest_a, cue_inputs):
        del cue_inputs
        wave = jnp.asarray(wave_rest_a)
        return jnp.where(wave < 911.6, 3.0e-4 * (wave / 911.6) ** 2, 0.0)

    wave_rest = np.geomspace(50.0, 2000.0, 512)
    age_grid = np.linspace(0.02, 3.0, 24)
    filters = JaxFilterSet.from_curves(["wide"], [np.linspace(4500.0, 8500.0, 32)], [np.ones(32)])
    names = [
        "log10_mass",
        "z",
        "tau_gyr",
        "tage_gyr",
        "logzsol",
        "gas_logu",
        "gas_logn_h",
        "gas_logoh",
        "gas_f_esc",
        "gas_f_dust",
    ]
    space = JaxParameterSpace(names=names, priors={name: UniformJaxPrior(-20.0, 20.0) for name in names})
    base_modules = [delayed_sfh_module(age_grid), analytic_stellar_module()]
    pure_model = build_jax_sed_model(
        [*base_modules, cue_nebular_module(emulator_with_lyc_floor)],
        wave_rest,
        filters,
        space,
    )
    extended_model = build_jax_sed_model(
        [*base_modules, cue_nebular_module(emulator_with_lyc_floor, lyc_continuum_apply=physical_lyc_continuum)],
        wave_rest,
        filters,
        space,
    )
    theta = jnp.asarray([10.0, 0.4, 1.5, 2.0, -0.2, -2.5, 2.0, -0.3, 0.0, 0.0])

    pure = pure_model.run_modules(theta)
    extended = extended_model.run_modules(theta)
    lyc = np.asarray(pure.wave_rest_a) < 911.6

    assert np.all(np.asarray(pure.nebular_continuum_lum_lsun_per_a)[lyc] == 0.0)
    assert np.any(np.asarray(extended.nebular_continuum_lum_lsun_per_a)[lyc] > 0.0)
    assert np.all(np.asarray(extended.total_lum_lsun_per_a)[lyc] > 0.0)


@pytest.mark.jaxcigale
def test_fsps_lyc_continuum_table_toggle_is_differentiable(tmp_path):
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")

    from composed.experimental.jaxcigale.cue import derive_cue_inputs_from_stellar_spectrum
    from composed.experimental.jaxcigale.dependencies import require_jax
    from composed.experimental.jaxcigale.fsps_nebular import FspsNebularContinuumTable

    jax, jnp = require_jax()

    table_path = tmp_path / "toy.cont"
    wave = np.asarray([100.0, 300.0, 800.0, 1000.0])
    logz_grid = [-1.0, 0.0]
    age_grid = [1.0e6, 2.0e6]
    logu_grid = [-3.0, -2.0]
    lines = ["#4 cols 8 rows 2 logZ 2 Age 2 logU", " ".join(f"{x:.8e}" for x in wave)]
    for iz, logz in enumerate(logz_grid):
        for ia, age in enumerate(age_grid):
            for iu, logu in enumerate(logu_grid):
                scale = 10.0 ** (0.1 * iz + 0.2 * ia + 0.3 * iu)
                spectrum = scale * np.asarray([1.0e-62, 2.0e-62, 3.0e-62, 9.0e-62])
                lines.append(f"{logz:.8e} {age:.8e} {logu:.8e}")
                lines.append(" ".join(f"{x:.8e}" for x in spectrum))
    table_path.write_text("\n".join(lines) + "\n")

    table = FspsNebularContinuumTable.from_file(table_path)
    apply = table.make_lyc_continuum_apply(effective_age_yr=1.5e6)
    model_wave = jnp.asarray([120.0, 500.0, 700.0, 1200.0])
    stellar_wave = jnp.asarray([100.0, 300.0, 800.0, 1200.0, 2000.0])
    stellar_lum = jnp.asarray([1.0e-5, 2.0e-5, 3.0e-5, 1.0e-4, 2.0e-4])

    def total_lyc_continuum(logu):
        cue_inputs = derive_cue_inputs_from_stellar_spectrum(
            stellar_wave,
            stellar_lum,
            logu=logu,
            logn_h=2.0,
            gas_logoh=-0.5,
            log_no=-0.1,
            log_co=-0.1,
            f_esc=0.0,
            f_dust=0.0,
        )
        return jnp.sum(apply(model_wave, cue_inputs))

    values = np.asarray(apply(
        model_wave,
        derive_cue_inputs_from_stellar_spectrum(
            stellar_wave,
            stellar_lum,
            logu=-2.5,
            logn_h=2.0,
            gas_logoh=-0.5,
            log_no=-0.1,
            log_co=-0.1,
            f_esc=0.0,
            f_dust=0.0,
        ),
    ))
    grad = float(jax.grad(total_lyc_continuum)(jnp.asarray(-2.5)))

    assert np.all(np.isfinite(values))
    assert np.all(values[:3] > 0.0)
    assert values[3] == 0.0
    assert np.isfinite(grad)
