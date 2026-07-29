from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from composed.backends.mock import MockBackend
from composed.data import SEDDataset
from composed.likelihood import GaussianPhotometricLikelihood
from composed.noise import ConditionalCatalogNoise
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior


def _require_neural_noise_dependencies() -> None:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nflows") is None:
        pytest.skip("ConditionalCatalogNoise fitting requires torch and nflows.")


@pytest.fixture(scope="module")
def correlated_catalog_noise():
    _require_neural_noise_dependencies()
    rng = np.random.default_rng(710)
    magnitudes = rng.uniform(20.0, 24.0, size=(2048, 2))
    shared = rng.normal(0.0, 0.12, size=(magnitudes.shape[0], 1))
    independent = rng.normal(0.0, 0.035, size=magnitudes.shape)
    log10_sigma = -6.0 + 0.18 * (magnitudes - 22.0) + shared + independent
    sigma = 10.0**log10_sigma
    noise = ConditionalCatalogNoise.fit(
        magnitudes,
        sigma,
        band_names=("g", "r"),
        flux_unit="maggies",
        seed=711,
        hidden_features=32,
        num_transforms=3,
        num_blocks=2,
        epochs=18,
        batch_size=256,
        validation_split=0.1,
        patience=8,
        device="cpu",
    )
    return noise, magnitudes, sigma


@pytest.mark.sbi
def test_conditional_catalog_noise_recovers_brightness_trend_and_residual_correlation(
    correlated_catalog_noise,
):
    noise, _, _ = correlated_catalog_noise
    rng = np.random.default_rng(712)
    flux_at_22 = 10.0 ** (-0.4 * np.full((2000, 2), 22.0))
    samples_at_22 = noise.sample(flux_at_22, rng=rng)
    correlation = np.corrcoef(np.log10(samples_at_22).T)[0, 1]

    flux_bright = 10.0 ** (-0.4 * np.full((1000, 2), 20.5))
    flux_faint = 10.0 ** (-0.4 * np.full((1000, 2), 23.5))
    bright_sigma = noise.sample(flux_bright, rng=np.random.default_rng(713))
    faint_sigma = noise.sample(flux_faint, rng=np.random.default_rng(714))

    assert correlation > 0.35
    assert correlation == pytest.approx(0.92, abs=0.55)
    assert np.all(np.median(faint_sigma, axis=0) > np.median(bright_sigma, axis=0))


@pytest.mark.sbi
def test_conditional_catalog_noise_validates_band_order_shape_flux_and_support(
    correlated_catalog_noise,
):
    noise, _, _ = correlated_catalog_noise
    noise.validate_for(band_names=("g", "r"), flux_unit="maggies")
    with pytest.raises(ValueError, match="band order"):
        noise.validate_for(band_names=("r", "g"), flux_unit="maggies")
    with pytest.raises(ValueError, match="flux unit"):
        noise.validate_for(band_names=("g", "r"), flux_unit="jy")
    with pytest.raises(ValueError, match="expected 2 bands"):
        noise.sample(np.ones(3), rng=np.random.default_rng(715))
    with pytest.raises(ValueError, match="strictly positive"):
        noise.sample(np.asarray([1.0, 0.0]), rng=np.random.default_rng(716))
    with pytest.raises(TypeError, match="explicit numpy.random.Generator"):
        noise.sample(np.ones(2), rng=None)

    original_policy = noise.support_policy
    noise.support_policy = "raise"
    try:
        with pytest.raises(ValueError, match="outside its training support"):
            noise.sample(
                10.0 ** (-0.4 * np.asarray([30.0, 30.0])),
                rng=np.random.default_rng(717),
            )
    finally:
        noise.support_policy = original_policy


@pytest.mark.sbi
def test_conditional_catalog_noise_filters_invalid_complete_rows_and_records_provenance():
    _require_neural_noise_dependencies()
    rng = np.random.default_rng(718)
    magnitudes = rng.uniform(20.0, 22.0, size=(128, 2))
    sigma = 10.0 ** (-6.0 + 0.1 * (magnitudes - 21.0))
    magnitudes[3, 0] = np.nan
    sigma[9, 1] = 0.0

    with pytest.warns(RuntimeWarning, match="filtered 2/128"):
        noise = ConditionalCatalogNoise.fit(
            magnitudes,
            sigma,
            band_names=("g", "r"),
            seed=719,
            hidden_features=8,
            num_transforms=1,
            num_blocks=1,
            epochs=1,
            batch_size=32,
            validation_split=0.0,
            device="cpu",
            row_selection="toy complete-row selection",
        )

    assert noise.provenance["n_input_rows"] == 128
    assert noise.provenance["n_training_rows"] == 126
    assert noise.provenance["n_rejected_rows"] == 2
    assert noise.provenance["row_selection"] == "toy complete-row selection"
    assert len(noise.provenance["catalog_array_sha256"]) == 64
    with pytest.warns(RuntimeWarning, match="filtered 2/128"):
        repeated = ConditionalCatalogNoise.fit(
            magnitudes,
            sigma,
            band_names=("g", "r"),
            seed=719,
            hidden_features=8,
            num_transforms=1,
            num_blocks=1,
            epochs=1,
            batch_size=32,
            validation_split=0.0,
            device="cpu",
            row_selection="toy complete-row selection",
        )
    assert repeated.specification()["model_state_sha256"] == noise.specification()["model_state_sha256"]
    with pytest.raises(ValueError, match="invalid row indices"):
        ConditionalCatalogNoise.fit(
            magnitudes,
            sigma,
            band_names=("g", "r"),
            invalid_rows="raise",
            epochs=1,
        )


@pytest.mark.sbi
def test_conditional_catalog_noise_save_load_and_seeded_simulation_are_reproducible(
    correlated_catalog_noise,
    tmp_path,
):
    noise, _, _ = correlated_catalog_noise
    checkpoint = noise.save(tmp_path / "survey_noise")
    loaded = ConditionalCatalogNoise.load(checkpoint, device="cpu")
    model_flux = 10.0 ** (-0.4 * np.asarray([22.0, 22.0]))

    first = noise.sample(model_flux, rng=np.random.default_rng(720))
    second = loaded.sample(model_flux, rng=np.random.default_rng(720))
    assert np.array_equal(first, second)
    assert loaded.specification()["model_state_sha256"] == noise.specification()["model_state_sha256"]

    data = SEDDataset(
        ("g", "r"),
        flux=model_flux,
        sigma=np.asarray([1.0e-6, 1.0e-6]),
        flux_unit="maggies",
    )
    likelihood = GaussianPhotometricLikelihood(
        MockBackend(model_flux, band_names=("g", "r")),
        data,
        ParameterSpace(("z",), {"z": UniformPrior(0.0, 1.0)}),
        model_discrepancy=0.1,
    )
    draw_a, sigma_a = likelihood.simulate_with_uncertainty(
        [0.5],
        noise,
        rng=np.random.default_rng(721),
    )
    draw_b, sigma_b = likelihood.simulate_with_uncertainty(
        [0.5],
        loaded,
        rng=np.random.default_rng(721),
    )
    assert np.array_equal(sigma_a, sigma_b)
    assert np.array_equal(draw_a, draw_b)


def test_conditional_catalog_noise_rejects_unrepresentable_log_sigma_before_power():
    noise = ConditionalCatalogNoise(
        band_names=("g",),
        flux_unit="maggies",
        magnitude_min=np.asarray([20.0]),
        magnitude_max=np.asarray([24.0]),
        estimator_configuration={"theta_dim": 1, "x_dim": 1},
        estimator_state={"placeholder": np.asarray([1.0])},
        theta_mean=np.asarray([0.0]),
        theta_std=np.asarray([1.0]),
        x_mean=np.asarray([22.0]),
        x_std=np.asarray([1.0]),
        history={},
        provenance={},
        support_policy="raise",
    )

    class ExtremeEstimator:
        def sample(self, context, num_samples, seed):
            del context, num_samples, seed
            return np.asarray([[400.0]])

    noise._estimator = ExtremeEstimator()
    with pytest.raises(RuntimeError, match="floating-point range"):
        noise.sample(
            10.0 ** (-0.4 * np.asarray([22.0])),
            rng=np.random.default_rng(722),
        )
