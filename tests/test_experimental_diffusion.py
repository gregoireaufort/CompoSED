import importlib.util

import numpy as np
import pytest


def test_importing_experimental_diffusion_module_is_lightweight():
    import inftools.experimental.diffusion as diffusion

    assert hasattr(diffusion, "FeatureMetadata")
    assert hasattr(diffusion, "ConditionalDiffusionEstimator")


def test_constructing_diffusion_without_torch_gives_helpful_error(monkeypatch):
    import inftools.experimental.diffusion as diffusion

    monkeypatch.setattr(diffusion, "torch", None)
    monkeypatch.setattr(diffusion, "nn", None)
    meta = diffusion.FeatureMetadata.from_names(["a", "b"])
    with pytest.raises(ImportError, match="requires the optional torch"):
        diffusion.ConditionalDiffusionEstimator(meta, model="mlp")


@pytest.mark.diffusion
def test_resolve_torch_device_cpu_and_bad_explicit_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    from inftools.experimental.diffusion import resolve_torch_device

    device = resolve_torch_device("cpu", validate=True)
    assert device.type == "cpu"

    with pytest.raises(RuntimeError, match="not usable"):
        resolve_torch_device("not_a_real_device", validate=True, allow_fallback=False)


@pytest.mark.diffusion
def test_feature_metadata_and_mask_tie_magerrs_to_mags():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    import torch
    from inftools.experimental.diffusion import FeatureMetadata, make_training_mask

    meta = FeatureMetadata.from_groups(
        {
            "mags": ["u", "g"],
            "magerrs": ["u_err", "g_err"],
            "params": ["z", "log10_mass"],
        }
    )
    mask = make_training_mask(
        (8, meta.n_features),
        meta,
        unknown_fraction={"mags": 1.0, "params": 0.0},
        device=torch.device("cpu"),
    )
    assert mask.shape == (8, 6)
    assert torch.all(mask[:, meta.groups["mags"]] == mask[:, meta.groups["magerrs"]])
    assert torch.all(mask[:, meta.groups["params"]])


@pytest.mark.diffusion
def test_training_mask_can_tie_all_observation_state_groups_by_band():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    import torch
    from inftools.experimental.diffusion import FeatureMetadata, make_training_mask

    meta = FeatureMetadata.from_groups(
        {
            "photometry": ["g", "r"],
            "uncertainty": ["g_sigma", "r_sigma"],
            "availability": ["g_available", "r_available"],
            "censoring": ["g_censored", "r_censored"],
            "parameters": ["z"],
        }
    )
    tied = ("photometry", "uncertainty", "availability", "censoring")
    mask = make_training_mask(
        (32, meta.n_features),
        meta,
        unknown_fraction={group: 0.5 for group in tied},
        device=torch.device("cpu"),
        tie_groups=tied,
    )

    reference = mask[:, meta.groups["photometry"]]
    for group in tied[1:]:
        assert torch.equal(reference, mask[:, meta.groups[group]])


@pytest.mark.diffusion
def test_diffusion_fit_sample_and_known_feature_clamping(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata

    rng = np.random.default_rng(10)
    z = rng.uniform(0.0, 1.0, size=48)
    mass = rng.normal(10.0, 0.2, size=48)
    mag_g = 22.0 + z - 0.1 * (mass - 10.0)
    mag_r = 21.5 + 0.5 * z - 0.1 * (mass - 10.0)
    x_train = np.column_stack([mag_g, mag_r, z, mass])

    meta = FeatureMetadata.from_groups({"mags": ["g", "r"], "params": ["z", "log10_mass"]})
    estimator = ConditionalDiffusionEstimator(
        meta,
        model="mlp",
        hidden_features=16,
        model_config={"mlp_blocks": 1, "emb_dim": 16, "time_hidden": 16},
        sigma_min=0.05,
        sigma_max=1.0,
        learning_rate=1e-3,
        device="cpu",
    )
    history = estimator.fit(
        x_train,
        mask_config={"unknown_fraction": {"mags": 0.0, "params": 1.0}},
        epochs=1,
        batch_size=16,
        seed=11,
        clamp_known_in_xt=True,
        loss_on_unknown_only=True,
    )
    assert np.isfinite(history["train_loss"][-1])

    known = np.array([[mag_g[0], mag_r[0], np.nan, np.nan]])
    mask = np.array([[True, True, False, False]])
    samples = estimator.sample(known, mask, num_samples=5, steps=3, sampler="edm_euler")
    assert samples.shape == (1, 5, 4)
    assert np.all(np.isfinite(samples))
    assert np.allclose(samples[0, :, 0], mag_g[0])
    assert np.allclose(samples[0, :, 1], mag_r[0])

    path = tmp_path / "diffusion.pt"
    estimator.save(path)
    loaded = ConditionalDiffusionEstimator.load(path, device="cpu")
    loaded_samples = loaded.sample(known, mask, num_samples=3, steps=2, sampler="edm_euler")
    assert loaded_samples.shape == (1, 3, 4)


@pytest.mark.diffusion
def test_diffusion_stays_float32_when_global_default_dtype_is_float64():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    import torch
    from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        rng = np.random.default_rng(12)
        x_train = rng.normal(size=(24, 3))
        meta = FeatureMetadata.from_groups({"mags": ["g"], "params": ["z", "log10_mass"]})
        estimator = ConditionalDiffusionEstimator(
            meta,
            model="mlp",
            hidden_features=8,
            model_config={"mlp_blocks": 1, "emb_dim": 8, "time_hidden": 8},
            sigma_min=0.05,
            sigma_max=1.0,
            learning_rate=1e-3,
            device="cpu",
        )
        assert next(estimator.score_model.parameters()).dtype == torch.float32
        estimator.fit(
            x_train,
            mask_config={"unknown_fraction": {"mags": 0.0, "params": 1.0}},
            epochs=1,
            batch_size=8,
            seed=13,
        )
        known = np.array([[x_train[0, 0], np.nan, np.nan]])
        mask = np.array([[True, False, False]])
        samples = estimator.sample(known, mask, num_samples=2, steps=2, sampler="edm_euler")
        assert samples.shape == (1, 2, 3)
        assert np.all(np.isfinite(samples))
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.diffusion
def test_score_model_forward_shape_if_torch_available():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed.")

    import torch
    from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata

    meta = FeatureMetadata.from_groups({"mags": ["g", "r"], "params": ["z"]})
    estimator = ConditionalDiffusionEstimator(
        meta,
        model="hybrid_sed",
        hidden_features=8,
        model_config={"emb_dim": 8, "time_hidden": 8, "cnn_blocks": 1, "param_blocks": 1, "fusion_blocks": 1},
        device="cpu",
    )
    x = torch.zeros((4, meta.n_features), dtype=torch.float32)
    mask = torch.ones_like(x, dtype=torch.bool)
    out = estimator.score_model(x, x, torch.full((4,), 0.5), mask)
    assert out.shape == x.shape
