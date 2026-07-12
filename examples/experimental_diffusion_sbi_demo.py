"""Small experimental conditional-diffusion SBI demo.

The toy feature vector is:

    [g_mag, r_mag, redshift, log10_mass]

The model is trained to condition on magnitudes and infer the two physical
parameters.  The numbers are deliberately synthetic; this example checks the
mechanics of masked joint sampling, not astrophysical realism.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata


def main() -> None:
    if importlib.util.find_spec("torch") is None:
        raise ImportError("This example requires torch. Install the diffusion extra or install torch.")

    rng = np.random.default_rng(123)
    n_train = 512
    z = rng.uniform(0.0, 2.0, size=n_train)
    log10_mass = rng.normal(10.0, 0.35, size=n_train)

    # Toy photometry in magnitudes.  Lower magnitudes are brighter.
    g_mag = 23.0 + 0.8 * z - 0.25 * (log10_mass - 10.0) + rng.normal(0.0, 0.05, n_train)
    r_mag = 22.5 + 0.5 * z - 0.20 * (log10_mass - 10.0) + rng.normal(0.0, 0.05, n_train)
    x_train = np.column_stack([g_mag, r_mag, z, log10_mass])

    meta = FeatureMetadata.from_groups(
        {
            "mags": ["g", "r"],
            "params": ["z", "log10_mass"],
        }
    )

    estimator = ConditionalDiffusionEstimator(
        meta,
        model="mlp",
        hidden_features=64,
        model_config={"mlp_blocks": 2, "emb_dim": 32, "time_hidden": 64},
        sigma_min=0.03,
        sigma_max=1.5,
        learning_rate=1e-3,
        device="auto",
    )
    estimator.fit(
        x_train,
        mask_config={"unknown_fraction": {"mags": 0.0, "params": 1.0}},
        epochs=10,
        batch_size=128,
        seed=4,
        clamp_known_in_xt=True,
        loss_on_unknown_only=True,
        verbose=True,
    )

    observed = np.array([[g_mag[0], r_mag[0], np.nan, np.nan]])
    known_mask = np.array([[True, True, False, False]])
    samples = estimator.sample(
        observed,
        known_mask,
        num_samples=256,
        steps=30,
        sampler="edm_euler",
    )

    param_samples = samples[0, :, 2:]
    print("True [z, log10_mass]:", np.array([z[0], log10_mass[0]]))
    print("Posterior mean:", np.mean(param_samples, axis=0))
    print("Posterior std:", np.std(param_samples, axis=0))
    print("Known magnitudes are clamped:", np.allclose(samples[0, :, :2], observed[:, :2]))


if __name__ == "__main__":
    main()
