# Photometric SBI Quickstart

This is the short experiment pattern for a catalog-scale photometric SBI run.
The point is not to hide the model; it is to keep the notebook organized around
scientific steps.

## Minimal Pipeline

```python
from composed import (
    Diffusion, Gaussian, ParameterSpace, Problem, SEDDataset, Simulate,
    UniformPrior, fit, load_filter_set,
)

filters = load_filter_set(["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"])

priors = ParameterSpace(
    names=["zred", "log10_mass", "dust2", "logzsol"],
    priors={
        "zred": UniformPrior(0.05, 1.5),
        "log10_mass": UniformPrior(8.0, 11.5),
        "dust2": UniformPrior(0.0, 0.8),
        "logzsol": UniformPrior(-1.0, 0.2),
    },
)

def noise_fn(flux):
    return 0.08 * abs(flux)

data = SEDDataset(
    band_names=filters.names,
    flux=observed_flux,
    sigma=observed_sigma,
    flux_unit="maggies",
)

problem = Problem(
    backend=backend,
    parameters=priors,
    data=data,
    likelihood=Gaussian(),
    filters=filters,
)

result = fit(
    problem,
    method=Diffusion(epochs=200, batch_size=2048, num_samples=512, steps=64, device="auto"),
    training=Simulate(
        n=100_000,
        noise_fn=noise_fn,
        infer=["zred", "log10_mass"],
        feature_transform="abmag",
        n_workers=8,
        batch_size=256,
        executor="process",
    ),
    seed=7,
)

samples = result.samples
```

This route always simulates from the declared `Problem`. It cannot accept an
unrelated pre-existing training table.

## Pre-existing Training Pairs

A presampled forward model, numerical simulation, or empirical labeled catalog
is complete in itself and should not be wrapped in a fictitious `Problem`:

```python
from composed import MAF, SBITrainingSet, train_sbi

training = SBITrainingSet.from_arrays(
    theta_train,
    x_train,
    theta_names=["zred", "log10_mass"],
    x_names=list(filters.names),
    source="presampled_forward_model_v2",
    finite="drop",
)

posterior = train_sbi(
    training,
    MAF(epochs=200, batch_size=2048, device="auto"),
    seed=7,
)
samples = posterior.sample(x_obs, num_samples=512)
```

## What Happens Scientifically

1. `Problem.parameters` draws physical parameters from the declared priors.
2. `Problem.simulate` calls the same backend and likelihood simulation path
   used by deterministic inference, including parameter mapping.
3. `noise_fn(flux)` defines the Gaussian observational noise in the same flux
   units as the backend photometry.
4. The generated active-band fluxes are converted to the selected feature
   convention, for example `flux`, `log10_flux`, or `abmag`.
5. The diffusion model trains on the joint vector
   `[photometry_features, inferred_parameters]`.
6. For the observed `Problem.data`, photometry coordinates are clamped known
   and parameter coordinates are sampled.

## Where Masks Enter

The Problem training-set generator uses its `SEDDataset` active-band
convention. Masked bands are excluded before training. Upper limits require an
explicit SBI feature encoding and are currently rejected by this convenience
route rather than silently treated as detections.

The diffusion mask is a different object: it controls which entries in the
joint vector are known during score-model training and posterior sampling.  The
default curriculum emphasizes `parameters | photometry`, but also includes
forward and mixed masks so the model learns more of the joint distribution.

## Important Audit Points

- `problem.specification()` records the backend, priors, mapping, and data model.
- `backend.predict_photometry(params, filters)` defines the physics.
- `noise_fn(flux)` defines the training noise model.
- `feature_transform` defines the units passed to the neural estimator.
- `infer=[...]` controls which parameters are sampled at inference time.
- `result.inference_state` is the reusable trained MAF or diffusion estimator.
- SBI `InferenceResult.logp` and `map_estimate` remain `None` when the method
  provides samples but no posterior density.

For a runnable no-FSPS example, see
`examples/minimal_photometric_diffusion_sbi.py`.
