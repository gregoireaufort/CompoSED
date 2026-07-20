# Photometric MAF SBI Quickstart

The stable CompoSED `0.1` SBI path trains a conditional masked autoregressive
flow for physical parameters given measured photometry and its uncertainty:

```text
q(theta | measured flux, sigma)
```

The training simulator and observed object use the same `Problem`, active-band
mask, flux units, mass normalization, and context encoding.

## Simulate And Fit One Problem

```python
from composed import (
    Gaussian,
    MAF,
    ParameterSpace,
    PhotometricContext,
    Problem,
    SEDDataset,
    Simulate,
    UniformPrior,
    fit,
)

parameters = ParameterSpace(
    names=["zred", "log10_mass", "dust2", "logzsol"],
    priors={
        "zred": UniformPrior(0.05, 2.0),
        "log10_mass": UniformPrior(8.0, 12.0),
        "dust2": UniformPrior(0.0, 1.0),
        "logzsol": UniformPrior(-1.0, 0.2),
    },
)

data = SEDDataset(
    band_names=filters.names,
    flux=observed_flux_maggies,
    sigma=observed_sigma_maggies,
    flux_unit="maggies",
)

problem = Problem(
    backend=backend,
    parameters=parameters,
    data=data,
    likelihood=Gaussian(),
    filters=filters,
)

def noise_fn(noiseless_flux):
    sigma_floor = 1.0e-12
    fractional_error = 0.08
    return sigma_floor + fractional_error * abs(noiseless_flux)

result = fit(
    problem,
    method=MAF(
        hidden_features=128,
        num_transforms=6,
        epochs=200,
        batch_size=2048,
        validation_split=0.1,
        patience=20,
        num_samples=512,
        inference_batch_size=8192,
        device="auto",
    ),
    training=Simulate(
        n=100_000,
        noise_fn=noise_fn,
        infer=["zred", "log10_mass"],
        context=PhotometricContext("snr_logsigma"),
        n_workers=8,
        batch_size=256,
        executor="process",
    ),
    seed=7,
)

samples = result.samples
posterior = result.inference_state
posterior.save("runs/fsps_photoz_maf")
```

`samples` has shape `(num_samples, n_parameters)` for the observed object.
The fitted `UniformPrior` and `LogUniformPrior` bounds are encoded through
invertible transforms, so the MAF produces physical samples inside prior
support rather than relying on post-hoc rejection.

## Why The Default Context Uses S/N And Sigma

`PhotometricContext("snr_logsigma")` supplies, in deterministic band order:

```text
[flux / sigma, log10(sigma / reference_flux)]
```

Both terms are needed. Signal-to-noise alone loses the absolute flux scale;
adding sigma makes the original measured flux recoverable. This representation
also accepts negative noisy fluxes, unlike AB magnitudes or logarithmic flux.
The default `reference_flux=1` is expressed in the declared dataset flux unit
and is recorded in the checkpoint schema.

The simulator stores the exact sigma returned by `noise_fn` for every training
realization. At inference, passing an `SEDDataset` makes the trained posterior
use `data.active_flux`, `data.active_sigma`, `data.active_band_names`, and
`data.flux_unit` after checking them against the training schema.

## Reuse On A Catalog

```python
posterior_samples = posterior.sample(
    catalog_flux_maggies,
    sigma=catalog_sigma_maggies,
    input_units="native",
    num_samples=128,
    batch_size=8192,
    seed=8,
)
```

The output shape is `(n_objects, 128, n_parameters)`. Batching occurs over
objects inside `nflows`; CompoSED does not loop over individual SEDs.

After reloading a checkpoint, the same call is available:

```python
from composed import TrainedMAFSBI

posterior = TrainedMAFSBI.load("runs/fsps_photoz_maf", device="auto")
samples = posterior.sample(
    catalog_flux_maggies,
    sigma=catalog_sigma_maggies,
    input_units="native",
    num_samples=128,
    batch_size=8192,
    seed=8,
)
```

The checkpoint directory contains:

- `manifest.json`: architecture, parameter and band order, context convention,
  prior transforms, training history, package versions, and metadata;
- `standardizers.npz`: theta and context standardization arrays;
- `weights.pt`: tensor-only nflows state.

The training table is deliberately not copied into the checkpoint.

## Pre-existing Photometry And Labels

A presampled forward model, numerical simulation, or empirical labeled catalog
does not require a fictitious backend:

```python
from composed import MAF, SBITrainingSet, train_sbi

training = SBITrainingSet.from_photometry(
    theta_train,
    flux_train,
    sigma_train,
    theta_names=["zred", "log10_mass"],
    band_names=list(filters.names),
    source="presampled_forward_model_v2",
    parameter_space=parameters,
)

posterior = train_sbi(
    training,
    MAF(epochs=200, batch_size=2048, validation_split=0.1, device="auto"),
    seed=7,
)
samples = posterior.sample(
    observed_flux,
    sigma=observed_sigma,
    input_units="native",
    num_samples=512,
    seed=8,
)
```

Use `SBITrainingSet.from_arrays` instead when `x_train` is already a complete,
pre-encoded context table.

## Masks, Upper Limits, And Diagnostics

Masked bands are removed consistently from simulated flux, simulated sigma,
observed flux, and observed sigma. A trained posterior rejects a dataset with a
different active-band order or flux unit.

Censored upper limits do not yet have a stable SBI context encoding in `0.1`.
Problem-driven MAF training and `SEDDataset` posterior sampling reject them
explicitly rather than treating limits as detections.

Run simulation-based calibration on held-out simulations:

```python
diagnostics = posterior.diagnostics(
    posterior_samples,
    theta_true,
    x_test=context_test,
    output_dir="runs/fsps_photoz_maf/diagnostics",
)
```

Prior coverage, simulator fidelity, the training noise distribution, and
held-out rank/coverage diagnostics are part of the scientific model, not merely
neural-network tuning details.

For a complete known-posterior regression test, see
[`maf_validation.md`](maf_validation.md) and run
`examples/validate_maf_photometric_sbi.py`.
