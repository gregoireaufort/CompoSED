# Photometric SBI Quickstart

The stable CompoSED `0.1` SBI path trains either a conditional masked
autoregressive flow (`MAF`) or Gaussian-mixture density network (`MDN`) for
physical parameters given measured photometry and its uncertainty:

```text
q(theta | measured flux, sigma)
```

The training simulator and observed object use the same `Problem`, active-band
mask, flux units, mass normalization, and context encoding.

## Simulate And Fit One Problem

```python
from composed import (
    ConditionalCatalogNoise,
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
    names=["zred", "log10_mass", "dust2", "logzsol", "gas_logz"],
    priors={
        "zred": UniformPrior(0.05, 2.0),
        "log10_mass": UniformPrior(8.0, 12.0),
        "dust2": UniformPrior(0.0, 1.0),
        "logzsol": UniformPrior(-1.0, 0.2),
        "gas_logz": UniformPrior(-1.0, 0.2),
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
    likelihood=Gaussian(photometric_model_discrepancy=0.05),
    filters=filters,
)

survey_noise = ConditionalCatalogNoise.fit(
    catalog_ab_magnitudes,
    catalog_sigma_maggies,
    band_names=filters.names,
    flux_unit="maggies",
    seed=6,
    catalog_source="survey catalog used for this analysis",
    row_selection="complete rows in the selected bands",
)

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
        noise_model=survey_noise,
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

from composed.plot import plot_effective_prior

plot_effective_prior(
    posterior.training_set,
    problem.parameters,
)
```

Here ``infer=["zred", "log10_mass"]`` selects the targets. Their array order
is always the corresponding order in ``problem.parameters.names``; inspect
``posterior.theta_names`` or ``result.parameter_names`` rather than relabelling
columns with the literal ``infer=`` list. CompoSED records the requested list
separately in ``result.metadata["requested_infer"]``.

`samples` has shape `(num_samples, n_parameters)` for the observed object.
The fitted `UniformPrior` and `LogUniformPrior` bounds are encoded through
invertible transforms, so the MAF produces physical samples inside prior
support rather than relying on post-hoc rejection.

The effective-prior plot compares accepted simulator rows with fresh draws
from `problem.parameters`. It should be inspected whenever
`failure_policy="resample"` is used, because backend-domain rejection can
change both marginal support and parameter correlations.

`ChoicePrior` may also be included in `Simulate.infer`. CompoSED enumerates the
Cartesian product of the declared choice axes and trains
`q(discrete_state | photometry)` alongside
`q(continuous_parameters | photometry, discrete_state)`. Returned discrete
values therefore lie exactly on the declared support, with no ordinal
interpolation. Use `posterior.discrete_probabilities(...)` for exact joint and
marginal category probabilities. For a mixed target, `posterior.log_prob(...)`
returns categorical log mass plus continuous log density in physical units.

## Use The Simpler MDN Alternative

The simulation, noise model, observation context, prior transforms, catalog
sampling, and diagnostics are identical. Replace only the inference method:

```python
from composed import MDN

result = fit(
    problem,
    method=MDN(
        n_components=8,
        hidden_features=128,
        num_blocks=3,
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
        noise_model=survey_noise,
        infer=["zred", "log10_mass"],
        context=PhotometricContext("snr_logsigma"),
    ),
    seed=7,
)

posterior = result.inference_state
posterior.save("runs/fsps_photoz_mdn")
```

The MDN is a mixture of diagonal Gaussians in transformed parameter space. Its
density is exactly normalized and cheap to evaluate. Multiple components can
represent distinct modes, but MAF is generally more expressive for curved or
high-dimensional posterior geometry. Install only PyTorch with
`python -m pip install -e ".[mdn]"`.

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

`ConditionalCatalogNoise` fits the genuinely joint conditional distribution

```text
q(log10 sigma_catalog[g, r, ...] | AB magnitude[g, r, ...])
```

using complete multiband rows in one fixed band order. Non-finite magnitudes
and non-positive uncertainties are either rejected or filtered as complete
rows, with the rejected count stored in provenance. The checkpoint also stores
the input-array hash, units, magnitude convention, standardization, random
seed, architecture, package versions, row selection, and training magnitude
support. Magnitudes outside that support are warned about or rejected; they
are never clamped.

The simulator stores raw `sigma_catalog` for every training realization. It
draws flux using

```text
sigma_draw^2 = sigma_catalog^2 + sigma_floor^2
               + (photometric_model_discrepancy * f_model)^2
```

The optional absolute `photometric_sigma_floor` and fractional model
discrepancy are declared on `Gaussian`; neither is added to the neural sigma
context. At observed inference, passing an `SEDDataset` makes the trained
posterior use `data.active_flux`, raw `data.active_sigma`,
`data.active_band_names`, and `data.flux_unit` after checking them against the
training schema. In particular, CompoSED does not estimate model discrepancy
from the observed flux.

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

Censored upper limits do not yet have a stable fixed-context SBI encoding in
`0.1`. Problem-driven MAF/MDN training and `SEDDataset` posterior sampling
reject them explicitly rather than treating limits as detections.

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

For validation guidance and the maintained regression tests, see
[`maf_validation.md`](maf_validation.md). End-to-end catalog workflows are in
the [tutorial guide](tutorials/index.md).
