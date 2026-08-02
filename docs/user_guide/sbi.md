# Simulation-Based Inference

CompoSED's stable neural posterior estimators are a conditional masked
autoregressive flow (MAF) and a mixture-density network (MDN).

## Problem-driven simulation

```python
result = fit(
    problem,
    MAF(
        hidden_features=256,
        num_transforms=6,
        num_blocks=3,
        epochs=300,
        batch_size=2048,
        device="auto",
    ),
    training=Simulate(
        n=300_000,
        noise_model=noise_model,
        infer=["zred", "log10_mass", "dust2"],
        context=PhotometricContext("snr_logsigma"),
        n_workers=8,
        batch_size=256,
        executor="process",
    ),
    seed=41,
)
```

For Problem-based SBI, ``infer=`` selects parameters; it does not define a
second numerical ordering. CompoSED arranges selected targets in the canonical
``problem.parameters.names`` order before building the training matrix. The
neural ``sample()`` and ``log_prob()`` methods use
``result.inference_state.theta_names`` in that canonical order.
``result.parameter_names`` preserves the inferred order but may additionally
contain conditioned or ``DeltaPrior`` columns. The sequence originally
supplied to ``infer=`` is retained as ``requested_infer`` in the training and
fit metadata. For SBI trained from pre-existing arrays,
``SBITrainingSet.theta_names`` remains the authoritative column order.

The simulation path:

1. samples `theta` from `Problem.parameters`;
2. applies the same parameter transform and backend as deterministic fitting;
3. applies mass normalization exactly as the likelihood;
4. draws measured flux from the declared noise model;
5. encodes measured flux and catalog sigma in deterministic band order;
6. trains $q(\theta\mid x)$.

`failure_policy="raise"` is the default. Choosing `"resample"` changes the
effective training measure to the prior conditional on simulator success and is
recorded in metadata. `warn_retry_fraction=0.05` warns when more than 5% of
attempted rows have failed; the warning does not stop the simulation. Use
`max_attempts` only when an explicit hard compute ceiling is required.

The effective prior is a standard simulation check:

```python
from composed.plot import plot_effective_prior

fig, axes = plot_effective_prior(
    result.inference_state.training_set,
    problem.parameters,
)
```

The filled distribution contains accepted simulator rows. Orange contours are
fresh draws from the declared prior. Any difference is caused by simulator
success conditioning, including correlations introduced by physical-domain
failures. For `ContinuitySFH`, an age expressed as a fraction of Universe age
still has to exceed the final fixed lookback-bin edge.

```{warning}
For learned survey-noise models, simulator success includes the noise model's
magnitude support. Very faint noiseless dropout-band predictions can therefore
truncate the training distribution even when the SPS backend itself succeeds.
Treat the effective-prior plot as a required check whenever failed rows are
resampled.
```

## Pre-existing pairs

If simulations or empirical labels already exist, do not construct an unrelated
`Problem`. Use:

```python
training = SBITrainingSet.from_photometry(
    theta=theta_train,
    flux=flux_train,
    sigma=sigma_train,
    theta_names=theta_names,
    band_names=band_names,
)
posterior = train_sbi(training, MAF(...), seed=41)
```

This mode makes the supplied paired dataset the explicit source of the neural
training distribution.

## Catalog reuse

```python
samples = posterior.sample(
    catalog_flux,
    sigma=catalog_sigma,
    input_units="native",
    num_samples=128,
    batch_size=8192,
    seed=42,
)
```

The shape is `(n_object, n_sample, n_parameter)`. Objects are batched inside
the neural estimator; CompoSED does not sample one SED at a time.

One trained MAF/MDN uses one fixed ordered band schema. Every catalog row must
contain finite flux and strictly positive sigma in all trained bands.
`SEDDataset.mask` cannot reduce the neural input dimension for an individual
object. For heterogeneous coverage, select a shared band set or train and route
one estimator per common coverage pattern.

The trained MAF/MDN stores the minimum and maximum of every encoded context
feature. At inference it warns when an object leaves that coordinate-wise box,
without clipping or changing the observation. It also warns when a bounded
posterior marginal has a 16-84% width below 1% of its prior coordinate while
its median lies within 5% of a prior edge. These checks are intentionally
minimal: passing them does not prove multivariate in-distribution support.

## Device policy

`device="auto"` selects CUDA, then MPS, then CPU. Neural tensors use float32 on
MPS because Metal does not support float64. Training inputs are checked and
converted consistently rather than inheriting a process-wide torch default
dtype.

## Validation

SBI quality depends on prior coverage, simulator fidelity, noise-model
fidelity, and training capacity. At minimum:

- use held-out simulations;
- inspect the effective accepted prior against the declared prior;
- run rank and coverage diagnostics;
- compare selected objects to a trusted Monte Carlo posterior;
- inspect posterior predictive photometry;
- test catalog regions near prior and survey-selection boundaries;
- record the simulation count, acceptance fraction, network, seed, and
  checkpoint schema.

See {doc}`../photometric_sbi_quickstart`, {doc}`../sbi_diagnostics`, and
{doc}`../maf_validation`.
