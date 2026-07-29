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

The simulation path:

1. samples `theta` from `Problem.parameters`;
2. applies the same parameter transform and backend as deterministic fitting;
3. applies mass normalization exactly as the likelihood;
4. draws measured flux from the declared noise model;
5. encodes measured flux and catalog sigma in deterministic band order;
6. trains $q(\theta\mid x)$.

`failure_policy="raise"` is the default. Choosing `"resample"` changes the
effective training measure to the prior conditional on simulator success and is
recorded in metadata.

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

## Device policy

`device="auto"` selects CUDA, then MPS, then CPU. Neural tensors use float32 on
MPS because Metal does not support float64. Training inputs are checked and
converted consistently rather than inheriting a process-wide torch default
dtype.

## Validation

SBI quality depends on prior coverage, simulator fidelity, noise-model
fidelity, and training capacity. At minimum:

- use held-out simulations;
- run rank and coverage diagnostics;
- compare selected objects to a trusted Monte Carlo posterior;
- inspect posterior predictive photometry;
- test catalog regions near prior and survey-selection boundaries;
- record the simulation count, acceptance fraction, network, seed, and
  checkpoint schema.

See {doc}`../photometric_sbi_quickstart`, {doc}`../sbi_diagnostics`, and
{doc}`../maf_validation`.
