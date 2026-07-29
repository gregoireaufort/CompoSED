# The CompoSED Mental Model

CompoSED separates five scientific responsibilities:

```text
backend + parameter space + data + likelihood = Problem
Problem + inference method = InferenceResult
```

## 1. Backend

The backend turns named physical parameters into noiseless observed model
quantities:

```python
model = backend.predict_photometry(params, filters)
spectrum = backend.predict_spectrum(params, wavelengths=wave_obs)
```

It owns the call to FSPS or CIGALE, redshifting, luminosity distance, and filter
integration. It must declare whether its output is absolute or per unit
surviving stellar mass.

## 2. Parameter space

`ParameterSpace.names` is the one canonical vector order. Its priors define the
measure sampled by grid, Monte Carlo, and simulator-based workflows.

```python
parameters = ParameterSpace(
    names=["zred", "log10_mass", "dust2"],
    priors={
        "zred": UniformPrior(0.01, 3.0),
        "log10_mass": UniformPrior(7.0, 12.0),
        "dust2": UniformPrior(0.0, 2.0),
    },
)
```

## 3. Data

`SEDDataset`, `SpectrumDataset`, or `SpectroPhotometricDataset` records the
observed arrays and the mask actually consumed by the likelihood. Data are
single-object containers; catalog workflows stack many such objects or use an
amortized neural posterior.

## 4. Likelihood

`Gaussian` configures the observational likelihood. The implementation aligns
bands by name, uses only active data, applies optional noise floors/model
discrepancy, evaluates censored upper limits, and applies mass normalization
according to the backend declaration.

## 5. Problem

`Problem` binds these pieces into one auditable statistical object:

```python
problem = Problem(
    backend=backend,
    parameters=parameters,
    data=data,
    likelihood=Gaussian(photometric_model_discrepancy=0.03),
    filters=filters,
)
```

An optional `parameter_transform` maps sampled values to backend values. Use it
for an explicit scientific transformation, not to hide units or rename an
unrelated quantity.

## 6. Inference

Traditional inference is one line once the problem exists:

```python
result = fit(problem, PocoMC(...), seed=11)
```

For an object with known redshift:

```python
result = fit(
    problem,
    PocoMC(...),
    conditions={"zred": 0.413},
    seed=11,
)
```

Problem-driven SBI adds the training experiment explicitly:

```python
result = fit(
    problem,
    MAF(device="auto"),
    training=Simulate(n=100_000, noise_model=noise_model),
    seed=11,
)
```

The returned `InferenceResult` uses one common sample/weight/parameter-name
contract irrespective of the inference engine.

## Audit trail

For every analysis, record:

- backend class, version, configuration, and external data grids;
- filter names and transmission-curve provenance;
- parameter order, prior classes, bounds, and conditions;
- dataset units, active masks, upper-limit convention, and selection cuts;
- mass normalization and mass reference;
- likelihood noise terms;
- random seeds and inference options;
- saved result fingerprint and posterior-predictive checks.
