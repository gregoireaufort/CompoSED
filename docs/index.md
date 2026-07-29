# CompoSED

**Composable Bayesian SED fitting and photo-z inference.**

CompoSED gives forward-model backends, observational likelihoods, and inference
methods one explicit interface. The package does not introduce a new stellar
population model. It lets the same scientific problem be evaluated with FSPS
or CIGALE and inferred with conventional Monte Carlo or amortized
simulation-based inference.

The central object is:

```python
problem = Problem(
    backend=backend,
    parameters=parameters,
    data=data,
    likelihood=Gaussian(),
    filters=filters,
)
```

The same `problem` can then be passed to a compatible inference method:

```python
result = fit(problem, PocoMC(...), seed=4)
```

or to neural posterior estimation:

```python
result = fit(
    problem,
    MAF(device="auto"),
    training=Simulate(n=100_000, noise_model=noise_model),
    seed=4,
)
```

CompoSED keeps the scientifically consequential choices visible: parameter
order, prior support, units, masks, censored upper limits, mass normalization,
backend configuration, simulation noise, and random seeds.

```{admonition} Release scope
:class: note
The stable release backends are FSPS and CIGALE v2022.0. Neural MAF and MDN
posterior estimators are stable optional layers. Conditional diffusion and fast
rest-frame catalog projection remain experimental and are labeled accordingly.
```

## Start here

- Read {doc}`install` for the backend-specific installation workflow.
- Read {doc}`getting_started` for a complete small fit.
- Read {doc}`user_guide/mental_model` before combining a backend and sampler.
- Check {doc}`capabilities` before choosing an inference method.
- Use {doc}`science/conventions` when auditing units and normalization.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
getting_started
capabilities
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/index
science/conventions
sfh_models
cigale_backend
```

```{toctree}
:maxdepth: 2
:caption: SBI and catalogs

photometric_sbi_quickstart
sbi_diagnostics
maf_validation
experimental_diffusion_sbi
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: API reference

reference/index
api_stability
```

```{toctree}
:maxdepth: 1
:caption: Development and validation

design
validation_provenance
environment
citations
release_checklist
contributing/documentation
```
