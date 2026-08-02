# Parameters, Priors, And Conditions

## Deterministic ordering

`ParameterSpace` is the bridge between a numerical vector and named backend
parameters:

```python
parameters = ParameterSpace(
    names=["zred", "log10_mass", "logzsol", "gas_logz", "dust2"],
    priors={
        "zred": UniformPrior(0.01, 4.0),
        "log10_mass": UniformPrior(7.0, 13.0),
        "logzsol": UniformPrior(-1.5, 0.3),
        "gas_logz": UniformPrior(-1.5, 0.3),
        "dust2": UniformPrior(0.0, 2.0),
    },
)
```

Dictionary insertion order is not used. `names` controls `to_dict`,
`from_dict`, prior sampling, sampler columns, result labels, and checkpoint
schemas.

```python
theta = parameters.from_dict(values)
values_again = parameters.to_dict(theta)
```

## Prior families

- `UniformPrior(low, high)` and `LogUniformPrior(low, high)` include both
  boundaries in `logpdf`.
- `NormalPrior` and `StudentTPrior` are unbounded continuous priors.
- `IntegerUniformPrior(low, high)` is inclusive and finite.
- `ChoicePrior(values)` assigns equal mass to listed numeric choices.
- `DeltaPrior(value)` is a point mass.

`sample_prior(n, rng)` returns shape `(n, ndim)`. Samples from valid finite
priors must have finite `log_prior`.

`ParameterSpace` represents a product of independent scalar priors; it does not
implement a general joint or conditional prior density. Deterministic physical
relations can be expressed with `Problem(parameter_transform=...)`, such as
mapping a sampled stellar metallicity to a gas metallicity. A genuinely
correlated stochastic prior should be represented by explicitly generated
training pairs for SBI, or by a custom inference workflow that evaluates that
joint density; it must not be hidden inside a backend transform.

## Fixed object-specific values

Use `conditions` when a quantity is known for the object but remains part of
the model:

```python
result = fit(
    problem,
    PocoMC(...),
    conditions={"zred": spectroscopic_redshift},
    seed=3,
)
```

CompoSED removes conditioned axes from the sampler, restores them in backend
calls, and returns deterministic columns in the normalized result. This works
more naturally than asking a continuous sampler to explore a point-mass prior.

For catalog SBI, condition values may vary object by object if they were
declared in the simulation context.

## Parameter transforms

`Problem(parameter_transform=...)` is an explicit map from sampled coordinates
to backend inputs. Common uses include transforming an age fraction into
`tage_gyr` at the proposed redshift or producing a tabular SFH.

The transform must preserve scientific meaning:

- parameter units should be clear in names or documentation;
- mass scaling must remain in the likelihood;
- transformed values should be validated against backend domain constraints;
- the transform configuration is included in the Problem fingerprint.

Named SFHs avoid most custom transformations; see {doc}`../sfh_models`.
