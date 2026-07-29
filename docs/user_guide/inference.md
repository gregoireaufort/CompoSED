# Traditional Inference

Once a `Problem` is defined, the selected method changes without changing the
backend or likelihood:

```python
result = fit(problem, method, seed=12)
```

## Continuous parameters

```python
result = fit(
    problem,
    Emcee(nwalkers=48, nsteps=2_000, burn=500),
    seed=12,
)
```

or:

```python
result = fit(
    problem,
    PocoMC(
        sampler_kwargs={"n_effective": 512, "n_active": 256},
        run_kwargs={"n_total": 4096, "progress": True},
    ),
    seed=12,
)
```

## Finite discrete parameters

`Grid` enumerates the Cartesian support implied by `ChoicePrior`,
`IntegerUniformPrior`, and fixed values:

```python
result = fit(problem, Grid(), seed=12)
```

It does not discretize a continuous prior implicitly.

## Mixed continuous/discrete parameters

Use `MixedTAMIS` or `MixedGibbs` when a CIGALE-like model includes both:

```python
result = fit(
    problem,
    MixedTAMIS(
        n_sample=4_000,
        T_max=50,
        n_comp=4,
        recycle=True,
    ),
    seed=12,
)
```

The discrete block uses its declared finite support. The continuous block is
sampled in a transformed space where required by the method. Inspect
`result.metadata` for adaptation and weighting diagnostics.

## Fixed values

```python
result = fit(
    problem,
    PocoMC(...),
    conditions={"zred": 0.32},
    seed=12,
)
```

Conditioned values are removed from the numerical sampler and restored in the
returned parameter table.

## Reproducibility

Always pass `seed`. CompoSED uses it for supported initialization and sampling
paths and records it in result metadata. For MCMC, inspect traces and
autocorrelation/ESS diagnostics; for importance samplers, inspect normalized
weights and ESS. A finite result is not by itself evidence of convergence.

See {doc}`../capabilities` for method/prior compatibility.
