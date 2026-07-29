# Inference Methods

## High-level configurations

```{automodule} composed.problem
:members: SamplerCapabilities, Sampler, Emcee, RandomWalk, Grid, MixedGibbs, MixedTAMIS, Laplace, TAMIS, PocoMC, fit
:show-inheritance:
```

## Batch execution

```{automodule} inftools.batch
:members: BatchFitFailure, BatchFitResult, fit_many
:show-inheritance:
```

## Low-level adapters

These functions are useful for custom analyses, but `Problem` plus `fit` is the
recommended stable entry point.

```{automodule} inftools.mcmc
:members: run_rw_metropolis, run_emcee
```

```{automodule} inftools.grid
:members: DiscreteGrid, ParameterBlocks, split_parameter_space, enumerate_discrete_grid, sample_discrete_grid, run_grid_sampler, run_mixed_gibbs, run_thresholded_mixed_gibbs
:show-inheritance:
```

```{automodule} inftools.mixed_tamis
:members: MixedTamisProposal, run_mixed_tamis
:show-inheritance:
```

```{automodule} inftools.pocomc_adapter
:members: pocomc_prior_from_parameter_space, run_pocomc
```
