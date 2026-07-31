# User Guide

This guide documents the release-ready photometric workflow. Sections that
describe spectrum generation, spectral likelihoods, or joint
spectrophotometric fits are explicitly experimental.

The guide follows the order in which a scientist constructs and audits an
analysis:

1. {doc}`mental_model`: compose the model, data, likelihood, and inference.
2. {doc}`data_and_filters`: declare observed arrays, masks, limits, and bands.
3. {doc}`parameters_and_priors`: fix parameter names, ordering, support, and
   object-specific conditions.
4. {doc}`backends`: select FSPS or CIGALE and inspect its physical convention.
5. {doc}`likelihoods`: define the statistical comparison to the observations.
6. {doc}`inference`: select a compatible traditional sampler.
7. {doc}`sbi`: simulate, train, validate, and reuse a neural posterior.
8. {doc}`catalogs`: evaluate finite grids or amortized posteriors over catalogs.
9. {doc}`diagnostics`: inspect MCMC mixing, weighted-sample degeneracy, and
   method-specific limitations.
10. {doc}`results_and_plots`: save, reload, summarize, and check predictions.

```{toctree}
:maxdepth: 1

mental_model
data_and_filters
parameters_and_priors
backends
likelihoods
inference
sbi
catalogs
diagnostics
results_and_plots
```
