# Results, Persistence, And Plots

All high-level inference methods return `InferenceResult`:

```text
samples       (n_sample, n_parameter)
weights       (n_sample,), normalized
logp          (n_sample,) or None
parameter_names
posterior_median
map_estimate  or None
chain         optional sampler-native chain
metadata
```

Sample-only neural methods may have `logp=None`; CompoSED does not invent a MAP
estimate.

## Summaries

```python
summary = posterior_summary(result, credible_interval=0.68)
```

The summary uses posterior weights. Importance/grid samples must not be treated
as unweighted draws unless they have first been resampled.

## Save and reload

```python
npz_path, metadata_path = save_inference_result(result, "runs/object_001")
loaded = load_inference_result(npz_path)
require_result_matches_problem(loaded, problem)
```

The numerical artifact contains a digest binding its scientific metadata.
Strict loading checks content and metadata integrity. The separate provenance
sidecar records machine and engine context.

## Corner and traces

```python
from composed.plot import plot_corner_hexbin, plot_traces

fig, axes = plot_corner_hexbin(result)
fig, axes = plot_traces(result)
```

The corner plot uses hexbins and can overlay a comparison posterior. Trace plots
use the sampler chain when available.

For simulator-generated SBI, compare accepted training rows with the declared
prior:

```python
from composed.plot import plot_effective_prior

fig, axes = plot_effective_prior(
    result.inference_state.training_set,
    problem.parameters,
)
```

This is a training-distribution check, not a posterior plot.

## Posterior predictive

```python
from composed.plot import plot_posterior_predictive

fig, axes = plot_posterior_predictive(
    result,
    problem,
    n_draw=200,
    seed=9,
)
```

This path verifies that the result belongs to the supplied `Problem`, calls the
same backend and parameter transform, plots the median and 16--84% model band,
and distinguishes detections, upper limits, and masked photometric points.

## What to inspect

- convergence or importance-weight ESS;
- posterior support against prior bounds;
- posterior predictive residuals;
- censored bands plotted as limits rather than detections;
- disagreement between independent inference methods;
- result/problem fingerprint match;
- seed, engine versions, grid hashes, and selection metadata.
