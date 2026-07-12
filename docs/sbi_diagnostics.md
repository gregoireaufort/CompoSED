# SBI Diagnostics

This page describes the stable diagnostic helpers in `inftools.diagnostics`.
They are intentionally independent of the neural estimator.  The same
functions can check samples from a MAF, a diffusion model, a saved catalog, or
another posterior sampler.

## Data Entering

Diagnostics consume posterior samples and, when available, true parameters:

```python
samples.shape == (n_objects, n_samples, n_parameters)
theta_true.shape == (n_objects, n_parameters)
```

If an estimator exposes `sample(x_obs, num_samples=...)`,
`sample_posterior_dataset` draws those samples and converts common estimator
return conventions into the object-first shape above.

## Transformations

The diagnostics do not change parameter units.  They compute:

- posterior means, medians, standard deviations, and 16-84% intervals;
- marginal ranks of the true value among posterior samples;
- central credible interval coverage curves;
- optional TARP coverage when the external `tarp` package is installed.

## Masks, Cuts, And Normalization

The diagnostic layer assumes masks and cuts have already been applied upstream.
For SBI, `x_test` should use the same feature convention that trained the
posterior estimator: same bands, units, upper-limit treatment, and feature
ordering.  No mass normalization or flux conversion happens here.

## Important Functions To Audit

- `sample_posterior_dataset`: estimator sampling and shape normalization.
- `rank_statistics`: rank percentiles used for calibration checks.
- `marginal_coverage`: central interval coverage.
- `run_sbi_diagnostics`: one-call wrapper that optionally writes arrays and
  plots.

## Minimal Example

```python
import numpy as np

from inftools.diagnostics import run_sbi_diagnostics

samples = np.random.normal(size=(100, 1000, 2))
theta_true = np.random.normal(size=(100, 2))

result = run_sbi_diagnostics(
    posterior_samples=samples,
    theta_true=theta_true,
    theta_names=["z", "log10_mass"],
    make_plots=False,
)

print(result["coverage"]["mean_coverage"])
```

## Sanity Checks

For simulated validation data, calibrated posteriors should have approximately
uniform rank percentiles and coverage curves close to the one-to-one line.
Systematic rank slopes, U-shapes, or coverage deficits are signs that the
posterior estimator is biased or overconfident.
