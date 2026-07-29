# First Fit

This toy problem is deliberately small, but it follows the same data flow as an
FSPS or CIGALE analysis. A deterministic backend returns photometry per unit
surviving stellar mass. The likelihood applies `10**log10_mass` exactly once,
and a finite grid evaluates three candidate masses.

```python
import numpy as np

from composed import (
    ChoicePrior,
    Gaussian,
    Grid,
    MassNormalization,
    ParameterSpace,
    Problem,
    SEDDataset,
    fit,
    posterior_summary,
)
from composed.backends.mock import MockBackend

data = SEDDataset(
    band_names=["g", "r"],
    flux=np.array([1.0, 2.0]),
    sigma=np.array([0.05, 0.10]),
    flux_unit="maggies",
)

backend = MockBackend(
    band_names=["g", "r"],
    flux=[1.0e-10, 2.0e-10],
    mass_normalization=MassNormalization.PER_SOLAR_MASS,
)

parameters = ParameterSpace(
    names=["log10_mass"],
    priors={"log10_mass": ChoicePrior([9.5, 10.0, 10.5])},
)

problem = Problem(
    backend=backend,
    parameters=parameters,
    data=data,
    likelihood=Gaussian(),
)

result = fit(problem, Grid(), seed=7)
print(posterior_summary(result))
```

## What happened

1. `SEDDataset` validated the one-dimensional flux, uncertainty, band-name,
   mask, and unit contract.
2. `ParameterSpace` fixed the order of the sampled vector.
3. `MockBackend` declared that its flux is per solar mass of surviving stars.
4. `Problem` built the backend-agnostic Gaussian photometric likelihood.
5. The likelihood multiplied the backend prediction by
   `10**log10_mass`; no backend guessed how mass should be handled.
6. `Grid` evaluated the finite prior support and returned an
   {class}`~composed.results.InferenceResult` with normalized weights.

The important lines to audit in a real analysis are the backend construction,
the `ParameterSpace`, the `SEDDataset` units and masks, and the
`MassNormalization` reported by the backend.

## Replace the toy pieces

For a real analysis:

- replace `MockBackend` with {class}`~composed.backends.fsps.FSPSBackend` or
  {class}`~composed.backends.cigale.CIGALEBackend`;
- load real filters with {func}`~composed.filters.load_filter_set` or use
  native CIGALE filter names;
- use continuous priors when the selected sampler supports them;
- choose an inference method from {doc}`capabilities`;
- save the result with {func}`~composed.results.save_inference_result`;
- inspect posterior predictions with
  {func}`~composed.plot.plot_posterior_predictive`.

The complete workflow is described in {doc}`user_guide/mental_model`.
