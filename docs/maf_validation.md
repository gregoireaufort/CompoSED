# Photometric MAF Validation

The stable CompoSED MAF path has two distinct validation layers: deterministic
software checks and simulation-based scientific calibration.

## Contract Tests

Run the focused SBI suite with:

```bash
python -m pytest -q \
  tests/test_sbi.py \
  tests/test_sbi_pipeline.py \
  tests/test_sbi_diagnostics.py
```

These tests check that:

- simulated flux and sigma use the likelihood's active-band mask;
- the noise model's sigma is retained in the conditioning vector;
- band order and flux units are checked at inference;
- bounded parameters round-trip through their transforms;
- posterior samples remain inside prior support;
- catalog rows are batched inside the neural estimator;
- save/load preserves transforms, context schema, and target ordering.

These are API, shape, unit, mask, and persistence tests. They do not establish
that a trained neural posterior is calibrated for a scientific analysis.

## Scientific Calibration

Calibration must use held-out simulations generated from the same `Problem`,
prior, filters, mask convention, and survey-noise model as the training set.
Do not reuse rows from the neural training table.

For each held-out object:

1. retain the simulator parameters as the known truth;
2. draw posterior samples conditioned on its measured flux and uncertainty;
3. compute rank statistics and marginal coverage;
4. compare posterior summaries with truth;
5. inspect boundary saturation and observation-support warnings.

The generic diagnostics live in `inftools.diagnostics`:

```python
from inftools.diagnostics import run_sbi_diagnostics

report = run_sbi_diagnostics(
    posterior_samples=posterior_samples,
    theta_true=theta_test,
    theta_names=posterior.theta_names,
)
```

The COSMOS2020 tutorials show the public end-to-end workflow for the two stable
backends:

- `notebooks/tutorials/01_fsps_maf_cosmos2020_catalog.ipynb`
- `notebooks/tutorials/02_cigale_maf_cosmos2020_catalog.ipynb`

Their held-out diagnostic cells generate fresh simulator truth and report rank
histograms, coverage, and posterior-summary residuals. These are simulator
calibration results, not validation against real galaxies. Good calibration
under the simulator does not rule out simulator-to-data shift.

The effective prior must also be inspected whenever failed backend evaluations
are resampled. The accepted simulation table can differ from the declared prior
if a physical-domain failure preferentially removes part of parameter space.
