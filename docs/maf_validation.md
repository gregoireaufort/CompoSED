# Photometric MAF Validation

The stable CompoSED 0.1 MAF path has two distinct validation layers.

## Contract Tests

The normal test suite checks deterministic behavior:

- simulated flux and sigma use the likelihood's active-band mask;
- the exact sigma returned by the noise model is retained;
- negative noisy flux is accepted by the default context;
- band order and flux units are checked at inference;
- uniform and log-uniform targets round-trip through bounded transforms;
- physical samples remain inside prior support;
- catalog rows are batched inside nflows;
- save/load preserves weights, standardizers, context schema, and target bounds.

These tests catch API, shape, unit, mask, and persistence regressions. They do
not establish that a trained neural posterior is calibrated.

## Known-Posterior Validation

Run:

```bash
python examples/validate_maf_photometric_sbi.py
```

The script uses only the public CompoSED workflow. It declares a backend,
uniform prior, `SEDDataset`, `Problem`, noise model, `Simulate`, and `MAF`. The
one-band model is

```text
flux_noiseless = amplitude
flux_measured ~ Normal(amplitude, sigma)
amplitude ~ Uniform(0, 1).
```

`sigma` is drawn independently for every object and supplied explicitly to the
MAF. For each held-out object the exact posterior is therefore a Gaussian
truncated to `[0, 1]`. Numerical quadrature gives its mean and width.

The validation compares learned and exact means/widths, checks empirical
coverage, verifies that no sample leaves the prior support, runs the generic SBI
diagnostics, and saves:

- `metrics.json`;
- `validation_arrays.npz` and its code/environment provenance sidecar;
- `run.provenance.json`;
- rank, coverage, and prediction plots;
- a reloadable MAF checkpoint.

The default thresholds are deliberately broad regression limits, not claims of
universal SBI accuracy. A real analysis must repeat simulation-based
calibration over its own prior, filters, masks, noise/depth distribution, and
forward model. In particular, a MAF trained at one fixed depth should not be
used silently on a heterogeneous-depth catalog.

For a shorter installation smoke test:

```bash
python examples/validate_maf_photometric_sbi.py \
  --n-train 4000 --n-test 64 --epochs 20 \
  --max-mean-mae 0.10 --max-std-mae 0.10 --coverage-tolerance 0.25
```

## COSMOS2020 Tutorial MAFs

After running tutorial 00 and either real-backend MAF tutorial, draw an
independent validation catalog from the same prior, backend, filters, and noise
distribution with:

```bash
python examples/validate_cosmos2020_maf_calibration.py fsps
python examples/validate_cosmos2020_maf_calibration.py cigale
```

This validation does not compare against LePhare and does not reuse the MAF
training simulations. It generates fresh simulator truth, then reports rank
histograms, marginal coverage, and posterior-median residuals. The output
records hashes of both the prepared COSMOS uncertainty catalog and the trained
MAF checkpoint.

For the 300,000-simulation, 256-hidden-feature tutorial checkpoints, a
512-object held-out run with 256 posterior samples per object gave:

| Backend | Mean absolute coverage error | Mean empirical coverage at 50/68/90/95% |
|---|---:|---|
| FSPS continuity SFH | 0.019 | 0.501 / 0.675 / 0.891 / 0.936 |
| CIGALE delayed-tau | 0.021 | 0.523 / 0.704 / 0.890 / 0.928 |

These are simulator-calibration results, not validation against real galaxies.
The single COSMOS2020 CIGALE posterior differs more visibly from its TAMIS
reference in age and attenuation. That distinction is scientifically useful:
good simulation-based calibration does not rule out simulator-to-data shift or
a difficult local real-data posterior.
