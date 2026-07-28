# Changelog

## 0.1.1

Scientific-safety patch release.

### Fixed

- Analytic catalog mass profiling no longer converts a non-positive amplitude
  into a numerically finite `log10_mass` near -307. Unbounded invalid optima
  now fail clearly; explicitly bounded fits report whether the optimum was
  clipped to a mass boundary.
- Fast rest-frame catalog projection no longer rescales W/nm luminosity merely
  because the wavelength coordinate was supplied in Angstrom.
- Fast projection now rejects backends that do not explicitly support a
  redshift-independent rest-spectrum grid and rejects filters not fully
  covered by that grid.
- The CIGALE MAF and TAMIS tutorials now use the same explicit 5% model-error
  term and validate cached comparison results against the current `Problem`.
- Tutorial notebooks ship without machine-specific saved outputs.
- Cached catalog mass marginalization now requires a continuous `Prior` and
  uses prior-density times irregular-grid cell width.
- Weighted posterior plots preserve the empirical posterior by resampling with
  replacement.
- `plot_posterior_predictive(result, problem)` validates the fitted Problem and
  reuses its parameter transform, filters, and photometric units.
- Problem-driven PocoMC no longer accepts a sampler-specific replacement
  prior. Conditioned SBI results restore condition columns and identify
  marginalized nuisance parameters.
- Physically invalid parameter combinations use a dedicated
  `ModelDomainError`: likelihoods map them to `-inf`, while configuration and
  shape errors remain visible.
- SBI simulation now fails on the first unsuccessful prior draw by default.
  Success-conditioned replacement draws require an explicit
  `failure_policy="resample"` and are recorded in training metadata.
- `Problem` fingerprints include referenced transform globals, structured
  priors, backend engine versions, and filter-curve hashes.
- Saved inference results and model grids include and verify a SHA256 of the
  numerical artifact. Model-grid loading verifies provenance by default.
- GitHub Actions now gates the core Python matrix, stable sampler adapters,
  neural SBI tests, notebook hygiene, and isolated wheel installation.

### Experimental status

- The finite-difference Laplace runner and the adapter for the separately
  installed historical `TAMIS` package are explicitly experimental and emit
  warnings when run. CompoSED's self-contained `MixedTAMIS` remains supported.
- The fast rest-frame catalog path is explicitly experimental in 0.1.1. It is
  currently enabled only for backends that declare the required capability;
  current FSPS models are rejected because their SFH evaluation requires
  redshift.

## 0.1.0

First public alpha release.

### Stable public surface

- Unified `Problem` and `fit` workflow for photometric, spectral, and joint
  spectrophotometric inference.
- Explicit masks, Gaussian errors, sigma floors, censored photometric upper
  limits, flux units, and surviving-stellar-mass normalization.
- FSPS and CIGALE v2022.0 forward-model backends.
- Named constant, exponential, delayed-tau, continuity, and tabular SFHs where
  supported by the selected backend.
- Grid, random-walk, emcee, self-contained mixed TAMIS/Gibbs, and PocoMC
  inference adapters.
- Stable conditional MAF and Gaussian-mixture MDN SBI with explicit
  measurement-noise conditioning, bounded target transforms, catalog batching,
  diagnostics, and reloadable checkpoints.
- Normalized inference results, provenance sidecars, posterior summaries, and
  posterior-comparison plots.

### Experimental

- Conditional diffusion remains in an explicit experimental namespace and is
  not part of the stable 0.1 API.
- The finite-difference Laplace runner, the adapter for the separately
  installed historical `TAMIS` package, and fast rest-frame catalog projection
  are experimental.

### Known limitations

- Censored upper limits do not yet have a stable MAF or MDN context encoding.
- Scientific calibration of an amortized posterior remains specific to its
  prior, backend, filters, masks, and noise distribution.
