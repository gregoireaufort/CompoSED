# Changelog

## Unreleased

### Changed

- Added a Sphinx/MyST/Furo documentation site with a workflow-oriented user
  guide, scientific-convention sheet, capability matrix, curated API
  reference, offline strict build, Read the Docs configuration, and
  documentation CI.
- The CIGALE tutorials now expose finite BC03 metallicity choices explicitly:
  MixedTAMIS samples the categorical block, while the MAF simulation
  marginalizes it when metallicity is not a neural target.
- Stable inference tutorials retain their latest executed plots for readable
  GitHub examples; users must still re-execute them with local provenance
  before interpreting numerical results.
- Photometric model discrepancy is now an explicit likelihood parameter:
  `sigma_eff^2 = sigma_catalog^2 + sigma_floor^2 + (eta * f_model)^2`.
  Detections, censored upper limits, catalog grids, and SBI simulations use the
  same theta-dependent variance and normalization.
- SBI contexts retain raw catalog uncertainty rather than a discrepancy term
  estimated from observed flux.
- Added `ConditionalCatalogNoise`, a serializable joint conditional MAF for
  `q(log10 sigma_catalog | noiseless AB magnitudes)` with fixed band order,
  support checks, and training provenance.
- `EmpiricalPhotometricNoise` is deprecated; release tutorials now share the
  learned COSMOS survey-noise model and explicit likelihood discrepancy.
- CIGALE v2022.0 provenance now records its native WMAP7 redshifting
  cosmology and exact mJy/maggie and spectral conversion constants.

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
- Catalog grid likelihoods now assess model finiteness in each object's active
  bands, so a non-finite value in a masked band cannot reject an otherwise
  valid model. Cached photometric grids use schema v3 and older grids must be
  rebuilt.
- Saved inference results bind the scientific JSON metadata to the NPZ archive;
  changing the Problem, sampler identity, or posterior summary is detected at
  strict load time.
- Nonpositive SFH timescales and invalid sampled redshifts or tabular histories
  consistently raise `ModelDomainError` and therefore receive zero posterior
  probability rather than aborting inference.
- MDN density validation uses the NumPy 1.x/2.x-compatible trapezoid helper.
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
