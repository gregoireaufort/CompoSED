# Changelog

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
- Grid, random-walk, emcee, Laplace, TAMIS, mixed TAMIS/Gibbs, and PocoMC
  inference adapters.
- Stable conditional MAF and Gaussian-mixture MDN SBI with explicit
  measurement-noise conditioning, bounded target transforms, catalog batching,
  diagnostics, and reloadable checkpoints.
- Normalized inference results, provenance sidecars, posterior summaries, and
  posterior-comparison plots.

### Experimental

- Conditional diffusion remains in an explicit experimental namespace and is
  not part of the stable 0.1 API.

### Known limitations

- Censored upper limits do not yet have a stable MAF or MDN context encoding.
- Scientific calibration of an amortized posterior remains specific to its
  prior, backend, filters, masks, and noise distribution.
