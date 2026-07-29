# API Stability

CompoSED is an alpha research package. The following boundary is intentional so
scientific analyses can distinguish supported interfaces from implementation
details.

## Stable user surface

The names exported by `composed` are the primary public API:

- observed-data containers;
- priors and `ParameterSpace`;
- named SFHs;
- `Problem`, `Gaussian`, `fit`, and stable sampler configurations;
- MAF and MDN SBI configurations and trained-posterior objects;
- normalized `InferenceResult` objects and provenance-aware persistence.

The FSPS and CIGALE backend classes under `composed.backends` are also stable.
The plotting functions under `composed.plot` and diagnostics under
`inftools.diagnostics` are supported public utilities.

## Advanced interfaces

Catalog grid construction, reusable model grids, low-level sampler adapters,
and direct likelihood construction are public for analyses that need them, but
they expose more implementation detail than `Problem` plus `fit`.

## Experimental interfaces

The following may change without the same compatibility guarantees:

- `composed.catalog_fast`;
- finite-difference Laplace inference;
- the historical external TAMIS adapter;
- `inftools.experimental.diffusion`;
- direct access to underscore-prefixed helpers.

Experimental status concerns interface maturity, not permission to perform
scientific validation. Results from any path still require provenance,
posterior predictive checks, and limiting-case tests.

## Deprecation policy

Stable names should be deprecated with a warning before removal. Scientific
convention changes, especially mass normalization or units, require an explicit
schema/version change and must not be silently reinterpreted.
