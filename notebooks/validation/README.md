# Validation Notebooks

These notebooks are intended to exercise the real scientific pipeline, not just
API plumbing. They call the same CIGALE v2022.0 and FSPS backend paths used by
the public release.

Current notebooks:

- `09_cigale_mixed_prior_validation.ipynb`: real CIGALE mock photometry and
  CIGALE refit under a mixed continuous/discrete prior, comparing grid, Gibbs,
  and mixed TAMIS samplers.

Fast analytic or toy-nebular notebooks belong in `notebooks/smoke/`, where they
can remain useful for debugging without being confused for scientific
validation.

Validation notebooks that load cached arrays must check provenance before
plotting. Use `composed.provenance.require_provenance(path)` before loading a
cached `.npz`; use `composed.provenance.save_npz_with_provenance(...)` when
creating one. At minimum, provenance should record the git commit/dirty flag,
engine versions, random seed, command arguments, `SPS_HOME`, SSP file path and
hash, and any input catalog or filter curves. If a sidecar is missing, the plot
should fail loudly rather than silently reusing a stale local product. See
`docs/validation_provenance.md`.
