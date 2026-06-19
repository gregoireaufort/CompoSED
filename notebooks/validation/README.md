# Validation Notebooks

These notebooks are intended to exercise the real scientific pipeline, not just
API plumbing. They should use the same effective modules that a science run
would use: DSPS stellar populations, Cue nebular emission, explicit dust/IGM
modules, and the real JAX likelihoods.

Current notebooks:

- `03_gordon16_dust_validation.ipynb`: validates the Gordon et al. dust law.
- `04_backend_cross_validation_single_sed.ipynb`: compares backend spectra for
  a single SED.
- `05_cigale_mock_jaxcigale_photometric_validation.ipynb`: cross-model
  photometric mock fit.
- `06_dsps_cue_spectrum_fitting_validation.ipynb`: real DSPS + Cue spectral
  self-consistency fit.
- `07_jades_like_dsps_cue_prism_validation.ipynb`: real DSPS + Cue high-redshift
  NIRSpec/PRISM-like spectral fit.
- `08_dsps_cue_joint_spectrophotometry_validation.ipynb`: real DSPS + Cue joint
  spectrum plus photometry fit, including an upper-limit band.
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
hash, Cue data directory and hash, and any input catalog or filter curves. If a
sidecar is missing, the plot should fail loudly rather than silently reusing a
stale local product. See `docs/validation_provenance.md`.
