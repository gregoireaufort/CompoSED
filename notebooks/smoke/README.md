# Smoke-Test Notebooks

These notebooks are fast plumbing checks. They may use analytic stellar
continua, toy nebular lines, or intentionally simplified data. They are useful
for testing array shapes, plotting code, and sampler wiring, but they should not
be used as evidence that the physical DSPS/Cue/CIGALE comparisons are correct.

Real scientific validation notebooks live in `notebooks/validation/`.

Current smoke notebooks include:

- `11_cigale_mixed_prior_smoke_test.ipynb`: small CIGALE mixed-prior wiring
  test covering the plain grid sampler, two-block Gibbs/SIR sampler, and mixed
  TAMIS sampler.
- `13_diffusion_gaussian_conditionals.ipynb`: analytic Gaussian conditional
  smoke test for experimental diffusion SBI. It checks mask/clamp behavior and
  compares sampled conditionals against exact Gaussian conditional means,
  covariances, ranks, and coverage curves.
- `14_diffusion_toy_sed_validation.ipynb`: nonlinear toy-SED conditional
  diffusion diagnostic. It trains on toy `ugrizYJH + theta` vectors, compares
  inverse and missing-band conditionals against a brute-force prior-bank
  reference, and checks the forward `mags | theta` conditional against the
  known toy noise model.
