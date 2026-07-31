# Smoke-Test Notebooks

These notebooks are fast plumbing checks. They may use analytic stellar
continua, toy nebular lines, or intentionally simplified data. They are useful
for testing array shapes, plotting code, and sampler wiring, but they should not
be used as evidence that the physical CIGALE or FSPS models are correct.

Real scientific validation notebooks live in `notebooks/validation/`.

Current smoke notebooks include:

- `11_cigale_mixed_prior_smoke_test.ipynb`: small CIGALE mixed-prior wiring
  test covering the plain grid sampler, two-block Gibbs/SIR sampler, and mixed
  TAMIS sampler.
