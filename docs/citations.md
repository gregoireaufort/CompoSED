# Citations and Acknowledgements

CompoSED is a thin inference and interface layer around several scientific
modeling codes. If a paper, talk, or data release uses one of these backends or
experimental modules, cite the underlying model code as well as CompoSED.

## Core physical-model citations

### CIGALE

Cite CIGALE when using `composed.backends.cigale` or when using CIGALE as a
reference model in validation notebooks.

- Boquien et al. 2019, *CIGALE: a python Code Investigating GALaxy Emission*,
  A&A, 622, A103.
- For older CIGALE methodology or module-specific usage, also check the CIGALE
  documentation and the relevant module references, including Noll et al. 2009
  where appropriate.

### FSPS and python-fsps

Cite FSPS when using `composed.backends.fsps` or FSPS-derived SSP data.

- Conroy, Gunn, & White 2009, *The Propagation of Uncertainties in Stellar
  Population Synthesis Modeling. I.*, ApJ, 699, 486.
- Conroy & Gunn 2010, *The Propagation of Uncertainties in Stellar Population
  Synthesis Modeling. III.*, ApJ, 712, 833.
- If the Python wrapper is part of the reproducible workflow, acknowledge
  `python-fsps` in the software dependencies as well.

### DSPS

Cite DSPS when using `composed.experimental.jaxcigale` with DSPS stellar
population synthesis.

- Hearin et al. 2021, *DSPS: Differentiable Stellar Population Synthesis*,
  arXiv:2112.06830.

### Cue

Cite Cue when using the Cue/JAX nebular-emission path or when comparing against
Cue-derived nebular predictions.

- Li et al. 2024, *Cue: A Fast and Flexible Photoionization Emulator for
  Modeling Nebular Emission Powered By Almost Any Ionizing Source*,
  arXiv:2405.04598.

## Software infrastructure

The following packages are important infrastructure. Whether they need formal
citations depends on the venue and on how central they are to the result.

- `astropy`, for cosmology, units, and astronomy utilities.
- `sedpy`, for filter handling in FSPS/sedpy photometry paths.
- `numpy` and `scipy`, for numerical array operations.
- `jax` and `numpyro`, for differentiable models and NUTS in the experimental
  JAX-CIGALE path.
- `torch` and `nflows`, for MAF-based simulation-based inference.
- `emcee`, `pocomc`, and other samplers when their algorithms are central to
  the reported inference.

## Suggested acknowledgement text

> This work used CompoSED, which interfaces Bayesian SED-fitting likelihoods and
> samplers with CIGALE, FSPS/python-fsps, DSPS, and Cue. We thank the developers
> of these public scientific modeling codes and cite the corresponding model
> papers where those components were used.
