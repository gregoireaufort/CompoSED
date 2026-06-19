# Environment Notes

See [`docs/install.md`](install.md) for the public installation workflow.

This page records the local full-stack environment used during current
CompoSED validation.  It is not meant to replace upstream backend installation
instructions.

## Local Full-Stack Choices

- Python 3.11.
- NumPy 1.26 for compatibility with CIGALE 2022-era dependencies.
- CIGALE target release: `v2022.0` from
  <https://gitlab.lam.fr/cigale/cigale/-/tree/v2022.0>.
- CPU JAX for portable validation:
  `jax==0.4.38`, `jaxlib==0.4.38`, `numpyro==0.17.0`.
- `astro-sedpy==0.4.1` for the FSPS/sedpy photometry path.
- `SPS_HOME` points at the local FSPS data directory.
- `CUE_DATA_DIR` points at a local clone of
  <https://github.com/yi-jia-li/cue>, specifically `src/cue/data`.
- `DSPS_CONTINUUM_SSP_FILE` points at the continuum SSP HDF5 file used by the
  JAX-CIGALE DSPS module.

The current local science environment can be checked with:

```bash
python scripts/check_environment.py --all
python -m pytest -q
```

The checker is the authoritative way to know which optional backends are usable
from the active Python interpreter.
