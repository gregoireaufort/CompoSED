# Environment Notes

See [`docs/install.md`](install.md) for the public installation workflow.

This page records the backend environment choices used for CompoSED v0.1
validation. It is not meant to replace upstream backend installation
instructions.

## Release Environment Choices

- Python 3.11.
- CIGALE target release: `v2022.0` from
  <https://gitlab.lam.fr/cigale/cigale/-/tree/v2022.0>.
- NumPy 1.23.5 in the CIGALE environment, preserving the upstream
  `sfhperiodic` implementation.
- `astro-sedpy==0.4.1` for the FSPS/sedpy photometry path.
- `SPS_HOME` points at the local FSPS data directory.

The two release backend environments can be checked independently:

```bash
python scripts/check_environment.py --fsps
python scripts/check_environment.py --cigale
python -m pytest -q
```

The checker reports which optional backends and inference dependencies are
visible from the active Python interpreter. The environment recipes are
`envs/composed-fsps.yml` and `envs/composed-cigale.yml`.
