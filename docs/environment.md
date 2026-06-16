# Unified Local Environment

This project is easiest to use from one environment containing the optional
FSPS, CIGALE, JAX-CIGALE, and SBI stacks.  The local development environment is
called `composed`.

The important compatibility choices are:

- Python 3.11.
- NumPy 1.26, because CIGALE 2022 and older astronomy packages are safer on the
  NumPy 1.x line.
- CPU JAX with `jax==0.4.38`, `jaxlib==0.4.38`, and `numpyro==0.17.0`.
- CIGALE 2022.0, installed from the local checkout rather than a newer CIGALE
  release.
- `astro-sedpy==0.4.1`, not the newer `sedpy` package, because the FSPS path uses
  `sedpy.observate`.
- `scikit-learn`, because Cue's public data loader imports it.
- `SPS_HOME=/Users/gregoire/Work/FSPS` for python-fsps.

The environment was created locally with:

```bash
mamba create -y -n composed --override-channels -c conda-forge \
  python=3.11 pip numpy=1.26 scipy=1.12 astropy=6.1 matplotlib \
  pytest ipykernel jupyterlab pandas h5py tqdm configobj rich emcee corner \
  jax=0.4.38 jaxlib=0.4.38 numpyro=0.17 dsps scikit-learn

conda env config vars set -n composed SPS_HOME=/Users/gregoire/Work/FSPS
conda activate composed

python -m pip install fsps==0.4.7 astro-sedpy==0.4.1 torch nflows pocomc
python -m pip install --no-deps -e /Users/gregoire/Work/cigale-v2022.0
python -m pip install --no-deps -e /Users/gregoire/Documents/Sedfitting/CompoSED
python -m ipykernel install --user --name composed --display-name "Python (composed)"
```

Validation commands:

```bash
conda activate composed
python -c "import fsps, pcigale, jax, numpyro, dsps, torch, nflows, sklearn, composed"
pytest -q
```

On this machine the full test suite passed in the `composed` environment:

```text
180 passed, 3 skipped
```
