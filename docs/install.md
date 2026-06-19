# Installing CompoSED

CompoSED is an inference/interface layer.  It does not vendor the scientific
engines it can call.  The stable rule is:

1. install CompoSED in a Python environment;
2. install the backend engine you need using that engine's upstream
   instructions;
3. declare any data/model-grid paths with environment variables;
4. run `scripts/check_environment.py` before running a notebook or fit.

This keeps the scientific provenance visible.  FSPS grids, CIGALE releases,
DSPS SSP files, and Cue emulator data are part of the model definition, not
generic Python utilities.

## Core Install

The core install is deliberately lightweight.  It is enough for data containers,
priors, parameter spaces, Gaussian likelihoods, mock backends, plotting, and
basic samplers.

```bash
git clone https://github.com/YOUR_ORG/CompoSED.git
cd CompoSED

mamba env create -f envs/composed-core.yml
conda activate composed-core

python -m pip install -e ".[dev,plot,samplers,notebooks]"
python scripts/check_environment.py --core
python -m pytest -q
```

The core tests should not require FSPS, CIGALE, JAX, DSPS, Cue, or torch.

## FSPS Backend

CompoSED's FSPS backend uses `python-fsps`, `astro-sedpy`, and FSPS stellar
population data files.  Install FSPS following the upstream python-fsps/FSPS
instructions:

<https://python-fsps.readthedocs.io/en/latest/installation/>

The crucial runtime variable is:

```bash
export SPS_HOME=/path/to/fsps
```

`python-fsps` requires this variable and will fail to import if it is missing or
points at the wrong location.

One clean setup is:

```bash
mamba env create -f envs/composed-fsps.yml
conda activate composed-fsps

# Follow the upstream FSPS/python-fsps instructions, then:
export SPS_HOME=/path/to/fsps

python -m pip install -e ".[dev,plot,samplers,notebooks,fsps]"
python scripts/check_environment.py --fsps
python examples/validate_fsps_backend.py
```

The validation script checks a real `FSPSBackend` call against a direct
`python-fsps` + `sedpy` calculation in the same environment.

## CIGALE Backend

CompoSED's CIGALE backend targets CIGALE `v2022.0` for reproducibility.  Install
CIGALE from the upstream release/tutorial, not from an unconstrained latest
package:

<https://gitlab.lam.fr/cigale/cigale/-/tree/v2022.0>

The important practical requirement is that `pcigale` imports in the same
Python environment as CompoSED:

```bash
python -c "import pcigale; print(pcigale.__file__)"
```

One clean setup is:

```bash
mamba env create -f envs/composed-cigale.yml
conda activate composed-cigale

# Follow the upstream CIGALE v2022.0 install/setup instructions, then:
python -m pip install -e ".[dev,plot,samplers,notebooks,cigale]"

python scripts/check_environment.py --cigale
python examples/cigale_photometry_demo.py
```

CompoSED does not hide CIGALE's database/module setup.  If CIGALE cannot build
one SED through `pcigale.warehouse.SedWarehouse`, CompoSED cannot use it either.

## JAX-CIGALE, DSPS, and Cue

`composed.experimental.jaxcigale` is experimental.  It uses JAX/NumPyro for the
graph and NUTS, DSPS for stellar populations, and optionally the public Cue
emulator data for nebular emission.

For a CPU validation environment:

```bash
mamba env create -f envs/composed-science-cpu.yml
conda activate composed-science-cpu
python -m pip install -e ".[dev,plot,samplers,notebooks,jaxcigale]"
```

Cue data are not committed to this repository.  Clone the public Cue repository
and point CompoSED at the data directory:

```bash
git clone --depth 1 https://github.com/yi-jia-li/cue.git external/cue
export CUE_DATA_DIR=$PWD/external/cue/src/cue/data
```

DSPS/Cue validation with nebular emission also needs a continuum SSP resource
that is consistent with the JAX-CIGALE stellar module:

```bash
export DSPS_CONTINUUM_SSP_FILE=/path/to/fsps_continuum_ssp_data.h5
```

Then check:

```bash
python scripts/check_environment.py --jaxcigale
python scripts/check_environment.py --cue
```

Use `--cue` when you expect the public Cue data files to be present.  The Cue
loader uses old public pickle files; `scikit-learn` may warn about the version
used to create those pickles.  Treat that warning as provenance information and
record it for validation runs.

## SBI / Neural Posterior Estimation

The SBI layer uses torch and nflows.  It can run without FSPS or CIGALE if you
train from a pre-existing `(theta, x)` dataset.

```bash
python -m pip install -e ".[sbi]"
python scripts/check_environment.py --sbi
python examples/sbi_mock_photometry_demo.py
```

GPU/MPS/CUDA choices are torch/JAX installation issues rather than CompoSED
API choices.  Use the platform-specific torch/JAX instructions for the machine
you intend to run on.

## Full Local Science Stack

For development and validation on a machine where all upstream engines are
available:

```bash
mamba env create -f envs/composed-science-cpu.yml
conda activate composed-science-cpu

# Follow upstream CIGALE v2022.0 setup.
# Follow upstream FSPS/python-fsps setup.
export SPS_HOME=/path/to/fsps
export CUE_DATA_DIR=$PWD/external/cue/src/cue/data
export DSPS_CONTINUUM_SSP_FILE=/path/to/fsps_continuum_ssp_data.h5

python -m pip install -e ".[all]"
python scripts/check_environment.py --all
python -m pytest -q
```

This is the environment intended for the validation notebooks in
`notebooks/validation/`.

## What The Checker Means

`scripts/check_environment.py` checks visibility from the active interpreter:

- Python and core CompoSED imports;
- `SPS_HOME`, `fsps`, and `sedpy` for FSPS;
- `pcigale` for CIGALE;
- JAX, NumPyro, DSPS, h5py, dill, and scikit-learn for JAX-CIGALE;
- `CUE_DATA_DIR` and expected public Cue files for Cue;
- torch and nflows for SBI.

It does not install anything and it does not prove scientific validity.  It is a
pre-flight check that the intended backend can be reached before a long run.

