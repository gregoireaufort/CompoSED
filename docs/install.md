# Installing CompoSED

CompoSED is an inference/interface layer.  It does not vendor the scientific
engines it can call.  The stable rule is:

1. install CompoSED in a Python environment;
2. install the backend engine you need using that engine's upstream
   instructions;
3. declare any data/model-grid paths with environment variables;
4. run `scripts/check_environment.py` before running a notebook or fit.

This keeps the scientific provenance visible. FSPS grids and the selected
CIGALE release are part of the model definition, not generic Python utilities.

## Core Install

The core install is deliberately lightweight.  It is enough for data containers,
priors, parameter spaces, Gaussian likelihoods, mock backends, plotting, and
basic samplers.

```bash
git clone https://github.com/gregoireaufort/CompoSED.git
cd CompoSED

conda env create -f envs/composed-core.yml
conda activate composed-core

python -m pip install -e ".[dev,plot]"
python scripts/check_environment.py --core
python -m pytest -q
```

The core tests should not require FSPS, CIGALE, or torch.
Install traditional samplers with `python -m pip install -e ".[samplers]"`.
Jupyter is deliberately not part of the default environment; users who want a
kernel can install `ipykernel` in the environment they actually use.

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
conda env create -f envs/composed-fsps.yml
conda activate composed-fsps

# Follow the upstream FSPS/python-fsps instructions, then:
export SPS_HOME=/path/to/fsps

python -m pip install -e ".[fsps,plot]"
python scripts/check_environment.py --fsps
python examples/validate_fsps_backend.py
```

The validation script checks a real `FSPSBackend` call against a direct
`python-fsps` + `sedpy` calculation in the same environment.

This environment intentionally does not install notebooks, pocomc, emcee,
nflows, or torch.  Add those only when the analysis needs them:

```bash
# MAF/nflows SBI path (also includes MDN).
python -m pip install -e ".[sbi]"

# Smaller MDN-only SBI path.
python -m pip install -e ".[mdn]"

# Experimental conditional diffusion path, only when explicitly needed.
python -m pip install -e ".[diffusion]"

# Traditional samplers such as emcee/pocomc.
python -m pip install -e ".[samplers]"

# Only if this env should appear as a Jupyter kernel.
python -m pip install ipykernel
python -m ipykernel install --user --name composed-fsps
```

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
conda env create -f envs/composed-cigale.yml
conda activate composed-cigale

# Follow the upstream CIGALE v2022.0 install/setup instructions, then:
python -m pip install -e ".[cigale,plot]"

python scripts/check_environment.py --cigale
python examples/cigale_photometry_demo.py
```

The dedicated recipe pins NumPy 1.23.5 because CIGALE v2022.0's native
`sfhperiodic` module, used for an exact constant SFH, still references the
removed `np.float` alias. CompoSED reports this incompatibility explicitly on
newer NumPy and does not silently replace a constant history with an
approximately constant long-timescale model. Exponential and delayed-tau
native modules do not have this specific limitation. The recipe also pins
`setuptools<81` because CIGALE v2022.0 imports the legacy `pkg_resources`
module at runtime.

CompoSED does not hide CIGALE's database/module setup.  If CIGALE cannot build
one SED through `pcigale.warehouse.SedWarehouse`, CompoSED cannot use it either.

As with FSPS, keep the backend environment small and install neural or sampler
layers only when needed:

```bash
python -m pip install -e ".[sbi]"        # MAF/nflows SBI, plus MDN
python -m pip install -e ".[mdn]"        # MDN only
python -m pip install -e ".[diffusion]"  # experimental diffusion SBI
python -m pip install -e ".[samplers]"   # emcee, PocoMC, and SciPy sampler helpers
```

## SBI / Neural Posterior Estimation

The stable neural SBI layer provides both MAF/nflows and a smaller
Gaussian-mixture MDN. Both can run without FSPS or CIGALE when trained from a
pre-existing paired photometric dataset.

```bash
python -m pip install -e ".[sbi]"
python scripts/check_environment.py --sbi
python examples/sbi_mock_photometry_demo.py
```

The MDN only requires torch:

```bash
python -m pip install -e ".[mdn]"
python scripts/check_environment.py --mdn
python examples/sbi_mdn_mock_photometry_demo.py
```

The conditional diffusion path is experimental and uses torch only:

```bash
python -m pip install -e ".[diffusion]"
python examples/minimal_photometric_diffusion_sbi.py
```

GPU/MPS/CUDA choices are torch installation issues rather than CompoSED API
choices. Use the platform-specific torch instructions for the machine you
intend to run on.

FSPS and CIGALE can be installed together when their dependency constraints
permit it, but separate backend environments are the documented and tested
release configuration. This keeps CIGALE v2022.0's older NumPy requirement
from constraining an FSPS/SBI environment.

## What The Checker Means

`scripts/check_environment.py` checks visibility from the active interpreter:

- Python and core CompoSED imports;
- `SPS_HOME`, `fsps`, and `sedpy` for FSPS;
- `pcigale` for CIGALE;
- torch and nflows for MAF SBI, and torch alone for MDN or diffusion;
- SciPy, emcee, PocoMC, and tqdm for the traditional sampler adapters.

It does not install anything and it does not prove scientific validity.  It is a
pre-flight check that the intended backend can be reached before a long run.
Use `--samplers`, `--diffusion`, or `--all` to check those complete inference
stacks explicitly.
