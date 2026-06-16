# CompoSED

Composable Bayesian SED fitting and photo-z inference.

```python
import numpy as np

from composed import GaussianPhotometricLikelihood, ParameterSpace, SEDDataset
from composed.backends.mock import MockBackend
from composed.priors import DeltaPrior

data = SEDDataset(
    band_names=["g", "r"],
    flux=np.array([1.0, 2.0]),
    sigma=np.array([0.1, 0.2]),
)

backend = MockBackend(flux=[1.1, 1.8], band_names=["g", "r"])
space = ParameterSpace(names=["z"], priors={"z": DeltaPrior(0.5)})
like = GaussianPhotometricLikelihood(backend, data, space)

print(like.log_prob(np.array([0.5])))
```

Backends expose `predict_photometry(params, filters) -> ModelPhotometry` and
may expose `predict_spectrum(params, wavelengths=...) -> ModelSpectrum`. They
must declare their `MassNormalization`. The likelihood multiplies by
`10**log10_mass` only for `MassNormalization.PER_SOLAR_MASS`.

## Spectral likelihood

Spectra use observed-frame Angstrom and observed `f_lambda` in
`erg s^-1 cm^-2 Angstrom^-1`.

```python
import numpy as np

from composed import GaussianSpectralLikelihood, ParameterSpace, SpectrumDataset
from composed.backends.mock import MockBackend
from composed.priors import DeltaPrior

data = SpectrumDataset(
    wavelength=np.array([5000.0, 5100.0, 5200.0]),
    flux=np.array([1.0, 1.2, 0.9]),
    sigma=np.array([0.1, 0.1, 0.1]),
)

backend = MockBackend(
    flux=[],
    spectrum_wavelength=[5000.0, 5100.0, 5200.0],
    spectrum_flux=[1.0, 1.1, 0.95],
)
space = ParameterSpace(names=["z"], priors={"z": DeltaPrior(0.0)})
like = GaussianSpectralLikelihood(backend, data, space)

print(like.log_prob([0.0]))
```

`GaussianSpectralLikelihood` requests the model on the active data wavelength
grid, applies the dataset mask to wavelength/flux/sigma together, and applies
mass normalization with the same explicit rule as the photometric likelihood.
Calibration polynomials, covariance matrices, and instrumental convolution are
not included in this first pass.

## FSPS backend

`FSPSBackend` is optional and requires `python-fsps`, FSPS stellar population
grids, `sedpy`, and `astropy`. Configure `SPS_HOME` before constructing the
backend.

```python
import numpy as np

from composed.backends.fsps import FSPSBackend
from composed.filters import FilterSet

from sedpy.observate import load_filters

filters = FilterSet(load_filters(["sdss_g0", "sdss_r0"]), names=["sdss_g0", "sdss_r0"])
backend = FSPSBackend()

phot = backend.predict_photometry(
    {
        "zred": 0.1,
        "logzsol": -0.3,
        "dust2": 0.2,
        "tabular_time_gyr": np.array([0.01, 1.0, 5.0]),
        "tabular_sfr_msun_per_yr": np.array([1.0, 1.0, 0.2]),
    },
    filters,
)
print(dict(zip(phot.band_names, phot.flux)))
```

The backend returns observed-frame photometry in maggies and observed-frame
spectra in `f_lambda` cgs per Angstrom. With the default
`MassNormalization.PER_SOLAR_MASS`, the tabular SFH is normalized to one solar
mass formed and the likelihood is responsible for applying `10**log10_mass`.

## CIGALE backend

`CIGALEBackend` is optional and requires CIGALE/`pcigale` and its database. It
uses CIGALE's `SedWarehouse.get_sed` API and returns observed-frame maggies for
photometry and observed-frame `f_lambda` cgs per Angstrom for spectra. Native
CIGALE filter names can be passed as strings; sedpy filters are also supported
via `photometry_mode="sedpy"`.

In `composed`, CIGALE is deliberately treated as a per-solar-mass backend.
SFH module `normalise=True` is enforced, and the Gaussian likelihood applies
`10**log10_mass`.

```python
from composed.backends.cigale import build_cigale_backend_and_parameter_space
from composed.filters import FilterSet
from composed.priors import UniformPrior

modules = ["sfhdelayed", "bc03", "redshifting"]
module_parameters = {
    "sfhdelayed": {
        "tau_main": {"range": [500.0, 5000.0]},
        "age_main": {"values": [1000, 3000, 5000], "dtype": "int"},
    },
    "bc03": {
        "imf": 1,
        "metallicity": {"values": [0.008, 0.02]},
    },
    "redshifting": {
        "redshift": {"name": "z", "range": [0.0, 2.0]},
    },
}

backend, space = build_cigale_backend_and_parameter_space(
    modules,
    module_parameters,
    additional_priors={"log10_mass": UniformPrior(8.0, 12.0)},
)

filters = FilterSet(["sdss.u", "sdss.g", "sdss.r"])
phot_per_msun = backend.predict_photometry(
    {"tau_main": 2000.0, "age_main": 3000, "metallicity": 0.02, "z": 0.5},
    filters,
)
```

See `examples/cigale_photometry_demo.py` and `docs/cigale_backend.md`.

## Running Real FSPS Validation Locally

The normal test suite can run without FSPS. Real FSPS validation requires:

- `python-fsps`
- `sedpy`
- `astropy`
- FSPS stellar population grids
- `SPS_HOME` set to the FSPS data directory

Run the optional pytest integration checks with:

```bash
python -m pytest -q -m fsps
```

Run the standalone numerical validation script with:

```bash
python examples/validate_fsps_backend.py
```

The script compares `FSPSBackend` against an independent direct
`python-fsps` + `sedpy` calculation in the same environment. It checks flux
shape, finite positive maggies, relative flux agreement, and AB magnitude
agreement. CI or lightweight development environments may skip these tests when
FSPS, sedpy, or `SPS_HOME` are unavailable.

## Simulation-Based Inference / MAF Posterior Estimator

`inftools.sbi` adds an optional Masked Autoregressive Flow posterior estimator
using `torch` and `nflows`. These dependencies are not required for importing
`composed` or the rest of `inftools`; constructing the estimator will raise a
helpful `ImportError` if they are missing.

Conceptual pipeline:

1. Define a `ParameterSpace`.
2. Wrap a backend with `GaussianPhotometricLikelihood`.
3. Define a flux-noise function, for example
   `sigma = sigma_floor + frac_error * abs(flux)`.
4. Simulate `(theta, x)` training pairs.
5. Train a conditional MAF estimator `q(theta | x)`.
6. Condition on observed active-band fluxes and draw posterior samples.

```python
import numpy as np

from inftools.sbi import simulate_training_set, train_maf_posterior

def noise_fn(flux):
    return 0.02 + 0.05 * np.abs(flux)

theta_train, x_train = simulate_training_set(
    parameter_space,
    likelihood,
    n=1000,
    noise_fn=noise_fn,
    rng=np.random.default_rng(1),
    batch_size=16,
    n_workers=4,
    executor="process",
)

estimator = train_maf_posterior(
    theta_train,
    x_train,
    hidden_features=64,
    num_transforms=3,
    epochs=50,
    batch_size=128,
)

samples = estimator.sample(x_obs, num_samples=10000)
```

SBI quality depends strongly on prior coverage, simulator fidelity, noise
modeling, and diagnostic checks. The simulator produces the same active-band
flux vector convention consumed by the Gaussian likelihood.

For expensive forward models such as FSPS, `simulate_training_set` can split
prior draws into chunks and evaluate them in worker processes. Each worker keeps
its own simulator/backend object alive across chunks, which avoids rebuilding
the stellar population machinery for every object. Process mode requires the
simulator and `noise_fn` to be pickleable; if you are calling it from a notebook,
define those at top level or run from a small script.

See `notebooks/cosmos2020_sbi_fsps_gpu_timing.ipynb` for a COSMOS2020 +
FSPS + MAF setup focused on GPU/MPS posterior-sampling timing.

## Catalog-scale fitting

For finite photometric grids, use the catalog grid evaluator so the backend
model grid is computed once and the Gaussian likelihood is evaluated against
all objects with chunked array operations:

```python
from composed import run_photometric_grid_catalog

result = run_photometric_grid_catalog(
    backend,
    datasets,          # list[SEDDataset], same band order
    parameter_space,   # finite-valued or fixed priors
    filters=filters,
    object_chunk_size=1024,
    model_chunk_size=4096,
)

z_map = result.map_estimates[:, parameter_space.names.index("z")]
```

For independent per-object samplers, use `inftools.fit_many` to run a
single-object fitting function across a catalog with serial, thread, or process
execution. For process execution, build fragile backend state such as FSPS or
CIGALE inside the worker function rather than trying to pickle a live backend.

## Saving and plotting inference runs

The intended fitting pipeline is:

1. declare a backend, parameter space, and likelihood;
2. declare the observed `SEDDataset` and/or `SpectrumDataset`;
3. run a sampler;
4. normalize the sampler output;
5. save the result;
6. make diagnostic plots.

```python
from composed import normalize_sampling_result, save_inference_result
from composed.plot import (
    plot_corner_hexbin,
    plot_posterior_predictive_sed,
    plot_traces,
)

raw = run_mixed_tamis(posterior, parameter_space, ...)
result = normalize_sampling_result(
    raw,
    parameter_space,
    sampler_name="mixed_tamis",
)
save_inference_result(result, "runs/object_001")

plot_corner_hexbin(result)
plot_traces(result)
plot_posterior_predictive_sed(
    result,
    backend,
    parameter_space,
    photometry=dataset,
    filters=filters,
)
```

`InferenceResult.weights` are always normalized. MCMC-like outputs use uniform
weights by default, while grid/TAMIS-style outputs use `weights_norm` from the
sampler metadata when available. Results are saved as an `.npz` array file plus
a JSON sidecar containing metadata and posterior summaries.

## Experimental JAX-CIGALE

`composed.experimental.jaxcigale` is a JAX-native, CIGALE-inspired prototype.
It does not call `pcigale`; instead it keeps the CIGALE idea of a fixed ordered
module chain while making each module a pure JAX operation after setup.

Optional dependencies:

```bash
pip install "composed[jaxcigale]"
```

Minimal analytic-stellar demo:

```python
import numpy as np

from composed.experimental.jaxcigale import (
    JaxFilterSet,
    JaxParameterSpace,
    UniformJaxPrior,
    analytic_stellar_module,
    build_jax_sed_model,
    delayed_sfh_module,
    no_nebular_module,
    redshift_module,
)

wave_rest = np.linspace(900.0, 20000.0, 512)
age_grid = np.linspace(0.02, 8.0, 64)
filter_wave = np.linspace(4000.0, 9000.0, 128)
filters = JaxFilterSet.from_curves(["wide"], [filter_wave], [np.ones_like(filter_wave)])

space = JaxParameterSpace(
    names=["log10_mass", "z", "tau_gyr", "tage_gyr", "logzsol"],
    priors={
        "log10_mass": UniformJaxPrior(8.0, 12.0),
        "z": UniformJaxPrior(0.0, 3.0),
        "tau_gyr": UniformJaxPrior(0.2, 8.0),
        "tage_gyr": UniformJaxPrior(0.2, 10.0),
        "logzsol": UniformJaxPrior(-1.0, 0.3),
    },
)

model = build_jax_sed_model(
    [delayed_sfh_module(age_grid), analytic_stellar_module(), no_nebular_module(), redshift_module()],
    wave_rest,
    filters,
    space,
)
```

For science runs, replace `analytic_stellar_module()` with
`dsps_stellar_module(ssp_data)`. Nebular emission is currently an explicit graph
slot: `no_nebular_module()` can be replaced by `nebular_emulator_module(...)`
once a CLOUDY/Cue-style emulator is validated. See
`docs/experimental_jaxcigale.md` and
`examples/experimental_jaxcigale_photometry_demo.py`.
