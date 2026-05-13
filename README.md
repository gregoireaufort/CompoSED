# sedinfer

Small interfaces for Bayesian SED fitting and photo-z inference.

```python
import numpy as np

from sedinfer import GaussianPhotometricLikelihood, ParameterSpace, SEDDataset
from sedinfer.backends.mock import MockBackend
from sedinfer.priors import DeltaPrior

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
must declare their `MassNormalization`. The likelihood multiplies by
`10**log10_mass` only for `MassNormalization.PER_SOLAR_MASS`.

## FSPS backend

`FSPSBackend` is optional and requires `python-fsps`, FSPS stellar population
grids, `sedpy`, and `astropy`. Configure `SPS_HOME` before constructing the
backend.

```python
import numpy as np

from sedinfer.backends.fsps import FSPSBackend
from sedinfer.filters import FilterSet

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

The backend returns observed-frame fluxes in maggies. With the default
`MassNormalization.PER_SOLAR_MASS`, the tabular SFH is normalized to one solar
mass formed and the likelihood is responsible for applying `10**log10_mass`.

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
`sedinfer` or the rest of `inftools`; constructing the estimator will raise a
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

See `notebooks/cosmos2020_sbi_fsps_gpu_timing.ipynb` for a COSMOS2020 +
FSPS + MAF setup focused on GPU/MPS posterior-sampling timing.
