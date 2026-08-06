# CompoSED

Composable Bayesian SED fitting and photo-z inference.

The release-ready CompoSED workflow is **photometry only**. Backend spectrum
generation and spectral or joint spectrophotometric fitting remain
experimental interfaces: they are available for development and validation,
but are not part of the supported analysis pipeline yet.

The [hosted documentation](https://gregoireaufort.github.io/CompoSED/) contains
the workflow-oriented user guide, scientific conventions, capability matrix,
and generated API reference. Its source starts at [`docs/index.md`](docs/index.md).
A local HTML build is:

```bash
python -m pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Installation

The core CompoSED package is lightweight, but the FSPS and CIGALE scientific
engines have their own upstream installation and data requirements. Start with
the install guide:

The installable distribution is named **`composed-sed`**, while the Python
package remains **`composed`**. This distinction is intentional: the unrelated
PyPI distribution named `composed` is not this project. A source checkout is
installed with the commands below and imported with `import composed`.

```bash
python -m pip install -e ".[dev,plot]"
python scripts/check_environment.py --core
```

Then install only the backend stack you need. FSPS users must configure
`SPS_HOME`; CIGALE users should install the upstream `v2022.0` release.

Optional inference layers are installed independently:

```bash
python -m pip install -e ".[samplers,mc-diagnostics]"  # samplers plus ArviZ diagnostics
python -m pip install -e ".[sbi]"       # MAF: torch and nflows
python -m pip install -e ".[mdn]"       # MDN only: torch
```

See [`docs/install.md`](docs/install.md) for the supported environment recipes
and backend checks.

## COSMOS2020 tutorial suite

The compact end-to-end tutorials in [`notebooks/tutorials`](notebooks/tutorials)
all use the same selected COSMOS2020 ugrizYJH catalog and the same representative
galaxy:

- FSPS continuity-SFH + uncertainty-conditioned MAF for 100,000 objects;
- CIGALE delayed-tau + uncertainty-conditioned MAF for 100,000 objects;
- CIGALE delayed-tau + self-contained TAMIS for one object;
- FSPS continuity-SFH + PocoMC for the same object.

Run `00_prepare_cosmos2020_ugrizYJH.ipynb` once to create the ignored 6.7 MB
tutorial artifact from the full FARMER FITS catalog. The remaining notebooks
never parse the source catalog. Set `COMPOSED_TUTORIAL_QUICK=1` for a short MAF
installation check; normal catalog runs use 300,000 prior simulations.

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
must declare their `MassNormalization`. The likelihood multiplies by
`10**log10_mass` only for `MassNormalization.PER_SOLAR_MASS`. In the public
contract, `log10_mass` is the present-day surviving stellar mass and per-mass
backend outputs are normalized by that same quantity.

`log10_mass` is directly comparable to CIGALE's `bayes.stellar.m_star`.
Prospector's usual `mass`/`logmass` parameter is formed mass instead; multiply
it by Prospector's surviving fraction `mfrac` before comparing it with
CompoSED.

The stable traditional inference paths are grid, random walk, emcee,
self-contained mixed Gibbs/TAMIS, and PocoMC. The finite-difference Laplace
runner and the adapter named `TAMIS` for a separately installed historical
package are experimental in 0.1.1 and emit a warning. Use `MixedTAMIS` for the
self-contained CompoSED implementation.

## Acknowledgements and citations

CompoSED is an inference/interface layer around scientific modeling codes. If
you use the CIGALE or FSPS paths, cite those projects as well as CompoSED. See
`docs/citations.md` for the current citation checklist and a short
acknowledgement template.

## FSPS backend

`FSPSBackend` is optional and requires `python-fsps`, FSPS stellar population
grids, `sedpy`, and `astropy`. Configure `SPS_HOME` before constructing the
backend.

```python
from composed import DelayedTauSFH
from composed.backends.fsps import FSPSBackend
from composed.filters import FilterSet

from sedpy.observate import load_filters

filters = FilterSet(load_filters(["sdss_g0", "sdss_r0"]), names=["sdss_g0", "sdss_r0"])
backend = FSPSBackend(
    sfh=DelayedTauSFH(),
    sp_kwargs={"add_igm_absorption": True},
)

phot = backend.predict_photometry(
    {
        "zred": 0.1,
        "logzsol": -0.3,
        "gas_logz": -0.3,
        "dust2": 0.2,
        "tage_gyr": 5.0,
        "tau_gyr": 1.5,
    },
    filters,
)
print(dict(zip(phot.band_names, phot.flux)))
```

The stable backend output is observed-frame photometry in maggies. Its
experimental spectrum output uses observed-frame `f_lambda` cgs per Angstrom.
With the default
`MassNormalization.PER_SOLAR_MASS`, the tabular SFH is internally normalized
to one solar mass formed. The backend then divides the FSPS spectrum by
`sp.stellar_mass`, returning luminosity per one solar mass of surviving stars.
The likelihood applies `10**log10_mass` once. Whether FSPS includes stellar
remnants follows `add_stellar_remnants` and is recorded in model metadata.

CompoSED supplies named `constant`, `exponential`, `delayed_tau`,
`continuity`, and `tabular` SFHs for FSPS. Priors remain explicit in
`ParameterSpace`; the SFH object only maps scalar parameters to the validated
tabular history. See [`docs/sfh_models.md`](docs/sfh_models.md) for equations,
age conventions, normalization, and backend support.

## CIGALE backend

`CIGALEBackend` is optional and requires CIGALE/`pcigale` and its database. It
uses CIGALE's `SedWarehouse.get_sed` API and returns observed-frame maggies for
photometry. Its experimental spectrum output uses observed-frame `f_lambda`
cgs per Angstrom. Native
CIGALE filter names can be passed as strings; sedpy filters are also supported
via `photometry_mode="sedpy"`.

In `composed`, CIGALE is deliberately treated as a per-surviving-stellar-mass
backend. SFH module `normalise=True` first gives a unit-formed-mass SED; the
backend divides it by CIGALE's `stellar.m_star`, and the Gaussian likelihood
then applies `10**log10_mass` once.

```python
from composed import DelayedTauSFH
from composed.backends.cigale import build_cigale_backend_and_parameter_space
from composed.filters import FilterSet
from composed.priors import UniformPrior

modules = ["bc03", "redshifting"]
module_parameters = {
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
    additional_priors={
        "log10_mass": UniformPrior(8.0, 12.0),
        "tage_gyr": UniformPrior(0.1, 5.0),
        "tau_gyr": UniformPrior(0.1, 5.0),
    },
    sfh=DelayedTauSFH(),
)

filters = FilterSet(["sdss.up", "sdss.gp", "sdss.rp"])
phot_per_stellar_msun = backend.predict_photometry(
    {"tau_gyr": 2.0, "tage_gyr": 3.0, "metallicity": 0.02, "z": 0.5},
    filters,
)
```

Named SFHs use one cross-backend contract: CompoSED first evaluates the
canonical tabular history, then sends it to FSPS directly or projects it onto
CIGALE's chronological 1 Myr grid. Constant, exponential, delayed-tau,
continuity, and arbitrary tabular histories therefore work with both engines.
Native CIGALE SFH modules remain available through the original module-list
API when `sfh` is omitted. See
[`docs/cigale_backend.md`](docs/cigale_backend.md),
[`docs/sfh_models.md`](docs/sfh_models.md), and the maintained
[`COSMOS2020 tutorials`](notebooks/tutorials/README.md).

The in-memory CompoSED SFH bridge does not patch the CIGALE installation or
write temporary SFH files. Its source and projection convention are included
in saved Problem provenance.

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

The marked integration test compares `FSPSBackend` against an independent
direct `python-fsps` + `sedpy` calculation in the same environment. It checks
flux shape, finite positive maggies, relative flux agreement, and AB magnitude
agreement. CI or lightweight development environments may skip it when FSPS,
sedpy, or `SPS_HOME` are unavailable.

FSPS spectra are converted with the fixed constants
`L_sun = 3.828e33 erg/s`, `1 pc = 3.085677581491367e18 cm`, and the AB
zero point `3631 Jy`. CompoSED does not adjust these constants to match another
backend; they are recorded in FSPS model and Problem provenance.

## Simulation-Based Inference / Neural Posterior Estimators

The stable `0.1` SBI methods are a conditional Masked Autoregressive Flow
(`MAF`) and a conditional Gaussian-mixture density network (`MDN`). Both learn
`q(theta | measured flux, measurement sigma)`. The default context contains
`flux/sigma` and `log10(sigma)` for every active band, so negative noisy fluxes
are valid and heteroscedastic catalog depths are conditioned on explicitly.

```python
from composed import (
    ConditionalCatalogNoise, Gaussian, MAF, PhotometricContext,
    Problem, SEDDataset, Simulate, fit,
)

# Fit this once from complete rows in the survey catalog. Magnitudes are AB;
# catalog_sigma contains the raw catalog uncertainty in maggies.
survey_noise = ConditionalCatalogNoise.fit(
    catalog_magnitudes,
    catalog_sigma,
    band_names=filters.names,
    flux_unit="maggies",
    seed=6,
)

problem = Problem(
    backend=backend,
    parameters=parameter_space,
    data=data,
    likelihood=Gaussian(photometric_model_discrepancy=0.05),
    filters=filters,
)

result = fit(
    problem,
    method=MAF(
        epochs=200,
        batch_size=2048,
        validation_split=0.1,
        patience=20,
        num_samples=512,
        inference_batch_size=8192,
        device="auto",
    ),
    training=Simulate(
        n=100_000,
        noise_model=survey_noise,
        infer=["zred", "log10_mass"],
        context=PhotometricContext("snr_logsigma"),
    ),
    seed=7,
)

samples = result.samples
posterior = result.inference_state
posterior.save("runs/photoz_maf")
```

In Problem-based SBI, `infer=` selects target names but does not establish a
separate column convention. Targets are ordered according to
`problem.parameters.names`. `posterior.theta_names` is the canonical inferred
subset used by `sample()` and `log_prob()`; `result.parameter_names` preserves
that order while possibly adding conditioned or `DeltaPrior` columns. The
originally requested sequence is retained in
`result.metadata["requested_infer"]` for provenance. Pre-existing-array SBI
instead uses the explicitly supplied `SBITrainingSet.theta_names` order.

The three uncertainties in this workflow are deliberately separate:

- `sigma_catalog` is the raw survey uncertainty. It is stored in
  `SEDDataset.sigma`, sampled by `ConditionalCatalogNoise`, and supplied to the
  neural context.
- `eta=photometric_model_discrepancy` is a dimensionless model-error
  amplitude. It is part of the likelihood, not part of the catalog.
- `sigma_draw**2 = sigma_catalog**2 + sigma_floor**2
  + (eta * f_model)**2` is used to draw and score flux.

The model term is evaluated from each proposed `f_model`, so its
theta-dependent Gaussian normalization is retained. Observed inference always
uses the catalog's raw sigma; CompoSED never substitutes `eta * f_obs`.
`ConditionalCatalogNoise` models the joint multiband distribution
`q(log10 sigma_catalog | noiseless AB magnitudes)`, records its exact band
order and training support, and warns or fails on extrapolation without
clamping.

Simulation fails on the first invalid prior draw by default. This prevents a
backend failure from silently changing the training distribution. If replacing
failed simulations is scientifically intended, request it explicitly:

> **Effective-prior warning:** `failure_policy="resample"` trains on the
> declared prior *conditioned on successful simulation and noise-model
> support*. For dropout objects, a learned survey-noise model can reject very
> faint noiseless blue-band predictions. Always inspect the accepted effective
> prior; it may be narrower and correlated even when the declared priors are
> independent.

```python
training = Simulate(
    n=100_000,
    noise_model=survey_noise,
    infer=["zred", "log10_mass"],
    failure_policy="resample",
    warn_retry_fraction=0.05,
)
```

The returned training metadata then labels the theta distribution
`"simulator_success_conditioned"` and records its acceptance fraction and
failure examples. Once failed attempts exceed `warn_retry_fraction`, CompoSED
warns without stopping or clipping the run. Set an explicit
`max_attempts=N` only when a hard compute budget is required.

Inspect the effective training prior whenever rows were replaced:

```python
from composed.plot import plot_effective_prior

plot_effective_prior(
    result.inference_state.training_set,
    problem.parameters,
)
```

For redshift-dependent galaxy ages,
`age_kind="fraction_of_universe"` guarantees
`tage <= age_universe(z)`. For `ContinuitySFH`, it does **not** guarantee that
the age exceeds the last fixed lookback-bin edge. The age-fraction prior and
`lookback_edges_gyr` must still be chosen compatibly; rejected rows remain
part of the effective-prior check above.

At inference, MAF and MDN checkpoints compare the encoded observation vector
with each training feature's minimum and maximum. Extrapolation emits a
warning but is not clipped. A second warning identifies bounded posterior
marginals that are both very narrow and pressed against a prior edge. These
are deliberately simple triage checks, not a multivariate out-of-distribution
test.

Stable MAF and MDN checkpoints have a fixed ordered band schema. Every object
passed to one checkpoint must provide finite flux and positive sigma in all of
those bands; per-object masks do not change the neural input dimension. For a
catalog with heterogeneous coverage, train one estimator per common coverage
pattern (and route objects accordingly), or restrict the catalog to a shared
band set. Availability-mask conditioning is not part of the stable MAF/MDN
interface.

Use the same workflow with a small, closed-form mixture posterior by changing
only the method:

```python
from composed import MDN

result = fit(
    problem,
    method=MDN(
        n_components=8,
        hidden_features=128,
        num_blocks=3,
        epochs=200,
        device="auto",
    ),
    training=Simulate(
        n=100_000,
        noise_model=survey_noise,
        infer=["zred", "log10_mass"],
        context=PhotometricContext("snr_logsigma"),
    ),
    seed=7,
)
```

`MDN` requires only PyTorch (`pip install -e ".[mdn]"`). Each component has a
diagonal covariance in transformed parameter space; the mixture can represent
multiple modes, while `MAF` remains the more expressive default for strongly
curved high-dimensional posteriors. Both expose physical-space `sample` and
`log_prob` methods for continuous targets and use the same prior-support
transforms. MAF additionally supports inferred `ChoicePrior` axes through an
exact categorical posterior over their Cartesian support and a continuous flow
conditioned on the selected category. Its `log_prob` is therefore a categorical
log mass plus a continuous log density. MDN remains continuous-only.

Uniform and log-uniform targets use invertible bounded transforms, so physical
samples remain inside prior support without rejection. Catalog inference is
batched inside the neural estimator:

```python
catalog_samples = posterior.sample(
    catalog_flux,
    sigma=catalog_sigma,
    input_units="native",
    num_samples=128,
    batch_size=8192,
    seed=8,
)
```

`device="auto"` validates CUDA, then MPS, then CPU using float32. Checkpoints
store tensor weights, standardizers, the ordered context schema, target prior
transforms, versions, and training metadata without duplicating the simulation
table. For pre-existing photometry use `SBITrainingSet.from_photometry`.

See [`docs/photometric_sbi_quickstart.md`](docs/photometric_sbi_quickstart.md)
and the COSMOS2020 tutorial suite above. The tests described in
[`docs/maf_validation.md`](docs/maf_validation.md) check the MAF software
contract, including shapes, transforms, persistence, and finite densities.
They are not a substitute for held-out calibration with the simulator, noise
model, selection, and prior used by a particular scientific analysis.

## Sampler diagnostics

Traditional inference results use one sampler-aware diagnostics entry point:

```python
from composed import diagnose
from composed.plot import plot_diagnostics, plot_traces

report = diagnose(result)
print(report.summary())

plot_diagnostics(result, report)
plot_traces(result)
```

MCMC results use rank-normalized R-hat, bulk/tail ESS, MCSE, and acceptance
telemetry where chain structure permits it. PocoMC and TAMIS instead report
importance-weight ESS, entropy/perplexity, maximum weight, and available
adaptation or evidence information. Exact grids report posterior support
concentration without calling it Monte Carlo convergence. ArviZ is installed
with the separate `mc-diagnostics` extra and imported lazily.

Use `save_inference_result(result, path, diagnostics=report)` to bind the
report into the saved result metadata. See
[`docs/user_guide/diagnostics.md`](docs/user_guide/diagnostics.md).

## SBI diagnostics

`inftools.diagnostics` provides estimator-agnostic checks for posterior samples.
It expects samples in object-first order:

```python
samples.shape == (n_objects, n_samples, n_parameters)
```

The diagnostics compute posterior summaries, rank statistics, marginal coverage
curves, and prediction-vs-truth plots.  They can sample any estimator exposing
`sample(x_obs, num_samples=...)`, or they can consume precomputed posterior
samples directly.

```python
from inftools.diagnostics import run_sbi_diagnostics

diagnostics = run_sbi_diagnostics(
    posterior_samples=samples,
    theta_true=theta_true,
    theta_names=["z", "log10_mass"],
    output_dir="runs/sbi_check",
)
```

TARP diagnostics are optional and require the external `tarp` package. Normal
rank and coverage diagnostics do not require torch, nflows, or TARP. See
[`docs/sbi_diagnostics.md`](docs/sbi_diagnostics.md).

## Parallel MixedTAMIS evaluation

MixedTAMIS remains serial by default. Expensive independent target evaluations
within each adaptation round can be distributed explicitly:

```python
result = fit(
    problem,
    MixedTAMIS(
        T_max=100,
        n_per_iter=1000,
        n_workers=8,
        batch_size=32,
        mp_context="spawn",
    ),
    seed=61,
)
```

The process pool persists for the complete run. Each process owns its backend
state and, for CIGALE, lazily creates its own `SedWarehouse`. Proposal draws,
adaptation, and final AMIS recycling remain in the parent process. Keep
`n_workers=1` when using a local function or lambda that cannot be pickled.
When fitting many galaxies with `inftools.fit_many`, parallelize either across
galaxies or within each sampler rather than enabling both at full core count.

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

Cached per-solar-mass grids use the lower-level catalog API:

```python
from composed.catalog import (
    build_photometric_model_grid,
    evaluate_catalog_model_grid_likelihood,
)
```

They can profile or marginalize over a separate
`log10_mass_grid`. For marginalization, that numerical grid is only an
integration grid: pass the same continuous `Prior` declared by the scientific
model. CompoSED multiplies its density by each irregular-grid cell width.
Sampler-specific arrays of mass weights are rejected.

Analytic mass profiling assumes a prior that is flat in `log10_mass` over the
declared bounds. A cached grid inherits those bounds automatically from an
excluded `UniformPrior`. Profiling is an excellent high-S/N approximation to
flat-log-mass marginalization, but not a mathematical identity. Informative or
otherwise non-flat mass priors require an explicit `log10_mass_grid`; CompoSED
uses the prior recorded when the grid was built unless an override is supplied
deliberately.

The separate rest-frame fast-projection API is experimental in 0.1.1. It
accepts only backends that explicitly declare a redshift-independent
rest-spectrum grid. CIGALE configurations with redshift-independent SFHs can
support that contract; current FSPS and redshift-aware CIGALE SFHs are rejected
because their SFH evaluation needs redshift. Every requested filter must be
covered by the rest wavelength grid.

For independent per-object samplers, use `inftools.fit_many` to run a
single-object fitting function across a catalog with serial, thread, or process
execution. For process execution, build fragile backend state such as FSPS or
CIGALE inside the worker function rather than trying to pickle a live backend.

## Runtime expectations

There is no hardware-independent wall time: stellar-population calls, filter
count, parameter dimension, process count, and neural device dominate different
phases. The practical scaling is predictable:

| Phase | Main scaling | What normally dominates |
|---|---|---|
| FSPS/CIGALE simulation | accepted prior draws | backend SED construction |
| MixedTAMIS/PocoMC | likelihood calls for one object | backend SED construction |
| MAF/MDN training | simulations plus epochs | simulations first, then CPU/GPU training |
| Trained catalog inference | objects x posterior draws | batched neural sampling |

The release tutorials print simulation, training, and catalog-inference timing
separately. Benchmark quick mode on the target machine before scheduling a full
300,000-simulation run; do not extrapolate neural inference timing from SPS
simulation throughput.

## One fitting workflow

The public fitting path composes backend, ordered priors, data, likelihood, and
sampler in one explicit `Problem`. The object exposes the three statistical
quantities separately: `log_prior`, `log_likelihood`, and `log_posterior`.

```python
import numpy as np

from composed import (
    Emcee,
    Gaussian,
    ParameterSpace,
    Problem,
    SEDDataset,
    UniformPrior,
    MassNormalization,
    fit,
    save_inference_result,
)
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.filters import FilterSet
from composed.plot import plot_corner_hexbin, plot_posterior_predictive, plot_traces

class LinearBackend(SEDBackend):
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        amplitude = params["amplitude"]
        return ModelPhotometry(filters.names, np.asarray([amplitude, 2.0 * amplitude]))

filters = FilterSet(("g", "r"))
backend = LinearBackend()
parameters = ParameterSpace(
    names=("amplitude",),
    priors={"amplitude": UniformPrior(0.0, 5.0)},
)
data = SEDDataset(
    band_names=filters.names,
    flux=np.asarray([1.0, 2.0]),
    sigma=np.asarray([0.1, 0.2]),
    flux_unit="maggies",
    metadata={"filters": filters},
)
problem = Problem(
    backend=backend,
    parameters=parameters,
    data=data,
    filters=filters,
    likelihood=Gaussian(),
)

result = fit(
    problem,
    sampler=Emcee(nwalkers=32, nsteps=1000, burnin=200),
    x0=np.asarray([1.0]),
    seed=42,
)
save_inference_result(result, "runs/object_001")

plot_corner_hexbin(result)
plot_traces(result)
plot_posterior_predictive(result, problem)
```

`parameter_transform` is the auditable physical bridge from sampled values to
backend inputs, for example from `tau_gyr` and an age fraction to FSPS tabular
SFH arrays. It may be omitted when parameter names already match the backend.

`InferenceResult.weights` are always normalized. MCMC-like outputs use uniform
weights by default, while grid/TAMIS-style outputs use `weights_norm` from the
sampler metadata when available. Results are saved as an `.npz` array file plus
a JSON sidecar containing metadata and posterior summaries.

## Validation provenance

Validation scripts that cache arrays should write provenance sidecars with
`composed.provenance.save_npz_with_provenance`; see
[`docs/validation_provenance.md`](docs/validation_provenance.md).

## Development Transparency

CompoSED has been developed with extensive assistance from AI coding tools,
principally OpenAI Codex and Anthropic Claude. AI assistance has been used for
implementation, testing, documentation, and adversarial code review.
Scientific design decisions, validation, and responsibility for the released
code remain with the maintainer.
