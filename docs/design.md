# CompoSED Design Notes

## Package Architecture

`composed` is split around stable interfaces rather than physical model details:

- `data.py` contains observed-data containers.
- `priors.py` and `parameters.py` define scalar priors and the ordered parameter vector contract.
- `filters.py` contains a thin `FilterSet` wrapper for backend filter objects.
- `backends/` contains forward-model implementations behind a common interface.
- `likelihood.py` evaluates backend-agnostic photometric and spectral likelihoods.
- `problem.py` composes backend, parameter transform, data, likelihood, and sampler-facing callbacks.
- `sfh.py` contains named, backend-independent SFH equations and their explicit backend adapters.
- `transforms/` contains lower-level physical or catalog-specific transforms, such as Pop-COSMOS utilities.
- `runners.py` provides light glue to inference tools.

The likelihood and parameter-space layers should stay small and deterministic. Backend-specific physics belongs in backends or transforms.

## Backend Contract

Every backend exposes:

```python
predict_photometry(params, filters) -> ModelPhotometry
```

`ModelPhotometry.flux` is a one-dimensional linear flux vector with an explicit
`flux_unit` (maggies by default). `ModelPhotometry.band_names` names every
element so the likelihood can align model and observed bands without assuming
positional order. Duplicate names are invalid.

Backends may also expose spectra:

```python
predict_spectrum(params, wavelengths=None, wavelength_range=None, resolution=None) -> ModelSpectrum
```

The first-pass spectral contract is intentionally simple and auditable:

- `wavelengths` and `wavelength_range` are observed-frame Angstrom.
- `ModelSpectrum.flux` is observed `f_lambda` in `erg s^-1 cm^-2 Angstrom^-1`.
- If `wavelengths` is supplied, the backend should return the model sampled on
  exactly that grid.
- Instrumental resolution convolution is not implemented yet. Passing
  `resolution` raises `NotImplementedError` in the current FSPS and CIGALE
  backends.

Backends must declare `mass_normalization`. A per-mass backend must additionally
declare `mass_reference` explicitly; the base class does not supply a default:

```python
mass_normalization: MassNormalization
mass_reference: MassReference
```

Backends may use any internal physics library, but they should raise ordinary Python numerical exceptions for invalid model states and return finite model fluxes for valid states.

FSPS and CIGALE are both backend implementations of this same contract. FSPS
accepts either a named CompoSED SFH or explicit tabular arrays and forwards
ordinary FSPS parameters into `python-fsps`. CIGALE accepts either a supported
named SFH plus an ordered list of non-SFH modules, or the original complete
native module list. It calls `pcigale.warehouse.SedWarehouse`. Neither backend
owns the likelihood or the priors.

The production FSPS backend reuses one mutable `StellarPopulation` for speed,
but resets every parameter overridden by the preceding call before evaluating
the next object. Valid per-call FSPS parameters are discovered from
`sp.params.all_params`; unknown names raise rather than being ignored.

## Mass Normalization

Mass scaling is explicit. In the stable public API, `log10_mass` always means
the present-day surviving stellar mass. A backend must choose one of:

- `MassNormalization.PER_SOLAR_MASS`: model fluxes are per one solar mass of surviving stars. The backend declares `MassReference.SURVIVING_STELLAR_MASS`; the likelihood requires `log10_mass` and multiplies model flux by `10**log10_mass`.
- `MassNormalization.ABSOLUTE`: model fluxes are already absolute. The likelihood never applies `log10_mass`, even if that parameter exists.

The likelihood never infers this behavior from parameter names, backend class
names, or flux magnitudes. A backend declaring a formed-mass reference is
rejected rather than allowing `log10_mass` to change scientific meaning.

FSPS and CIGALE still use formed mass internally. FSPS normalizes its tabular
SFH to one solar mass formed, evaluates the population, and divides the
spectrum by `sp.stellar_mass`. CIGALE enforces `normalise=True`, verifies
`sfh.integrated == 1`, and divides by `stellar.m_star`. Both backends record the
formed mass, surviving mass, surviving fraction, and public mass reference in
model metadata before the likelihood applies the fitted amplitude.

Cached model grids serialize `mass_reference` and the mass-convention schema.
Grids written before the surviving-mass convention are rejected and must be
rebuilt; they cannot be safely relabeled because their flux amplitudes differ.

## Physical Transforms

Stable named SFHs live in `composed/sfh.py`. They consume scalar named values,
produce a canonical increasing time-since-onset grid in Gyr and SFR in solar
masses per year, and record their conventions in metadata. Priors remain in
`ParameterSpace`. Backend-specific translations are explicit: FSPS receives a
tabular history, while the supported CIGALE subset maps to native v2022.0 SFH
modules.

Lower-level physical transforms live under `composed/transforms/`. For
example, Pop-COSMOS continuity-SFH utilities convert catalog theta rows into
FSPS-ready tabular SFHs.

Transforms should be pure functions where possible. They should not own backend instances, caches, multiprocessing pools, or global state.

## Backend-Agnostic Likelihood

`likelihood.py` must remain backend-agnostic because it defines the statistical contract:

- expose `log_prior`, `log_likelihood`, and `log_posterior` separately,
- apply masks consistently to fluxes and uncertainties,
- align photometry by band name,
- sample spectra on the active data wavelength grid,
- apply only declared mass normalization,
- return `-inf` for controlled backend numerical failures,
- raise clear errors for API/configuration mismatches.

Backend-specific parameter aliases such as `z`, `zred`, or `redshift` must be handled by transforms or backends, not by the likelihood.

The compatibility method `log_prob` means `log_posterior`. Sampler adapters
that own a prior, such as PocoMC, must consume `log_likelihood` and apply their
prior exactly once. In `fit(problem, PocoMC(...))`, that sampler prior is
always translated from `Problem.parameters`. A sampler-local replacement
`prior` or `bounds` is rejected because it would disagree with saved Problem
provenance.

## Public Problem Contract

`Problem` is the user-level composition object. Its optional
`parameter_transform(params)` is the explicit bridge from sampled coordinates
to backend inputs, such as tabular SFH arrays. A `Problem` exposes:

```python
problem.log_prior(theta)
problem.log_likelihood(theta)
problem.log_posterior(theta)
problem.simulate(theta, noise_model, rng)
problem.simulate_with_uncertainty(theta, noise_model, rng)
```

Photometry and spectroscopy can be combined with
`SpectroPhotometricDataset`; their likelihood terms are summed and the prior is
added once. `fit(problem, method, seed=...)` performs inference-method
capability checks and normalizes outputs to `InferenceResult`.

For photometric SBI, ``simulate_with_uncertainty`` returns the same active-band
flux vector as the likelihood plus the raw ``sigma_catalog`` sampled by the
survey noise model. If the Problem declares an absolute floor or fractional
model discrepancy, the flux draw uses
``sigma_draw^2 = sigma_catalog^2 + sigma_floor^2 + (eta f_model)^2``.
Those extra likelihood terms are not added to the neural uncertainty context.
Observed inference likewise consumes the raw sigma stored in the dataset.

Problem-driven MAF SBI is explicit:

```python
fit(problem, MAF(...), training=Simulate(n=100_000, noise_model=survey_noise))
```

Those training pairs are sampled from `problem.parameters` and generated by
the Problem simulator, so they share the deterministic fit's backend parameter
mapping, units, masks, and mass normalization. The default
``PhotometricContext("snr_logsigma")`` encodes, in active-band order,
``[flux / sigma_catalog, log10(sigma_catalog / reference_flux)]``. This is
invertible given the recorded reference flux, accepts negative noisy flux, and
makes heterogeneous catalog depths explicit. The recommended
``ConditionalCatalogNoise`` learns the complete multiband conditional
``q(log10 sigma_catalog | noiseless AB magnitudes)``. Its fixed band order,
units, support, input hash, row filtering, standardization, architecture,
package versions, and random seed are serialized with the learned flow.

Training simulation does not silently repair an invalid prior. The default
`Simulate.failure_policy` is `"raise"`: one failed forward model stops the run
and reports its parameter vector. A scientist may explicitly select
`failure_policy="resample"`, but the resulting theta table is then sampled from
the declared prior conditional on simulator success. That fact, the acceptance
fraction, and failure examples are stored in the training metadata. For coupled
physical support such as age versus redshift, prefer a valid
parameterization such as `age_kind="fraction_of_universe"` rather than
success-conditioned rejection.

Uniform and log-uniform inferred parameters are mapped to an unconstrained
neural target space before MAF training and mapped back after sampling. This
prevents out-of-prior samples without rejection. Normal-prior parameters use an
identity target transform. Other prior classes are rejected by the stable MAF
path rather than assigned an implicit transform.

Pre-existing photometric pairs use ``SBITrainingSet.from_photometry`` and the
same context encoder. Already encoded generic arrays use
``SBITrainingSet.from_arrays`` plus ``train_sbi`` without declaring an unrelated
Problem. Sample-only methods may return ``InferenceResult.logp=None``; no MAP is
invented in that case.

Values supplied through ``fit(..., conditions={...})`` are restored as
deterministic columns in a Problem-driven SBI ``InferenceResult``. Parameters
deliberately omitted from ``Simulate.infer`` remain marginalized and are named
in result metadata; CompoSED does not fabricate values for them. A
posterior-predictive SED therefore requires posterior columns for every
Problem parameter.

Trained MAF checkpoints contain tensor weights, standardization arrays, ordered
parameter/band/context schema, prior transforms, package versions, and training
history. They deliberately do not duplicate the simulation table. Loading a
checkpoint reconstructs the same physical parameter bounds and native
photometry encoding. Catalog sampling batches objects inside nflows.

`Problem.specification()` is the immutable scientific identity used to compare
results. It includes observed-array hashes, masks, units, structured priors,
backend configuration and engine versions, filter-curve hashes, and the code
plus referenced closure/global values of `parameter_transform`. Saved model
grids additionally record this model-building specification. Numerical result
archives and grids are content-hashed and verified on load by default. Engine
source files are identified by content hash, and an FSPS tree by its git
revision and tracked dirty state, so identical installations at different
absolute paths have the same scientific identity. Machine-local paths remain
available in the separate run-provenance sidecar.
The high-level fit seed controls prior/noise simulation, MAF weight
initialization, minibatch order, and the initial posterior draw. Reused catalog
sampling accepts its own explicit seed.

Censored upper limits do not yet have a stable SBI context in version 0.1 and
are rejected explicitly. Experimental conditional diffusion remains under
``inftools.experimental`` and is not exported from the stable ``composed``
namespace.

## Fast Catalog Projection

Fast rest-frame catalog projection is experimental in CompoSED 0.1.1. It is
not a generic replacement for normal backend evaluation. A backend must
explicitly declare that its rest spectrum can be tabulated independently of
redshift. CIGALE configurations with redshift-independent SFHs can do so;
current FSPS and redshift-aware CIGALE SFHs cannot because their SFH evaluation
requires the requested redshift. The projector also requires full rest-grid
coverage for every requested filter and raises when a band is unavailable.

When a cached per-mass model grid is marginalized over ``log10_mass``, the
mass grid is a quadrature grid rather than a prior. A continuous ``Prior`` is
required, prior density is multiplied by midpoint-cell width, and unbounded
priors require an explicit finite integration domain. Irregular numerical
spacing therefore cannot silently change the declared mass prior.

Raw dimensionless filter curves use the photon-counting AB definition. CIGALE
database filters are a separate input convention because their stored `tr`
arrays are already normalized mJy kernels. Generic wavelength arrays must
declare their units; sedpy filters are recognized as Angstrom-valued curves.

Fast projection uses Astropy Planck18 by default. A different cosmology, for
example WMAP7 when reproducing CIGALE v2022.0, must be passed explicitly.
Catalog redshifts are not rounded by default. When grouping is requested, the
input and evaluated redshifts are both retained and positive redshifts may not
round to the special 10 pc `z=0` convention.

`GaussianSpectralLikelihood` uses `SpectrumDataset.active_arrays()`, requests a
model spectrum at the active observed wavelengths, checks that the returned
wavelength grid matches the data grid, and then evaluates a diagonal Gaussian
likelihood in the dataset flux units. Calibration polynomials, covariance
matrices, line masks, and instrumental line-spread functions are intentionally
left out of the first pass.
