# CompoSED Design Notes

## Package Architecture

`composed` is split around stable interfaces rather than physical model details:

- `data.py` contains observed-data containers.
- `priors.py` and `parameters.py` define scalar priors and the ordered parameter vector contract.
- `filters.py` contains a thin `FilterSet` wrapper for backend filter objects.
- `backends/` contains forward-model implementations behind a common interface.
- `likelihood.py` evaluates backend-agnostic photometric and spectral likelihoods.
- `problem.py` composes backend, parameter transform, data, likelihood, and sampler-facing callbacks.
- `transforms/` contains physical or catalog-specific parameter transforms, such as SFH utilities.
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

Backends must also declare:

```python
mass_normalization: MassNormalization
```

Backends may use any internal physics library, but they should raise ordinary Python numerical exceptions for invalid model states and return finite model fluxes for valid states.

FSPS and CIGALE are both backend implementations of this same contract. FSPS
accepts FSPS-ready parameters such as tabular SFHs and forwards ordinary FSPS
parameters into `python-fsps`. CIGALE accepts an ordered CIGALE module list plus
module-parameter specifications and calls `pcigale.warehouse.SedWarehouse`.
Neither backend owns the likelihood.

The production FSPS backend reuses one mutable `StellarPopulation` for speed,
but resets every parameter overridden by the preceding call before evaluating
the next object. Valid per-call FSPS parameters are discovered from
`sp.params.all_params`; unknown names raise rather than being ignored.

## Mass Normalization

Mass scaling is explicit. A backend must choose one of:

- `MassNormalization.PER_SOLAR_MASS`: model fluxes are per solar mass formed. The likelihood requires a `log10_mass` parameter and multiplies model flux by `10**log10_mass`.
- `MassNormalization.ABSOLUTE`: model fluxes are already absolute. The likelihood never applies `log10_mass`, even if that parameter exists.

The likelihood never infers this behavior from parameter names, backend class names, or flux magnitudes.

The CIGALE backend currently exposes only `PER_SOLAR_MASS`: SFH module
`normalise=True` is enforced and verified through `sfh.integrated` when
available. This mirrors the FSPS convention that the backend returns normalized
photometry and the likelihood applies the explicit `log10_mass`.

## Physical Transforms

Physical transforms live under `composed/transforms/`. For example, Pop-COSMOS continuity-SFH utilities convert catalog theta rows into FSPS-ready tabular SFHs.

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
prior exactly once.

## Public Problem Contract

`Problem` is the user-level composition object. Its optional
`parameter_transform(params)` is the explicit bridge from sampled coordinates
to backend inputs, such as tabular SFH arrays. A `Problem` exposes:

```python
problem.log_prior(theta)
problem.log_likelihood(theta)
problem.log_posterior(theta)
problem.simulate(theta, noise_fn, rng)
```

Photometry and spectroscopy can be combined with
`SpectroPhotometricDataset`; their likelihood terms are summed and the prior is
added once. `fit(problem, method, seed=...)` performs inference-method
capability checks and normalizes outputs to `InferenceResult`.

Problem-driven SBI is explicit:

```python
fit(problem, MAF(...), training=Simulate(n=100_000, noise_fn=noise_fn))
```

Those training pairs are sampled from `problem.parameters` and generated by
`problem.simulate`, so they share the deterministic fit's backend parameter
mapping, units, masks, and mass normalization. Pre-existing paired arrays use
`SBITrainingSet.from_arrays` plus `train_sbi` without declaring an unrelated
Problem. Sample-only methods may return `InferenceResult.logp=None`; no MAP is
invented in that case.

## Fast Catalog Projection

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
