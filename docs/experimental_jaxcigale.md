# Experimental JAX-CIGALE

`composed.experimental.jaxcigale` is a JAX-native, CIGALE-inspired prototype.
It is not a wrapper around `pcigale`.  The point is to keep the useful CIGALE
idea of an ordered module chain while making the numerical graph pure,
fixed-shape, and differentiable.

## Architecture

Each module has two phases:

1. **setup**: Python code prepares static choices such as wavelength grids,
   filters, SSP tables, template arrays, and fixed model-family choices;
2. **apply**: a pure JAX function maps continuous parameters and an `SEDState`
   to a new `SEDState`.

Users may swap module families between runs, but the module list is fixed once
the model is built. That is the key compromise that keeps the plug-and-play
feel without asking JAX/NUTS to differentiate over model identity.

## First Supported Chain

The first useful chain is photometry-only:

```python
modules = [
    delayed_sfh_module(age_grid_gyr),
    dsps_stellar_module(ssp_data),      # or analytic_stellar_module for tests
    no_nebular_module(),                # emulator slot
    calzetti_attenuation_module(),
    madau_igm_module(),
    redshift_module(),
]
```

The backend returns maggies through fixed filter curves. The SFH is normalized
to one solar mass formed, and `JaxSedModel.predict_photometry` applies
`10**log10_mass` once, centrally.

## Module Variants

The experimental graph now has a small menu of auditable module families.  The
choice is still static per run: swap modules when building the model, then keep
the compiled graph fixed for JIT/NUTS.

SFH modules:

- `delayed_sfh_module` and `delayed_sfh_cosmic_time_module`
- `exponential_sfh_module` and `exponential_sfh_cosmic_time_module`
- `constant_sfh_module` and `constant_sfh_cosmic_time_module`
- `powerlaw_sfh_module` and `powerlaw_sfh_cosmic_time_module`
- `continuity_sfh_module`
- `continuity_sfh_cosmic_time_module`

The cosmic-time versions output the increasing cosmic-time table expected by
DSPS/FSPS-style CSP calculations.  All of these normalize the SFR integral to
one solar mass formed, leaving `log10_mass` as the only mass scaling.

`continuity_sfh_cosmic_time_module` is the redshift-aware non-parametric
variant.  Users provide lookback-time edges beginning at 0 Gyr, for example
`[0.0, 0.03, 0.1, 0.3, 1.0]`; the module appends `age_universe(z)` at runtime.
That means the oldest bin is always `1 Gyr-age_universe(z)` for the current
redshift proposal, or for a fixed redshift supplied through
`fixed_parameters={"z": ...}`.  The age-of-universe calculation is buffered in
a setup-time interpolation table, so fitting `z` does not call the cosmology
integral at every NUTS step.

Dust attenuation modules:

- `calzetti_attenuation_module`
- `modified_starburst_attenuation_module`
- `gordon16_rvfa_extinction_module`
- `smc_screen_attenuation_module`

Dust emission modules:

- `modified_blackbody_dust_module`
- `two_temperature_dust_module`

Both dust-emission modules normalize the IR shape to the luminosity absorbed by
the preceding attenuation module.  They are simple validation models, not a
replacement for a full CIGALE Dale/Draine-style library.

## Nebular Emission

`no_nebular_module()` keeps the graph slot explicit. A future CLOUDY/Cue-like
emulator should implement:

```python
continuum, lines = emulator_apply(wave_rest_a, logu=..., density=..., f_esc=..., logzsol=...)
```

and then be plugged in through `nebular_emulator_module`.

There is now a more specific experimental Cue-style block:

```python
from composed.experimental.jaxcigale import cue_nebular_module

modules = [
    delayed_sfh_module(age_grid_gyr),
    dsps_stellar_module(ssp_data),
    cue_nebular_module(emulator_apply),
    calzetti_attenuation_module(),
    madau_igm_module(),
    redshift_module(),
]
```

The stellar module remains unchanged. The Cue block reads the existing
``SEDState.wave_rest_a`` and ``SEDState.stellar_lum_lsun_per_a`` and derives
the nebular-emulator inputs internally:

1. convert stellar ``L_lambda`` to ``L_nu``;
2. fit the four Cue ionizing power-law segments split at HeII, OII, HeI, and HI;
3. compute the intrinsic hydrogen-ionizing photon rate ``Q_H``;
4. apply the simple CIGALE-like gas budget
   ``Q_H,gas = max(1 - f_esc - f_dust, floor) * Q_H``;
5. derive or read gas abundance parameters such as ``[O/H]``, ``log(N/O)``,
   and ``log(C/O)``;
6. call a user-supplied JAX-compatible ``emulator_apply`` function that returns
   nebular continuum and line luminosity densities in ``Lsun / Angstrom``.

This means the original table-based nebular module can still be used in a
different graph. Cue-specific calculations do not leak into the stellar
module. The current repository includes ``toy_cue_emulator`` only for tests and
plumbing examples; it is not a scientific CLOUDY emulator.

The public Cue TensorFlow/PCA emulator has also been ported to JAX in
``composed.experimental.jaxcigale.cue_port``. The port reads Cue's public
``speculator_*.pkl`` and ``pca_*.pkl`` files, without importing TensorFlow, and
reproduces the public NumPy/PCA calculation for continuum and lines. It can be
used as a ``cue_nebular_module`` adapter:

```python
from composed.experimental.jaxcigale import CueJaxPort, cue_nebular_module

cue_port = CueJaxPort.from_public_cue_data_dir("/path/to/cue/src/cue/data")
modules = [
    delayed_sfh_module(age_grid_gyr),
    dsps_stellar_module(ssp_data),
    cue_nebular_module(cue_port.make_nebular_apply(line_sigma_a=1.0)),
    calzetti_attenuation_module(),
    madau_igm_module(),
    redshift_module(),
]
```

By default this keeps the public Cue continuum convention: no nebular
continuum is predicted blueward of 912 A.  For diagnostics where we want an
FSPS/CLOUDY-style LyC nebular continuum, keep the Cue emulator pure and add an
explicit table extension:

```python
from composed.experimental.jaxcigale import load_fsps_lyc_continuum_apply

lyc_continuum = load_fsps_lyc_continuum_apply(
    sps_home="/path/to/FSPS",
    isoc_type="mist",
    effective_age_yr=1.0e6,
)
modules = [
    delayed_sfh_module(age_grid_gyr),
    dsps_stellar_module(ssp_data),
    cue_nebular_module(
        cue_port.make_nebular_apply(line_sigma_a=1.0),
        lyc_continuum_apply=lyc_continuum,
    ),
]
```

The FSPS table stores continuum in ``Lsun / Hz / Q``.  The JAX adapter
interpolates that fixed table in gas metallicity, effective nebular age, and
``logU``; scales by the gas-powered ``Q_H`` derived from the stellar spectrum;
and converts to ``Lsun / Angstrom``.  The default Cue path remains unchanged
unless this extension is passed explicitly.

The diagnostic comparison script is:

```bash
PYTHONPATH=/path/to/CompoSED \
CUE_DATA_DIR=/path/to/cue/src/cue/data \
python examples/experimental_cue_jax_port_comparison.py
```

It saves continuum/line spectra comparisons and residual plots under
``outputs/experimental_cue_jax_port_comparison``.

The gas metallicity convention is deliberately flexible. If ``gas_logoh`` is
present, it is used directly. Otherwise the Cue block falls back to
``stellar_logzsol + gas_stellar_logoh_offset``. That is the hook for fitting a
physical relation between stellar and nebular composition instead of forcing
them to be identical.

## NUTS

`run_numpyro_nuts` is the first inference runner. It accepts a vector-valued
`JaxSedModel` posterior, so it does not need a NumPyro model with named sample
sites. This keeps the prototype close to the existing `composed` parameter
vector convention.

Finite prior bounds are sampled in an unconstrained coordinate system and then
mapped back to the physical parameters returned to the user. This avoids making
NUTS bounce directly off hard uniform-prior walls.

The first DSPS recovery script is:

```bash
PYTHONPATH=/path/to/CompoSED \
python examples/experimental_jaxcigale_dsps_nuts_recovery.py
```

Set `DSPS_SSP_FILE` or pass `--ssp-file` to point at
`ssp_data_fsps_v3.2_lgmet_age.h5`. The script generates one deterministic mock
SED with finite error bars, fits `log10_mass`, `z`, `logzsol`, and `dust2`, and
holds the delayed-SFH nuisance parameters fixed. It saves:

- `outputs/experimental_jaxcigale_dsps_nuts_recovery/dsps_nuts_recovery_samples.npz`
- `outputs/experimental_jaxcigale_dsps_nuts_recovery/dsps_nuts_recovery_summary.json`

For GPU checks, set the platform before JAX imports:

```bash
python examples/experimental_jaxcigale_dsps_nuts_recovery.py --jax-platform cuda
python examples/experimental_jaxcigale_dsps_nuts_recovery.py --jax-platform mps --precision float32
```

The MPS path must use float32. The observed-spectrum to photometry calculation
applies `10**log10_mass` before filter integration so per-solar-mass broadband
fluxes do not flush to zero in float32.

The assessment notebook is:

```text
notebooks/experimental_jaxcigale_nuts_assessment.ipynb
```

It is the physicist-facing "does the graph make sense?" document. It contains:

- a synthetic broadband filter set from `u` through `H`;
- the JAX model prediction;
- an independent NumPy + direct DSPS reference calculation for the same
  SFH, attenuation, IGM, redshift, luminosity distance, and filter integral;
- spectrum, filter, photometry, trace, marginal-posterior, residual, and
  timing plots;
- a switch to fit only `log10_mass`, `z`, `logzsol`, and `dust2`, or to also
  fit the delayed-SFH shape parameters `tau_gyr` and `tage_gyr`;
- an optional multi-mock sweep once the single-object recovery is sane.

The notebook defaults are intended for a real CPU/x64 check. For a fast smoke
test without editing the notebook, override the counts from the shell:

```bash
COMPOSED_JAXCIGALE_WARMUP=20 \
COMPOSED_JAXCIGALE_SAMPLES=30 \
COMPOSED_JAXCIGALE_POSTERIOR_PREDICTIVE=10 \
jupyter nbconvert --to notebook --execute notebooks/experimental_jaxcigale_nuts_assessment.ipynb
```

Useful notebook switches are:

- `COMPOSED_JAXCIGALE_FIT_SFH_SHAPE=1`
- `COMPOSED_JAXCIGALE_MULTI_MOCK=1`
- `COMPOSED_JAXCIGALE_ADD_NOISE=1`

The first joint spectrum-plus-photometry validation notebook is:

```text
notebooks/validation/08_jades_like_joint_spectrophotometry.ipynb
```

It simulates a JADES-like NIRSpec PRISM spectrum plus approximate NIRCam-like
broad bands.  Blue bands are encoded as real upper limits using
`GaussianPhotometricData`, and the fit uses `GaussianSpectroPhotometricData` so
the spectrum and photometry contribute to the same log posterior.
- `COMPOSED_JAXCIGALE_OUTPUT_DIR=/path/to/output`

## Current Limits

- Photometry only; spectra need LSF convolution and masking policy.
- DSPS requires user-provided SSP data.
- Dust emission starts with a simple modified blackbody.
- IGM is an approximate Madau-like differentiable prescription.
- The CIGALE bridge is intentionally restricted and does not promise exact
  `pcigale` parameter parity.
