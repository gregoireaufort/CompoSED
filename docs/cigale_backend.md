# CIGALE Backend

`CIGALEBackend` wraps CIGALE's low-level `pcigale.warehouse.SedWarehouse`
interface behind the standard `composed` backend contract:

```python
predict_photometry(params, filters) -> ModelPhotometry
```

The backend is optional. Importing `composed` does not require CIGALE, but
constructing `CIGALEBackend` requires the `pcigale` package and its database.

## Units And Filters

CIGALE `SED` objects use:

- wavelength grid in nm,
- luminosity density in W / nm,
- `SED.fnu` and `SED.compute_fnu(filter_name)` in mJy.

`CIGALEBackend` returns maggies, matching the rest of `composed`.
The conversion uses the AB definition `1 maggie = 3631 Jy = 3.631e6 mJy`.
The spectral mJy-to-`f_lambda` conversion uses
`c = 2.99792458e18 Angstrom/s`. These exact values are recorded in backend
metadata and regression-tested. No solar-luminosity constant is introduced:
native CIGALE W/nm and mJy are preserved through these conversions.

Observed-frame photometry should include CIGALE's `redshifting` module. CIGALE's
current `redshifting` module also applies its built-in IGM attenuation while
redshifting the spectrum. In the pinned CIGALE v2022.0 engine this module uses
the WMAP7 cosmology. CompoSED therefore defaults its named-SFH age conversion
to Astropy `WMAP7`, and the native WMAP7 convention is written into Problem
and model provenance. Passing a custom `cosmology=` can alter CompoSED's
named-SFH age conversion, but it cannot alter the upstream CIGALE
`redshifting` cosmology; that restricted scope is also recorded.

Two photometry modes are supported:

- `photometry_mode="cigale"`: `filters` should contain native CIGALE filter
  names such as `"sdss.up"`; the backend calls `sed.compute_fnu`. CIGALE
  v2022.0 uses the primed SDSS names `sdss.up/gp/rp/ip/zp`.
- `photometry_mode="sedpy"`: `filters` should contain sedpy filter objects;
  the backend converts CIGALE `fnu` to `f_lambda` and integrates with sedpy.
- `photometry_mode="auto"` chooses native CIGALE mode when all filters are
  strings, otherwise sedpy mode.

## Mass Normalization

CIGALE is used in `composed` as a per-surviving-stellar-mass backend. The
backend declares:

```python
MassNormalization.PER_SOLAR_MASS
```

and enforces `normalise=True` for every SFH module whose name starts with
`sfh`. This first produces an SED for one solar mass formed. The backend
requires `SED.info['stellar.m_star']` and divides photometry and spectra by that
value. The likelihood then multiplies the per-surviving-mass output by
`10**log10_mass`.

Thus `log10_mass` has the same meaning for CIGALE and FSPS: present-day
surviving stellar mass. The original `sfh.integrated`, `stellar.m_star`, and
their ratio remain available in `ModelPhotometry.metadata` or
`ModelSpectrum.metadata` for inspection.

The standalone numerical check can be run in the CIGALE environment with:

```bash
python examples/validate_cigale_mass_normalization.py
```

It compares the backend against an independent direct `SedWarehouse` call and
explicit division by `stellar.m_star`.

## Parameter Specs

For the stable cross-backend SFH subset, prefer a named SFH and omit the SFH
module from `modules`:

```python
from composed import DelayedTauSFH, UniformPrior

backend, parameter_space = build_cigale_backend_and_parameter_space(
    modules=["bc03", "redshifting"],
    module_parameters={
        "bc03": {"imf": 1, "metallicity": [0.008, 0.02]},
        "redshifting": {"redshift": {"name": "z", "range": [0.05, 2.0]}},
    },
    additional_priors={
        "log10_mass": UniformPrior(8.0, 12.0),
        "tage_gyr": UniformPrior(0.1, 5.0),
        "tau_gyr": UniformPrior(0.1, 5.0),
    },
    sfh=DelayedTauSFH(),
)
```

CompoSED maps constant, exponential, and delayed-tau histories to native
CIGALE v2022.0 modules with `normalise=True`. See `docs/sfh_models.md` for the
equations and unit conversion. Continuity and arbitrary tabular histories are
not silently approximated in this backend.

The original native-module API remains available for every CIGALE SFH. Module
parameters are specified as a nested dictionary:

```python
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
```

Supported variable specs:

- `{"range": [low, high], "scale": "linear"}` -> uniform prior
- `{"range": [low, high], "scale": "log"}` -> log-uniform prior
- `{"values": [...]}` or a direct list -> discrete choice prior
- `{"dtype": "int"}` with a linear range -> discrete integer prior

Fixed scalars are passed directly to CIGALE and are not part of the theta
vector.

Unsupported module parameters, missing required values, and upstream CIGALE
configuration errors are intentionally not translated into generic numerical
failures. They retain the original actionable exception and stop simulation.
Likewise, `Simulate.failure_policy="raise"` is the default: CompoSED does not
silently resample an invalid CIGALE prior draw. A user may explicitly request
`failure_policy="resample"`, in which case the simulator-success conditioning
and rejected draws are recorded in training metadata.

Use:

```python
backend, parameter_space = build_cigale_backend_and_parameter_space(
    modules,
    module_parameters,
    additional_priors={"log10_mass": UniformPrior(8.0, 12.0)},
)
```

to build a backend and a matching deterministic `ParameterSpace`.
