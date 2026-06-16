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

Observed-frame photometry should include CIGALE's `redshifting` module. CIGALE's
current `redshifting` module also applies its built-in IGM attenuation while
redshifting the spectrum.

Two photometry modes are supported:

- `photometry_mode="cigale"`: `filters` should contain native CIGALE filter
  names such as `"sdss.u"`; the backend calls `sed.compute_fnu`.
- `photometry_mode="sedpy"`: `filters` should contain sedpy filter objects;
  the backend converts CIGALE `fnu` to `f_lambda` and integrates with sedpy.
- `photometry_mode="auto"` chooses native CIGALE mode when all filters are
  strings, otherwise sedpy mode.

## Mass Normalization

CIGALE is used in `composed` as a per-solar-mass backend. The backend declares:

```python
MassNormalization.PER_SOLAR_MASS
```

and enforces `normalise=True` for every SFH module whose name starts with
`sfh`. The likelihood is then responsible for multiplying model photometry by
`10**log10_mass`.

This keeps the same convention as the FSPS backend: backends make normalized
photometry; likelihoods apply the explicit mass parameter.

## Parameter Specs

Module parameters are specified as a nested dictionary:

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

Use:

```python
backend, parameter_space = build_cigale_backend_and_parameter_space(
    modules,
    module_parameters,
    additional_priors={"log10_mass": UniformPrior(8.0, 12.0)},
)
```

to build a backend and a matching deterministic `ParameterSpace`.
