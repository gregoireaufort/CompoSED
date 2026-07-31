# Forward-Model Backends

Every backend implements the stable photometric contract:

```python
predict_photometry(params, filters) -> ModelPhotometry
```

FSPS and CIGALE currently also implement
`predict_spectrum(params, wavelengths=...) -> ModelSpectrum`, but this is an
experimental interface and does not make spectral fitting production-ready.

The likelihood never imports FSPS or CIGALE. It consumes only these model
containers and the backend's explicit mass-normalization declaration.

## FSPS

```python
from composed import DelayedTauSFH
from composed.backends.fsps import FSPSBackend

backend = FSPSBackend(
    sfh=DelayedTauSFH(age_kind="fraction_of_universe"),
    sp_kwargs={
        "add_igm_absorption": True,
        "add_neb_emission": True,
    },
    default_z_key="zred",
)
```

FSPS dependencies are imported lazily, but construction requires a usable
`fsps` installation and `SPS_HOME`. The backend:

1. constructs or validates the SFH;
2. obtains rest-frame $L_\lambda$ from FSPS;
3. applies the declared redshift and luminosity distance;
4. integrates observed $f_\lambda$ through sedpy filters;
5. returns maggies.

See {doc}`../install` and the backend API reference.

## CIGALE

```python
from composed import DelayedTauSFH
from composed.backends.cigale import build_cigale_backend_and_parameter_space

backend, parameters = build_cigale_backend_and_parameter_space(
    modules=["bc03", "dustatt_modified_starburst", "redshifting"],
    module_parameters=module_parameters,
    additional_priors=additional_priors,
    sfh=DelayedTauSFH(),
)
```

The CIGALE backend calls `pcigale.warehouse.SedWarehouse` and targets upstream
CIGALE v2022.0. SFH modules are normalized, and native CIGALE mJy photometry is
converted to maggies. Module order remains the caller's scientific choice.

See {doc}`../cigale_backend` for parameter specifications and native module
details.

## Redshift names

Backends accept `z`, `zred`, or `redshift` and expose `default_z_key` for
ambiguous configurations. Pick one name in the `ParameterSpace` and use it
consistently. The likelihood does not search for redshift aliases.

## Domain failures

Invalid physical states should raise `ModelDomainError`. Public posterior
evaluation maps controlled domain failures to `-inf`; API mismatches such as
wrong shapes or missing required parameters remain clear exceptions.

## Backend audit checklist

- Which engine/version and data grids are used?
- Is IGM or nebular emission enabled?
- Which cosmology computes age and luminosity distance?
- What rest-frame luminosity units enter the redshift calculation?
- What observed flux unit leaves the backend?
- Is output `PER_SOLAR_MASS` or `ABSOLUTE`?
- If per mass, is the reference surviving stellar mass?
- Does each parameter affect the intended engine option?
