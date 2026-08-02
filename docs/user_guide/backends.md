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

IGM absorption is not enabled unless `add_igm_absorption=True` is passed to
`sp_kwargs`, following the python-FSPS default. Photo-z analyses spanning the
Lyman break should normally make this choice explicit rather than relying on a
default. Nebular and dust emission are enabled by the current CompoSED FSPS
configuration unless explicitly disabled.

### FSPS defaults changed by CompoSED

CompoSED uses python-fsps defaults except where the universal tabular-SFH
contract or an explicit galaxy-SED convention requires otherwise:

| Option | python-fsps 0.4.7 | CompoSED default | Reason |
|---|---:|---:|---|
| `sfh` | `0` | `3` | Required to supply the canonical tabular SFH |
| `zcontinuous` | `0` | `1` | Interpolate continuously in stellar metallicity |
| `add_neb_emission` | `False` | `True` | CompoSED galaxy-SED convention; may be disabled explicitly |
| `add_dust_emission` | `True` | `True` | Unchanged |
| `add_igm_absorption` | `False` | `False` | Unchanged; enable explicitly for Lyman-break work |
| `compute_vega_mags` | `False` | `False` | Unchanged; CompoSED returns AB-based maggies |

Every value may be overridden through `sp_kwargs`. The effective constructor
arguments are included in Problem provenance.

FSPS treats stellar metallicity (`logzsol`) and gas metallicity (`gas_logz`) as
separate parameters. If `gas_logz` is omitted, python-FSPS uses its own default;
CompoSED does not silently copy `logzsol` into it. A user may fit both values
independently or tie them explicitly with `Problem(parameter_transform=...)`.
That choice should be recorded because it changes the physical relation between
the stellar continuum and nebular emission.

For an explicit one-to-one tie:

```python
from composed import Gaussian, Problem

def tie_gas_to_stellar_metallicity(params):
    backend_params = dict(params)
    backend_params["gas_logz"] = backend_params["logzsol"]
    return backend_params

problem = Problem(
    backend=backend,
    parameters=parameters,  # contains logzsol, but not gas_logz
    data=data,
    filters=filters,
    likelihood=Gaussian(),
    parameter_transform=tie_gas_to_stellar_metallicity,
)
```

More general deterministic relations can be written in the same transform.
The transform is applied before every backend evaluation and recorded in the
Problem fingerprint. When nebular emission is enabled and `logzsol` is explicit
but `gas_logz` is absent, the backend emits a one-time warning rather than
silently implying that the two metallicities are tied.

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
converted to maggies. Named CompoSED SFHs use the same canonical history as
FSPS and are projected in memory onto CIGALE's 1 Myr grid. Module order remains
the caller's scientific choice. Listing a native CIGALE SFH module with
`sfh=None` retains the original CIGALE behavior.

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
