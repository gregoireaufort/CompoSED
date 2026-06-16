# Experimental FSPS Stellar Module For CIGALE

This is a research prototype, intentionally separate from the stable
`composed` backends.

The goal is to replace CIGALE's built-in stellar SSP module, such as `bc03`,
with a module that calls `python-fsps`, while leaving the rest of CIGALE's chain
unchanged:

```python
["sfhdelayed", "fsps_stellar", "nebular", "dustatt_modified_starburst", "redshifting"]
```

The module lives under:

```text
composed.experimental.cigale_modules.fsps_stellar
```

and is registered at runtime:

```python
from composed.experimental.cigale_fsps_stellar import register_cigale_fsps_stellar_module

register_cigale_fsps_stellar_module()
```

This appends the experimental module directory to `pcigale.sed_modules.__path__`.
It does not copy files into a CIGALE checkout.

## Current Behavior

`fsps_stellar`:

- consumes the SFH placed on the CIGALE `SED` object by an upstream SFH module;
- evaluates stellar-only FSPS spectra;
- splits the SFH into young and old components using `separation_age`;
- adds CIGALE-compatible `stellar.young` and `stellar.old` contributions;
- fills the stellar metadata used by downstream CIGALE modules such as
  `nebular` and dust attenuation modules.

The module is still experimental, but its conventions are now explicit:

- CIGALE BC03 metallicities are absolute `Z` values.
- FSPS metallicities are passed as `logzsol`.
- `composed.experimental.cigale_fsps_stellar_conventions` defines the mapping
  between the two. The default matched-BC03 convention is `Z_sun = 0.02`, so
  `Z=0.02` maps to `logzsol=0`.
- CIGALE BC03 IMF `0` maps to FSPS `imf_type=0` (Salpeter).
- CIGALE BC03 IMF `1` maps to FSPS `imf_type=1` (Chabrier).
- Other FSPS IMF types can be used, but they are not treated as direct BC03
  matches.

When the module runs, it records both CIGALE-style and FSPS-style metadata:

```text
stellar.metallicity
stellar.logzsol
stellar.fsps.imf_type
stellar.fsps.imf_label
stellar.fsps.logzsol
stellar.fsps.z_sun
stellar.fsps.equivalent_metallicity
stellar.fsps.zcontinuous
```

The prototype approximates some stellar bookkeeping quantities from FSPS
attributes and from the generated spectrum. It should not yet be treated as
production-equivalent to `bc03`, but it is intended to be robust enough for
broadband CIGALE experiments.

## Convention Helpers

Use these helpers when constructing BC03-vs-FSPS comparisons:

```python
from composed.experimental.cigale_fsps_stellar_conventions import (
    fsps_parameters_from_cigale_bc03,
)

mapping = fsps_parameters_from_cigale_bc03(imf=1, metallicity=0.008)

bc03_params = {
    "imf": 1,
    "metallicity": mapping.metallicity,
    "separation_age": 10,
}
fsps_params = {
    "imf_type": mapping.imf_type,
    "logzsol": mapping.logzsol,
    "z_sun": mapping.z_sun,
    "zcontinuous": 1,
    "separation_age": 10,
}
nebular_params = {
    "logU": -2.0,
    "zgas": mapping.zgas,
    "f_esc": 0.0,
    "f_dust": 0.0,
    "emission": True,
}
```

## CIGALE Cache Gotcha

Some CIGALE modules cache wavelength-grid-dependent arrays. That is dangerous
when one script alternates between BC03 and FSPS grids. Use the helper below for
mixed-grid diagnostics:

```python
from composed.experimental.cigale_fsps_stellar import make_mixed_grid_sed_warehouse

warehouse = make_mixed_grid_sed_warehouse()
```

This configures `nocache` for `fsps_stellar`, `nebular`,
`dustatt_modified_starburst`, and `redshifting`.

## Smoke Test

Use an environment with `pcigale`, `python-fsps`, and `SPS_HOME` configured:

```bash
SPS_HOME=/path/to/FSPS \
PYTHONPATH=/path/to/composed \
python - <<'PY'
from composed.experimental.cigale_fsps_stellar import register_cigale_fsps_stellar_module
from pcigale.warehouse import SedWarehouse

register_cigale_fsps_stellar_module()

sed = SedWarehouse(nocache=["fsps_stellar"]).get_sed(
    ["sfhdelayed", "fsps_stellar", "nebular", "redshifting"],
    [
        {"age_main": 50, "tau_main": 30.0, "normalise": True},
        {"logzsol": -0.3, "separation_age": 10},
        {"logU": -2.0, "zgas": 0.014, "emission": True},
        {"redshift": 0.1},
    ],
)
print(sed.info["stellar.m_star"], sed.info["stellar.n_ly"])
PY
```

The optional pytest marker is:

```bash
pytest -q -m cigale_fsps
```

The lightweight test suite also includes convention tests that do not require
CIGALE or FSPS.
