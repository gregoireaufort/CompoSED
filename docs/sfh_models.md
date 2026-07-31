# Named Star-Formation Histories

CompoSED separates the SFH equation from both the prior and the stellar
population engine. A named SFH reads scalar values from the backend parameter
dictionary and produces a validated history with:

- time since star-formation onset in Gyr, strictly increasing;
- SFR in solar masses per year, finite and non-negative;
- explicit metadata recording the model and parameter convention.

The stable models are:

| Name | Equation or input | FSPS | CIGALE v2022.0 |
|---|---|---:|---:|
| `constant` | `SFR(t) = constant` | yes | `sfhperiodic`, one rectangular episode |
| `exponential` | `SFR(t) proportional to exp(-t/tau)` | yes | `sfh2exp`, no burst |
| `delayed_tau` | `SFR(t) proportional to t exp(-t/tau)` | yes | `sfhdelayed`, no burst |
| `continuity` | piecewise-constant adjacent SFR ratios | yes | no |
| `tabular` | user-supplied time and SFR arrays | yes | no |

The exact CIGALE constant adapter uses upstream `sfhperiodic`. In CIGALE
v2022.0 that module still references `np.float`, so the dedicated
`envs/composed-cigale.yml` recipe pins NumPy 1.23.5. On newer NumPy CompoSED
raises a targeted error instead of changing the SFH equation.

The CIGALE adapters call native upstream modules. Continuity and arbitrary
tabular histories are deliberately not emulated through temporary files in the
production CIGALE backend. Users can still use any native CIGALE SFH module by
listing it directly in `CIGALEBackend.modules` and omitting `sfh=`.

## Scalar Parameters And Priors

The SFH object names the scalar parameters it consumes; `ParameterSpace` owns
their priors. The default delayed-tau parameters are `tage_gyr` and `tau_gyr`:

```python
from composed import DelayedTauSFH, ParameterSpace, UniformPrior
from composed.backends.fsps import FSPSBackend

backend = FSPSBackend(sfh=DelayedTauSFH())
parameters = ParameterSpace(
    names=["zred", "log10_mass", "tage_gyr", "tau_gyr"],
    priors={
        "zred": UniformPrior(0.05, 2.0),
        "log10_mass": UniformPrior(8.0, 12.0),
        "tage_gyr": UniformPrior(0.1, 10.0),
        "tau_gyr": UniformPrior(0.1, 5.0),
    },
)
```

For a redshift-aware age, sample a fraction of the Universe age:

```python
sfh = DelayedTauSFH(
    age="age_fraction",
    age_kind="fraction_of_universe",
    tau="tau_gyr",
)
```

At every evaluation CompoSED computes
`tage_gyr = age_fraction * age_universe(z)`. Fractions must lie in `(0, 1]`.
An absolute `tage_gyr` is also checked against the Universe age whenever a
redshift is available. The default cosmology is Astropy `Planck18` for both
named backends unless another cosmology is supplied.

This parameterization enforces the cosmic upper bound only. In particular,
`ContinuitySFH` also requires the resulting galaxy age to exceed its last
fixed `lookback_edges_gyr` value. A broad redshift/age-fraction prior can still
produce invalid continuity histories at young ages. Choose compatible edges
and priors, or use explicit simulator resampling and inspect the resulting
effective prior. The current fixed-bin construction is otherwise unchanged.

## Normalization

Constant, exponential, delayed-tau, and continuity histories are numerically
normalized to form one solar mass before reaching FSPS. Native CIGALE SFH
modules are called with `normalise=True`, which has the same internal formed
mass convention. The backends then convert their spectra to luminosity per one
solar mass of surviving stars. Consequently the public `log10_mass` remains
present-day surviving stellar mass; the SFH object never applies mass scaling.

`TabularSFH` preserves its input amplitude. `FSPSBackend` normalizes it only
when its declared `mass_normalization` is `PER_SOLAR_MASS`; with `ABSOLUTE`, the
supplied SFR amplitude is retained.

## Continuity Convention

`ContinuitySFH` defines bins in lookback time from the fitted galaxy age. Its
parameters are ordered recent-to-old:

```text
logsfr_ratio_i = log10(SFR_recent_bin / SFR_next_older_bin)
```

A positive value therefore means that the more recent bin has the higher SFR.
The fixed `lookback_edges_gyr` begin at zero, and the fitted galaxy age is
appended as the oldest edge. The age must exceed the last fixed edge. At bin
boundaries the generated FSPS grid uses a short explicit transition width
(`boundary_epsilon_gyr=1e-5`, or 10 kyr, by default). This remains negligible
relative to the default 10 Myr narrowest bin, but survives python-fsps's
lower-precision work arrays and prevents distinct boundaries from collapsing
to duplicate times.

## CIGALE Example

With a named SFH, omit the native SFH module from the module list:

```python
from composed import DelayedTauSFH, UniformPrior
from composed.backends.cigale import build_cigale_backend_and_parameter_space

backend, parameters = build_cigale_backend_and_parameter_space(
    modules=["bc03", "redshifting"],
    module_parameters={
        "bc03": {"imf": 1, "metallicity": [0.008, 0.02]},
        "redshifting": {"redshift": {"name": "z", "range": [0.05, 2.0]}},
    },
    additional_priors={
        "log10_mass": UniformPrior(8.0, 12.0),
        "tage_gyr": UniformPrior(0.1, 10.0),
        "tau_gyr": UniformPrior(0.1, 5.0),
    },
    sfh=DelayedTauSFH(),
)
```

CIGALE v2022.0 evaluates these native histories on a 1 Myr grid. CompoSED
converts Gyr parameters to the nearest integer Myr and records no claim that a
finite FSPS grid and CIGALE's native discrete history are pointwise identical.

The real-engine adapter check is:

```bash
SPS_HOME=/path/to/fsps python examples/validate_named_sfh_backends.py fsps
python examples/validate_named_sfh_backends.py cigale
```
