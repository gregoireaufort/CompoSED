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
| `constant` | `SFR(t) = constant` | yes | yes |
| `exponential` | `SFR(t) proportional to exp(-t/tau)` | yes | yes |
| `delayed_tau` | `SFR(t) proportional to t exp(-t/tau)` | yes | yes |
| `continuity` | piecewise-constant adjacent SFR ratios | yes | yes |
| `tabular` | user-supplied time and SFR arrays | yes | yes |

All five named models first produce the same canonical `SFHHistory`. FSPS
receives that table directly. CIGALE receives an in-memory projection onto its
native chronological 1 Myr bins, normalized so
`sum(SFR [Msun/yr]) * 1e6 yr = 1 Msun` formed. The process-local module is
registered at runtime; CompoSED does not patch the CIGALE installation and does
not create temporary files. Users can still use any native CIGALE SFH module
by listing it directly in `CIGALEBackend.modules` and omitting `sfh=`.

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
redshift is available. The default follows each backend's declared
convention: FSPS uses its backend cosmology, while CIGALE named-SFH ages use
`WMAP7` to match CIGALE v2022.0 redshifting. A supplied cosmology changes
named-SFH age conversion but does not replace CIGALE's internal redshifting
cosmology.

### Time-grid resolution

The constant, exponential, and delayed-tau models evaluate the SFH on a
linear time-since-onset grid with `n_time=256` by default. This is adequate only
when the shortest physically important SFH timescale is resolved by several
grid intervals. Very short `tau_gyr` values in an old population can therefore
have a correctly normalized but poorly resolved SFH shape.

For analyses allowing short bursts or `tau_gyr` near or below about 0.2 Gyr,
repeat a representative calculation with a finer grid, for example
`DelayedTauSFH(n_time=2048)`, and verify that the photometry or posterior is
stable. The relevant convergence criterion is the ratio of the shortest SFH
timescale to `tage_gyr / (n_time - 1)`; 0.2 Gyr is a warning scale, not a
universal physical boundary.

Increasing `n_time` costs linearly in SFH array construction and memory. A few
thousand points are normally negligible beside an SPS call. Very large values
can still matter for hundreds of thousands of SBI simulations, especially
with several process workers. FSPS receives the full table; CIGALE first
integrates it into its fixed 1 Myr representation, so increasing `n_time`
beyond convergence cannot increase CIGALE's final time resolution. Benchmark
representative end-to-end simulations rather than only the SFH function.

This parameterization enforces the cosmic upper bound only. In particular,
`ContinuitySFH` also requires the resulting galaxy age to exceed its last
fixed `lookback_edges_gyr` value. A broad redshift/age-fraction prior can still
produce invalid continuity histories at young ages. Choose compatible edges
and priors, or use explicit simulator resampling and inspect the resulting
effective prior. The current fixed-bin construction is otherwise unchanged.

## Normalization

Constant, exponential, delayed-tau, and continuity histories are numerically
normalized to form one solar mass. The named CIGALE bridge independently
checks its projected 1 Myr bins against the same formed-mass convention. Native
CIGALE SFH modules are called with `normalise=True`. The backends then convert
their spectra to luminosity per one solar mass of surviving stars.
Consequently the public `log10_mass` remains present-day surviving stellar
mass; the SFH object never applies mass scaling.

`TabularSFH` preserves its input amplitude. A per-solar-mass backend normalizes
that amplitude before synthesis; FSPS with `MassNormalization.ABSOLUTE`
retains it. CIGALE supports only the per-solar-mass mode, so its tabular bridge
always normalizes to one solar mass formed.

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

CIGALE v2022.0 evaluates stellar populations from a 1 Myr SFH array. CompoSED
rounds the history endpoint to the nearest integer Myr, integrates the
piecewise-linear canonical SFH into those bins, and records the source age,
integer CIGALE age, time-scale adjustment, normalization, bridge schema, and
source hash. The residual FSPS/CIGALE difference then comes from engine and
SSP conventions rather than different named SFH equations.

The backend SFH checks are:

```bash
SPS_HOME=/path/to/fsps python -m pytest -q -m fsps tests/test_fsps_backend.py
python -m pytest -q -m cigale tests/test_cigale_backend.py tests/test_cigale_tabular_sfh.py
```
