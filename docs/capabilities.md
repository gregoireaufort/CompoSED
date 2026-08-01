# Capability Matrix

This page describes the supported combinations in CompoSED 0.1. It is a
selection guide, not a claim that every optional scientific engine is installed
in the same environment.

```{admonition} Release scope
:class: warning
Only the pure photometric pipeline is release-ready. Spectrum generation,
spectral likelihoods, and joint spectrophotometric fits are experimental.
```

## Forward models

| Backend | Photometry | Spectrum generation | Named SFHs | Status |
|---|---:|---:|---:|---|
| FSPS | stable | experimental | yes | photometry stable |
| CIGALE v2022.0 | stable | experimental | all stable named SFHs plus native modules | photometry stable |
| Mock | testing | experimental testing | not applicable | testing |
| Fast rest-frame catalog projection | experimental | no | restricted | experimental |

FSPS requires `python-fsps`, sedpy, and `SPS_HOME`. CIGALE is installed
separately from its upstream v2022.0 release. See {doc}`install`.

## Observations

| Data | Gaussian detections | Masks | Upper limits | Status |
|---|---:|---:|---:|---|
| Photometry | yes | yes | one-sided Gaussian CDF | stable |
| Spectrum | first pass | first pass | no | experimental |
| Joint spectrophotometry | first pass | first pass | photometry only | experimental |

Photometric upper limits are represented in linear flux units. The model is
compared to a flux ceiling; an AB limiting magnitude must therefore be
converted explicitly before constructing the dataset.

## Traditional inference

| Method | Continuous | Finite discrete | Fixed by `conditions` | Status |
|---|---:|---:|---:|---|
| `RandomWalk` | yes | no | yes | stable |
| `Emcee` | yes | no | yes | stable |
| `Grid` | no | yes | yes | stable |
| `MixedGibbs` | yes | yes | yes | stable |
| `MixedTAMIS` | yes | yes | yes | stable |
| `PocoMC` | yes | no | yes | stable |
| `Laplace` | yes | no | limited | experimental |
| external `TAMIS` adapter | yes | no | limited | experimental |

`conditions={"zred": value}` removes a fixed parameter from the sampled space
while preserving it in backend calls and returned results. This is preferable
to asking a continuous-only sampler to traverse a `DeltaPrior`.

The methods share `diagnose(result)` but not a fictitious universal convergence
statistic. MCMC uses optional ArviZ R-hat/ESS/MCSE where chain structure permits it;
PocoMC and TAMIS use normalized-weight and adaptation diagnostics; grids report
exact posterior support concentration. See {doc}`user_guide/diagnostics`.

## Simulation-based inference

| Method | Conditional density | Catalog batching | Device selection | Status |
|---|---:|---:|---:|---|
| MAF / nflows | yes | yes | auto, CUDA, MPS, CPU | stable |
| MDN | yes | yes | auto, CUDA, MPS, CPU | stable |
| Conditional diffusion | sample-only | yes | auto, CUDA, MPS, CPU | experimental |

The stable photometric SBI context conditions explicitly on both the measured
flux and its catalog uncertainty. Upper-limit/censoring states are not yet part
of the stable MAF/MDN context.

## Parameter-prior compatibility

- `UniformPrior`, `LogUniformPrior`, `NormalPrior`, and `StudentTPrior` describe
  continuous axes.
- `ChoicePrior` and `IntegerUniformPrior` describe finite discrete axes.
- `DeltaPrior` describes a fixed value, but `conditions` is the clearer
  object-specific interface for continuous samplers.
- MAF and MDN use explicit transforms for supported prior families; inspect
  {doc}`photometric_sbi_quickstart` before making a discrete quantity a neural
  target.
