# Monte Carlo And Weighted Diagnostics

A finite posterior sample is not by itself evidence that an inference run is
reliable. CompoSED exposes one sampler-aware entry point:

```python
from composed import diagnose
from composed.plot import plot_diagnostics, plot_traces

report = diagnose(result)
print(report.summary())

diagnostic_figure, axes = plot_diagnostics(result, report)
trace_figure, trace_axes = plot_traces(result)
```

The report does not apply the same statistic to every algorithm. It identifies
the inference family and computes only quantities with a valid interpretation.

## What Is Reported

| Method | Main diagnostics | Not claimed |
|---|---|---|
| Random-walk MCMC | bulk/tail ESS, MCSE, autocorrelation scale, acceptance | R-hat from one chain |
| Emcee | rank-normalized R-hat, bulk/tail ESS, MCSE, walker acceptance | independent-chain proof from interacting walkers |
| Mixed Gibbs | MCMC metrics, discrete transition rate, visited states | exactness when an approximate thresholded kernel was requested |
| PocoMC | weight ESS, entropy/perplexity, maximum weight, evidence uncertainty | MCMC R-hat |
| TAMIS / MixedTAMIS | final weight ESS, beta and iteration-ESS histories, weight concentration | convergence from beta alone |
| Finite grid | posterior effective support and weight concentration | Monte Carlo convergence |
| Laplace | optimizer status and Hessian conditioning | sampling convergence |
| MAF / MDN | held-out rank, coverage, OOD, and boundary checks | chain ESS for independent neural draws |

ArviZ computes the rank-normalized MCMC quantities and is installed by the
`mc-diagnostics` optional dependency. Imports remain lazy, so core CompoSED and
weighted diagnostics do not require ArviZ.

```bash
python -m pip install -e ".[mc-diagnostics]"
```

Current ArviZ releases require a newer NumPy than the documented CIGALE
v2022.0 environment. A CIGALE run can save its post-burn-in chain normally;
load that `InferenceResult` and run `diagnose` in a modern analysis environment
that has ArviZ. No CIGALE import or backend evaluation is needed for this step.

## Interpreting Warnings

The defaults warn at R-hat above 1.01, bulk or tail ESS below 400, relative
importance-weight ESS below 0.1, or a maximum normalized importance weight
above 0.1. These are screening thresholds, not universal scientific laws.
Change them explicitly when the precision required by an analysis differs:

```python
report = diagnose(
    result,
    rhat_threshold=1.01,
    min_ess=1_000,
    min_relative_weight_ess=0.2,
)
```

For a single MCMC chain, ESS and MCSE remain available but R-hat is reported as
unavailable. For Emcee, walkers interact through the ensemble proposal. Their
R-hat is useful for screening stuck walkers, but independently initialized
ensembles are a stronger replication test.

Importance-weight ESS measures concentration of the normalized weights. It
does not establish that an adaptive proposal found every posterior mode. Repeat
important PocoMC or TAMIS analyses with independent seeds and compare posterior
summaries, evidence where applicable, and posterior-predictive distributions.

## Saving The Report

Bind the report into the same hashed scientific artifact as the posterior:

```python
from composed import save_inference_result

save_inference_result(
    result,
    "runs/object_001",
    diagnostics=report,
)
```

After loading, the JSON-compatible report is available as
`loaded_result.diagnostics`. Diagnostics are not recomputed during loading,
because doing so could reinterpret an old chain with changed software.

## Audit Checklist

- Inspect the chain after burn-in, not the adaptation segment.
- Check every fitted parameter, including weakly constrained nuisance axes.
- Inspect discrete-state occupancy and transitions for mixed models.
- For weighted methods, compare ESS with the total number of evaluated rows.
- Repeat adaptive importance and sequential Monte Carlo runs with new seeds.
- Check posterior predictions against detections, upper limits, and masks.
- Treat diagnostics as numerical evidence, not a replacement for model checks.
