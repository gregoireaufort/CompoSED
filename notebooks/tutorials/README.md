# COSMOS2020 Tutorials

Run the notebooks in numerical order:

1. `00_prepare_cosmos2020_ugrizYJH.ipynb` reads the full FARMER catalog once and writes the shared 100,000-object artifact.
2. `01_fsps_maf_cosmos2020_catalog.ipynb` trains an FSPS continuity-SFH MAF and summarizes 100,000 posteriors.
3. `02_cigale_maf_cosmos2020_catalog.ipynb` repeats the catalog workflow with native CIGALE delayed-tau modeling.
4. `03_cigale_tamis_single_galaxy.ipynb` fits one stored galaxy with CIGALE and CompoSED's self-contained mixed-proposal TAMIS. The BC03 metallicity choices form an explicit categorical block.
5. `04_fsps_pocomc_single_galaxy.ipynb` fits the same galaxy with FSPS continuity SFHs and PocoMC.

Set the source catalog path before running notebook 00:

```bash
export COSMOS2020_FARMER_FITS=/path/to/cosmos2020_farmer.fits
```

The prepared `.npz` catalog and all inference outputs are local ignored
artifacts. Set `COMPOSED_TUTORIAL_QUICK=1` before opening an SBI notebook for a
short installation check; the normal configuration uses 300,000 simulations
and all 100,000 catalog objects.

The four inference notebooks are distributed with their latest executed plots
and numerical summaries so the complete workflows are readable on GitHub.
Their local input and model artifacts remain ignored. Re-execute them after
preparing the shared catalog before using the numbers scientifically; saved
outputs are demonstrations, not provenance for a new checkout.

Both MAF notebooks use the same public facade as the sampler tutorials:
`fit(problem, MAF(...), training=Simulate(...))`. The returned
`InferenceResult` contains the single-galaxy posterior and retains the trained
amortized estimator for catalog inference.

COSMOS2020/LePhare values shown in the plots are comparison estimates, not ground truth.
All four inference notebooks use the same effective uncertainty convention,
`sigma_eff(theta)^2 = sigma_catalog^2 + (0.05 f_model(theta))^2`. The raw
COSMOS2020 `sigma_catalog` is retained in every observed dataset and in the MAF
context. The 5% model-discrepancy term is declared on `Gaussian`, evaluated
only after the forward model, and used internally for likelihood evaluation
and SBI flux draws. It is never constructed from the observed flux.

The two MAF tutorials also share one serialized
`ConditionalCatalogNoise` model trained on complete COSMOS2020 magnitude and
raw-sigma rows. It learns the joint survey distribution
`q(log10 sigma_catalog | noiseless AB magnitude)` in the fixed ugrizYJH order.
The tutorial noise model uses `support_policy="raise"` during simulation:
prior draws whose noiseless magnitudes lie outside the complete-row COSMOS
training support are rejected rather than extrapolated or clamped. These and
any rare backend/noise failures are explicitly replaced with
`failure_policy="resample"`. The training metadata records the failures and
labels the resulting parameter distribution as simulator-success-conditioned.

The CIGALE MAF simulation prior includes the same finite BC03 metallicity
choices as the mixed-TAMIS tutorial. Metallicity is an explicit categorical
neural target: CompoSED learns its exact posterior probability on the declared
BC03 support and conditions the continuous flow on that category. It never
interpolates continuously between BC03 templates.

The CIGALE notebooks require the upstream v2022.0 engine and database described
in `docs/install.md`. The FSPS notebooks require python-fsps, sedpy, and a valid
`SPS_HOME`. PocoMC and the MAF stack are installed through the `samplers` and
`sbi` extras respectively; the self-contained TAMIS path does not use the
separate historical `TAMIS` Python package.

After training either MAF, use the held-out simulation and diagnostics cells in
the corresponding tutorial to inspect ranks, marginal coverage, and posterior
residuals without treating catalog estimates as truth.
