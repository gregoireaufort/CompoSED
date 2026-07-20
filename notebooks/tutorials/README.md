# COSMOS2020 Tutorials

Run the notebooks in numerical order:

1. `00_prepare_cosmos2020_ugrizYJH.ipynb` reads the full FARMER catalog once and writes the shared 100,000-object artifact.
2. `01_fsps_maf_cosmos2020_catalog.ipynb` trains an FSPS continuity-SFH MAF and summarizes 100,000 posteriors.
3. `02_cigale_maf_cosmos2020_catalog.ipynb` repeats the catalog workflow with native CIGALE delayed-tau modeling.
4. `03_cigale_tamis_single_galaxy.ipynb` fits one stored galaxy with CIGALE and CompoSED's self-contained mixed-proposal TAMIS. Its categorical block is empty here, so this is ordinary continuous TAMIS.
5. `04_fsps_pocomc_single_galaxy.ipynb` fits the same galaxy with FSPS continuity SFHs and PocoMC.

Set the source catalog path before running notebook 00:

```bash
export COSMOS2020_FARMER_FITS=/path/to/cosmos2020_farmer.fits
```

The prepared `.npz` catalog and all inference outputs are local ignored
artifacts. Set `COMPOSED_TUTORIAL_QUICK=1` before opening an SBI notebook for a
short installation check; the normal configuration uses 300,000 simulations
and all 100,000 catalog objects.

Both MAF notebooks use the same public facade as the sampler tutorials:
`fit(problem, MAF(...), training=Simulate(...))`. The returned
`InferenceResult` contains the single-galaxy posterior and retains the trained
amortized estimator for catalog inference.

COSMOS2020/LePhare values shown in the plots are comparison estimates, not ground truth.
All four inference notebooks use the same effective uncertainty convention,
`sigma_eff^2 = sigma_catalog^2 + (0.05 |flux|)^2`. The 5% term is an explicit
model-error floor; it is included both when simulating SBI training data and
when conditioning/evaluating the observed catalog.

The CIGALE notebooks require the upstream v2022.0 engine and database described
in `docs/install.md`. The FSPS notebooks require python-fsps, sedpy, and a valid
`SPS_HOME`. PocoMC and the MAF stack are installed through the `samplers` and
`sbi` extras respectively; the self-contained TAMIS path does not use the
separate historical `TAMIS` Python package.

After training either MAF, its held-out simulator calibration can be generated
without retraining:

```bash
python examples/validate_cosmos2020_maf_calibration.py fsps
python examples/validate_cosmos2020_maf_calibration.py cigale
```
