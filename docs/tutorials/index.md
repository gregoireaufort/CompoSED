# Tutorials

The committed tutorial suite uses one prepared COSMOS2020 ugrizYJH subset so
each notebook can focus on the inference workflow rather than catalog parsing.

1. `00_prepare_cosmos2020_ugrizYJH.ipynb` prepares the ignored tutorial data
   artifact from the FARMER catalog.
2. `01_fsps_maf_cosmos2020_catalog.ipynb` trains an FSPS continuity-SFH MAF and
   applies it to a large catalog.
3. `02_cigale_maf_cosmos2020_catalog.ipynb` trains a CIGALE delayed-tau MAF.
   Its BC03 metallicity is an exact categorical target rather than a relaxed
   continuous approximation.
4. `03_cigale_tamis_single_galaxy.ipynb` fits one object with mixed continuous
   and discrete CIGALE parameters using `MixedTAMIS`.
5. `04_fsps_pocomc_single_galaxy.ipynb` fits the same object with FSPS and
   PocoMC.

The notebooks live in
[`notebooks/tutorials`](https://github.com/gregoireaufort/CompoSED/tree/main/notebooks/tutorials).
Their local README records the input artifact and quick-run switches.

```{admonition} Execution policy
:class: warning
Documentation builds do not execute these notebooks. Full FSPS/CIGALE
simulation and neural training can take hours and require external engines and
catalog data. Execute them in a validated backend environment and retain their
run provenance.
```

The notebooks are the maintained user-facing workflows. Focused executable
checks for individual contracts live in the test suite.
