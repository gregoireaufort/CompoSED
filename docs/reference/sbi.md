# Simulation-Based Inference

## High-level CompoSED API

```{automodule} composed.sbi
:members: PhotometricContext, SBITrainingSet, PhotometricTrainingSet, Simulate, MAF, MDN, TrainedMAFSBI, TrainedMDNSBI, MAFCatalogSummary, MAFPhotometricSBIResult, MDNPhotometricSBIResult, SBISimulationFailureWarning, SBIContextSupportWarning, SBIPosteriorSaturationWarning, simulate_sbi_training_set, simulate_photometric_training_set, train_maf_photometric_sbi, train_mdn_photometric_sbi, train_sbi, transform_photometry
:show-inheritance:
```

## NumPy-facing estimators

```{automodule} inftools.sbi
:members: Standardizer, SBISimulationFailureWarning, MAFPosteriorEstimator, MDNPosteriorEstimator, build_maf, build_mdn, simulate_training_set, train_maf_posterior_from_dataset, train_maf_posterior, train_mdn_posterior_from_dataset, train_mdn_posterior, sample_posterior
:show-inheritance:
```

## Experimental diffusion

```{automodule} inftools.experimental.diffusion
:members: FeatureMetadata, ConditionalDiffusionEstimator, resolve_torch_device, make_training_mask, make_curriculum_training_mask, make_condition_mask
:show-inheritance:
```
