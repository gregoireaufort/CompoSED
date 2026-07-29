# Public API Inventory

This checklist maps the stable `composed` namespace and its documented
submodules to the reference pages. It is maintained deliberately; adding an
export without a documentation category is a release-review failure.

## Data and model contracts

- Data: `SEDDataset`, `SpectrumDataset`, `SpectroPhotometricDataset`
- Model outputs: `ModelPhotometry`, `ModelSpectrum`, `SEDBackend`
- Filters: `FilterSet`, `load_filter_set`
- Units: `MassNormalization`, `MassReference`
- Domain failures: `ModelDomainError`

## Parameters and physical histories

- Parameter order: `ParameterSpace`
- Priors: `UniformPrior`, `NormalPrior`, `StudentTPrior`,
  `LogUniformPrior`, `IntegerUniformPrior`, `ChoicePrior`, `DeltaPrior`
- SFHs: `SFHModel`, `SFHHistory`, `ConstantSFH`, `DelayedTauSFH`,
  `ExponentialSFH`, `ContinuitySFH`, `TabularSFH`, `make_sfh`,
  `available_sfh_models`
- Derived histories: `DerivedSFHQuantities`, `derive_sfh_quantities`

## Statistical problem

- Likelihood configuration: `Gaussian`
- Direct likelihoods: `GaussianPhotometricLikelihood`,
  `GaussianSpectralLikelihood`
- Composition: `Problem`
- Execution: `fit`

## Traditional inference

- Generic configuration: `Sampler`, `SamplerCapabilities`
- Stable methods: `Grid`, `RandomWalk`, `Emcee`, `MixedGibbs`,
  `MixedTAMIS`, `PocoMC`
- Experimental methods: `Laplace`, external `TAMIS`

## Simulation-based inference

- Simulation: `Simulate`, `PhotometricContext`, `SBITrainingSet`,
  `PhotometricTrainingSet`
- Methods: `MAF`, `MDN`
- Trained estimators: `TrainedMAFSBI`, `TrainedMDNSBI`
- Training helpers: `simulate_sbi_training_set`,
  `simulate_photometric_training_set`, `train_sbi`,
  `train_maf_photometric_sbi`, `train_mdn_photometric_sbi`
- Catalog/result helpers: `MAFCatalogSummary`, `MAFPhotometricSBIResult`,
  `MDNPhotometricSBIResult`, `transform_photometry`
- Noise: `ConditionalCatalogNoise`, `EmpiricalPhotometricNoise`

## Results and provenance

- Results: `InferenceResult`, `InferenceFailure`, `normalize_sampling_result`,
  `posterior_summary`
- Persistence: `save_inference_result`, `load_inference_result`
- Scientific identity: `problem_fingerprint`,
  `require_result_matches_problem`
- Artifact provenance: `collect_run_provenance`, `provenance_path_for`,
  `read_provenance`, `require_provenance`, `write_provenance`,
  `save_npz_with_provenance`, `sha256_file`,
  `verify_artifact_provenance`

## Catalog paths

- Finite grids: `CatalogGridResult`, `run_photometric_grid_catalog`
- Experimental projection: `RestFrameSpectralGrid`,
  `RedshiftFilterOperator`, `NativeCatalogFitResult`,
  `build_restframe_spectral_grid`, `build_redshift_filter_operator`,
  `project_rest_grid_to_photometric_grid`,
  `fit_catalog_with_restframe_grid`, `save_restframe_spectral_grid`,
  `load_restframe_spectral_grid`

Backend classes and plotting functions live in their documented submodules
rather than the top-level namespace.
