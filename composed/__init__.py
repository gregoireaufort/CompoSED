"""Composable Bayesian SED fitting and photo-z inference."""

from importlib.metadata import PackageNotFoundError, version

from composed.data import SEDDataset, SpectroPhotometricDataset, SpectrumDataset
from composed.derived import DerivedSFHQuantities, derive_sfh_quantities
from composed.catalog import CatalogGridResult, run_photometric_grid_catalog
from composed.catalog_fast import (
    NativeCatalogFitResult,
    RedshiftFilterOperator,
    RestFrameSpectralGrid,
    build_redshift_filter_operator,
    build_restframe_spectral_grid,
    fit_catalog_with_restframe_grid,
    load_restframe_spectral_grid,
    project_rest_grid_to_photometric_grid,
    save_restframe_spectral_grid,
)
from composed.likelihood import GaussianPhotometricLikelihood, GaussianSpectralLikelihood
from composed.filters import FilterSet, load_filter_set
from composed.parameters import ParameterSpace
from composed.noise import EmpiricalPhotometricNoise
from composed.priors import DeltaPrior, LogUniformPrior, NormalPrior, StudentTPrior, UniformPrior
from composed.problem import (
    Emcee,
    Gaussian,
    Grid,
    Laplace,
    MixedGibbs,
    MixedTAMIS,
    PocoMC,
    Problem,
    RandomWalk,
    Sampler,
    SamplerCapabilities,
    TAMIS,
    fit,
)
from composed.provenance import (
    collect_run_provenance,
    provenance_path_for,
    read_provenance,
    require_provenance,
    save_npz_with_provenance,
    sha256_file,
    write_provenance,
)
from composed.results import (
    InferenceResult,
    InferenceFailure,
    load_inference_result,
    normalize_sampling_result,
    posterior_summary,
    save_inference_result,
)
from composed.sbi import (
    MAF,
    MAFCatalogSummary,
    MAFPhotometricSBIResult,
    PhotometricContext,
    PhotometricTrainingSet,
    SBITrainingSet,
    Simulate,
    TrainedMAFSBI,
    simulate_photometric_training_set,
    simulate_sbi_training_set,
    train_sbi,
    train_maf_photometric_sbi,
    transform_photometry,
)
from composed.sfh import (
    ConstantSFH,
    ContinuitySFH,
    DelayedTauSFH,
    ExponentialSFH,
    SFHHistory,
    SFHModel,
    TabularSFH,
    available_sfh_models,
    make_sfh,
)
from composed.units import MassNormalization, MassReference

try:
    __version__ = version("composed")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "ConstantSFH",
    "ContinuitySFH",
    "DelayedTauSFH",
    "ExponentialSFH",
    "SFHHistory",
    "SFHModel",
    "TabularSFH",
    "available_sfh_models",
    "make_sfh",
    "DeltaPrior",
    "DerivedSFHQuantities",
    "CatalogGridResult",
    "GaussianPhotometricLikelihood",
    "GaussianSpectralLikelihood",
    "Gaussian",
    "Grid",
    "InferenceResult",
    "InferenceFailure",
    "Laplace",
    "FilterSet",
    "LogUniformPrior",
    "MassNormalization",
    "MassReference",
    "MAF",
    "MAFCatalogSummary",
    "MAFPhotometricSBIResult",
    "PhotometricContext",
    "MixedGibbs",
    "MixedTAMIS",
    "NormalPrior",
    "StudentTPrior",
    "NativeCatalogFitResult",
    "ParameterSpace",
    "PocoMC",
    "Problem",
    "PhotometricTrainingSet",
    "SBITrainingSet",
    "RedshiftFilterOperator",
    "RestFrameSpectralGrid",
    "SEDDataset",
    "Sampler",
    "SamplerCapabilities",
    "Simulate",
    "SpectroPhotometricDataset",
    "SpectrumDataset",
    "RandomWalk",
    "TAMIS",
    "TrainedMAFSBI",
    "Emcee",
    "EmpiricalPhotometricNoise",
    "UniformPrior",
    "build_redshift_filter_operator",
    "build_restframe_spectral_grid",
    "collect_run_provenance",
    "fit_catalog_with_restframe_grid",
    "derive_sfh_quantities",
    "fit",
    "load_restframe_spectral_grid",
    "load_inference_result",
    "load_filter_set",
    "normalize_sampling_result",
    "posterior_summary",
    "project_rest_grid_to_photometric_grid",
    "provenance_path_for",
    "read_provenance",
    "require_provenance",
    "run_photometric_grid_catalog",
    "simulate_photometric_training_set",
    "simulate_sbi_training_set",
    "save_inference_result",
    "save_npz_with_provenance",
    "save_restframe_spectral_grid",
    "sha256_file",
    "train_maf_photometric_sbi",
    "train_sbi",
    "transform_photometry",
    "write_provenance",
]
