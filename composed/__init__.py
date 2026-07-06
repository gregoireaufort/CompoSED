"""Composable Bayesian SED fitting and photo-z inference."""

from composed.data import SEDDataset, SpectrumDataset
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
from composed.parameters import ParameterSpace
from composed.priors import DeltaPrior, LogUniformPrior, NormalPrior, UniformPrior
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
    load_inference_result,
    normalize_sampling_result,
    posterior_summary,
    save_inference_result,
)
from composed.units import MassNormalization

__all__ = [
    "DeltaPrior",
    "CatalogGridResult",
    "GaussianPhotometricLikelihood",
    "GaussianSpectralLikelihood",
    "InferenceResult",
    "LogUniformPrior",
    "MassNormalization",
    "NormalPrior",
    "NativeCatalogFitResult",
    "ParameterSpace",
    "RedshiftFilterOperator",
    "RestFrameSpectralGrid",
    "SEDDataset",
    "SpectrumDataset",
    "UniformPrior",
    "build_redshift_filter_operator",
    "build_restframe_spectral_grid",
    "collect_run_provenance",
    "fit_catalog_with_restframe_grid",
    "load_restframe_spectral_grid",
    "load_inference_result",
    "normalize_sampling_result",
    "posterior_summary",
    "project_rest_grid_to_photometric_grid",
    "provenance_path_for",
    "read_provenance",
    "require_provenance",
    "run_photometric_grid_catalog",
    "save_inference_result",
    "save_npz_with_provenance",
    "save_restframe_spectral_grid",
    "sha256_file",
    "write_provenance",
]
