"""Composable Bayesian SED fitting and photo-z inference."""

from composed.data import SEDDataset, SpectrumDataset
from composed.catalog import CatalogGridResult, run_photometric_grid_catalog
from composed.likelihood import GaussianPhotometricLikelihood, GaussianSpectralLikelihood
from composed.parameters import ParameterSpace
from composed.priors import DeltaPrior, LogUniformPrior, NormalPrior, UniformPrior
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
    "ParameterSpace",
    "SEDDataset",
    "SpectrumDataset",
    "UniformPrior",
    "load_inference_result",
    "normalize_sampling_result",
    "posterior_summary",
    "run_photometric_grid_catalog",
    "save_inference_result",
]
