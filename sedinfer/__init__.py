"""Small interfaces for Bayesian SED fitting and photo-z inference."""

from sedinfer.data import SEDDataset, SpectrumDataset
from sedinfer.catalog import CatalogGridResult, run_photometric_grid_catalog
from sedinfer.likelihood import GaussianPhotometricLikelihood, GaussianSpectralLikelihood
from sedinfer.parameters import ParameterSpace
from sedinfer.priors import DeltaPrior, LogUniformPrior, NormalPrior, UniformPrior
from sedinfer.units import MassNormalization

__all__ = [
    "DeltaPrior",
    "CatalogGridResult",
    "GaussianPhotometricLikelihood",
    "GaussianSpectralLikelihood",
    "LogUniformPrior",
    "MassNormalization",
    "NormalPrior",
    "ParameterSpace",
    "SEDDataset",
    "SpectrumDataset",
    "UniformPrior",
    "run_photometric_grid_catalog",
]
