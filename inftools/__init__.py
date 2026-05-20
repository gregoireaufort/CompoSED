# inftools/__init__.py

import importlib

from .core import Posterior, SamplingResult
from .transforms import BoxLogitTransform

__all__ = [
    "Posterior",
    "SamplingResult",
    "BatchFitFailure",
    "BatchFitResult",
    "fit_many",
    "run_laplace",
    "finite_difference_hessian",
    "run_emcee",
    "run_rw_metropolis",
    "run_tamis",
    "run_pocomc",
    "DiscreteGrid",
    "ParameterBlocks",
    "conditional_continuous_posterior",
    "enumerate_discrete_grid",
    "full_theta_from_blocks",
    "run_grid_sampler",
    "run_mixed_gibbs",
    "run_mixed_tamis",
    "sample_discrete_grid",
    "split_parameter_space",
    "plotting",
    "BoxLogitTransform",
    "MAFPosteriorEstimator",
    "simulate_training_set",
    "train_maf_posterior",
]


def __getattr__(name):
    if name in {"BatchFitFailure", "BatchFitResult", "fit_many"}:
        from .batch import BatchFitFailure, BatchFitResult, fit_many

        return {
            "BatchFitFailure": BatchFitFailure,
            "BatchFitResult": BatchFitResult,
            "fit_many": fit_many,
        }[name]
    if name in {"run_laplace", "finite_difference_hessian"}:
        from .laplace import finite_difference_hessian, run_laplace

        return {"run_laplace": run_laplace, "finite_difference_hessian": finite_difference_hessian}[name]
    if name in {"run_emcee", "run_rw_metropolis"}:
        from .mcmc import run_emcee, run_rw_metropolis

        return {"run_emcee": run_emcee, "run_rw_metropolis": run_rw_metropolis}[name]
    if name == "run_tamis":
        from .tamis_adapter import run_tamis

        return run_tamis
    if name == "run_pocomc":
        from .pocomc_adapter import run_pocomc

        return run_pocomc
    if name in {
        "DiscreteGrid",
        "ParameterBlocks",
        "conditional_continuous_posterior",
        "enumerate_discrete_grid",
        "full_theta_from_blocks",
        "run_grid_sampler",
        "run_mixed_gibbs",
        "sample_discrete_grid",
        "split_parameter_space",
    }:
        from .grid import (
            DiscreteGrid,
            ParameterBlocks,
            conditional_continuous_posterior,
            enumerate_discrete_grid,
            full_theta_from_blocks,
            run_grid_sampler,
            run_mixed_gibbs,
            sample_discrete_grid,
            split_parameter_space,
        )

        return {
            "DiscreteGrid": DiscreteGrid,
            "ParameterBlocks": ParameterBlocks,
            "conditional_continuous_posterior": conditional_continuous_posterior,
            "enumerate_discrete_grid": enumerate_discrete_grid,
            "full_theta_from_blocks": full_theta_from_blocks,
            "run_grid_sampler": run_grid_sampler,
            "run_mixed_gibbs": run_mixed_gibbs,
            "sample_discrete_grid": sample_discrete_grid,
            "split_parameter_space": split_parameter_space,
        }[name]
    if name == "run_mixed_tamis":
        from .mixed_tamis import run_mixed_tamis

        return run_mixed_tamis
    if name == "plotting":
        return importlib.import_module(".plotting", __name__)
    if name in {"MAFPosteriorEstimator", "simulate_training_set", "train_maf_posterior"}:
        from .sbi import MAFPosteriorEstimator, simulate_training_set, train_maf_posterior

        return {
            "MAFPosteriorEstimator": MAFPosteriorEstimator,
            "simulate_training_set": simulate_training_set,
            "train_maf_posterior": train_maf_posterior,
        }[name]
    raise AttributeError(f"module 'inftools' has no attribute {name!r}")
