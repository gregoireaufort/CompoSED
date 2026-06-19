from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from inftools.grid import full_theta_from_blocks, split_parameter_space
from composed.data import SEDDataset
from composed.likelihood import _backend_params_and_mass_scale, _normal_logcdf


@dataclass
class CatalogGridResult:
    """Vectorized photometric grid result for a catalog of SEDs.

    ``samples`` has shape ``(n_grid, n_parameters)``. ``logp`` and
    ``weights_norm`` have shape ``(n_objects, n_grid)`` so each object keeps its
    own posterior over the shared grid.
    """

    samples: np.ndarray
    logp: np.ndarray
    weights_norm: np.ndarray
    map_estimates: np.ndarray
    map_indices: np.ndarray
    parameter_names: tuple[str, ...]
    band_names: tuple[str, ...]
    meta: dict = field(default_factory=dict)


def run_photometric_grid_catalog(
    backend,
    datasets: Sequence[SEDDataset],
    parameter_space,
    filters=None,
    *,
    sigma_floor: float | None = None,
    model_chunk_size: int = 2048,
    object_chunk_size: int = 512,
    max_grid_size: int | None = 1_000_000,
    store_model_flux: bool = False,
) -> CatalogGridResult:
    """Evaluate one finite photometric grid against many SEDs.

    This is the catalog-scale version of the plain CIGALE-style grid
    calculation. The backend is called once per grid point to build
    ``model_flux[grid, band]``. The Gaussian likelihood is then evaluated for
    all objects using chunked NumPy broadcasting over
    ``data_flux[object, band]`` and ``sigma[object, band]``.

    All datasets must use the same band order. Individual objects may still
    have different masks; masked bands contribute neither residual nor log
    determinant.
    """

    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("run_photometric_grid_catalog requires at least one SEDDataset.")
    if model_chunk_size <= 0:
        raise ValueError("model_chunk_size must be positive.")
    if object_chunk_size <= 0:
        raise ValueError("object_chunk_size must be positive.")
    if sigma_floor is not None and float(sigma_floor) < 0.0:
        raise ValueError("sigma_floor must be non-negative.")

    band_names, data_flux, data_sigma, active_mask, upper_limit, upper_limit_mask = _stack_catalog_arrays(
        datasets, sigma_floor=sigma_floor
    )
    if filters is None:
        filters = datasets[0].metadata.get("filters")

    samples, log_prior = _finite_grid_theta(parameter_space, max_grid_size=max_grid_size)
    model_flux, model_valid = _predict_model_grid_flux(
        backend=backend,
        samples=samples,
        parameter_space=parameter_space,
        filters=filters,
        band_names=band_names,
    )
    valid_grid = model_valid & np.isfinite(log_prior)

    logp = _catalog_gaussian_logp(
        data_flux=data_flux,
        data_sigma=data_sigma,
        active_mask=active_mask,
        upper_limit=upper_limit,
        upper_limit_mask=upper_limit_mask,
        model_flux=model_flux,
        log_prior=log_prior,
        valid_grid=valid_grid,
        model_chunk_size=int(model_chunk_size),
        object_chunk_size=int(object_chunk_size),
    )
    weights = _normalize_logp_rows(logp)
    map_indices = np.asarray([int(np.nanargmax(row)) for row in logp], dtype=int)
    map_estimates = samples[map_indices]

    meta = {
        "active_mask": active_mask,
        "upper_limit": upper_limit,
        "upper_limit_mask": upper_limit_mask,
        "valid_grid": valid_grid,
        "sigma_floor": sigma_floor,
        "model_chunk_size": int(model_chunk_size),
        "object_chunk_size": int(object_chunk_size),
    }
    if store_model_flux:
        meta["model_flux"] = model_flux

    return CatalogGridResult(
        samples=samples,
        logp=logp,
        weights_norm=weights,
        map_estimates=map_estimates,
        map_indices=map_indices,
        parameter_names=tuple(parameter_space.names),
        band_names=band_names,
        meta=meta,
    )


def _stack_catalog_arrays(
    datasets: Sequence[SEDDataset],
    *,
    sigma_floor: float | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    band_names = tuple(datasets[0].band_names)
    flux_rows = []
    sigma_rows = []
    mask_rows = []
    upper_limit_rows = []
    upper_limit_mask_rows = []
    for i, dataset in enumerate(datasets):
        if tuple(dataset.band_names) != band_names:
            raise ValueError(
                "All catalog datasets must have the same band order; "
                f"object 0 has {band_names}, object {i} has {tuple(dataset.band_names)}."
            )
        mask = np.asarray(dataset.active_mask, dtype=bool)
        if not np.any(mask):
            raise ValueError(f"Object {i} has no active bands.")
        sigma = np.asarray(dataset.sigma, dtype=float)
        if sigma_floor is not None:
            sigma = np.sqrt(sigma**2 + float(sigma_floor) ** 2)
        upper_mask = mask & np.asarray(dataset.upper_limit_mask, dtype=bool)
        detection_mask = mask & ~upper_mask
        flux_rows.append(np.where(detection_mask, np.asarray(dataset.flux, dtype=float), 0.0))
        sigma_rows.append(np.where(mask, sigma, 1.0))
        mask_rows.append(mask)
        upper_limit_rows.append(np.where(upper_mask, np.asarray(dataset.upper_limit, dtype=float), 0.0))
        upper_limit_mask_rows.append(upper_mask)
    return (
        band_names,
        np.asarray(flux_rows, dtype=float),
        np.asarray(sigma_rows, dtype=float),
        np.asarray(mask_rows, dtype=bool),
        np.asarray(upper_limit_rows, dtype=float),
        np.asarray(upper_limit_mask_rows, dtype=bool),
    )


def _finite_grid_theta(parameter_space, max_grid_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    blocks = split_parameter_space(parameter_space)
    if blocks.continuous_indices:
        names = ", ".join(blocks.continuous_names)
        raise ValueError(
            "run_photometric_grid_catalog only supports finite-valued or fixed parameters. "
            f"Continuous parameter(s) found: {names}."
        )

    from inftools.grid import enumerate_discrete_grid

    grid = enumerate_discrete_grid(parameter_space, max_size=max_grid_size)
    samples = np.asarray(
        [full_theta_from_blocks(parameter_space, np.empty(0), values) for values in grid.points],
        dtype=float,
    )
    log_prior = np.asarray([parameter_space.log_prior(theta) for theta in samples], dtype=float)
    return samples, log_prior


def _predict_model_grid_flux(backend, samples, parameter_space, filters, band_names):
    model_flux = np.full((samples.shape[0], len(band_names)), np.nan, dtype=float)
    model_valid = np.zeros(samples.shape[0], dtype=bool)
    for i, theta in enumerate(samples):
        params = parameter_space.to_dict(theta)
        try:
            backend_params, mass_scale = _backend_params_and_mass_scale(
                params,
                backend,
                quantity_name="photometry",
            )
            model = backend.predict_photometry(backend_params, filters)
            aligned = _align_model_flux(model, band_names)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        model_flux[i] = mass_scale * aligned
        model_valid[i] = np.all(np.isfinite(model_flux[i]))
    return model_flux, model_valid


def _align_model_flux(model, band_names: Sequence[str]) -> np.ndarray:
    flux = np.asarray(model.flux, dtype=float)
    names = tuple(str(name) for name in getattr(model, "band_names", ()))
    if len(names) != flux.size:
        raise ValueError("ModelPhotometry band_names length must match flux length.")
    if len(set(names)) != len(names):
        raise ValueError("ModelPhotometry band_names must be unique.")
    missing = [name for name in band_names if name not in names]
    if missing:
        raise ValueError(f"Model photometry is missing active catalog band(s): {', '.join(missing)}")
    lookup = {name: i for i, name in enumerate(names)}
    return np.asarray([flux[lookup[name]] for name in band_names], dtype=float)


def _catalog_gaussian_logp(
    *,
    data_flux,
    data_sigma,
    active_mask,
    upper_limit,
    upper_limit_mask,
    model_flux,
    log_prior,
    valid_grid,
    model_chunk_size,
    object_chunk_size,
) -> np.ndarray:
    n_objects = data_flux.shape[0]
    n_grid = model_flux.shape[0]
    logp = np.full((n_objects, n_grid), -np.inf, dtype=float)
    detection_mask = active_mask & ~upper_limit_mask
    inv_sigma2 = np.where(detection_mask, 1.0 / data_sigma**2, 0.0)
    logdet = np.sum(np.where(detection_mask, np.log(2.0 * np.pi * data_sigma**2), 0.0), axis=1)

    for g0 in range(0, n_grid, model_chunk_size):
        g1 = min(g0 + model_chunk_size, n_grid)
        local_valid = valid_grid[g0:g1]
        if not np.any(local_valid):
            continue
        grid_indices = np.arange(g0, g1)[local_valid]
        model = model_flux[grid_indices]
        for o0 in range(0, n_objects, object_chunk_size):
            o1 = min(o0 + object_chunk_size, n_objects)
            diff = data_flux[o0:o1, None, :] - model[None, :, :]
            chi2 = np.sum(diff**2 * inv_sigma2[o0:o1, None, :], axis=2)
            log_like = -0.5 * (chi2 + logdet[o0:o1, None])
            local_upper_mask = upper_limit_mask[o0:o1]
            if np.any(local_upper_mask):
                z = (upper_limit[o0:o1, None, :] - model[None, :, :]) / data_sigma[o0:o1, None, :]
                log_like += np.sum(np.where(local_upper_mask[:, None, :], _normal_logcdf(z), 0.0), axis=2)
            logp[o0:o1, grid_indices] = log_prior[grid_indices][None, :] + log_like

    if not np.all(np.any(np.isfinite(logp), axis=1)):
        bad = np.where(~np.any(np.isfinite(logp), axis=1))[0]
        raise RuntimeError(f"No finite grid point for catalog object(s): {bad.tolist()}")
    return logp


def _normalize_logp_rows(logp: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(logp, dtype=float)
    for i, row in enumerate(logp):
        finite = np.isfinite(row)
        if not np.any(finite):
            raise RuntimeError(f"Cannot normalize catalog weights for object {i}: all logp are non-finite.")
        max_logp = np.max(row[finite])
        weights[i, finite] = np.exp(row[finite] - max_logp)
        weights[i] /= np.sum(weights[i])
    return weights
