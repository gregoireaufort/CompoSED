from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from inftools.grid import full_theta_from_blocks, split_parameter_space
from composed.data import SEDDataset
from composed.likelihood import _backend_params_and_mass_scale, _normal_logcdf
from composed.units import MassNormalization


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


@dataclass
class PhotometricModelGrid:
    """Mass-normalized photometric model grid cached independently of data.

    ``flux`` has shape ``(n_models, n_bands)`` and is stored in the backend's
    output units, normally maggies for CompoSED photometric backends.  For the
    CIGALE-like workflow this grid is explicitly per unit stellar mass:
    ``backend.mass_normalization == MassNormalization.PER_SOLAR_MASS`` and no
    ``log10_mass`` scale has been applied yet.
    """

    samples: np.ndarray
    flux: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    parameter_names: tuple[str, ...]
    band_names: tuple[str, ...]
    mass_normalization: MassNormalization
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save the cached grid to a NumPy ``.npz`` file."""

        save_photometric_model_grid(self, path)


@dataclass
class CatalogProfileGridResult:
    """Catalog likelihood result using a cached mass-normalized model grid.

    ``profile_logp`` stores the log prior over non-mass grid parameters plus
    the Gaussian likelihood evaluated at each model's analytic best mass scale.
    If a ``log10_mass_grid`` was supplied, ``marginal_logp`` stores the same
    quantity marginalized over that mass grid with the declared mass prior.
    """

    model_grid: PhotometricModelGrid
    profile_logp: np.ndarray
    profile_weights_norm: np.ndarray
    profile_map_indices: np.ndarray
    profile_map_estimates: np.ndarray
    log10_mass_profile: np.ndarray
    mass_scale_profile: np.ndarray
    marginal_logp: np.ndarray | None = None
    marginal_weights_norm: np.ndarray | None = None
    marginal_map_indices: np.ndarray | None = None
    marginal_map_estimates: np.ndarray | None = None
    log10_mass_grid: np.ndarray | None = None
    mass_posterior_norm: np.ndarray | None = None
    log10_mass_quantiles: np.ndarray | None = None
    band_names: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)


def build_photometric_model_grid(
    backend,
    parameter_space,
    filters=None,
    *,
    band_names: Sequence[str] | None = None,
    excluded_parameters: Sequence[str] = ("log10_mass",),
    max_grid_size: int | None = 1_000_000,
) -> PhotometricModelGrid:
    """Build a reusable photometric model grid without applying mass scaling.

    This is the CompoSED analogue of CIGALE's model-grid step.  Every finite
    non-mass parameter combination is sent to the backend once.  The returned
    flux grid is then reusable for many catalog objects and many choices of
    mass prior/likelihood handling.

    Parameters listed in ``excluded_parameters`` are intentionally absent from
    the model grid.  This is normally just ``log10_mass``: CIGALE-like mass is
    handled later by profile or marginal likelihood evaluation, not by
    recomputing the SED.
    """

    excluded = tuple(str(name) for name in excluded_parameters)
    samples, names, log_prior = _finite_grid_theta_excluding(parameter_space, excluded, max_grid_size=max_grid_size)
    if samples.shape[0] == 0:
        raise ValueError("Cannot build an empty photometric model grid.")

    if band_names is None:
        if filters is not None and hasattr(filters, "names"):
            band_names = tuple(str(name) for name in filters.names)
        elif filters is not None:
            band_names = tuple(str(name) for name in filters)
        else:
            raise ValueError("band_names are required when filters do not expose names.")
    band_names = tuple(str(name) for name in band_names)

    mass_norm = MassNormalization(getattr(backend, "mass_normalization", None))
    model_flux = np.full((samples.shape[0], len(band_names)), np.nan, dtype=float)
    model_valid = np.zeros(samples.shape[0], dtype=bool)
    for i, row in enumerate(samples):
        params = {name: float(value) for name, value in zip(names, row)}
        try:
            model = backend.predict_photometry(params, filters)
            aligned = _align_model_flux(model, band_names)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        model_flux[i] = aligned
        model_valid[i] = np.all(np.isfinite(aligned)) and np.all(aligned >= 0.0)

    return PhotometricModelGrid(
        samples=samples,
        flux=model_flux,
        log_prior=log_prior,
        valid=model_valid & np.isfinite(log_prior),
        parameter_names=names,
        band_names=band_names,
        mass_normalization=mass_norm,
        meta={
            "excluded_parameters": excluded,
            "units": "backend photometry units; CompoSED CIGALEBackend uses maggies",
            "mass_scale_applied": False,
        },
    )


def evaluate_catalog_model_grid_likelihood(
    model_grid: PhotometricModelGrid,
    datasets: Sequence[SEDDataset],
    *,
    sigma_floor: float | None = None,
    log10_mass_grid: Sequence[float] | None = None,
    log10_mass_bounds: tuple[float, float] | None = None,
    log10_mass_prior: Sequence[float] | None = None,
    model_chunk_size: int = 2048,
    object_chunk_size: int = 512,
    mass_chunk_size: int = 128,
    store_mass_posterior: bool = False,
) -> CatalogProfileGridResult:
    """Evaluate a cached mass-normalized model grid against a catalog.

    The model grid is computed once; this function does only likelihood work.
    For detections, the profile mass scale is the usual weighted least-squares
    normalization

    ``A_hat = sum(f_obs f_model / sigma^2) / sum(f_model^2 / sigma^2)``.

    Upper limits are then evaluated at that profiled mass.  If
    ``log10_mass_grid`` is supplied, the likelihood is additionally evaluated
    on that mass grid, giving a pseudo-posterior over mass and a mass-marginal
    posterior over the non-mass model grid.
    """

    if model_grid.mass_normalization != MassNormalization.PER_SOLAR_MASS:
        raise ValueError(
            "evaluate_catalog_model_grid_likelihood expects a PER_SOLAR_MASS model grid. "
            f"Got {model_grid.mass_normalization}."
        )
    if model_chunk_size <= 0 or object_chunk_size <= 0 or mass_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive.")
    if sigma_floor is not None and float(sigma_floor) < 0.0:
        raise ValueError("sigma_floor must be non-negative.")

    datasets = tuple(datasets)
    band_names, data_flux, data_sigma, active_mask, upper_limit, upper_limit_mask = _stack_catalog_arrays(
        datasets, sigma_floor=sigma_floor
    )
    if tuple(model_grid.band_names) != band_names:
        raise ValueError(
            "Model grid band order does not match catalog band order: "
            f"grid={tuple(model_grid.band_names)}, catalog={band_names}."
        )

    log10_mass_grid_arr, log_mass_prior_weights, mass_bounds = _prepare_mass_grid_and_prior(
        log10_mass_grid=log10_mass_grid,
        log10_mass_bounds=log10_mass_bounds,
        log10_mass_prior=log10_mass_prior,
    )
    profile = _catalog_profile_mass_logp(
        data_flux=data_flux,
        data_sigma=data_sigma,
        active_mask=active_mask,
        upper_limit=upper_limit,
        upper_limit_mask=upper_limit_mask,
        model_flux=model_grid.flux,
        log_prior=model_grid.log_prior,
        valid_grid=model_grid.valid,
        model_chunk_size=int(model_chunk_size),
        object_chunk_size=int(object_chunk_size),
        log10_mass_bounds=mass_bounds,
        require_finite=log10_mass_grid_arr is None,
    )

    marginal_logp = None
    marginal_weights = None
    marginal_map_indices = None
    marginal_map_estimates = None
    mass_posterior_norm = None
    log10_mass_quantiles = None
    if log10_mass_grid_arr is not None:
        marginal = _catalog_marginal_mass_logp(
            data_flux=data_flux,
            data_sigma=data_sigma,
            active_mask=active_mask,
            upper_limit=upper_limit,
            upper_limit_mask=upper_limit_mask,
            model_flux=model_grid.flux,
            log_prior=model_grid.log_prior,
            valid_grid=model_grid.valid,
            log10_mass_grid=log10_mass_grid_arr,
            log_mass_prior_weights=log_mass_prior_weights,
            model_chunk_size=int(model_chunk_size),
            object_chunk_size=int(object_chunk_size),
            mass_chunk_size=int(mass_chunk_size),
            store_mass_posterior=store_mass_posterior,
        )
        marginal_logp = marginal["marginal_logp"]
        marginal_weights = _normalize_logp_rows(marginal_logp)
        marginal_map_indices = np.asarray([int(np.nanargmax(row)) for row in marginal_logp], dtype=int)
        marginal_map_estimates = model_grid.samples[marginal_map_indices]
        mass_posterior_norm = marginal["mass_posterior_norm"]
        log10_mass_quantiles = marginal["log10_mass_quantiles"]

    if np.all(np.any(np.isfinite(profile["profile_logp"]), axis=1)):
        profile_weights = _normalize_logp_rows(profile["profile_logp"])
        profile_map_indices = np.asarray([int(np.nanargmax(row)) for row in profile["profile_logp"]], dtype=int)
        profile_map_estimates = model_grid.samples[profile_map_indices]
    else:
        profile_weights = np.zeros_like(profile["profile_logp"], dtype=float)
        profile_map_indices = np.full(profile["profile_logp"].shape[0], -1, dtype=int)
        profile_map_estimates = np.full((profile["profile_logp"].shape[0], model_grid.samples.shape[1]), np.nan)
    return CatalogProfileGridResult(
        model_grid=model_grid,
        profile_logp=profile["profile_logp"],
        profile_weights_norm=profile_weights,
        profile_map_indices=profile_map_indices,
        profile_map_estimates=profile_map_estimates,
        log10_mass_profile=profile["log10_mass_profile"],
        mass_scale_profile=profile["mass_scale_profile"],
        marginal_logp=marginal_logp,
        marginal_weights_norm=marginal_weights,
        marginal_map_indices=marginal_map_indices,
        marginal_map_estimates=marginal_map_estimates,
        log10_mass_grid=log10_mass_grid_arr,
        mass_posterior_norm=mass_posterior_norm,
        log10_mass_quantiles=log10_mass_quantiles,
        band_names=band_names,
        meta={
            "active_mask": active_mask,
            "upper_limit": upper_limit,
            "upper_limit_mask": upper_limit_mask,
            "sigma_floor": sigma_floor,
            "log10_mass_bounds": mass_bounds,
            "mass_prior": "uniform over supplied log10_mass_grid" if log_mass_prior_weights is not None else None,
        },
    )


def save_photometric_model_grid(grid: PhotometricModelGrid, path: str | Path) -> None:
    """Save a :class:`PhotometricModelGrid` for later catalog evaluation."""

    path = Path(path)
    np.savez_compressed(
        path,
        samples=np.asarray(grid.samples, dtype=float),
        flux=np.asarray(grid.flux, dtype=float),
        log_prior=np.asarray(grid.log_prior, dtype=float),
        valid=np.asarray(grid.valid, dtype=bool),
        parameter_names=np.asarray(grid.parameter_names, dtype=object),
        band_names=np.asarray(grid.band_names, dtype=object),
        mass_normalization=np.asarray(grid.mass_normalization.value, dtype=object),
        meta=np.asarray(json.dumps(grid.meta, sort_keys=True, default=str), dtype=object),
    )


def load_photometric_model_grid(path: str | Path) -> PhotometricModelGrid:
    """Load a model grid saved by :func:`save_photometric_model_grid`."""

    data = np.load(Path(path), allow_pickle=True)
    return PhotometricModelGrid(
        samples=np.asarray(data["samples"], dtype=float),
        flux=np.asarray(data["flux"], dtype=float),
        log_prior=np.asarray(data["log_prior"], dtype=float),
        valid=np.asarray(data["valid"], dtype=bool),
        parameter_names=tuple(str(name) for name in data["parameter_names"]),
        band_names=tuple(str(name) for name in data["band_names"]),
        mass_normalization=MassNormalization(str(data["mass_normalization"].item())),
        meta=json.loads(str(data["meta"].item())) if "meta" in data.files else {},
    )


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


def _finite_grid_theta_excluding(
    parameter_space,
    excluded_names: Sequence[str],
    *,
    max_grid_size: int | None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Enumerate finite grid rows after removing nuisance parameters.

    This is intentionally separate from ``enumerate_discrete_grid`` because
    CIGALE-like mass scaling often removes a continuous ``log10_mass`` from the
    forward-model grid while keeping all other axes finite.
    """

    from inftools.grid import finite_prior_values

    excluded = set(str(name) for name in excluded_names)
    names: list[str] = []
    values_by_axis: list[np.ndarray] = []
    for name in parameter_space.names:
        if name in excluded:
            continue
        prior = parameter_space.priors.get(name)
        fixed = _fixed_prior_value(prior)
        if fixed is not None:
            values = np.asarray([fixed], dtype=float)
        else:
            values = finite_prior_values(prior)
            if values is None:
                raise ValueError(
                    "build_photometric_model_grid only supports finite-valued or fixed non-mass parameters. "
                    f"Parameter {name!r} is continuous; exclude it or use a finite prior."
                )
        names.append(str(name))
        values_by_axis.append(np.asarray(values, dtype=float))

    if not values_by_axis:
        return np.empty((1, 0), dtype=float), (), np.asarray([0.0], dtype=float)

    size = int(np.prod([axis.size for axis in values_by_axis], dtype=np.int64))
    if max_grid_size is not None and size > int(max_grid_size):
        raise ValueError(
            f"Model grid has {size} points, larger than max_grid_size={max_grid_size}. "
            "Reduce the grid or raise max_grid_size explicitly."
        )

    from itertools import product

    samples = np.asarray(list(product(*values_by_axis)), dtype=float)
    log_prior = np.asarray(
        [
            _log_prior_for_named_values(parameter_space, names, row, excluded=excluded)
            for row in samples
        ],
        dtype=float,
    )
    return samples, tuple(names), log_prior


def _fixed_prior_value(prior) -> float | None:
    if prior is None:
        return None
    if type(prior).__name__ == "DeltaPrior":
        return float(prior.value)
    return None


def _log_prior_for_named_values(parameter_space, names: Sequence[str], values: Sequence[float], *, excluded: set[str]) -> float:
    value_by_name = {name: float(value) for name, value in zip(names, values)}
    total = 0.0
    for name in parameter_space.names:
        if name in excluded:
            continue
        prior = parameter_space.priors.get(name)
        if prior is None:
            continue
        logp = prior.logpdf(value_by_name[name])
        if not np.isfinite(logp):
            return -np.inf
        total += float(logp)
    return float(total)


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


def _prepare_mass_grid_and_prior(*, log10_mass_grid, log10_mass_bounds, log10_mass_prior):
    if log10_mass_bounds is not None:
        lo, hi = (float(log10_mass_bounds[0]), float(log10_mass_bounds[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("log10_mass_bounds must be finite and increasing.")
        bounds = (lo, hi)
    else:
        bounds = None

    if log10_mass_grid is None:
        if log10_mass_prior is not None:
            raise ValueError("log10_mass_prior requires log10_mass_grid.")
        return None, None, bounds

    grid = np.asarray(log10_mass_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("log10_mass_grid must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("log10_mass_grid must be finite.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("log10_mass_grid must be strictly increasing.")
    if bounds is None:
        bounds = (float(grid[0]), float(grid[-1]))
    else:
        keep = (grid >= bounds[0]) & (grid <= bounds[1])
        if not np.any(keep):
            raise ValueError("log10_mass_grid has no values inside log10_mass_bounds.")
        grid = grid[keep]

    if log10_mass_prior is None:
        log_weights = np.full(grid.size, -np.log(grid.size), dtype=float)
    else:
        prior = np.asarray(log10_mass_prior, dtype=float)
        if prior.shape != grid.shape:
            raise ValueError("log10_mass_prior must have the same shape as log10_mass_grid.")
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0) or np.sum(prior) <= 0.0:
            raise ValueError("log10_mass_prior must contain finite non-negative weights with positive sum.")
        log_weights = np.log(prior / np.sum(prior))
    return grid, log_weights, bounds


def _catalog_profile_mass_logp(
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
    log10_mass_bounds,
    require_finite=True,
):
    n_objects = data_flux.shape[0]
    n_grid = model_flux.shape[0]
    profile_logp = np.full((n_objects, n_grid), -np.inf, dtype=float)
    mass_scale = np.full((n_objects, n_grid), np.nan, dtype=float)
    log10_mass = np.full((n_objects, n_grid), np.nan, dtype=float)
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
            numerator = np.sum(data_flux[o0:o1, None, :] * model[None, :, :] * inv_sigma2[o0:o1, None, :], axis=2)
            denominator = np.sum(model[None, :, :] ** 2 * inv_sigma2[o0:o1, None, :], axis=2)
            scale = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.0)
            scale = _clip_mass_scale(scale, log10_mass_bounds)
            model_scaled = scale[:, :, None] * model[None, :, :]
            diff = data_flux[o0:o1, None, :] - model_scaled
            chi2 = np.sum(diff**2 * inv_sigma2[o0:o1, None, :], axis=2)
            log_like = -0.5 * (chi2 + logdet[o0:o1, None])
            local_upper_mask = upper_limit_mask[o0:o1]
            if np.any(local_upper_mask):
                z = (upper_limit[o0:o1, None, :] - model_scaled) / data_sigma[o0:o1, None, :]
                log_like += np.sum(np.where(local_upper_mask[:, None, :], _normal_logcdf(z), 0.0), axis=2)
            finite_scale = np.isfinite(scale) & (scale > 0.0)
            values = log_prior[grid_indices][None, :] + log_like
            values = np.where(finite_scale, values, -np.inf)
            profile_logp[o0:o1, grid_indices] = values
            mass_scale[o0:o1, grid_indices] = scale
            log10_mass[o0:o1, grid_indices] = np.where(finite_scale, np.log10(scale), np.nan)

    if require_finite and not np.all(np.any(np.isfinite(profile_logp), axis=1)):
        bad = np.where(~np.any(np.isfinite(profile_logp), axis=1))[0]
        raise RuntimeError(f"No finite profiled grid point for catalog object(s): {bad.tolist()}")
    return {
        "profile_logp": profile_logp,
        "mass_scale_profile": mass_scale,
        "log10_mass_profile": log10_mass,
    }


def _clip_mass_scale(scale: np.ndarray, log10_mass_bounds: tuple[float, float] | None) -> np.ndarray:
    scale = np.asarray(scale, dtype=float)
    if log10_mass_bounds is None:
        tiny = np.finfo(float).tiny
        return np.where(np.isfinite(scale) & (scale > tiny), scale, tiny)
    lo, hi = log10_mass_bounds
    return np.clip(scale, 10.0**lo, 10.0**hi)


def _catalog_marginal_mass_logp(
    *,
    data_flux,
    data_sigma,
    active_mask,
    upper_limit,
    upper_limit_mask,
    model_flux,
    log_prior,
    valid_grid,
    log10_mass_grid,
    log_mass_prior_weights,
    model_chunk_size,
    object_chunk_size,
    mass_chunk_size,
    store_mass_posterior,
):
    n_objects = data_flux.shape[0]
    n_grid = model_flux.shape[0]
    n_mass = log10_mass_grid.size
    marginal_logp = np.full((n_objects, n_grid), -np.inf, dtype=float)
    mass_logp_by_model = (
        np.full((n_objects, n_grid, n_mass), -np.inf, dtype=float) if store_mass_posterior else None
    )
    mass_marginal_logp = np.full((n_objects, n_mass), -np.inf, dtype=float)
    detection_mask = active_mask & ~upper_limit_mask
    inv_sigma2 = np.where(detection_mask, 1.0 / data_sigma**2, 0.0)
    logdet = np.sum(np.where(detection_mask, np.log(2.0 * np.pi * data_sigma**2), 0.0), axis=1)
    mass_scales = 10.0**log10_mass_grid

    for g0 in range(0, n_grid, model_chunk_size):
        g1 = min(g0 + model_chunk_size, n_grid)
        local_valid = valid_grid[g0:g1]
        if not np.any(local_valid):
            continue
        grid_indices = np.arange(g0, g1)[local_valid]
        model = model_flux[grid_indices]
        for o0 in range(0, n_objects, object_chunk_size):
            o1 = min(o0 + object_chunk_size, n_objects)
            local_mass_logp = np.full((o1 - o0, grid_indices.size, n_mass), -np.inf, dtype=float)
            for m0 in range(0, n_mass, mass_chunk_size):
                m1 = min(m0 + mass_chunk_size, n_mass)
                scaled = mass_scales[m0:m1][None, :, None] * model[:, None, :]
                diff = data_flux[o0:o1, None, None, :] - scaled[None, :, :, :]
                chi2 = np.sum(diff**2 * inv_sigma2[o0:o1, None, None, :], axis=3)
                log_like = -0.5 * (chi2 + logdet[o0:o1, None, None])
                local_upper_mask = upper_limit_mask[o0:o1]
                if np.any(local_upper_mask):
                    z = (upper_limit[o0:o1, None, None, :] - scaled[None, :, :, :]) / data_sigma[
                        o0:o1, None, None, :
                    ]
                    log_like += np.sum(
                        np.where(local_upper_mask[:, None, None, :], _normal_logcdf(z), 0.0),
                        axis=3,
                    )
                local_mass_logp[:, :, m0:m1] = (
                    log_prior[grid_indices][None, :, None]
                    + log_like
                    + log_mass_prior_weights[None, None, m0:m1]
                )
            marginal_logp[o0:o1, grid_indices] = _logsumexp(local_mass_logp, axis=2)
            local_mass_marginal = _logsumexp(local_mass_logp, axis=1)
            mass_marginal_logp[o0:o1] = np.logaddexp(mass_marginal_logp[o0:o1], local_mass_marginal)
            if store_mass_posterior:
                mass_logp_by_model[o0:o1, grid_indices, :] = local_mass_logp

    if not np.all(np.any(np.isfinite(marginal_logp), axis=1)):
        bad = np.where(~np.any(np.isfinite(marginal_logp), axis=1))[0]
        raise RuntimeError(f"No finite mass-marginal grid point for catalog object(s): {bad.tolist()}")

    mass_posterior_norm = None
    if store_mass_posterior:
        mass_posterior_norm = np.zeros_like(mass_logp_by_model, dtype=float)
        flat = mass_logp_by_model.reshape(n_objects, -1)
        norm = _logsumexp(flat, axis=1)
        with np.errstate(under="ignore"):
            mass_posterior_norm = np.exp(mass_logp_by_model - norm[:, None, None])

    log10_mass_quantiles = np.empty((n_objects, 3), dtype=float)
    for i in range(n_objects):
        finite = np.isfinite(mass_marginal_logp[i])
        if not np.any(finite):
            log10_mass_quantiles[i] = np.nan
            continue
        weights = np.zeros(n_mass, dtype=float)
        max_logp = np.max(mass_marginal_logp[i, finite])
        weights[finite] = np.exp(mass_marginal_logp[i, finite] - max_logp)
        weights /= np.sum(weights)
        log10_mass_quantiles[i] = _weighted_quantile_grid(log10_mass_grid, weights, [0.16, 0.5, 0.84])

    return {
        "marginal_logp": marginal_logp,
        "mass_posterior_norm": mass_posterior_norm,
        "log10_mass_quantiles": log10_mass_quantiles,
    }


def _logsumexp(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_value = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(max_value)
    shifted = np.where(finite, values - max_value, -np.inf)
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = max_value + np.log(summed)
    out = np.where(finite, out, -np.inf)
    if axis is None:
        return np.asarray(out).reshape(()).item()
    return np.squeeze(out, axis=axis)


def _weighted_quantile_grid(grid: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if grid.ndim != 1 or weights.shape != grid.shape:
        raise ValueError("grid and weights must be one-dimensional arrays with matching shape.")
    if not np.isfinite(weights).all() or np.sum(weights) <= 0.0:
        return np.full(len(tuple(quantiles)), np.nan, dtype=float)
    weights = weights / np.sum(weights)
    cdf = np.cumsum(weights)
    return np.interp(np.asarray(tuple(quantiles), dtype=float), cdf, grid)


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
