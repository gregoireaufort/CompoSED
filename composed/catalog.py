from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from inftools.grid import _logsumexp, full_theta_from_blocks, split_parameter_space
from composed.data import SEDDataset
from composed.errors import ModelDomainError
from composed.likelihood import _backend_params_and_mass_scale, _normal_logcdf
from composed.priors import (
    ChoicePrior,
    DeltaPrior,
    IntegerUniformPrior,
    LogUniformPrior,
    NormalPrior,
    Prior,
    StudentTPrior,
    UniformPrior,
)
from composed.problem import _backend_configuration, _filter_specification, _stable_value
from composed.provenance import provenance_path_for, read_provenance, require_provenance, save_npz_with_provenance
from composed.units import (
    MASS_CONVENTION_SCHEMA,
    MassNormalization,
    MassReference,
    backend_mass_reference,
    canonical_photometric_flux_unit,
    convert_photometric_flux,
    validate_mass_reference,
)


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
    output units, normally maggies for CompoSED photometric backends. For the
    CIGALE-like workflow this grid is explicitly per unit surviving stellar mass:
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
    mass_reference: MassReference | None = MassReference.SURVIVING_STELLAR_MASS
    flux_unit: str = "maggies"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mass_normalization = MassNormalization(self.mass_normalization)
        self.mass_reference = validate_mass_reference(self.mass_normalization, self.mass_reference)

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
    Every reported ``log10_mass`` is present-day surviving stellar mass.
    ``mass_profile_at_boundary`` identifies models whose unconstrained
    analytic amplitude was clipped to an explicitly declared mass bound.
    """

    model_grid: PhotometricModelGrid
    profile_logp: np.ndarray
    profile_weights_norm: np.ndarray
    profile_map_indices: np.ndarray
    profile_map_estimates: np.ndarray
    log10_mass_profile: np.ndarray
    mass_scale_profile: np.ndarray
    mass_profile_at_boundary: np.ndarray
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
    mass_reference = backend_mass_reference(backend)
    model_flux = np.full((samples.shape[0], len(band_names)), np.nan, dtype=float)
    model_valid = np.zeros(samples.shape[0], dtype=bool)
    for i, row in enumerate(samples):
        params = {name: float(value) for name, value in zip(names, row)}
        try:
            model = backend.predict_photometry(params, filters)
            aligned = convert_photometric_flux(
                _align_model_flux(model, band_names),
                getattr(model, "flux_unit", "maggies"),
                "maggies",
            )
        except (ModelDomainError, FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        model_flux[i] = aligned
        # A successful backend evaluation makes this a valid grid row.
        # Individual catalog objects may mask different bands, so flux
        # finiteness is checked later against each object's active mask.
        model_valid[i] = True

    return PhotometricModelGrid(
        samples=samples,
        flux=model_flux,
        log_prior=log_prior,
        valid=model_valid & np.isfinite(log_prior),
        parameter_names=names,
        band_names=band_names,
        mass_normalization=mass_norm,
        mass_reference=mass_reference,
        flux_unit="maggies",
        meta={
            "schema": "composed.photometric_model_grid.v3",
            "excluded_parameters": excluded,
            "excluded_parameter_priors": {
                name: _stable_value(parameter_space.priors[name])
                for name in excluded
                if name in parameter_space.priors
            },
            "scientific_specification": {
                "backend": _backend_configuration(backend),
                "parameters": list(parameter_space.names),
                "priors": {
                    name: _stable_value(parameter_space.priors[name])
                    for name in parameter_space.names
                },
                "filters": _filter_specification(filters),
                "band_names": list(band_names),
            },
            "units": "maggies",
            "mass_scale_applied": False,
            "mass_reference": getattr(mass_reference, "value", None),
            "mass_convention": MASS_CONVENTION_SCHEMA,
        },
    )


def evaluate_catalog_model_grid_likelihood(
    model_grid: PhotometricModelGrid,
    datasets: Sequence[SEDDataset],
    *,
    sigma_floor: float | None = None,
    model_discrepancy: float = 0.0,
    log10_mass_grid: Sequence[float] | None = None,
    log10_mass_bounds: tuple[float, float] | None = None,
    log10_mass_prior: Prior | None = None,
    model_chunk_size: int = 2048,
    object_chunk_size: int = 512,
    mass_chunk_size: int = 128,
    store_mass_posterior: bool = False,
) -> CatalogProfileGridResult:
    """Evaluate a cached mass-normalized model grid against a catalog.

    The model grid is computed once; this function does only likelihood work.
    ``model_discrepancy`` is the same dimensionless ``eta`` used by
    :class:`GaussianPhotometricLikelihood`; it is combined with the raw
    catalog sigma from each dataset after mass scaling. Non-zero eta requires
    an explicit mass grid because the usual analytic amplitude is then
    invalid.
    For detections, the profile mass scale is the usual weighted least-squares
    normalization

    ``A_hat = sum(f_obs f_model / sigma^2) / sum(f_model^2 / sigma^2)``.

    The analytic normalization is used only for detection-only catalogs. If
    any object contains an upper limit, ``log10_mass_grid`` is required and the
    complete censored likelihood is maximized over that grid. The same grid is
    also used for mass marginalization. In that case ``log10_mass_prior`` must
    be a continuous :class:`~composed.priors.Prior`; its density is multiplied
    by the integration-cell width on the (possibly irregular) mass grid.
    """

    if model_grid.mass_normalization != MassNormalization.PER_SOLAR_MASS:
        raise ValueError(
            "evaluate_catalog_model_grid_likelihood expects a PER_SOLAR_MASS model grid. "
            f"Got {model_grid.mass_normalization}."
        )
    validate_mass_reference(model_grid.mass_normalization, model_grid.mass_reference)
    if model_chunk_size <= 0 or object_chunk_size <= 0 or mass_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive.")
    if sigma_floor is not None and float(sigma_floor) < 0.0:
        raise ValueError("sigma_floor must be non-negative.")
    model_discrepancy = _validate_model_discrepancy(model_discrepancy)

    datasets = tuple(datasets)
    band_names, data_flux, data_sigma, active_mask, upper_limit, upper_limit_mask = _stack_catalog_arrays(
        datasets, sigma_floor=sigma_floor, target_flux_unit=model_grid.flux_unit
    )
    if tuple(model_grid.band_names) != band_names:
        raise ValueError(
            "Model grid band order does not match catalog band order: "
            f"grid={tuple(model_grid.band_names)}, catalog={band_names}."
        )

    try:
        declared_mass_prior = _declared_excluded_prior(model_grid, "log10_mass")
    except ValueError:
        # A custom Prior may be scientifically valid but not reconstructable
        # from generic JSON metadata. A fully explicit quadrature and Prior is
        # still auditable and must remain usable.
        if log10_mass_grid is None or log10_mass_prior is None:
            raise
        declared_mass_prior = None
    log10_mass_grid, log10_mass_bounds, log10_mass_prior = _apply_declared_mass_prior(
        declared_prior=declared_mass_prior,
        log10_mass_grid=log10_mass_grid,
        log10_mass_bounds=log10_mass_bounds,
        log10_mass_prior=log10_mass_prior,
    )
    log10_mass_grid_arr, log_mass_prior_weights, mass_bounds, mass_prior_meta = _prepare_mass_grid_and_prior(
        log10_mass_grid=log10_mass_grid,
        log10_mass_bounds=log10_mass_bounds,
        log10_mass_prior=log10_mass_prior,
    )
    if mass_prior_meta is None and declared_mass_prior is not None:
        mass_prior_meta = {
            "type": type(declared_mass_prior).__name__,
            "repr": repr(declared_mass_prior),
            "profile_bounds": mass_bounds,
            "handling": "analytic profile constrained to declared flat log10_mass support",
        }
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
        log10_mass_grid=log10_mass_grid_arr,
        mass_chunk_size=int(mass_chunk_size),
        require_finite=log10_mass_grid_arr is None,
        model_discrepancy=model_discrepancy,
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
            model_discrepancy=model_discrepancy,
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
        mass_profile_at_boundary=profile["mass_profile_at_boundary"],
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
            "model_discrepancy": model_discrepancy,
            "log10_mass_bounds": mass_bounds,
            "mass_prior": mass_prior_meta,
            "mass_reference": model_grid.mass_reference.value,
            "mass_convention": MASS_CONVENTION_SCHEMA,
        },
    )


def save_photometric_model_grid(grid: PhotometricModelGrid, path: str | Path) -> None:
    """Save a :class:`PhotometricModelGrid` for later catalog evaluation."""

    path = Path(path)
    save_npz_with_provenance(
        path,
        compressed=True,
        provenance_paths=grid.meta.get("provenance_paths"),
        extra={
            "artifact_type": "PhotometricModelGrid",
            "mass_convention": MASS_CONVENTION_SCHEMA,
            "grid_meta": grid.meta,
        },
        samples=np.asarray(grid.samples, dtype=float),
        flux=np.asarray(grid.flux, dtype=float),
        log_prior=np.asarray(grid.log_prior, dtype=float),
        valid=np.asarray(grid.valid, dtype=bool),
        parameter_names=np.asarray(grid.parameter_names, dtype=object),
        band_names=np.asarray(grid.band_names, dtype=object),
        mass_normalization=np.asarray(grid.mass_normalization.value, dtype=object),
        mass_reference=np.asarray(
            "" if grid.mass_reference is None else grid.mass_reference.value,
            dtype=object,
        ),
        flux_unit=np.asarray(grid.flux_unit, dtype=object),
        meta=np.asarray(json.dumps(grid.meta, sort_keys=True, default=str), dtype=object),
    )


def load_photometric_model_grid(
    path: str | Path,
    *,
    require_provenance_sidecar: bool = True,
) -> PhotometricModelGrid:
    """Load a model grid saved by :func:`save_photometric_model_grid`.

    Provenance and the archive content hash are verified by default. Set
    ``require_provenance_sidecar=False`` only to inspect a legacy grid.
    """

    path = Path(path)
    provenance = None
    if require_provenance_sidecar:
        provenance = require_provenance(path)
    elif provenance_path_for(path).exists():
        provenance = read_provenance(provenance_path_for(path))
    data = np.load(path, allow_pickle=True)
    if "mass_reference" not in data.files:
        raise ValueError(
            "Legacy photometric model grid has no mass_reference. Rebuild it with this "
            "CompoSED version; older grids were normalized by formed mass."
        )
    meta = json.loads(str(data["meta"].item())) if "meta" in data.files else {}
    if meta.get("schema") != "composed.photometric_model_grid.v3":
        raise ValueError(
            "Photometric model grid does not use the current per-object mask semantics. "
            "Rebuild it before scientific reuse."
        )
    if provenance is not None:
        meta["provenance"] = provenance
    return PhotometricModelGrid(
        samples=np.asarray(data["samples"], dtype=float),
        flux=np.asarray(data["flux"], dtype=float),
        log_prior=np.asarray(data["log_prior"], dtype=float),
        valid=np.asarray(data["valid"], dtype=bool),
        parameter_names=tuple(str(name) for name in data["parameter_names"]),
        band_names=tuple(str(name) for name in data["band_names"]),
        mass_normalization=MassNormalization(str(data["mass_normalization"].item())),
        mass_reference=(
            None
            if str(data["mass_reference"].item()) == ""
            else MassReference(str(data["mass_reference"].item()))
        ),
        flux_unit=(str(data["flux_unit"].item()) if "flux_unit" in data.files else "maggies"),
        meta=meta,
    )


def run_photometric_grid_catalog(
    backend,
    datasets: Sequence[SEDDataset],
    parameter_space,
    filters=None,
    *,
    sigma_floor: float | None = None,
    model_discrepancy: float = 0.0,
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
    determinant. ``model_discrepancy`` applies the same
    ``(eta * f_model)**2`` variance term as the scalar likelihood, including
    its theta-dependent log determinant and censored upper-limit terms.
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
    model_discrepancy = _validate_model_discrepancy(model_discrepancy)

    band_names, data_flux, data_sigma, active_mask, upper_limit, upper_limit_mask = _stack_catalog_arrays(
        datasets, sigma_floor=sigma_floor, target_flux_unit="maggies"
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
        model_discrepancy=model_discrepancy,
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
        "model_discrepancy": model_discrepancy,
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
    target_flux_unit: str = "maggies",
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    band_names = tuple(datasets[0].band_names)
    source_flux_unit = datasets[0].flux_unit
    target_flux_unit = canonical_photometric_flux_unit(target_flux_unit)
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
        if dataset.flux_unit != source_flux_unit:
            raise ValueError(
                "All catalog datasets must use the same flux_unit before stacking; "
                f"object 0 uses {source_flux_unit!r}, object {i} uses {dataset.flux_unit!r}."
            )
        mask = np.asarray(dataset.active_mask, dtype=bool)
        if not np.any(mask):
            raise ValueError(f"Object {i} has no active bands.")
        flux = convert_photometric_flux(dataset.flux, source_flux_unit, target_flux_unit)
        sigma = convert_photometric_flux(dataset.sigma, source_flux_unit, target_flux_unit)
        if sigma_floor is not None:
            floor = float(convert_photometric_flux(sigma_floor, source_flux_unit, target_flux_unit))
            sigma = np.sqrt(sigma**2 + floor**2)
        upper_mask = mask & np.asarray(dataset.upper_limit_mask, dtype=bool)
        detection_mask = mask & ~upper_mask
        flux_rows.append(np.where(detection_mask, flux, 0.0))
        sigma_rows.append(np.where(mask, sigma, 1.0))
        mask_rows.append(mask)
        upper = convert_photometric_flux(dataset.upper_limit, source_flux_unit, target_flux_unit)
        upper_limit_rows.append(np.where(upper_mask, upper, 0.0))
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
            aligned = convert_photometric_flux(
                _align_model_flux(model, band_names),
                getattr(model, "flux_unit", "maggies"),
                "maggies",
            )
        except (ModelDomainError, FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        model_flux[i] = mass_scale * aligned
        # As in the scalar likelihood, a non-finite value matters only when
        # that band is active for the object being evaluated.
        model_valid[i] = True
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
    model_discrepancy=0.0,
) -> np.ndarray:
    n_objects = data_flux.shape[0]
    n_grid = model_flux.shape[0]
    logp = np.full((n_objects, n_grid), -np.inf, dtype=float)
    detection_mask = active_mask & ~upper_limit_mask
    model_discrepancy = _validate_model_discrepancy(model_discrepancy)

    for g0 in range(0, n_grid, model_chunk_size):
        g1 = min(g0 + model_chunk_size, n_grid)
        local_valid = valid_grid[g0:g1]
        if not np.any(local_valid):
            continue
        grid_indices = np.arange(g0, g1)[local_valid]
        model = model_flux[grid_indices]
        for o0 in range(0, n_objects, object_chunk_size):
            o1 = min(o0 + object_chunk_size, n_objects)
            local_active = active_mask[o0:o1]
            object_model_valid = _object_model_finite(local_active, model)
            model_for_sigma = np.where(
                local_active[:, None, :],
                model[None, :, :],
                0.0,
            )
            sigma_eff = np.sqrt(
                data_sigma[o0:o1, None, :] ** 2
                + (model_discrepancy * model_for_sigma) ** 2
            )
            local_detection = detection_mask[o0:o1, None, :]
            diff = np.where(
                local_detection,
                data_flux[o0:o1, None, :] - model[None, :, :],
                0.0,
            )
            chi2 = np.sum(
                np.where(local_detection, diff**2 / sigma_eff**2, 0.0),
                axis=2,
            )
            logdet = np.sum(
                np.where(local_detection, np.log(2.0 * np.pi * sigma_eff**2), 0.0),
                axis=2,
            )
            log_like = -0.5 * (chi2 + logdet)
            local_upper_mask = upper_limit_mask[o0:o1]
            if np.any(local_upper_mask):
                z = np.where(
                    local_upper_mask[:, None, :],
                    (upper_limit[o0:o1, None, :] - model[None, :, :])
                    / sigma_eff,
                    0.0,
                )
                log_like += np.sum(np.where(local_upper_mask[:, None, :], _normal_logcdf(z), 0.0), axis=2)
            values = log_prior[grid_indices][None, :] + log_like
            logp[o0:o1, grid_indices] = np.where(object_model_valid, values, -np.inf)

    if not np.all(np.any(np.isfinite(logp), axis=1)):
        bad = np.where(~np.any(np.isfinite(logp), axis=1))[0]
        raise RuntimeError(f"No finite grid point for catalog object(s): {bad.tolist()}")
    return logp


def _object_model_finite(active_mask: np.ndarray, model_flux: np.ndarray) -> np.ndarray:
    """Return finite-model validity for every object/model pair.

    ``active_mask`` has shape ``(n_object, n_band)`` and ``model_flux`` has
    shape ``(n_model, n_band)``. A non-finite model value invalidates only
    objects that actually use that band, matching the scalar likelihood.
    """

    active_mask = np.asarray(active_mask, dtype=bool)
    model_flux = np.asarray(model_flux, dtype=float)
    return np.all(
        (~active_mask[:, None, :]) | np.isfinite(model_flux)[None, :, :],
        axis=2,
    )


def _validate_model_discrepancy(value: float) -> float:
    """Validate the dimensionless catalog likelihood discrepancy ``eta``."""

    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("model_discrepancy must be finite and non-negative.")
    return value


def _prepare_mass_grid_and_prior(*, log10_mass_grid, log10_mass_bounds, log10_mass_prior):
    """Prepare a continuous mass-prior quadrature on an explicit grid.

    The supplied grid is only a numerical integration grid. Its points are not
    interpreted as equally probable discrete choices. The prior density is
    evaluated at each point and multiplied by the width of that point's
    midpoint cell before normalization.
    """

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
        return None, None, bounds, None

    grid = np.asarray(log10_mass_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("log10_mass_grid must be a one-dimensional array with at least two points.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("log10_mass_grid must be finite.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("log10_mass_grid must be strictly increasing.")
    if log10_mass_prior is None:
        raise ValueError(
            "log10_mass_prior must be a continuous Prior when log10_mass_grid is supplied. "
            "The numerical grid does not define a prior."
        )
    if not isinstance(log10_mass_prior, Prior):
        raise TypeError(
            "log10_mass_prior must be a composed.priors.Prior instance, not an array of weights."
        )
    if isinstance(log10_mass_prior, (ChoicePrior, IntegerUniformPrior, DeltaPrior)):
        raise TypeError(
            "Cached mass marginalization requires a continuous log10_mass_prior. "
            "Put discrete mass choices in the full ParameterSpace grid instead."
        )

    prior_support = _finite_prior_support(log10_mass_prior)
    if prior_support is not None:
        if bounds is None:
            bounds = prior_support
        elif not np.allclose(bounds, prior_support, rtol=0.0, atol=1e-12):
            raise ValueError(
                "log10_mass_bounds would truncate or extend the declared bounded mass prior. "
                "Declare the intended bounds on the Prior itself."
            )
    elif bounds is None:
        raise ValueError(
            "An unbounded log10_mass_prior requires explicit finite log10_mass_bounds "
            "for numerical marginalization."
        )

    if not np.isclose(grid[0], bounds[0], rtol=0.0, atol=1e-12) or not np.isclose(
        grid[-1], bounds[1], rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "log10_mass_grid endpoints must equal the mass integration bounds. "
            f"Got grid [{grid[0]}, {grid[-1]}] and bounds {bounds}."
        )

    log_density = np.asarray([log10_mass_prior.logpdf(value) for value in grid], dtype=float)
    cell_widths = _grid_cell_widths(grid)
    log_unnormalized = log_density + np.log(cell_widths)
    normalization = _logsumexp(log_unnormalized)
    if not np.isfinite(normalization):
        raise ValueError("The declared log10_mass_prior has no finite density on log10_mass_grid.")
    log_weights = log_unnormalized - normalization
    prior_meta = {
        "type": type(log10_mass_prior).__name__,
        "repr": repr(log10_mass_prior),
        "integration_bounds": tuple(float(value) for value in bounds),
        "quadrature": "prior density times midpoint-cell width",
        "normalized_weights": np.exp(log_weights),
    }
    return grid, log_weights, bounds, prior_meta


def _declared_excluded_prior(model_grid: PhotometricModelGrid, name: str) -> Prior | None:
    """Recover an excluded parameter prior recorded when the grid was built."""

    excluded = tuple(model_grid.meta.get("excluded_parameters", ()))
    if name not in excluded:
        return None
    prior_spec = model_grid.meta.get("excluded_parameter_priors", {}).get(name)
    if prior_spec is None:
        prior_spec = (
            model_grid.meta.get("scientific_specification", {})
            .get("priors", {})
            .get(name)
        )
    if prior_spec is None:
        return None
    return _prior_from_stable_specification(prior_spec, parameter_name=name)


def _apply_declared_mass_prior(
    *,
    declared_prior: Prior | None,
    log10_mass_grid,
    log10_mass_bounds,
    log10_mass_prior,
):
    """Make the cached grid's declared mass prior the default scientific contract."""

    if declared_prior is None:
        return log10_mass_grid, log10_mass_bounds, log10_mass_prior

    if log10_mass_grid is not None:
        # An explicit prior remains a deliberate override, allowing one expensive
        # model grid to be reused for sensitivity tests. Otherwise the declared
        # ParameterSpace prior is authoritative.
        if log10_mass_prior is None:
            log10_mass_prior = declared_prior
        return log10_mass_grid, log10_mass_bounds, log10_mass_prior

    if log10_mass_prior is not None:
        raise ValueError("log10_mass_prior requires log10_mass_grid.")
    if not isinstance(declared_prior, UniformPrior):
        raise ValueError(
            "Analytic cached-grid mass profiling is valid only for a declared "
            "UniformPrior in log10_mass. The cached grid records "
            f"{type(declared_prior).__name__}; supply log10_mass_grid to perform "
            "numerical mass marginalization with that prior."
        )

    declared_bounds = (float(declared_prior.low), float(declared_prior.high))
    if log10_mass_bounds is None:
        log10_mass_bounds = declared_bounds
    else:
        supplied = tuple(float(value) for value in log10_mass_bounds)
        if not np.allclose(supplied, declared_bounds, rtol=0.0, atol=1e-12):
            raise ValueError(
                "log10_mass_bounds do not match the UniformPrior declared when the "
                f"cached grid was built: supplied={supplied}, declared={declared_bounds}."
            )
    return log10_mass_grid, log10_mass_bounds, log10_mass_prior


def _prior_from_stable_specification(specification, *, parameter_name: str) -> Prior:
    """Reconstruct a built-in scalar prior from deterministic grid metadata."""

    if not isinstance(specification, dict):
        raise ValueError(f"Cached prior metadata for {parameter_name!r} is malformed.")
    prior_type = str(specification.get("type", "")).rsplit(".", 1)[-1]
    configuration = specification.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Cached prior metadata for {parameter_name!r} has no configuration.")
    prior_classes = {
        "UniformPrior": UniformPrior,
        "NormalPrior": NormalPrior,
        "StudentTPrior": StudentTPrior,
        "LogUniformPrior": LogUniformPrior,
        "IntegerUniformPrior": IntegerUniformPrior,
        "ChoicePrior": ChoicePrior,
        "DeltaPrior": DeltaPrior,
    }
    prior_class = prior_classes.get(prior_type)
    if prior_class is None:
        raise ValueError(
            f"Cached prior type {prior_type!r} for excluded parameter {parameter_name!r} "
            "cannot be reconstructed. Supply an explicit numerical grid and Prior."
        )
    try:
        return prior_class(**configuration)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cached prior metadata for excluded parameter {parameter_name!r} is invalid."
        ) from exc


def _finite_prior_support(prior: Prior) -> tuple[float, float] | None:
    """Return finite support for bounded continuous priors."""

    if isinstance(prior, (UniformPrior, LogUniformPrior)):
        return float(prior.low), float(prior.high)
    return None


def _grid_cell_widths(grid: np.ndarray) -> np.ndarray:
    """Widths of midpoint cells spanning ``[grid[0], grid[-1]]``."""

    grid = np.asarray(grid, dtype=float)
    edges = np.empty(grid.size + 1, dtype=float)
    edges[0] = grid[0]
    edges[-1] = grid[-1]
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    widths = np.diff(edges)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("log10_mass_grid does not define positive integration cells.")
    return widths


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
    log10_mass_grid=None,
    mass_chunk_size=128,
    require_finite=True,
    model_discrepancy=0.0,
):
    model_discrepancy = _validate_model_discrepancy(model_discrepancy)
    if np.any(upper_limit_mask) or model_discrepancy > 0.0:
        if log10_mass_grid is None:
            raise ValueError(
                "Mass profiling with upper limits or non-zero model_discrepancy requires an explicit "
                "log10_mass_grid because the detection-only analytic normalization is no longer the "
                "likelihood optimum."
            )
        return _catalog_grid_profile_mass_logp(
            data_flux=data_flux,
            data_sigma=data_sigma,
            active_mask=active_mask,
            upper_limit=upper_limit,
            upper_limit_mask=upper_limit_mask,
            model_flux=model_flux,
            log_prior=log_prior,
            valid_grid=valid_grid,
            model_chunk_size=model_chunk_size,
            object_chunk_size=object_chunk_size,
            log10_mass_grid=np.asarray(log10_mass_grid, dtype=float),
            mass_chunk_size=int(mass_chunk_size),
            model_discrepancy=model_discrepancy,
        )

    n_objects = data_flux.shape[0]
    n_grid = model_flux.shape[0]
    profile_logp = np.full((n_objects, n_grid), -np.inf, dtype=float)
    mass_scale = np.full((n_objects, n_grid), np.nan, dtype=float)
    log10_mass = np.full((n_objects, n_grid), np.nan, dtype=float)
    mass_at_boundary = np.zeros((n_objects, n_grid), dtype=bool)
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
            local_active = active_mask[o0:o1]
            object_model_valid = _object_model_finite(local_active, model)
            local_detection = detection_mask[o0:o1, None, :]
            model_detection = np.where(local_detection, model[None, :, :], 0.0)
            numerator = np.sum(
                data_flux[o0:o1, None, :] * model_detection * inv_sigma2[o0:o1, None, :],
                axis=2,
            )
            denominator = np.sum(
                model_detection**2 * inv_sigma2[o0:o1, None, :],
                axis=2,
            )
            unconstrained_scale = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0.0,
            )
            scale = _clip_mass_scale(unconstrained_scale, log10_mass_bounds)
            at_boundary = _mass_scale_boundary_mask(unconstrained_scale, log10_mass_bounds)
            model_scaled = scale[:, :, None] * model_detection
            diff = np.where(
                local_detection,
                data_flux[o0:o1, None, :] - model_scaled,
                0.0,
            )
            chi2 = np.sum(diff**2 * inv_sigma2[o0:o1, None, :], axis=2)
            log_like = -0.5 * (chi2 + logdet[o0:o1, None])
            finite_scale = np.isfinite(scale) & (scale > 0.0) & object_model_valid
            values = log_prior[grid_indices][None, :] + log_like
            values = np.where(finite_scale, values, -np.inf)
            profile_logp[o0:o1, grid_indices] = values
            mass_scale[o0:o1, grid_indices] = np.where(finite_scale, scale, np.nan)
            log10_mass[o0:o1, grid_indices] = np.where(finite_scale, np.log10(scale), np.nan)
            mass_at_boundary[o0:o1, grid_indices] = at_boundary & finite_scale

    if require_finite and not np.all(np.any(np.isfinite(profile_logp), axis=1)):
        bad = np.where(~np.any(np.isfinite(profile_logp), axis=1))[0]
        if log10_mass_bounds is None:
            raise RuntimeError(
                "No finite positive analytic mass normalization for catalog object(s) "
                f"{bad.tolist()}. A non-positive unconstrained amplitude is not a physical "
                "stellar-mass estimate. Supply explicit log10_mass_bounds to report a "
                "flagged boundary solution, or use an explicit log10_mass_grid."
            )
        raise RuntimeError(f"No finite profiled grid point for catalog object(s): {bad.tolist()}")
    return {
        "profile_logp": profile_logp,
        "mass_scale_profile": mass_scale,
        "log10_mass_profile": log10_mass,
        "mass_profile_at_boundary": mass_at_boundary,
    }


def _catalog_grid_profile_mass_logp(
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
    log10_mass_grid,
    mass_chunk_size,
    model_discrepancy=0.0,
):
    """Maximize the complete detection+censoring likelihood over mass."""

    n_objects = data_flux.shape[0]
    n_grid = model_flux.shape[0]
    profile_logp = np.full((n_objects, n_grid), -np.inf, dtype=float)
    mass_scale = np.full((n_objects, n_grid), np.nan, dtype=float)
    log10_mass = np.full((n_objects, n_grid), np.nan, dtype=float)
    mass_at_boundary = np.zeros((n_objects, n_grid), dtype=bool)
    detection_mask = active_mask & ~upper_limit_mask
    model_discrepancy = _validate_model_discrepancy(model_discrepancy)
    scales = 10.0 ** np.asarray(log10_mass_grid, dtype=float)

    for g0 in range(0, n_grid, model_chunk_size):
        g1 = min(g0 + model_chunk_size, n_grid)
        local_valid = valid_grid[g0:g1]
        if not np.any(local_valid):
            continue
        grid_indices = np.arange(g0, g1)[local_valid]
        model = model_flux[grid_indices]
        for o0 in range(0, n_objects, object_chunk_size):
            o1 = min(o0 + object_chunk_size, n_objects)
            object_model_valid = _object_model_finite(active_mask[o0:o1], model)
            best_values = np.full((o1 - o0, grid_indices.size), -np.inf, dtype=float)
            best_mass_index = np.zeros((o1 - o0, grid_indices.size), dtype=int)
            for m0 in range(0, scales.size, mass_chunk_size):
                m1 = min(m0 + mass_chunk_size, scales.size)
                scaled = model[:, None, :] * scales[None, m0:m1, None]
                scaled_for_sigma = np.where(
                    active_mask[o0:o1, None, None, :],
                    scaled[None, :, :, :],
                    0.0,
                )
                sigma_eff = np.sqrt(
                    data_sigma[o0:o1, None, None, :] ** 2
                    + (model_discrepancy * scaled_for_sigma) ** 2
                )
                local_detection = detection_mask[o0:o1, None, None, :]
                diff = np.where(
                    local_detection,
                    data_flux[o0:o1, None, None, :] - scaled[None, :, :, :],
                    0.0,
                )
                chi2 = np.sum(
                    np.where(local_detection, diff**2 / sigma_eff**2, 0.0),
                    axis=3,
                )
                logdet = np.sum(
                    np.where(local_detection, np.log(2.0 * np.pi * sigma_eff**2), 0.0),
                    axis=3,
                )
                values = -0.5 * (chi2 + logdet)
                local_upper_mask = upper_limit_mask[o0:o1]
                if np.any(local_upper_mask):
                    z = np.where(
                        local_upper_mask[:, None, None, :],
                        (upper_limit[o0:o1, None, None, :] - scaled[None, :, :, :])
                        / sigma_eff,
                        0.0,
                    )
                    values += np.sum(
                        np.where(local_upper_mask[:, None, None, :], _normal_logcdf(z), 0.0),
                        axis=3,
                    )
                values += log_prior[grid_indices][None, :, None]
                values = np.where(object_model_valid[:, :, None], values, -np.inf)
                local_index = np.argmax(values, axis=2)
                local_best = np.take_along_axis(values, local_index[:, :, None], axis=2)[:, :, 0]
                improve = local_best > best_values
                best_values = np.where(improve, local_best, best_values)
                best_mass_index = np.where(improve, m0 + local_index, best_mass_index)

            profile_logp[o0:o1, grid_indices] = best_values
            finite_best = np.isfinite(best_values)
            log10_mass[o0:o1, grid_indices] = np.where(
                finite_best,
                log10_mass_grid[best_mass_index],
                np.nan,
            )
            mass_scale[o0:o1, grid_indices] = np.where(
                finite_best,
                scales[best_mass_index],
                np.nan,
            )
            mass_at_boundary[o0:o1, grid_indices] = finite_best & (
                (best_mass_index == 0) | (best_mass_index == scales.size - 1)
            )

    if not np.all(np.any(np.isfinite(profile_logp), axis=1)):
        bad = np.where(~np.any(np.isfinite(profile_logp), axis=1))[0]
        raise RuntimeError(f"No finite censored mass-profile grid point for catalog object(s): {bad.tolist()}")
    return {
        "profile_logp": profile_logp,
        "mass_scale_profile": mass_scale,
        "log10_mass_profile": log10_mass,
        "mass_profile_at_boundary": mass_at_boundary,
    }


def _clip_mass_scale(scale: np.ndarray, log10_mass_bounds: tuple[float, float] | None) -> np.ndarray:
    scale = np.asarray(scale, dtype=float)
    if log10_mass_bounds is None:
        return np.where(np.isfinite(scale) & (scale > 0.0), scale, np.nan)
    lo, hi = log10_mass_bounds
    return np.where(np.isfinite(scale), np.clip(scale, 10.0**lo, 10.0**hi), np.nan)


def _mass_scale_boundary_mask(
    unconstrained_scale: np.ndarray,
    log10_mass_bounds: tuple[float, float] | None,
) -> np.ndarray:
    """Flag analytic optima clipped to an explicitly declared mass boundary."""

    scale = np.asarray(unconstrained_scale, dtype=float)
    if log10_mass_bounds is None:
        return np.zeros(scale.shape, dtype=bool)
    lo, hi = log10_mass_bounds
    return np.isfinite(scale) & ((scale <= 10.0**lo) | (scale >= 10.0**hi))


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
    model_discrepancy=0.0,
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
    model_discrepancy = _validate_model_discrepancy(model_discrepancy)
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
            object_model_valid = _object_model_finite(active_mask[o0:o1], model)
            local_mass_logp = np.full((o1 - o0, grid_indices.size, n_mass), -np.inf, dtype=float)
            for m0 in range(0, n_mass, mass_chunk_size):
                m1 = min(m0 + mass_chunk_size, n_mass)
                scaled = mass_scales[m0:m1][None, :, None] * model[:, None, :]
                scaled_for_sigma = np.where(
                    active_mask[o0:o1, None, None, :],
                    scaled[None, :, :, :],
                    0.0,
                )
                sigma_eff = np.sqrt(
                    data_sigma[o0:o1, None, None, :] ** 2
                    + (model_discrepancy * scaled_for_sigma) ** 2
                )
                local_detection = detection_mask[o0:o1, None, None, :]
                diff = np.where(
                    local_detection,
                    data_flux[o0:o1, None, None, :] - scaled[None, :, :, :],
                    0.0,
                )
                chi2 = np.sum(
                    np.where(local_detection, diff**2 / sigma_eff**2, 0.0),
                    axis=3,
                )
                logdet = np.sum(
                    np.where(local_detection, np.log(2.0 * np.pi * sigma_eff**2), 0.0),
                    axis=3,
                )
                log_like = -0.5 * (chi2 + logdet)
                local_upper_mask = upper_limit_mask[o0:o1]
                if np.any(local_upper_mask):
                    z = np.where(
                        local_upper_mask[:, None, None, :],
                        (upper_limit[o0:o1, None, None, :] - scaled[None, :, :, :])
                        / sigma_eff,
                        0.0,
                    )
                    log_like += np.sum(
                        np.where(local_upper_mask[:, None, None, :], _normal_logcdf(z), 0.0),
                        axis=3,
                    )
                values = (
                    log_prior[grid_indices][None, :, None]
                    + log_like
                    + log_mass_prior_weights[None, None, m0:m1]
                )
                local_mass_logp[:, :, m0:m1] = np.where(
                    object_model_valid[:, :, None],
                    values,
                    -np.inf,
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


def _weighted_quantile_grid(grid: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Quantiles of a piecewise-constant density over grid midpoint cells."""

    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if grid.ndim != 1 or weights.shape != grid.shape:
        raise ValueError("grid and weights must be one-dimensional arrays with matching shape.")
    if not np.isfinite(weights).all() or np.sum(weights) <= 0.0:
        return np.full(len(tuple(quantiles)), np.nan, dtype=float)
    quantiles = np.asarray(tuple(quantiles), dtype=float)
    if np.any(~np.isfinite(quantiles)) or np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("quantiles must lie in [0, 1].")

    weights = weights / np.sum(weights)
    edges = np.empty(grid.size + 1, dtype=float)
    edges[0] = grid[0]
    edges[-1] = grid[-1]
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    cdf_edges = np.concatenate(([0.0], np.cumsum(weights)))

    output = np.empty(quantiles.size, dtype=float)
    for i, quantile in enumerate(quantiles):
        if quantile <= 0.0:
            output[i] = edges[0]
            continue
        if quantile >= 1.0:
            output[i] = edges[-1]
            continue
        cell = min(int(np.searchsorted(cdf_edges, quantile, side="right") - 1), weights.size - 1)
        if weights[cell] <= 0.0:
            output[i] = edges[cell]
            continue
        fraction = (quantile - cdf_edges[cell]) / weights[cell]
        output[i] = edges[cell] + fraction * (edges[cell + 1] - edges[cell])
    return output


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
