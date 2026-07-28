from __future__ import annotations

import ast
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from composed.backends.base import ModelSpectrum
from composed.catalog import (
    PhotometricModelGrid,
    _finite_grid_theta_excluding,
    _normalize_logp_rows,
    evaluate_catalog_model_grid_likelihood,
)
from composed.data import SEDDataset
from composed.filters import FilterSet
from composed.priors import Prior
from composed.provenance import provenance_path_for, read_provenance, require_provenance, save_npz_with_provenance
from composed.units import (
    MASS_CONVENTION_SCHEMA,
    MassNormalization,
    MassReference,
    backend_mass_reference,
    validate_mass_reference,
)

MJY_PER_MAGGIE = 3631.0e3
AB_ZERO_FNU_W_M2_HZ = 3631.0e-26
C_NM_PER_S = 299_792_458.0e9
PARSEC_M = 3.085677581491367e16


class ExperimentalFastCatalogWarning(UserWarning):
    """Warning emitted by the restricted fast rest-frame catalog path."""


def _warn_experimental_fast_catalog() -> None:
    warnings.warn(
        "Fast rest-frame catalog projection is experimental in CompoSED 0.1.1. "
        "It currently supports only backends that explicitly declare a "
        "redshift-independent rest-spectrum capability, and it requires every "
        "requested filter to be covered by the rest wavelength grid.",
        ExperimentalFastCatalogWarning,
        stacklevel=3,
    )


@dataclass
class RestFrameSpectralGrid:
    """Mass-normalized rest-frame spectral model grid.

    ``luminosity_w_per_nm`` has shape ``(n_models, n_wave)``.  Each row is a
    rest-frame luminosity-density spectrum before cosmological redshifting and
    IGM attenuation.  The expected unit is W/nm per declared mass
    normalization, normally one solar mass of surviving stars.
    """

    wavelength_nm: np.ndarray
    luminosity_w_per_nm: np.ndarray
    samples: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    parameter_names: tuple[str, ...]
    mass_normalization: MassNormalization
    mass_reference: MassReference | None = MassReference.SURVIVING_STELLAR_MASS
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mass_normalization = MassNormalization(self.mass_normalization)
        self.mass_reference = validate_mass_reference(self.mass_normalization, self.mass_reference)

    def save(self, path: str | Path) -> None:
        """Save the rest-frame grid to a NumPy ``.npz`` file."""

        save_restframe_spectral_grid(self, path)


def save_restframe_spectral_grid(grid: RestFrameSpectralGrid, path: str | Path) -> None:
    """Save a rest-frame spectral grid for later catalog fitting.

    The saved arrays are the expensive forward-model product.  Loading this
    file later lets users change the catalog, masks, redshifts, filters, and
    mass prior without re-running the backend forward model, provided the
    physical model grid itself is unchanged.
    """

    path = Path(path)
    save_npz_with_provenance(
        path,
        compressed=True,
        provenance_paths=grid.meta.get("provenance_paths"),
        extra={
            "artifact_type": "RestFrameSpectralGrid",
            "mass_convention": MASS_CONVENTION_SCHEMA,
            "grid_meta": grid.meta,
        },
        wavelength_nm=np.asarray(grid.wavelength_nm, dtype=float),
        luminosity_w_per_nm=np.asarray(grid.luminosity_w_per_nm, dtype=float),
        samples=np.asarray(grid.samples, dtype=float),
        log_prior=np.asarray(grid.log_prior, dtype=float),
        valid=np.asarray(grid.valid, dtype=bool),
        parameter_names=np.asarray(grid.parameter_names, dtype=object),
        mass_normalization=np.asarray(grid.mass_normalization.value, dtype=object),
        mass_reference=np.asarray(
            "" if grid.mass_reference is None else grid.mass_reference.value,
            dtype=object,
        ),
        meta=np.asarray(json.dumps(grid.meta, sort_keys=True, default=str), dtype=object),
    )


def load_restframe_spectral_grid(
    path: str | Path,
    *,
    require_provenance_sidecar: bool = False,
) -> RestFrameSpectralGrid:
    """Load a rest-frame spectral grid saved by ``save_restframe_spectral_grid``."""

    path = Path(path)
    provenance = None
    if require_provenance_sidecar:
        provenance = require_provenance(path)
    elif provenance_path_for(path).exists():
        provenance = read_provenance(provenance_path_for(path))
    data = np.load(path, allow_pickle=True)
    if "mass_reference" not in data.files:
        raise ValueError(
            "Legacy rest-frame spectral grid has no mass_reference. Rebuild it with this "
            "CompoSED version; older grids were normalized by formed mass."
        )
    meta = _decode_saved_meta(data["meta"].item()) if "meta" in data.files else {}
    if provenance is not None:
        meta["provenance"] = provenance
    return RestFrameSpectralGrid(
        wavelength_nm=np.asarray(data["wavelength_nm"], dtype=float),
        luminosity_w_per_nm=np.asarray(data["luminosity_w_per_nm"], dtype=float),
        samples=np.asarray(data["samples"], dtype=float),
        log_prior=np.asarray(data["log_prior"], dtype=float),
        valid=np.asarray(data["valid"], dtype=bool),
        parameter_names=tuple(str(name) for name in data["parameter_names"]),
        mass_normalization=MassNormalization(str(data["mass_normalization"].item())),
        mass_reference=(
            None
            if str(data["mass_reference"].item()) == ""
            else MassReference(str(data["mass_reference"].item()))
        ),
        meta=meta,
    )


@dataclass(frozen=True)
class RedshiftFilterOperator:
    """Linear map from rest-frame luminosity density to observed maggies.

    ``matrix`` has shape ``(n_bands, n_wave)`` and is applied as
    ``luminosity_w_per_nm @ matrix.T``.  The matrix contains filter
    transmission, luminosity distance, IGM attenuation, and unit conversion.
    """

    redshift: float
    wavelength_nm: np.ndarray
    band_names: tuple[str, ...]
    matrix: np.ndarray
    valid_bands: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class NativeCatalogFitResult:
    """Catalog result from rest-frame grid projection.

    The posterior arrays have the same object/model layout as
    ``CatalogProfileGridResult`` but the observed photometric grids are built
    on the fly for each rounded redshift group. The boolean
    ``mass_profile_at_boundary`` array flags explicitly bounded mass optima.
    """

    rest_grid: RestFrameSpectralGrid
    redshifts: np.ndarray
    redshift_values: np.ndarray
    evaluated_redshifts: np.ndarray
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


def build_restframe_spectral_grid(
    backend,
    parameter_space,
    *,
    wavelengths_nm: Sequence[float] | None = None,
    excluded_parameters: Sequence[str] = ("log10_mass", "z", "zred", "redshift"),
    max_grid_size: int | None = 1_000_000,
) -> RestFrameSpectralGrid:
    """Build the expensive rest-frame SED grid once.

    This experimental path is the native CompoSED analogue of CIGALE's cached
    pre-redshift grid. It enumerates finite non-mass, non-redshift parameters,
    asks a backend with an explicit capability declaration for rest-frame
    luminosity density, and stores all spectra on one wavelength grid in nm
    and W/nm.
    """

    _warn_experimental_fast_catalog()
    if not bool(getattr(backend, "supports_fast_catalog_restframe", False)):
        raise NotImplementedError(
            f"{type(backend).__name__} does not declare support for the experimental "
            "redshift-independent rest-frame catalog grid. In particular, FSPS SFH "
            "evaluation currently requires a redshift and must use the ordinary backend "
            "or cached photometric-grid path."
        )

    samples, names, log_prior = _finite_grid_theta_excluding(
        parameter_space,
        tuple(excluded_parameters),
        max_grid_size=max_grid_size,
    )
    if samples.shape[0] == 0:
        raise ValueError("Cannot build an empty rest-frame spectral grid.")
    mass_norm = MassNormalization(getattr(backend, "mass_normalization", None))
    mass_reference = backend_mass_reference(backend)

    requested_wave = None if wavelengths_nm is None else _validate_wavelength_nm(wavelengths_nm)
    wavelength_grid = requested_wave
    spectra: list[np.ndarray | None] = [None] * samples.shape[0]
    valid = np.zeros(samples.shape[0], dtype=bool)

    for i, row in enumerate(samples):
        params = {name: float(value) for name, value in zip(names, row)}
        try:
            model = backend.predict_rest_spectrum(params, wavelengths=requested_wave)
            wave_nm, luminosity_w_per_nm = _coerce_rest_spectrum_to_w_per_nm(model)
            if wavelength_grid is None:
                wavelength_grid = wave_nm
            elif not np.array_equal(wave_nm, wavelength_grid):
                luminosity_w_per_nm = np.interp(wavelength_grid, wave_nm, luminosity_w_per_nm, left=np.nan, right=np.nan)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        spectra[i] = luminosity_w_per_nm
        valid[i] = np.all(np.isfinite(luminosity_w_per_nm)) and np.all(luminosity_w_per_nm >= 0.0)

    if wavelength_grid is None:
        raise RuntimeError("No valid rest-frame spectra were produced.")
    luminosity = np.full((samples.shape[0], wavelength_grid.size), np.nan, dtype=float)
    for i, spectrum in enumerate(spectra):
        if spectrum is not None:
            luminosity[i] = spectrum

    return RestFrameSpectralGrid(
        wavelength_nm=wavelength_grid,
        luminosity_w_per_nm=luminosity,
        samples=samples,
        log_prior=log_prior,
        valid=valid & np.isfinite(log_prior),
        parameter_names=names,
        mass_normalization=mass_norm,
        mass_reference=mass_reference,
        meta={
            "excluded_parameters": tuple(excluded_parameters),
            "wavelength_unit": "nm",
            "luminosity_unit": "W/nm",
            "mass_scale_applied": False,
            "mass_reference": getattr(mass_reference, "value", None),
            "mass_convention": MASS_CONVENTION_SCHEMA,
        },
    )


def build_redshift_filter_operator(
    wavelength_nm: Sequence[float],
    filters: FilterSet | Sequence[object],
    redshift: float,
    *,
    igm_model: str | Callable[[np.ndarray, float], np.ndarray] | None = "cigale",
    luminosity_distance_m: float | None = None,
    cosmology=None,
) -> RedshiftFilterOperator:
    """Build one linear redshift/filter operator.

    CIGALE filter names and CIGALE ``wl``/``tr`` objects contain a kernel that
    is already normalized to produce mJy when integrated against ``f_lambda``.
    Ordinary tabulated transmission curves are instead integrated with the AB
    photon-counting convention

    ``<f_nu> = integral(lambda T f_lambda d lambda) /
    (c integral(T / lambda d lambda))``.

    Generic ``wavelength`` arrays must declare ``wavelength_unit``.  Sedpy
    filters are recognized as Angstrom-valued photon-response curves.  The
    default cosmology is Astropy Planck18 and never depends on whether CIGALE
    happens to be importable.
    """

    wave_rest = _validate_wavelength_nm(wavelength_nm)
    z = float(redshift)
    if not np.isfinite(z) or z < 0.0:
        raise ValueError("redshift must be finite and non-negative.")
    filter_set = filters if isinstance(filters, FilterSet) else FilterSet(tuple(filters))
    d_l_m = (
        _luminosity_distance_m(z, cosmology=cosmology)
        if luminosity_distance_m is None
        else float(luminosity_distance_m)
    )
    if not np.isfinite(d_l_m) or d_l_m <= 0.0:
        raise ValueError("luminosity_distance_m must be finite and positive.")

    wave_obs = wave_rest * (1.0 + z)
    igm = _igm_transmission(wave_obs, z, igm_model)
    trapz_weights = _trapezoid_weights(wave_rest)
    observed_trapz_weights = (1.0 + z) * trapz_weights

    matrix = np.full((len(filter_set), wave_rest.size), np.nan, dtype=float)
    valid_bands = np.zeros(len(filter_set), dtype=bool)
    for i, filter_obj in enumerate(filter_set.filters):
        filt_wave, filt_trans, response_kind = _filter_curve_nm(filter_obj)
        if wave_obs[0] > filt_wave[0] or wave_obs[-1] < filt_wave[-1]:
            continue
        trans_on_model = np.interp(wave_obs, filt_wave, filt_trans, left=0.0, right=0.0)
        if response_kind == "cigale_mjy_kernel":
            row = trapz_weights * trans_on_model * igm / (4.0 * np.pi * d_l_m**2) / MJY_PER_MAGGIE
        elif response_kind == "photon":
            # Use the same quadrature grid for numerator and denominator. This
            # makes a flat f_nu spectrum exactly invariant under the operator
            # up to floating-point precision, including on coarse model grids.
            denominator = C_NM_PER_S * np.sum(observed_trapz_weights * trans_on_model / wave_obs)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(f"Filter {filter_set.names[i]!r} has zero or invalid AB normalization.")
            row = (
                trapz_weights
                * wave_obs
                * trans_on_model
                * igm
                / (4.0 * np.pi * d_l_m**2 * denominator * AB_ZERO_FNU_W_M2_HZ)
            )
        elif response_kind == "energy":
            denominator = C_NM_PER_S * np.sum(observed_trapz_weights * trans_on_model / wave_obs**2)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(f"Filter {filter_set.names[i]!r} has zero or invalid AB normalization.")
            row = (
                trapz_weights
                * trans_on_model
                * igm
                / (4.0 * np.pi * d_l_m**2 * denominator * AB_ZERO_FNU_W_M2_HZ)
            )
        else:  # pragma: no cover - guarded by _filter_curve_nm.
            raise ValueError(f"Unsupported filter response kind {response_kind!r}.")
        if np.all(np.isfinite(row)):
            matrix[i] = row
            valid_bands[i] = True

    return RedshiftFilterOperator(
        redshift=z,
        wavelength_nm=wave_rest,
        band_names=tuple(filter_set.names),
        matrix=matrix,
        valid_bands=valid_bands,
        meta={
            "output_unit": "maggies",
            "igm_model": None if igm_model is None else str(igm_model),
            "luminosity_distance_m": d_l_m,
            "cosmology": _cosmology_label(cosmology),
        },
    )


def project_rest_grid_to_photometric_grid(
    rest_grid: RestFrameSpectralGrid,
    operator: RedshiftFilterOperator,
    *,
    age_parameter: str | None = "age",
    age_unit: str = "Myr",
    reject_older_than_universe: bool = True,
    cosmology=None,
) -> PhotometricModelGrid:
    """Project a rest-frame spectral grid into observed photometry at one z."""

    if rest_grid.mass_normalization != MassNormalization.PER_SOLAR_MASS:
        raise ValueError(
            "Native fast catalog mode expects a PER_SOLAR_MASS rest-frame grid. "
            f"Got {rest_grid.mass_normalization}."
        )
    validate_mass_reference(rest_grid.mass_normalization, rest_grid.mass_reference)
    if not np.array_equal(rest_grid.wavelength_nm, operator.wavelength_nm):
        raise ValueError("Rest grid wavelength and operator wavelength grids do not match.")
    if not np.all(operator.valid_bands):
        unavailable = [
            name for name, valid in zip(operator.band_names, operator.valid_bands) if not valid
        ]
        raise ValueError(
            "The experimental fast catalog path requires the rest wavelength grid to "
            "cover every requested filter. Unavailable band(s): "
            + ", ".join(unavailable)
        )
    flux = rest_grid.luminosity_w_per_nm @ operator.matrix.T
    valid = rest_grid.valid & np.all(np.isfinite(flux), axis=1) & np.all(flux >= 0.0, axis=1)
    valid &= _age_validity_mask(
        rest_grid,
        operator.redshift,
        age_parameter=age_parameter,
        age_unit=age_unit,
        reject=reject_older_than_universe,
        cosmology=cosmology,
    )
    return PhotometricModelGrid(
        samples=rest_grid.samples,
        flux=flux,
        log_prior=rest_grid.log_prior,
        valid=valid,
        parameter_names=rest_grid.parameter_names,
        band_names=operator.band_names,
        mass_normalization=rest_grid.mass_normalization,
        mass_reference=rest_grid.mass_reference,
        meta={
            "source": "RestFrameSpectralGrid projected with RedshiftFilterOperator",
            "redshift": operator.redshift,
            "operator_meta": dict(operator.meta),
            "mass_reference": rest_grid.mass_reference.value,
            "mass_convention": MASS_CONVENTION_SCHEMA,
        },
    )


def fit_catalog_with_restframe_grid(
    rest_grid: RestFrameSpectralGrid,
    datasets: Sequence[SEDDataset],
    redshifts: Sequence[float],
    filters: FilterSet | Sequence[object],
    *,
    redshift_decimals: int | None = None,
    igm_model: str | Callable[[np.ndarray, float], np.ndarray] | None = "cigale",
    cosmology=None,
    sigma_floor: float | None = None,
    log10_mass_grid: Sequence[float] | None = None,
    log10_mass_bounds: tuple[float, float] | None = None,
    log10_mass_prior: Prior | None = None,
    model_chunk_size: int = 2048,
    object_chunk_size: int = 512,
    mass_chunk_size: int = 128,
    age_parameter: str | None = "age",
    age_unit: str = "Myr",
    reject_older_than_universe: bool = True,
) -> NativeCatalogFitResult:
    """Fit many catalog objects by redshift-projecting one rest-frame grid."""

    _warn_experimental_fast_catalog()
    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("fit_catalog_with_restframe_grid requires at least one dataset.")
    z = np.asarray(redshifts, dtype=float)
    if z.shape != (len(datasets),):
        raise ValueError("redshifts must have one value per dataset.")
    if not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("redshifts must be finite and non-negative.")
    z_eval = np.round(z, int(redshift_decimals)) if redshift_decimals is not None else z
    rounded_to_zero = (z > 0.0) & (z_eval <= 0.0)
    if np.any(rounded_to_zero):
        bad = np.where(rounded_to_zero)[0]
        raise ValueError(
            "Redshift rounding mapped positive observed redshift(s) to zero, which would invoke the 10 pc convention. "
            f"Use redshift_decimals=None or more precision. Object indices: {bad.tolist()}"
        )

    n_objects = len(datasets)
    n_models = rest_grid.samples.shape[0]
    profile_logp = np.full((n_objects, n_models), -np.inf, dtype=float)
    log10_mass_profile = np.full((n_objects, n_models), np.nan, dtype=float)
    mass_scale_profile = np.full((n_objects, n_models), np.nan, dtype=float)
    mass_profile_at_boundary = np.zeros((n_objects, n_models), dtype=bool)
    marginal_logp = None
    mass_posterior_norm = None
    log10_mass_quantiles = None
    effective_mass_grid = None
    mass_prior_meta = None
    if log10_mass_grid is not None:
        marginal_logp = np.full((n_objects, n_models), -np.inf, dtype=float)
        log10_mass_quantiles = np.full((n_objects, 3), np.nan, dtype=float)

    band_names = tuple(datasets[0].band_names)
    operators = {}
    for z_value in np.unique(z_eval):
        rows = np.where(z_eval == z_value)[0]
        operator = build_redshift_filter_operator(
            rest_grid.wavelength_nm,
            filters,
            float(z_value),
            igm_model=igm_model,
            cosmology=cosmology,
        )
        operators[float(z_value)] = operator
        phot_grid = project_rest_grid_to_photometric_grid(
            rest_grid,
            operator,
            age_parameter=age_parameter,
            age_unit=age_unit,
            reject_older_than_universe=reject_older_than_universe,
            cosmology=cosmology,
        )
        result = evaluate_catalog_model_grid_likelihood(
            phot_grid,
            [datasets[i] for i in rows],
            sigma_floor=sigma_floor,
            log10_mass_grid=log10_mass_grid,
            log10_mass_bounds=log10_mass_bounds,
            log10_mass_prior=log10_mass_prior,
            model_chunk_size=model_chunk_size,
            object_chunk_size=object_chunk_size,
            mass_chunk_size=mass_chunk_size,
        )
        profile_logp[rows] = result.profile_logp
        log10_mass_profile[rows] = result.log10_mass_profile
        mass_scale_profile[rows] = result.mass_scale_profile
        mass_profile_at_boundary[rows] = result.mass_profile_at_boundary
        if result.log10_mass_grid is not None:
            if effective_mass_grid is None:
                effective_mass_grid = result.log10_mass_grid.copy()
                mass_prior_meta = result.meta.get("mass_prior")
            elif not np.array_equal(effective_mass_grid, result.log10_mass_grid):
                raise RuntimeError("Mass quadrature changed between redshift groups.")
        if marginal_logp is not None and result.marginal_logp is not None:
            marginal_logp[rows] = result.marginal_logp
            log10_mass_quantiles[rows] = result.log10_mass_quantiles
            if result.mass_posterior_norm is not None:
                if mass_posterior_norm is None:
                    mass_posterior_norm = np.full(
                        (n_objects,) + result.mass_posterior_norm.shape[1:],
                        np.nan,
                        dtype=float,
                    )
                mass_posterior_norm[rows] = result.mass_posterior_norm

    profile_weights = _normalize_logp_rows(profile_logp)
    profile_map_indices = np.asarray([int(np.nanargmax(row)) for row in profile_logp], dtype=int)
    profile_map_estimates = rest_grid.samples[profile_map_indices]

    marginal_weights = None
    marginal_map_indices = None
    marginal_map_estimates = None
    if marginal_logp is not None:
        marginal_weights = _normalize_logp_rows(marginal_logp)
        marginal_map_indices = np.asarray([int(np.nanargmax(row)) for row in marginal_logp], dtype=int)
        marginal_map_estimates = rest_grid.samples[marginal_map_indices]

    return NativeCatalogFitResult(
        rest_grid=rest_grid,
        redshifts=z,
        redshift_values=np.unique(z_eval),
        evaluated_redshifts=z_eval,
        profile_logp=profile_logp,
        profile_weights_norm=profile_weights,
        profile_map_indices=profile_map_indices,
        profile_map_estimates=profile_map_estimates,
        log10_mass_profile=log10_mass_profile,
        mass_scale_profile=mass_scale_profile,
        mass_profile_at_boundary=mass_profile_at_boundary,
        marginal_logp=marginal_logp,
        marginal_weights_norm=marginal_weights,
        marginal_map_indices=marginal_map_indices,
        marginal_map_estimates=marginal_map_estimates,
        log10_mass_grid=effective_mass_grid,
        mass_posterior_norm=mass_posterior_norm,
        log10_mass_quantiles=log10_mass_quantiles,
        band_names=band_names,
        meta={
            "mode": "native_rest_grid_projection",
            "redshift_decimals": redshift_decimals,
            "input_redshifts": z.copy(),
            "evaluated_redshifts": z_eval.copy(),
            "cosmology": _cosmology_label(cosmology),
            "operators_by_redshift": operators,
            "sigma_floor": sigma_floor,
            "mass_prior": mass_prior_meta,
        },
    )


def _validate_wavelength_nm(wavelength_nm: Sequence[float]) -> np.ndarray:
    wave = np.asarray(wavelength_nm, dtype=float)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("wavelength_nm must be a one-dimensional array with at least two points.")
    if not np.all(np.isfinite(wave)) or np.any(np.diff(wave) <= 0.0):
        raise ValueError("wavelength_nm must be finite and strictly increasing.")
    return wave


def _decode_saved_meta(value) -> dict:
    text = str(value)
    if not text:
        return {}
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        # Older experimental files used ``str(dict)`` before the public loader
        # existed.  Accept that local format so users can still inspect them.
        out = ast.literal_eval(text)
    return dict(out) if isinstance(out, Mapping) else {"value": out}


def _coerce_rest_spectrum_to_w_per_nm(model: ModelSpectrum) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(model.wavelength, dtype=float)
    lum = np.asarray(model.flux, dtype=float)
    wave_unit = str(model.wavelength_unit).lower()
    flux_unit = str(model.flux_unit).lower()
    if wave_unit in {"angstrom", "a", "aa"}:
        wave = wave / 10.0
    elif wave_unit != "nm":
        raise ValueError(f"Unsupported rest-spectrum wavelength unit {model.wavelength_unit!r}; expected nm or Angstrom.")
    if flux_unit in {"w/nm", "w nm-1"}:
        pass
    elif flux_unit in {"w/angstrom", "w/a", "w/aa"}:
        # L per nm = L per Angstrom * 10 Angstrom/nm. This conversion depends
        # on the luminosity-density unit, not on the wavelength-coordinate unit.
        lum = lum * 10.0
    else:
        raise ValueError(f"Unsupported rest-spectrum luminosity unit {model.flux_unit!r}; expected W/nm.")
    wave = _validate_wavelength_nm(wave)
    if lum.shape != wave.shape:
        raise ValueError("Rest-spectrum wavelength and luminosity arrays must have matching shapes.")
    return wave, lum


def _filter_curve_nm(filter_obj) -> tuple[np.ndarray, np.ndarray, str]:
    if isinstance(filter_obj, str):
        if filter_obj.startswith("line."):
            raise NotImplementedError(
                "Native fast catalog mode currently handles broadband filter curves, not CIGALE line.* pseudo-filters."
            )
        try:
            from pcigale.data import SimpleDatabase as Database
        except ImportError as exc:
            raise ImportError("CIGALE filter names require pcigale to be installed.") from exc
        with Database("filters") as db:
            filt = db.get(name=filter_obj)
        return _validate_wavelength_nm(filt.wl), np.asarray(filt.tr, dtype=float), "cigale_mjy_kernel"

    if hasattr(filter_obj, "wavelength_nm"):
        wave = np.asarray(filter_obj.wavelength_nm, dtype=float)
        response_kind = str(getattr(filter_obj, "response_type", "photon")).lower()
    elif hasattr(filter_obj, "wl"):
        wave = np.asarray(filter_obj.wl, dtype=float)
        response_kind = "cigale_mjy_kernel"
    elif hasattr(filter_obj, "wavelength"):
        wave = np.asarray(filter_obj.wavelength, dtype=float)
        module_name = type(filter_obj).__module__
        if hasattr(filter_obj, "wavelength_unit"):
            unit = str(filter_obj.wavelength_unit).lower()
        elif module_name.startswith("sedpy.") or hasattr(filter_obj, "ab_zero_counts"):
            unit = "angstrom"
        else:
            raise ValueError(
                "Generic filter objects exposing wavelength must also declare wavelength_unit; "
                "use wavelength_nm for an unambiguous nm curve."
            )
        if unit in {"angstrom", "a", "aa"}:
            wave = wave / 10.0
        elif unit != "nm":
            raise ValueError(f"Unsupported filter wavelength unit {unit!r}; expected nm or Angstrom.")
        response_kind = str(getattr(filter_obj, "response_type", "photon")).lower()
    else:
        raise ValueError("Filter objects must expose wavelength_nm, wavelength, or wl.")

    if hasattr(filter_obj, "transmission"):
        trans = np.asarray(filter_obj.transmission, dtype=float)
    elif hasattr(filter_obj, "tr"):
        trans = np.asarray(filter_obj.tr, dtype=float)
    else:
        raise ValueError("Filter objects must expose transmission or tr.")

    wave = _validate_wavelength_nm(wave)
    if trans.shape != wave.shape or not np.all(np.isfinite(trans)):
        raise ValueError("Filter transmission must be finite and match the wavelength shape.")
    if np.any(trans < 0.0):
        raise ValueError("Filter transmission must be non-negative.")
    if response_kind not in {"photon", "energy", "cigale_mjy_kernel"}:
        raise ValueError("Filter response_type must be 'photon' or 'energy'.")
    return wave, trans, response_kind


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    weights = np.empty_like(x, dtype=float)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return weights


def _igm_transmission(
    observed_wavelength_nm: np.ndarray,
    redshift: float,
    igm_model: str | Callable[[np.ndarray, float], np.ndarray] | None,
) -> np.ndarray:
    if igm_model is None or redshift <= 0.0:
        return np.ones_like(observed_wavelength_nm, dtype=float)
    if callable(igm_model):
        trans = np.asarray(igm_model(observed_wavelength_nm, redshift), dtype=float)
    elif str(igm_model).lower() == "cigale":
        from pcigale.sed_modules.redshifting import igm_transmission

        trans = np.asarray(igm_transmission(observed_wavelength_nm, float(redshift)), dtype=float)
    else:
        raise ValueError("igm_model must be None, 'cigale', or a callable.")
    if trans.shape != observed_wavelength_nm.shape or not np.all(np.isfinite(trans)):
        raise ValueError("IGM transmission must be finite and match the wavelength shape.")
    return trans


def _default_cosmology():
    try:
        from astropy.cosmology import Planck18
    except ImportError as exc:  # pragma: no cover - astropy is a core dependency.
        raise ImportError("Fast catalog projection requires astropy for its default Planck18 cosmology.") from exc
    return Planck18


def _cosmology_label(cosmology) -> str:
    cosmo = _default_cosmology() if cosmology is None else cosmology
    return str(getattr(cosmo, "name", type(cosmo).__name__))


def _luminosity_distance_m(redshift: float, *, cosmology=None) -> float:
    if redshift <= 0.0:
        return 10.0 * PARSEC_M
    cosmo = _default_cosmology() if cosmology is None else cosmology
    distance = cosmo.luminosity_distance(float(redshift))
    try:
        return float(distance.to("m").value)
    except AttributeError:
        return float(distance)


def _age_validity_mask(
    rest_grid: RestFrameSpectralGrid,
    redshift: float,
    *,
    age_parameter: str | None,
    age_unit: str,
    reject: bool,
    cosmology=None,
) -> np.ndarray:
    if not reject or age_parameter is None or age_parameter not in rest_grid.parameter_names:
        return np.ones(rest_grid.samples.shape[0], dtype=bool)
    idx = rest_grid.parameter_names.index(age_parameter)
    age = np.asarray(rest_grid.samples[:, idx], dtype=float)
    if age_unit.lower() in {"myr", "megayr"}:
        age_myr = age
    elif age_unit.lower() in {"gyr", "gigayr"}:
        age_myr = 1000.0 * age
    else:
        raise ValueError("age_unit must be 'Myr' or 'Gyr'.")
    return age_myr <= _age_universe_myr(redshift, cosmology=cosmology)


def _age_universe_myr(redshift: float, *, cosmology=None) -> float:
    cosmo = _default_cosmology() if cosmology is None else cosmology
    age = cosmo.age(float(redshift))
    try:
        return float(age.to("Myr").value)
    except AttributeError:
        return float(age)


__all__ = [
    "ExperimentalFastCatalogWarning",
    "NativeCatalogFitResult",
    "RedshiftFilterOperator",
    "RestFrameSpectralGrid",
    "build_redshift_filter_operator",
    "build_restframe_spectral_grid",
    "fit_catalog_with_restframe_grid",
    "load_restframe_spectral_grid",
    "project_rest_grid_to_photometric_grid",
    "save_restframe_spectral_grid",
]
