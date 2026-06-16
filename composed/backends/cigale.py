from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from composed.backends.base import ModelPhotometry, ModelSpectrum, SEDBackend
from composed.filters import FilterSet
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, IntegerUniformPrior, LogUniformPrior, Prior, UniformPrior
from composed.units import MassNormalization

REDSHIFT_KEYS = ("z", "zred", "redshift")
MJY_PER_MAGGIE = 3631.0e3
C_A_PER_S = 2.99792458e18


@dataclass
class CIGALEBackend(SEDBackend):
    """CIGALE forward-model backend using ``pcigale.warehouse.SedWarehouse``.

    ``modules`` is the ordered CIGALE SED module chain. ``module_parameters`` is
    a nested mapping keyed by module name and then CIGALE parameter name. Values
    may be fixed scalars, finite numeric choices, or prior specifications such
    as ``{"range": [low, high], "scale": "linear"}``. Variable specifications
    are consumed from the flat ``params`` dictionary passed to
    ``predict_photometry``; helper functions in this module build a matching
    ``ParameterSpace``.

    CIGALE SED conventions used here:

    - ``SED.wavelength_grid`` is in nm.
    - ``SED.luminosity`` is luminosity density in W / nm.
    - ``SED.fnu`` and ``SED.compute_fnu(filter_name)`` return observed flux
      density in mJy after the CIGALE ``redshifting`` module has supplied the
      luminosity distance.
    - Native CIGALE filter photometry is converted from mJy to maggies.

    This backend deliberately supports only
    ``MassNormalization.PER_SOLAR_MASS``. For SFH modules, ``normalise=True`` is
    enforced on every call, so the likelihood remains the only place where
    ``10**log10_mass`` may be applied.
    """

    modules: Sequence[str]
    module_parameters: Mapping[str, Mapping[str, Any]] | None = None
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    default_z_key: str = "redshift"
    photometry_mode: str = "auto"
    cache_warehouse: bool = True
    nocache_modules: Sequence[str] | None = None
    clear_photometry_cache: bool = False
    strict_unknown_parameters: bool = True
    require_redshifting: bool = True
    require_sfh_integrated_info: bool = True
    mass_normalization_tolerance: float = 1e-8
    _warehouse: Any = field(default=None, init=False, repr=False)
    _entries: tuple["_ParameterEntry", ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        modules = tuple(str(module) for module in self.modules)
        if not modules:
            raise ValueError("CIGALEBackend requires at least one CIGALE module.")
        self.modules = modules

        self.mass_normalization = MassNormalization(self.mass_normalization)
        if self.mass_normalization != MassNormalization.PER_SOLAR_MASS:
            raise ValueError(
                "CIGALEBackend currently supports only MassNormalization.PER_SOLAR_MASS; "
                "SFH module normalise=True is enforced so likelihood.py applies mass scaling."
            )
        if self.default_z_key not in REDSHIFT_KEYS:
            raise ValueError(f"default_z_key must be one of {REDSHIFT_KEYS}.")
        if self.photometry_mode not in {"auto", "cigale", "sedpy"}:
            raise ValueError("photometry_mode must be one of: auto, cigale, sedpy.")
        if float(self.mass_normalization_tolerance) < 0.0:
            raise ValueError("mass_normalization_tolerance must be non-negative.")
        if not _module_available("pcigale"):
            raise ImportError(
                "CIGALEBackend requires the optional pcigale package. Install CIGALE/pcigale "
                "before constructing CIGALEBackend."
            )

        module_parameters = _normalize_module_parameters(self.module_parameters)
        unknown_modules = set(module_parameters) - set(modules)
        if unknown_modules:
            names = ", ".join(sorted(unknown_modules))
            raise ValueError(f"module_parameters contains module(s) not present in modules: {names}")
        self.module_parameters = module_parameters
        self._entries = _resolve_parameter_entries(modules, module_parameters)

    def predict_photometry(self, params: Mapping[str, Any], filters: FilterSet | Sequence[object]) -> ModelPhotometry:
        """Predict observed-frame filter photometry in maggies."""

        filter_set = coerce_filter_set(filters)
        sed, module_list = self._sed_from_params(params)

        mode = self._resolve_photometry_mode(filter_set)
        try:
            if mode == "cigale":
                flux_maggies = self._native_cigale_photometry(sed, filter_set)
            else:
                flux_maggies = self._sedpy_photometry(sed, filter_set)
        finally:
            if self.clear_photometry_cache:
                _clear_cigale_photometry_caches()

        if flux_maggies.shape != (len(filter_set),):
            raise ValueError(f"CIGALE photometry shape {flux_maggies.shape}; expected {(len(filter_set),)}.")
        if not np.all(np.isfinite(flux_maggies)):
            raise FloatingPointError("CIGALE photometry contains non-finite values.")
        if np.any(flux_maggies < 0.0):
            raise FloatingPointError("CIGALE photometry contains negative fluxes.")

        return ModelPhotometry(
            band_names=filter_set.names,
            flux=flux_maggies,
            metadata={
                "backend": "cigale",
                "photometry_mode": mode,
                "modules": module_list,
                "cigale_info": dict(getattr(sed, "info", {})),
            },
        )

    def predict_spectrum(
        self,
        params: Mapping[str, Any],
        wavelengths: Sequence[float] | None = None,
        wavelength_range: tuple[float, float] | None = None,
        resolution: float | None = None,
    ) -> ModelSpectrum:
        """Predict observed-frame ``f_lambda`` in cgs per Angstrom.

        CIGALE provides the observed spectrum as ``sed.fnu`` in mJy after the
        ``redshifting`` module has set the luminosity distance. This method
        converts that to ``f_lambda`` on the CIGALE wavelength grid, then
        optionally interpolates to requested observed-frame Angstrom
        wavelengths.
        """

        if resolution is not None:
            raise NotImplementedError("CIGALEBackend spectral resolution convolution is not implemented yet.")
        sed, module_list = self._sed_from_params(params)
        wave_a = np.asarray(getattr(sed, "wavelength_grid"), dtype=float) * 10.0
        fnu_mjy = np.asarray(getattr(sed, "fnu"), dtype=float)
        flam_cgs_per_a = _fnu_mjy_to_flambda_cgs_per_a(wave_a, fnu_mjy)
        wave_out, flux_out = _sample_or_clip_spectrum(wave_a, flam_cgs_per_a, wavelengths, wavelength_range)
        if not np.all(np.isfinite(flux_out)):
            raise FloatingPointError("CIGALE spectrum contains non-finite flux values after sampling.")
        if np.any(flux_out < 0.0):
            raise FloatingPointError("CIGALE spectrum contains negative flux values after sampling.")
        return ModelSpectrum(
            wavelength=wave_out,
            flux=flux_out,
            wavelength_unit="angstrom",
            flux_unit="erg/s/cm^2/angstrom",
            metadata={
                "backend": "cigale",
                "modules": module_list,
                "cigale_info": dict(getattr(sed, "info", {})),
                "spectrum_frame": "observed",
                "mass_normalization": self.mass_normalization.value,
            },
        )

    def _sed_from_params(self, params: Mapping[str, Any]):
        params = dict(params)
        module_list, parameter_list = self._build_cigale_configuration(params)
        sed = self._sed_warehouse().get_sed(module_list, parameter_list)
        if self.mass_normalization == MassNormalization.PER_SOLAR_MASS:
            self._validate_per_solar_mass_sed(sed)
        return sed, module_list

    def _sed_warehouse(self):
        if self.cache_warehouse and self._warehouse is not None:
            return self._warehouse

        from pcigale.warehouse import SedWarehouse

        nocache = list(self.nocache_modules) if self.nocache_modules is not None else None
        warehouse = SedWarehouse(nocache=nocache)
        if self.cache_warehouse:
            self._warehouse = warehouse
        return warehouse

    def _build_cigale_configuration(self, params: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
        configs = {module: {} for module in self.modules}
        used = set()

        for entry in self._entries:
            if entry.prior is None:
                value = entry.fixed_value
            else:
                if entry.flat_name not in params:
                    raise KeyError(f"Missing CIGALE backend parameter {entry.flat_name!r}.")
                value = params[entry.flat_name]
                used.add(entry.flat_name)
            configs[entry.module][entry.parameter] = _coerce_cigale_value(value, entry.dtype)

        self._enforce_sfh_normalise(configs)
        self._inject_redshift(configs, params, used)
        self._check_unknown_parameters(params, used)

        return list(self.modules), [configs[module] for module in self.modules]

    def _enforce_sfh_normalise(self, configs: dict[str, dict[str, Any]]) -> None:
        for module in self.modules:
            if not _is_sfh_module(module):
                continue
            existing = configs[module].get("normalise", True)
            if not _truthy(existing):
                raise ValueError(
                    f"CIGALE SFH module {module!r} has normalise={existing!r}; "
                    "composed requires normalise=True so mass scaling remains explicit."
                )
            configs[module]["normalise"] = True

    def _inject_redshift(self, configs: dict[str, dict[str, Any]], params: Mapping[str, Any], used: set[str]) -> None:
        redshift_module = _first_redshifting_module(self.modules)
        if self.require_redshifting and redshift_module is None:
            raise ValueError("CIGALEBackend requires a redshifting module for observed-frame photometry.")
        if redshift_module is None or "redshift" in configs[redshift_module]:
            return

        key, z = self._get_redshift(params)
        configs[redshift_module]["redshift"] = z
        used.add(key)

    def _check_unknown_parameters(self, params: Mapping[str, Any], used: set[str]) -> None:
        if not self.strict_unknown_parameters:
            return
        ignored = {"log10_mass", "logmass"}
        unknown = set(params) - set(used) - ignored
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"Unexpected parameter(s) for CIGALEBackend: {names}")

    def _get_redshift(self, params: Mapping[str, Any]) -> tuple[str, float]:
        ordered_keys = (self.default_z_key,) + tuple(key for key in REDSHIFT_KEYS if key != self.default_z_key)
        for key in ordered_keys:
            if key in params:
                z = float(params[key])
                if not np.isfinite(z) or z < 0.0:
                    raise ValueError(f"Redshift parameter {key!r} must be finite and non-negative.")
                return key, z
        raise ValueError("Missing redshift parameter. Provide one of: z, zred, redshift.")

    def _validate_per_solar_mass_sed(self, sed) -> None:
        info = getattr(sed, "info", {})
        if "sfh.integrated" not in info:
            if self.require_sfh_integrated_info:
                raise ValueError(
                    "CIGALE SED does not expose info['sfh.integrated']; cannot verify per-solar-mass normalization."
                )
            return
        formed_mass = float(info["sfh.integrated"])
        if not np.isfinite(formed_mass) or not np.isclose(
            formed_mass, 1.0, rtol=float(self.mass_normalization_tolerance), atol=float(self.mass_normalization_tolerance)
        ):
            raise ValueError(
                "CIGALE SED is not normalized to one solar mass formed "
                f"(info['sfh.integrated']={formed_mass!r})."
            )

    def _resolve_photometry_mode(self, filter_set: FilterSet) -> str:
        if self.photometry_mode != "auto":
            return self.photometry_mode
        return "cigale" if all(isinstance(f, str) for f in filter_set.filters) else "sedpy"

    @staticmethod
    def _native_cigale_photometry(sed, filter_set: FilterSet) -> np.ndarray:
        flux_mjy = []
        for filter_obj in filter_set.filters:
            if isinstance(filter_obj, str):
                filter_name = filter_obj
            else:
                filter_name = getattr(filter_obj, "name", None)
            if filter_name is None:
                raise ValueError("Native CIGALE photometry requires filter names or filter objects with a name attribute.")
            flux_mjy.append(float(sed.compute_fnu(str(filter_name))))
        return np.asarray(flux_mjy, dtype=float) / MJY_PER_MAGGIE

    @staticmethod
    def _sedpy_photometry(sed, filter_set: FilterSet) -> np.ndarray:
        try:
            from sedpy.observate import getSED
        except ImportError as exc:
            raise ImportError("CIGALEBackend sedpy photometry mode requires sedpy.") from exc

        wave_a = np.asarray(getattr(sed, "wavelength_grid"), dtype=float) * 10.0
        fnu_mjy = np.asarray(getattr(sed, "fnu"), dtype=float)
        flam_cgs_per_a = _fnu_mjy_to_flambda_cgs_per_a(wave_a, fnu_mjy)
        mags = getSED(wave_a, flam_cgs_per_a, list(filter_set.filters), linear_flux=False)
        return 10.0 ** (-0.4 * np.asarray(mags, dtype=float))


def build_cigale_parameter_space(
    modules: Sequence[str],
    module_parameters: Mapping[str, Mapping[str, Any]] | None,
    additional_priors: Mapping[str, Prior] | None = None,
) -> ParameterSpace:
    """Build a deterministic ``ParameterSpace`` from CIGALE parameter specs.

    Parameters are ordered as ``additional_priors`` first, then CIGALE module
    parameters in module order and insertion order. This makes it convenient to
    put ``log10_mass`` first while keeping backend parameters reproducible.
    """

    entries = _resolve_parameter_entries(tuple(str(module) for module in modules), _normalize_module_parameters(module_parameters))
    names: list[str] = []
    priors: dict[str, Prior] = {}
    if additional_priors:
        for name, prior in additional_priors.items():
            names.append(str(name))
            priors[str(name)] = prior
    for entry in entries:
        if entry.prior is None:
            continue
        if entry.flat_name in priors:
            raise ValueError(f"Duplicate parameter name {entry.flat_name!r}.")
        names.append(entry.flat_name)
        priors[entry.flat_name] = entry.prior
    return ParameterSpace(names=names, priors=priors)


def build_cigale_backend_and_parameter_space(
    modules: Sequence[str],
    module_parameters: Mapping[str, Mapping[str, Any]] | None,
    additional_priors: Mapping[str, Prior] | None = None,
    **backend_kwargs,
) -> tuple[CIGALEBackend, ParameterSpace]:
    """Construct a ``CIGALEBackend`` and matching ``ParameterSpace``."""

    backend = CIGALEBackend(modules=modules, module_parameters=module_parameters, **backend_kwargs)
    parameter_space = build_cigale_parameter_space(modules, module_parameters, additional_priors=additional_priors)
    return backend, parameter_space


def coerce_filter_set(filters: FilterSet | Sequence[object]) -> FilterSet:
    if filters is None:
        raise ValueError("CIGALEBackend requires a FilterSet or sequence of filters.")
    if isinstance(filters, FilterSet):
        return filters
    return FilterSet(filters=tuple(filters))


@dataclass(frozen=True)
class _ParameterEntry:
    module: str
    parameter: str
    flat_name: str
    prior: Prior | None
    fixed_value: Any = None
    dtype: str = "float"


def _normalize_module_parameters(module_parameters: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if module_parameters is None:
        return {}
    return {str(module): dict(params) for module, params in module_parameters.items()}


def _resolve_parameter_entries(
    modules: Sequence[str], module_parameters: Mapping[str, Mapping[str, Any]]
) -> tuple[_ParameterEntry, ...]:
    variable_counts = Counter()
    for module in modules:
        for parameter, spec in module_parameters.get(module, {}).items():
            if _spec_is_variable(spec):
                variable_counts[str(parameter)] += 1

    entries: list[_ParameterEntry] = []
    for module in modules:
        for parameter, spec in module_parameters.get(module, {}).items():
            parameter = str(parameter)
            default_name = parameter if variable_counts[parameter] <= 1 else f"{module}.{parameter}"
            entries.append(_parse_parameter_spec(module, parameter, spec, default_name))

    flat_names = [entry.flat_name for entry in entries if entry.prior is not None]
    duplicates = sorted(name for name, count in Counter(flat_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate flat CIGALE parameter name(s): {', '.join(duplicates)}")
    return tuple(entries)


def _parse_parameter_spec(module: str, parameter: str, spec: Any, default_name: str) -> _ParameterEntry:
    if isinstance(spec, MappingABC):
        dtype = str(spec.get("dtype", "float"))
        flat_name = str(spec.get("name", spec.get("alias", default_name)))
        if "range" in spec:
            values = tuple(spec["range"])
            if len(values) != 2:
                raise ValueError(f"{module}.{parameter} range specs must contain exactly two values.")
            low, high = values
            scale = str(spec.get("scale", "linear")).lower()
            if dtype in {"int", "integer"}:
                if scale not in {"linear", "lin"}:
                    raise ValueError(f"{module}.{parameter} integer ranges only support linear scale.")
                prior = IntegerUniformPrior(int(low), int(high))
            elif scale in {"log", "loguniform", "log-uniform"}:
                prior = LogUniformPrior(float(low), float(high))
            elif scale in {"linear", "lin"}:
                prior = UniformPrior(float(low), float(high))
            else:
                raise ValueError(f"Unsupported scale {scale!r} for {module}.{parameter}.")
            return _ParameterEntry(module, parameter, flat_name, prior, dtype=dtype)
        if "values" in spec:
            return _entry_from_values(module, parameter, spec["values"], flat_name, dtype)
        if "fixed" in spec:
            return _ParameterEntry(module, parameter, flat_name, None, fixed_value=spec["fixed"], dtype=dtype)
        if "value" in spec:
            return _ParameterEntry(module, parameter, flat_name, None, fixed_value=spec["value"], dtype=dtype)
        raise ValueError(
            f"CIGALE parameter spec for {module}.{parameter} must contain one of: range, values, fixed, value."
        )

    if _is_sequence(spec):
        return _entry_from_values(module, parameter, spec, default_name, "float")
    return _ParameterEntry(module, parameter, default_name, None, fixed_value=spec, dtype=_infer_dtype(spec))


def _entry_from_values(module: str, parameter: str, values: Sequence[Any], flat_name: str, dtype: str) -> _ParameterEntry:
    values = list(values)
    if not values:
        raise ValueError(f"{module}.{parameter} values list must not be empty.")
    if len(values) == 1:
        return _ParameterEntry(module, parameter, flat_name, None, fixed_value=values[0], dtype=dtype)
    try:
        prior = ChoicePrior([float(value) for value in values])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Variable values for {module}.{parameter} must be finite numeric values.") from exc
    return _ParameterEntry(module, parameter, flat_name, prior, dtype=dtype)


def _spec_is_variable(spec: Any) -> bool:
    if isinstance(spec, MappingABC):
        if "range" in spec:
            return True
        if "values" in spec:
            return len(list(spec["values"])) > 1
        return False
    if _is_sequence(spec):
        return len(list(spec)) > 1
    return False


def _is_sequence(value: Any) -> bool:
    return isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def _infer_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    return "float" if isinstance(value, float) else "object"


def _coerce_cigale_value(value: Any, dtype: str) -> Any:
    if dtype in {"int", "integer"}:
        return int(round(float(value)))
    if dtype in {"bool", "boolean"}:
        return _truthy(value)
    if dtype in {"float", "double"}:
        return float(value)
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _is_sfh_module(module: str) -> bool:
    return module.split(".", 1)[0].lower().startswith("sfh")


def _first_redshifting_module(modules: Sequence[str]) -> str | None:
    for module in modules:
        if module.split(".", 1)[0] == "redshifting":
            return module
    return None


def _fnu_mjy_to_flambda_cgs_per_a(wave_a: Sequence[float], fnu_mjy: Sequence[float]) -> np.ndarray:
    wave_a = np.asarray(wave_a, dtype=float)
    fnu_mjy = np.asarray(fnu_mjy, dtype=float)
    if wave_a.ndim != 1 or fnu_mjy.ndim != 1 or wave_a.shape != fnu_mjy.shape:
        raise ValueError("CIGALE wavelength and fnu arrays must be matching 1D arrays.")
    if wave_a.size < 2 or np.any(np.diff(wave_a) <= 0.0):
        raise FloatingPointError("CIGALE wavelength grid must be strictly increasing.")
    if not np.all(np.isfinite(wave_a)) or not np.all(np.isfinite(fnu_mjy)):
        raise FloatingPointError("CIGALE spectrum contains non-finite values.")
    return (fnu_mjy * 1e-26) * C_A_PER_S / wave_a**2


def _sample_or_clip_spectrum(
    wavelength: Sequence[float],
    flux: Sequence[float],
    requested_wavelengths: Sequence[float] | None,
    wavelength_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(wavelength, dtype=float)
    spec = np.asarray(flux, dtype=float)
    if requested_wavelengths is not None:
        out_wave = np.asarray(requested_wavelengths, dtype=float)
        if out_wave.ndim != 1:
            raise ValueError("Requested wavelengths must be one-dimensional.")
        out_flux = np.interp(out_wave, wave, spec, left=np.nan, right=np.nan)
    else:
        out_wave = wave.copy()
        out_flux = spec.copy()

    if wavelength_range is not None:
        lo, hi = wavelength_range
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("wavelength_range must be a finite increasing (min, max) pair.")
        keep = (out_wave >= lo) & (out_wave <= hi)
        out_wave = out_wave[keep]
        out_flux = out_flux[keep]
    return out_wave, out_flux


def _module_available(name: str) -> bool:
    module = sys.modules.get(name)
    if module is not None:
        return True
    return importlib.util.find_spec(name) is not None


def _clear_cigale_photometry_caches() -> None:
    """Clear CIGALE native photometry caches that grow with continuous redshift.

    CIGALE keys its native filter-integration caches by wavelength-grid size,
    filter name, nebular line width, and redshift.  That is excellent for a
    finite CIGALE grid, but stochastic samplers can generate thousands of
    unique redshifts and therefore thousands of never-reused cache entries.
    """

    try:
        from pcigale.sed import SED
        from pcigale.sed import utils as sed_utils
    except Exception:  # pragma: no cover - pcigale is optional
        return

    SED.cache_filters.clear()
    sed_utils.dx_cache.clear()
    sed_utils.best_grid_cache.clear()


__all__ = [
    "CIGALEBackend",
    "MJY_PER_MAGGIE",
    "build_cigale_backend_and_parameter_space",
    "build_cigale_parameter_space",
]
