"""Small, explicit photometric noise models for simulation-based inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from composed.units import canonical_photometric_flux_unit, convert_photometric_flux


@dataclass
class ConditionalCatalogNoise:
    """Joint survey uncertainty model ``q(log10 sigma_catalog | magnitude)``.

    One training row contains all bands in ``band_names``. The conditional MAF
    therefore learns both brightness trends and correlations between survey
    uncertainties. At simulation time the input is the full noiseless model
    flux vector in ``flux_unit``; it is converted to AB magnitude without
    clipping, checked against the training support, and used to draw one
    strictly positive catalog-sigma vector.

    The random generator is explicit. This object returns only
    ``sigma_catalog``. A CompoSED :class:`~composed.problem.Problem` separately
    combines it with any declared model discrepancy when drawing flux.
    """

    band_names: tuple[str, ...]
    flux_unit: str
    magnitude_min: np.ndarray
    magnitude_max: np.ndarray
    estimator_configuration: Mapping[str, object]
    estimator_state: Mapping[str, np.ndarray]
    theta_mean: np.ndarray
    theta_std: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    history: Mapping[str, Sequence[float]]
    provenance: Mapping[str, object]
    support_policy: str = "warn"
    sampling_device: str | None = "cpu"
    _estimator: object | None = field(default=None, init=False, repr=False, compare=False)
    _warned_out_of_support: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.band_names = tuple(str(name) for name in self.band_names)
        if not self.band_names or len(set(self.band_names)) != len(self.band_names):
            raise ValueError("band_names must be a non-empty sequence of unique names.")
        self.flux_unit = canonical_photometric_flux_unit(self.flux_unit)
        n_band = len(self.band_names)
        self.magnitude_min = _finite_vector(self.magnitude_min, n_band, "magnitude_min")
        self.magnitude_max = _finite_vector(self.magnitude_max, n_band, "magnitude_max")
        if np.any(self.magnitude_max <= self.magnitude_min):
            raise ValueError("Every magnitude support interval must have max > min.")
        self.theta_mean = _finite_vector(self.theta_mean, n_band, "theta_mean")
        self.theta_std = _positive_vector(self.theta_std, n_band, "theta_std")
        self.x_mean = _finite_vector(self.x_mean, n_band, "x_mean")
        self.x_std = _positive_vector(self.x_std, n_band, "x_std")
        self.estimator_configuration = dict(self.estimator_configuration)
        self.estimator_state = {
            str(name): np.asarray(value).copy()
            for name, value in self.estimator_state.items()
        }
        if not self.estimator_state:
            raise ValueError("estimator_state cannot be empty.")
        self.history = {str(name): list(values) for name, values in self.history.items()}
        self.provenance = dict(self.provenance)
        policy = str(self.support_policy).strip().lower()
        if policy not in {"warn", "raise", "ignore"}:
            raise ValueError("support_policy must be 'warn', 'raise', or 'ignore'.")
        self.support_policy = policy

    @classmethod
    def fit(
        cls,
        catalog_magnitudes: np.ndarray,
        catalog_sigma: np.ndarray,
        *,
        band_names: Sequence[str],
        flux_unit: str = "maggies",
        seed: int | None = None,
        hidden_features: int = 64,
        num_transforms: int = 5,
        num_blocks: int = 2,
        learning_rate: float = 1.0e-3,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.1,
        patience: int | None = 20,
        min_delta: float = 0.0,
        device: str | None = "cpu",
        support_policy: str = "warn",
        invalid_rows: str = "filter",
        catalog_source: str | None = None,
        row_selection: str | None = None,
        verbose: bool = False,
    ) -> "ConditionalCatalogNoise":
        """Fit a joint conditional flow from complete multiband catalog rows.

        ``catalog_magnitudes`` are AB magnitudes. ``catalog_sigma`` must be raw
        catalog flux uncertainties in ``flux_unit`` and must not contain a
        model-discrepancy term. Rows with any non-finite magnitude or
        non-positive/non-finite sigma either raise or are removed as complete
        rows according to ``invalid_rows``; no band is imputed independently.
        """

        magnitudes = np.asarray(catalog_magnitudes, dtype=float)
        sigma = np.asarray(catalog_sigma, dtype=float)
        names = tuple(str(name) for name in band_names)
        if magnitudes.ndim != 2 or sigma.ndim != 2 or magnitudes.shape != sigma.shape:
            raise ValueError(
                "catalog_magnitudes and catalog_sigma must have the same two-dimensional shape."
            )
        if magnitudes.shape[1] != len(names):
            raise ValueError(
                f"band_names has {len(names)} entries but catalog arrays have {magnitudes.shape[1]} columns."
            )
        if not names or len(set(names)) != len(names):
            raise ValueError("band_names must be non-empty and unique.")
        if magnitudes.shape[0] < 2:
            raise ValueError("At least two catalog rows are required to fit ConditionalCatalogNoise.")
        invalid_policy = str(invalid_rows).strip().lower()
        if invalid_policy not in {"filter", "raise"}:
            raise ValueError("invalid_rows must be 'filter' or 'raise'.")

        valid = np.all(np.isfinite(magnitudes), axis=1)
        valid &= np.all(np.isfinite(sigma) & (sigma > 0.0), axis=1)
        rejected = np.flatnonzero(~valid)
        if rejected.size and invalid_policy == "raise":
            raise ValueError(
                "ConditionalCatalogNoise requires complete finite magnitude rows and strictly "
                f"positive finite sigma rows; invalid row indices begin {rejected[:10].tolist()}."
            )
        if rejected.size:
            warnings.warn(
                f"ConditionalCatalogNoise filtered {rejected.size}/{magnitudes.shape[0]} incomplete "
                "or invalid catalog rows; the count and row-index hash are stored in provenance.",
                RuntimeWarning,
                stacklevel=2,
            )
        magnitudes_fit = magnitudes[valid]
        sigma_fit = sigma[valid]
        if magnitudes_fit.shape[0] < 2:
            raise ValueError("Fewer than two complete catalog rows remain after filtering.")

        from inftools.sbi import MAFPosteriorEstimator

        estimator = MAFPosteriorEstimator(
            theta_dim=len(names),
            x_dim=len(names),
            hidden_features=int(hidden_features),
            num_transforms=int(num_transforms),
            num_blocks=int(num_blocks),
            learning_rate=float(learning_rate),
            device=device,
            standardize=True,
            initialization_seed=seed,
        )
        log10_sigma = np.log10(sigma_fit)
        history = estimator.fit(
            log10_sigma,
            magnitudes_fit,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=float(validation_split),
            patience=patience,
            min_delta=float(min_delta),
            seed=seed,
            verbose=bool(verbose),
        )
        state = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in estimator.flow.state_dict().items()
        }
        input_hash = _catalog_training_hash(
            magnitudes,
            sigma,
            band_names=names,
            flux_unit=flux_unit,
        )
        rejected_hash = sha256(np.asarray(rejected, dtype=np.int64).tobytes()).hexdigest()
        provenance = {
            "schema": "composed.conditional_catalog_noise.v1",
            "band_names": names,
            "flux_unit": canonical_photometric_flux_unit(flux_unit),
            "magnitude_convention": "AB: m = -2.5 log10(f_nu / 3631 Jy); maggies use m = -2.5 log10(flux)",
            "target_transform": "log10(sigma_catalog in flux_unit)",
            "conditioning_transform": "AB magnitude, then training-set standardization",
            "standardization": "per-band mean and standard deviation fit on retained training rows",
            "seed": None if seed is None else int(seed),
            "architecture": estimator.configuration(),
            "package_versions": {
                "composed": _package_version("composed-sed"),
                "numpy": str(np.__version__),
                "torch": str(getattr(estimator.torch, "__version__", "unknown")),
                "nflows": _package_version("nflows"),
            },
            "catalog_array_sha256": input_hash,
            "catalog_source": catalog_source,
            "row_selection": row_selection or "all caller-provided rows; complete multiband rows retained",
            "n_input_rows": int(magnitudes.shape[0]),
            "n_training_rows": int(magnitudes_fit.shape[0]),
            "n_rejected_rows": int(rejected.size),
            "rejected_row_indices_sha256": rejected_hash,
            "rejected_row_indices_first_100": tuple(int(value) for value in rejected[:100]),
            "training_magnitude_min": tuple(float(value) for value in np.min(magnitudes_fit, axis=0)),
            "training_magnitude_max": tuple(float(value) for value in np.max(magnitudes_fit, axis=0)),
        }
        result = cls(
            band_names=names,
            flux_unit=flux_unit,
            magnitude_min=np.min(magnitudes_fit, axis=0),
            magnitude_max=np.max(magnitudes_fit, axis=0),
            estimator_configuration=estimator.configuration(),
            estimator_state=state,
            theta_mean=estimator.theta_standardizer.mean,
            theta_std=estimator.theta_standardizer.std,
            x_mean=estimator.x_standardizer.mean,
            x_std=estimator.x_standardizer.std,
            history=history,
            provenance=provenance,
            support_policy=support_policy,
            sampling_device="cpu",
        )
        # Retain the fitted object for immediate use when it already lives on
        # CPU. Other devices are rebuilt lazily on CPU for process-safe survey
        # simulation.
        if str(estimator.device) == "cpu":
            result._estimator = estimator
        return result

    def validate_for(self, *, band_names: Sequence[str], flux_unit: str) -> None:
        """Fail if a simulator would silently reorder bands or mix flux units."""

        supplied_names = tuple(str(name) for name in band_names)
        if supplied_names != self.band_names:
            raise ValueError(
                "ConditionalCatalogNoise band order does not match the active simulator bands: "
                f"trained={self.band_names}, supplied={supplied_names}."
            )
        supplied_unit = canonical_photometric_flux_unit(flux_unit)
        if supplied_unit != self.flux_unit:
            raise ValueError(
                "ConditionalCatalogNoise flux unit does not match the simulator: "
                f"trained={self.flux_unit!r}, supplied={supplied_unit!r}."
            )

    def sample(
        self,
        model_flux: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``sigma_catalog`` for one or many noiseless full-band SEDs."""

        if rng is None or not isinstance(rng, np.random.Generator):
            raise TypeError("ConditionalCatalogNoise.sample requires an explicit numpy.random.Generator.")
        flux = np.asarray(model_flux, dtype=float)
        single = flux.ndim == 1
        if single:
            flux_batch = flux[None, :]
        elif flux.ndim == 2:
            flux_batch = flux
        else:
            raise ValueError("model_flux must have shape (n_bands,) or (n_objects, n_bands).")
        expected = len(self.band_names)
        if flux_batch.shape[1] != expected:
            raise ValueError(
                f"ConditionalCatalogNoise expected {expected} bands, got shape {flux.shape}."
            )
        if not np.all(np.isfinite(flux_batch)) or np.any(flux_batch <= 0.0):
            raise ValueError(
                "ConditionalCatalogNoise requires finite strictly positive noiseless flux in every band "
                "because AB magnitude is its conditioning variable."
            )
        flux_maggies = convert_photometric_flux(flux_batch, self.flux_unit, "maggies")
        magnitude = -2.5 * np.log10(flux_maggies)
        self._check_support(magnitude)
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        sampled_log_sigma = np.asarray(
            self._get_estimator().sample(magnitude, num_samples=1, seed=seed),
            dtype=float,
        )
        if magnitude.shape[0] == 1:
            sampled_log_sigma = sampled_log_sigma.reshape(1, expected)
        else:
            sampled_log_sigma = sampled_log_sigma[:, 0, :]
        minimum_log10 = np.log10(np.nextafter(0.0, 1.0))
        maximum_log10 = np.log10(np.finfo(float).max)
        if (
            not np.all(np.isfinite(sampled_log_sigma))
            or np.any(sampled_log_sigma < minimum_log10)
            or np.any(sampled_log_sigma > maximum_log10)
        ):
            raise RuntimeError(
                "ConditionalCatalogNoise flow produced log10 uncertainties "
                "outside the finite floating-point range."
            )
        sigma = 10.0**sampled_log_sigma
        if sigma.shape != flux_batch.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise RuntimeError("ConditionalCatalogNoise flow produced invalid catalog uncertainties.")
        return sigma[0] if single else sigma

    def __call__(
        self,
        model_flux: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        theta=None,
    ) -> np.ndarray:
        """Noise-function interface used by :class:`composed.Simulate`."""

        del theta
        if rng is None:
            raise TypeError("ConditionalCatalogNoise requires the simulator to pass an explicit rng.")
        return self.sample(model_flux, rng=rng)

    def specification(self) -> dict[str, object]:
        """Return the scientific configuration without embedding neural weights."""

        return {
            "name": type(self).__name__,
            "schema": self.provenance.get("schema"),
            "band_names": self.band_names,
            "flux_unit": self.flux_unit,
            "support_policy": self.support_policy,
            "magnitude_min": tuple(float(value) for value in self.magnitude_min),
            "magnitude_max": tuple(float(value) for value in self.magnitude_max),
            "estimator_configuration": dict(self.estimator_configuration),
            "training_provenance": dict(self.provenance),
            "model_state_sha256": _state_hash(self.estimator_state),
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save the learned flow, transforms, support, and provenance."""

        path = Path(path)
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"Noise checkpoint path is not a directory: {path}")
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"Noise checkpoint directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        state_names = tuple(self.estimator_state)
        arrays: dict[str, np.ndarray] = {
            "magnitude_min": self.magnitude_min,
            "magnitude_max": self.magnitude_max,
            "theta_mean": self.theta_mean,
            "theta_std": self.theta_std,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
        }
        for index, name in enumerate(state_names):
            arrays[f"state_{index:04d}"] = np.asarray(self.estimator_state[name])
        np.savez_compressed(path / "arrays.npz", **arrays)
        manifest = {
            "format": "composed.conditional_catalog_noise.v1",
            "band_names": self.band_names,
            "flux_unit": self.flux_unit,
            "support_policy": self.support_policy,
            "sampling_device": self.sampling_device,
            "estimator_configuration": dict(self.estimator_configuration),
            "state_names": state_names,
            "history": dict(self.history),
            "provenance": dict(self.provenance),
            "model_state_sha256": _state_hash(self.estimator_state),
        }
        with (path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = "cpu",
    ) -> "ConditionalCatalogNoise":
        """Load a learned survey-noise model on a requested sampling device."""

        path = Path(path)
        with (path / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != "composed.conditional_catalog_noise.v1":
            raise ValueError(f"Unsupported ConditionalCatalogNoise format {manifest.get('format')!r}.")
        with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
            state = {
                name: np.asarray(arrays[f"state_{index:04d}"])
                for index, name in enumerate(manifest["state_names"])
            }
            result = cls(
                band_names=tuple(manifest["band_names"]),
                flux_unit=str(manifest["flux_unit"]),
                magnitude_min=arrays["magnitude_min"],
                magnitude_max=arrays["magnitude_max"],
                estimator_configuration=dict(manifest["estimator_configuration"]),
                estimator_state=state,
                theta_mean=arrays["theta_mean"],
                theta_std=arrays["theta_std"],
                x_mean=arrays["x_mean"],
                x_std=arrays["x_std"],
                history=dict(manifest.get("history", {})),
                provenance=dict(manifest["provenance"]),
                support_policy=str(manifest["support_policy"]),
                sampling_device=device,
            )
        if _state_hash(result.estimator_state) != manifest.get("model_state_sha256"):
            raise ValueError("ConditionalCatalogNoise neural weights do not match the saved manifest digest.")
        return result

    def _check_support(self, magnitude: np.ndarray) -> None:
        outside = (magnitude < self.magnitude_min[None, :]) | (
            magnitude > self.magnitude_max[None, :]
        )
        if not np.any(outside) or self.support_policy == "ignore":
            return
        bands = tuple(
            self.band_names[index]
            for index in np.flatnonzero(np.any(outside, axis=0))
        )
        message = (
            "ConditionalCatalogNoise received model magnitudes outside its training support "
            f"in band(s) {bands}. Values are extrapolated without clamping."
        )
        if self.support_policy == "raise":
            raise ValueError(message)
        if not self._warned_out_of_support:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            self._warned_out_of_support = True

    def _get_estimator(self):
        if self._estimator is not None:
            return self._estimator
        from inftools.sbi import MAFPosteriorEstimator, Standardizer

        configuration = dict(self.estimator_configuration)
        configuration["device"] = self.sampling_device
        estimator = MAFPosteriorEstimator(**configuration)
        estimator.theta_standardizer = Standardizer(self.theta_mean, self.theta_std)
        estimator.x_standardizer = Standardizer(self.x_mean, self.x_std)
        state = {
            name: estimator.torch.as_tensor(value)
            for name, value in self.estimator_state.items()
        }
        estimator.flow.load_state_dict(state)
        estimator.flow.eval()
        estimator.history = {name: list(values) for name, values in self.history.items()}
        self._estimator = estimator
        return estimator

    def __getstate__(self):
        """Exclude the live torch module when pickling process workers."""

        state = self.__dict__.copy()
        state["_estimator"] = None
        state["_warned_out_of_support"] = False
        return state


@dataclass(frozen=True)
class EmpiricalPhotometricNoise:
    """Sample complete empirical catalog-uncertainty rows.

    ``sigma_rows`` has shape ``(n_objects, n_bands)`` in the same flux unit as
    the simulator. Sampling one complete row preserves empirical correlations
    in survey depth between bands. The returned uncertainty is

    ``sqrt((sigma_scale * empirical_sigma)**2 + (fractional_error * abs(flux))**2)``.

    Set ``fractional_error=0`` when model discrepancy is declared separately on
    ``Gaussian(photometric_model_discrepancy=...)``. This keeps the empirical
    row as raw ``sigma_catalog`` and applies model discrepancy exactly once.
    """

    sigma_rows: np.ndarray
    fractional_error: float = 0.05
    sigma_scale: float = 1.0
    band_names: Sequence[str] | None = None
    flux_unit: str | None = None

    def __post_init__(self) -> None:
        sigma = np.asarray(self.sigma_rows, dtype=float)
        if sigma.ndim != 2 or sigma.shape[0] == 0 or sigma.shape[1] == 0:
            raise ValueError("sigma_rows must have shape (n_objects, n_bands) with non-zero dimensions.")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma_rows must contain finite strictly positive uncertainties.")
        fractional_error = float(self.fractional_error)
        sigma_scale = float(self.sigma_scale)
        if not np.isfinite(fractional_error) or fractional_error < 0.0:
            raise ValueError("fractional_error must be finite and non-negative.")
        if not np.isfinite(sigma_scale) or sigma_scale <= 0.0:
            raise ValueError("sigma_scale must be finite and positive.")
        if self.band_names is None:
            band_names = tuple(f"band_{i}" for i in range(sigma.shape[1]))
        else:
            band_names = tuple(str(name) for name in self.band_names)
            if len(band_names) != sigma.shape[1] or len(set(band_names)) != len(band_names):
                raise ValueError("band_names must be unique and match sigma_rows columns.")

        object.__setattr__(self, "sigma_rows", sigma)
        object.__setattr__(self, "fractional_error", fractional_error)
        object.__setattr__(self, "sigma_scale", sigma_scale)
        object.__setattr__(self, "band_names", band_names)
        if self.flux_unit is not None:
            object.__setattr__(self, "flux_unit", str(self.flux_unit))

    def __call__(self, flux, *, rng: np.random.Generator | None = None, theta=None) -> np.ndarray:
        """Return one sigma vector matching a noiseless one-dimensional flux."""

        del theta
        flux = np.asarray(flux, dtype=float)
        if flux.shape != (self.sigma_rows.shape[1],):
            raise ValueError(
                f"Noise model expected flux shape {(self.sigma_rows.shape[1],)}, got {flux.shape}."
            )
        if not np.all(np.isfinite(flux)):
            raise ValueError("Noise model received non-finite flux.")
        if rng is None:
            rng = np.random.default_rng()
        row = self.sigma_rows[int(rng.integers(0, self.sigma_rows.shape[0]))]
        return np.hypot(self.sigma_scale * row, self.fractional_error * np.abs(flux))

    def specification(self) -> dict[str, object]:
        return {
            "name": type(self).__name__,
            "n_empirical_rows": int(self.sigma_rows.shape[0]),
            "band_names": self.band_names,
            "fractional_error": self.fractional_error,
            "sigma_scale": self.sigma_scale,
            "flux_unit": self.flux_unit,
        }


def _finite_vector(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape {(size,)}.")
    return array.copy()


def _positive_vector(values, size: int, name: str) -> np.ndarray:
    array = _finite_vector(values, size, name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return array


def _catalog_training_hash(
    magnitudes: np.ndarray,
    sigma: np.ndarray,
    *,
    band_names: Sequence[str],
    flux_unit: str,
) -> str:
    digest = sha256()
    for array in (np.asarray(magnitudes, dtype=np.float64), np.asarray(sigma, dtype=np.float64)):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps(tuple(band_names), separators=(",", ":")).encode("utf-8"))
    digest.update(canonical_photometric_flux_unit(flux_unit).encode("ascii"))
    return digest.hexdigest()


def _state_hash(state: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name in sorted(state):
        array = np.asarray(state[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _json_safe(value: Any):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
