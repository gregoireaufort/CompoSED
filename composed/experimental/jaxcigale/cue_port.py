from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from composed.experimental.jaxcigale.cue import (
    CueDerivedInputs,
    cue_logq_from_logu,
    cue_theta12_to_public_package_theta,
    zero_numerical_lyc_floor,
)
from composed.experimental.jaxcigale.dependencies import require_jax
from composed.experimental.jaxcigale.photometry import C_A_PER_S
from composed.units import LSUN_CGS


LINE_GROUP_NAMES = (
    "H1",
    "He1",
    "He2",
    "C1",
    "C2C3",
    "C4",
    "N",
    "O1",
    "O2",
    "O3",
    "ionE_1",
    "ionE_2",
    "S4",
    "Ar4",
    "Ne3",
    "Ne4",
)

LINE_GROUP_IONS = (
    ("H  1",),
    ("He 1",),
    ("He 2",),
    ("C  1",),
    ("C  2", "C  3"),
    ("C  4",),
    ("N  1", "N  2", "N  3"),
    ("O  1",),
    ("O  2",),
    ("O  3",),
    ("Mg 2", "Fe 2", "Si 2", "Al 2", "P  2", "S  2", "Cl 2", "Ar 2"),
    ("Al 3", "Si 3", "S  3", "Cl 3", "Ar 3", "Ne 2"),
    ("S  4",),
    ("Ar 4",),
    ("Ne 3",),
    ("Ne 4",),
)

CUE_ADDED_LINE_WAVELENGTHS_A = np.asarray(
    [
        4685.68,
        1550.77,
        1548.19,
        1750.00,
        2424.28,
        1882.71,
        1892.03,
        1406.02,
        4711.26,
        4740.12,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class CueSpeculatorWeights:
    """Pure-array copy of one public Cue Speculator neural network."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    alphas: tuple[np.ndarray, ...]
    betas: tuple[np.ndarray, ...]
    parameter_shift: np.ndarray
    parameter_scale: np.ndarray
    pca_shift: np.ndarray
    pca_scale: np.ndarray
    log_spectrum_shift: np.ndarray
    log_spectrum_scale: np.ndarray

    @classmethod
    def from_public_pickle(cls, path: str | Path) -> "CueSpeculatorWeights":
        """Load a public Cue ``speculator_*.pkl`` file without TensorFlow.

        The public pickle contains a TensorFlow ``ListWrapper`` reference even
        though the saved values are plain arrays. We provide a tiny fake class
        only while unpickling so the JAX port does not depend on TensorFlow.
        """

        dill = _require_dill()
        _install_fake_tensorflow_listwrapper()
        raw = dill.load(open(path, "rb"))
        return cls.from_public_pickle_payload(raw)

    @classmethod
    def from_public_pickle_payload(cls, raw: Sequence[object]) -> "CueSpeculatorWeights":
        if len(raw) < 10:
            raise ValueError("Cue Speculator pickle payload is shorter than expected.")
        return cls(
            weights=tuple(np.asarray(x, dtype=float) for x in raw[0]),
            biases=tuple(np.asarray(x, dtype=float) for x in raw[1]),
            alphas=tuple(np.asarray(x, dtype=float) for x in raw[2]),
            betas=tuple(np.asarray(x, dtype=float) for x in raw[3]),
            parameter_shift=np.asarray(raw[4], dtype=float),
            parameter_scale=np.asarray(raw[5], dtype=float),
            pca_shift=np.asarray(raw[6], dtype=float),
            pca_scale=np.asarray(raw[7], dtype=float),
            log_spectrum_shift=np.asarray(raw[8], dtype=float),
            log_spectrum_scale=np.asarray(raw[9], dtype=float),
        )

    def pca_coefficients_numpy(self, theta_public: np.ndarray) -> np.ndarray:
        theta = _as_2d_numpy(theta_public)
        layer = (theta - self.parameter_shift) / self.parameter_scale
        for w, b, alpha, beta in zip(self.weights[:-1], self.biases[:-1], self.alphas, self.betas):
            pre = layer @ w + b
            layer = (beta + (1.0 - beta) / (1.0 + np.exp(-alpha * pre))) * pre
        return (layer @ self.weights[-1] + self.biases[-1]) * self.pca_scale + self.pca_shift

    def pca_coefficients_jax(self, theta_public):
        jax, jnp = require_jax()
        theta = _as_2d_jax(theta_public)
        layer = (theta - jnp.asarray(self.parameter_shift)) / jnp.asarray(self.parameter_scale)
        for w, b, alpha, beta in zip(self.weights[:-1], self.biases[:-1], self.alphas, self.betas):
            pre = layer @ jnp.asarray(w) + jnp.asarray(b)
            layer = (jnp.asarray(beta) + (1.0 - jnp.asarray(beta)) * jax.nn.sigmoid(jnp.asarray(alpha) * pre)) * pre
        return (layer @ jnp.asarray(self.weights[-1]) + jnp.asarray(self.biases[-1])) * jnp.asarray(
            self.pca_scale
        ) + jnp.asarray(self.pca_shift)


@dataclass(frozen=True)
class CuePCAWeights:
    """Pure-array inverse-PCA state for one Cue continuum or line block."""

    components: np.ndarray
    mean: np.ndarray

    @classmethod
    def from_public_pickle(cls, path: str | Path) -> "CuePCAWeights":
        dill = _require_dill()
        _patch_numpy_for_public_cue()
        # The public PCA pickles need the Cue package classes in import scope.
        raw = dill.load(open(path, "rb"))
        return cls.from_public_pca_object(raw)

    @classmethod
    def from_public_pca_object(cls, raw: object) -> "CuePCAWeights":
        pca = getattr(raw, "PCA", None)
        if pca is None:
            raise ValueError("Cue PCA pickle does not expose a PCA attribute.")
        return cls(
            components=np.asarray(pca.components_, dtype=float),
            mean=np.asarray(pca.mean_, dtype=float),
        )

    def inverse_transform_numpy(self, coefficients: np.ndarray) -> np.ndarray:
        return np.asarray(coefficients) @ self.components + self.mean

    def inverse_transform_jax(self, coefficients):
        _, jnp = require_jax()
        return jnp.asarray(coefficients) @ jnp.asarray(self.components) + jnp.asarray(self.mean)


@dataclass(frozen=True)
class CueBlockPort:
    """One Cue emulator block: Speculator coefficients plus inverse PCA."""

    speculator: CueSpeculatorWeights
    pca: CuePCAWeights
    wavelength_a: np.ndarray

    def predict_log10_numpy(self, theta_public: np.ndarray) -> np.ndarray:
        coeff = self.speculator.pca_coefficients_numpy(theta_public)
        return self.pca.inverse_transform_numpy(coeff) * self.speculator.log_spectrum_scale + self.speculator.log_spectrum_shift

    def predict_log10_jax(self, theta_public):
        _, jnp = require_jax()
        coeff = self.speculator.pca_coefficients_jax(theta_public)
        return self.pca.inverse_transform_jax(coeff) * jnp.asarray(self.speculator.log_spectrum_scale) + jnp.asarray(
            self.speculator.log_spectrum_shift
        )

    def predict_numpy(self, theta_public: np.ndarray) -> np.ndarray:
        return 10.0 ** self.predict_log10_numpy(theta_public)

    def predict_jax(self, theta_public):
        _, jnp = require_jax()
        return 10.0 ** self.predict_log10_jax(theta_public)


@dataclass(frozen=True)
class CueJaxPort:
    """JAX port of the public Cue continuum and line emulators.

    The port reproduces Cue's public NumPy/TensorFlow/PCA calculation using
    JAX arrays. It does not require TensorFlow at evaluation time.
    """

    continuum: CueBlockPort
    line_blocks: tuple[CueBlockPort, ...]
    line_wavelength_a: np.ndarray
    line_sort_index: np.ndarray
    line_keep_index: np.ndarray

    @classmethod
    def from_public_cue_data_dir(cls, data_dir: str | Path) -> "CueJaxPort":
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Cue data directory does not exist: {data_dir}")
        _patch_numpy_for_public_cue()
        _make_public_cue_source_importable(data_dir)
        continuum_wave = np.genfromtxt(data_dir / "FSPSlam.dat")[122:]
        continuum = CueBlockPort(
            speculator=CueSpeculatorWeights.from_public_pickle(data_dir / "speculator_cont_new.pkl"),
            pca=CuePCAWeights.from_public_pickle(data_dir / "pca_cont_new.pkl"),
            wavelength_a=np.asarray(continuum_wave, dtype=float),
        )
        line_blocks = tuple(
            CueBlockPort(
                speculator=CueSpeculatorWeights.from_public_pickle(data_dir / f"speculator_line_new_{name}.pkl"),
                pca=CuePCAWeights.from_public_pickle(data_dir / f"pca_line_new_{name}.pkl"),
                wavelength_a=np.asarray([], dtype=float),
            )
            for name in LINE_GROUP_NAMES
        )
        grouped_line_wavelength = _public_cue_grouped_line_wavelengths(data_dir)
        line_sort_index, line_keep_index = _public_cue_line_sort_and_old_indices(grouped_line_wavelength)
        line_wavelength = grouped_line_wavelength[line_sort_index][line_keep_index]
        return cls(
            continuum=continuum,
            line_blocks=line_blocks,
            line_wavelength_a=line_wavelength,
            line_sort_index=line_sort_index,
            line_keep_index=line_keep_index,
        )

    def predict_continuum_native_numpy(self, theta_public: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.continuum.wavelength_a, self.continuum.predict_numpy(theta_public)

    def predict_continuum_native_jax(self, theta_public):
        return self.continuum.wavelength_a, self.continuum.predict_jax(theta_public)

    def predict_lines_native_numpy(self, theta_public: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pieces = [block.predict_log10_numpy(theta_public) for block in self.line_blocks]
        log_values = np.asarray(np.hstack(pieces))[:, self.line_sort_index][:, self.line_keep_index]
        return self.line_wavelength_a, 10.0**log_values

    def predict_lines_native_jax(self, theta_public):
        _, jnp = require_jax()
        pieces = [block.predict_log10_jax(theta_public) for block in self.line_blocks]
        log_values = jnp.asarray(jnp.hstack(pieces))[:, jnp.asarray(self.line_sort_index)][:, jnp.asarray(self.line_keep_index)]
        return self.line_wavelength_a, 10.0**log_values

    def make_nebular_apply(self, line_sigma_a: float = 1.0, lyc_numerical_floor_fraction: float = 1.0e-12):
        """Return a ``cue_nebular_module``-compatible JAX apply function."""

        line_sigma_a = float(line_sigma_a)
        lyc_numerical_floor_fraction = float(lyc_numerical_floor_fraction)

        def apply(wave_rest_a, theta12, cue_inputs: CueDerivedInputs):
            del theta12
            _, jnp = require_jax()
            wave = jnp.asarray(wave_rest_a)
            theta_public = cue_theta12_to_public_package_theta(cue_inputs)
            continuum_wave, continuum_raw = self.predict_continuum_native_jax(theta_public)
            line_wave, line_raw = self.predict_lines_native_jax(theta_public)
            logq_for_logu = cue_logq_from_logu(cue_inputs.logu, cue_inputs.logn_h)
            q_scale = 10.0 ** (cue_inputs.log_q_h_gas - logq_for_logu)

            # Public Cue continuum is native Lnu in erg/s/Hz for the reference
            # ionizing normalization. Convert to per-solar-mass Lsun/Hz using
            # the gas-powered Q_H derived from the stellar spectrum.
            continuum_lnu_lsun_per_hz = jnp.squeeze(continuum_raw, axis=0) / LSUN_CGS * q_scale
            continuum_lnu_on_grid = jnp.interp(wave, jnp.asarray(continuum_wave), continuum_lnu_lsun_per_hz, left=0.0, right=0.0)
            continuum_lsun_per_a = continuum_lnu_on_grid * C_A_PER_S / jnp.maximum(wave, 1.0) ** 2

            # Public Cue line output is native integrated line luminosity in
            # erg/s for the same reference ionizing normalization. Convert to
            # Lsun and spread each line over a Gaussian in wavelength so the
            # output is an L_lambda density on the model grid.
            line_lsun = jnp.squeeze(line_raw, axis=0) / LSUN_CGS * q_scale
            sigma = jnp.asarray(line_sigma_a)
            profile = jnp.exp(-0.5 * ((wave[:, None] - jnp.asarray(line_wave)[None, :]) / sigma) ** 2)
            profile = profile / (jnp.sqrt(2.0 * jnp.pi) * sigma)
            lines_lsun_per_a = jnp.sum(profile * line_lsun[None, :], axis=1)
            continuum_lsun_per_a = zero_numerical_lyc_floor(
                wave,
                continuum_lsun_per_a,
                floor_fraction=lyc_numerical_floor_fraction,
            )
            lines_lsun_per_a = zero_numerical_lyc_floor(
                wave,
                lines_lsun_per_a,
                floor_fraction=lyc_numerical_floor_fraction,
            )
            return continuum_lsun_per_a, lines_lsun_per_a

        return apply


def _as_2d_numpy(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        return theta[None, :]
    if theta.ndim != 2:
        raise ValueError("Cue theta must have shape (12,) or (n, 12).")
    return theta


def _as_2d_jax(theta):
    _, jnp = require_jax()
    theta = jnp.asarray(theta)
    if theta.ndim == 1:
        return theta[None, :]
    if theta.ndim != 2:
        raise ValueError("Cue theta must have shape (12,) or (n, 12).")
    return theta


def _public_cue_line_sort_and_old_indices(nn_wavelength: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wav_sorted_index = np.argsort(nn_wavelength)
    sorted_wavelength = nn_wavelength[wav_sorted_index]
    new_added = np.where(np.isin(np.round(sorted_wavelength, 2), np.round(CUE_ADDED_LINE_WAVELENGTHS_A, 2)))[0]
    line_old = np.arange(sorted_wavelength.size)[~np.isin(np.arange(sorted_wavelength.size), new_added)]
    if line_old.size != sorted_wavelength.size - len(CUE_ADDED_LINE_WAVELENGTHS_A):
        raise ValueError("Cue line-list selection is inconsistent with the expected added-line count.")
    return wav_sorted_index, line_old


def _public_cue_grouped_line_wavelengths(data_dir: Path) -> np.ndarray:
    names = np.load(data_dir / "lineList_replaceblnd_name.npy")
    wavelengths = np.load(data_dir / "lineList_wav.npy")
    order = np.argsort(wavelengths)
    sorted_names = names[order]
    sorted_wavelengths = wavelengths[order]
    elements = np.asarray([str(name)[:4].rstrip() for name in sorted_names])
    selections = [np.where(np.isin(elements, np.asarray(ion_set)))[0] for ion_set in LINE_GROUP_IONS]
    return sorted_wavelengths[np.concatenate(selections)]


def _require_dill():
    try:
        import dill
    except ImportError as exc:
        raise ImportError("Loading public Cue weights requires dill. Install it with `pip install dill`.") from exc
    return dill


def _install_fake_tensorflow_listwrapper() -> None:
    import importlib.machinery
    import sys
    import types

    class ListWrapper(list):
        pass

    def fake_module(name: str):
        module = sys.modules.setdefault(name, types.ModuleType(name))
        if getattr(module, "__spec__", None) is None:
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return module

    fake_module("tensorflow")
    fake_module("tensorflow.python")
    fake_module("tensorflow.python.trackable")
    data_structures = fake_module("tensorflow.python.trackable.data_structures")
    if not hasattr(data_structures, "ListWrapper"):
        data_structures.ListWrapper = ListWrapper


def _patch_numpy_for_public_cue() -> None:
    # Cue's public code was written before NumPy removed ``np.in1d``.
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # type: ignore[attr-defined]


def _make_public_cue_source_importable(data_dir: Path) -> None:
    """Make cloned public Cue classes importable for PCA unpickling."""

    import importlib.util
    import sys
    import types

    # Expected layout: <clone>/src/cue/data.
    cue_package_dir = Path(data_dir).resolve().parent
    source_root = cue_package_dir.parent
    if not cue_package_dir.exists():
        return
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    # Do not import ``cue.__init__``: it imports the TensorFlow continuum/line
    # modules. PCA unpickling only needs these class definitions.
    cue_pkg = sys.modules.get("cue")
    if cue_pkg is None:
        cue_pkg = types.ModuleType("cue")
        cue_pkg.__path__ = [str(cue_package_dir)]  # type: ignore[attr-defined]
        sys.modules["cue"] = cue_pkg
    for module_name in ("cont_pca", "line_pca"):
        full_name = f"cue.{module_name}"
        if full_name in sys.modules:
            continue
        path = cue_package_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load public Cue module source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
