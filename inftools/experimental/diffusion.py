"""Experimental masked conditional diffusion for array-based SBI.

The learned object is the joint feature vector, for example
``[magnitudes, magnitude_errors, physical_parameters]``.  During sampling, a
boolean mask marks known coordinates.  Known coordinates are reclamped at every
reverse-diffusion step, so the sampler can be used for inverse inference,
forward prediction, missing-band inpainting, or mixed conditionals.

This module is intentionally not tied to a survey catalog.  Dataset-specific
feature construction should live in notebooks or thin scripts that produce a
plain two-dimensional training array and a matching ``FeatureMetadata``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping
import warnings

import numpy as np

from inftools.sbi import Standardizer


try:  # Keep module importable in environments without torch.
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
except ImportError:  # pragma: no cover - exercised in environments without torch.
    torch = None
    nn = None


def _require_torch():
    if torch is None or nn is None:
        raise ImportError(
            "inftools.experimental.diffusion requires the optional torch dependency. "
            "Install it with, for example: pip install torch"
        )
    return torch, nn


def resolve_torch_device(
    device: str | None = "auto",
    *,
    validate: bool = True,
    allow_fallback: bool = True,
):
    """Return a usable torch device for diffusion training and sampling.

    ``device="auto"`` tries CUDA, then Apple MPS, then CPU.  Validation runs a
    tiny float32 tensor exercise on the candidate device so long jobs do not
    start on a nominally available but unusable accelerator.  If an explicitly
    requested accelerator fails and ``allow_fallback=False``, a clear
    ``RuntimeError`` is raised.
    """

    torch_mod, _ = _require_torch()
    requested = "auto" if device is None else str(device).lower()
    if requested == "auto":
        candidates: list[str] = []
        if torch_mod.cuda.is_available():
            candidates.append("cuda")
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            candidates.append("mps")
        candidates.append("cpu")
    else:
        candidates = [requested]
        if allow_fallback and requested != "cpu":
            candidates.append("cpu")

    failures: list[str] = []
    for candidate in candidates:
        try:
            candidate_device = torch_mod.device(candidate)
            if validate:
                _validate_float32_device(torch_mod, candidate_device)
            if candidate != requested and requested != "auto":
                warnings.warn(
                    f"Requested torch device {requested!r} failed validation; "
                    f"falling back to {candidate!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return candidate_device
        except Exception as exc:
            failures.append(f"{candidate}: {exc!r}")
            if requested != "auto" and not allow_fallback:
                raise RuntimeError(
                    f"Requested torch device {requested!r} is not usable for diffusion float32 workloads. "
                    f"Validation failure: {exc!r}"
                ) from exc

    raise RuntimeError("No usable torch device found. Validation failures: " + "; ".join(failures))


@dataclass(frozen=True)
class FeatureMetadata:
    """Column names and scientific groups for a diffusion feature vector.

    ``groups`` maps a group name such as ``"mags"`` or ``"params"`` to integer
    feature-column indices.  ``group_names`` stores the names inside each group
    in the same order.  The boolean masks used by the diffusion code have the
    same length as ``names`` and use ``True`` for known/conditioned features.
    """

    names: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]
    group_names: dict[str, tuple[str, ...]]

    @classmethod
    def from_groups(cls, groups: Mapping[str, list[str] | tuple[str, ...]]) -> "FeatureMetadata":
        names: list[str] = []
        index_groups: dict[str, tuple[int, ...]] = {}
        group_names: dict[str, tuple[str, ...]] = {}
        for group, group_cols in groups.items():
            cols = tuple(str(name) for name in group_cols)
            start = len(names)
            names.extend(cols)
            index_groups[str(group)] = tuple(range(start, start + len(cols)))
            group_names[str(group)] = cols
        return cls(names=tuple(names), groups=index_groups, group_names=group_names)

    @classmethod
    def from_names(cls, names: list[str] | tuple[str, ...], group: str = "features") -> "FeatureMetadata":
        names_tuple = tuple(str(name) for name in names)
        return cls(
            names=names_tuple,
            groups={str(group): tuple(range(len(names_tuple)))},
            group_names={str(group): names_tuple},
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureMetadata":
        names = tuple(str(name) for name in payload["names"])
        groups = {str(group): tuple(int(i) for i in cols) for group, cols in payload["groups"].items()}
        group_names = {
            str(group): tuple(str(name) for name in cols)
            for group, cols in payload.get("group_names", {}).items()
        }
        if not group_names:
            group_names = {group: tuple(names[i] for i in cols) for group, cols in groups.items()}
        meta = cls(names=names, groups=groups, group_names=group_names)
        meta.validate()
        return meta

    @property
    def n_features(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "n_features": self.n_features,
            "groups": {group: list(cols) for group, cols in self.groups.items()},
            "group_names": {group: list(names) for group, names in self.group_names.items()},
        }

    def validate(self) -> None:
        if self.n_features == 0:
            raise ValueError("FeatureMetadata must contain at least one feature.")
        seen: list[int] = []
        for group, cols in self.groups.items():
            cols_tuple = tuple(int(i) for i in cols)
            if len(cols_tuple) != len(self.group_names.get(group, ())):
                raise ValueError(f"group_names[{group!r}] must match groups[{group!r}] length.")
            seen.extend(cols_tuple)
        if sorted(seen) != list(range(self.n_features)):
            raise ValueError("FeatureMetadata groups must cover each feature column exactly once.")


def make_training_mask(
    batch_shape: tuple[int, int],
    feature_metadata: FeatureMetadata | Mapping[str, Any],
    unknown_fraction: Mapping[str, float | list[float] | tuple[float, ...]],
    device=None,
    known_bands: list[str] | tuple[str, ...] | None = None,
    known_params: list[str] | tuple[str, ...] | None = None,
    tie_magerr_to_mag: bool = True,
):
    """Return a boolean torch mask where ``True`` means known.

    ``unknown_fraction`` is group-wise.  A scalar value hides that fraction of
    each group's columns independently per row.  A short list, e.g.
    ``[0.0, 0.5, 1.0]``, means draw one of those missing-fraction regimes per
    row, making the masking curriculum visible in configuration files.
    """

    torch_mod, _ = _require_torch()
    meta = _coerce_feature_metadata(feature_metadata)
    batch_size, n_features = int(batch_shape[0]), int(batch_shape[1])
    if n_features != meta.n_features:
        raise ValueError(f"batch_shape has {n_features} features, but metadata has {meta.n_features}.")

    if device is None:
        device = torch_mod.device("cpu")
    mask = torch_mod.ones((batch_size, n_features), dtype=torch_mod.bool, device=device)

    for group, cols in meta.groups.items():
        frac_spec = unknown_fraction.get(group, 0.0)
        if isinstance(frac_spec, (int, float)) and float(frac_spec) <= 0.0:
            continue
        col_t = torch_mod.tensor(cols, dtype=torch_mod.long, device=device)
        frac = _draw_fraction_by_row(frac_spec, batch_size, device)
        keep = torch_mod.rand((batch_size, len(cols)), device=device) > frac
        mask[:, col_t] = keep

    if known_bands is not None:
        _force_named_group_mask(mask, meta, "mags", known_bands, device)
    if known_params is not None:
        _force_named_group_mask(mask, meta, "params", known_params, device)

    if tie_magerr_to_mag and "mags" in meta.groups and "magerrs" in meta.groups:
        mag_cols = meta.groups["mags"]
        magerr_cols = meta.groups["magerrs"]
        if len(mag_cols) != len(magerr_cols):
            raise ValueError(
                "Cannot tie magnitude-error mask to magnitude mask: "
                f"{len(mag_cols)} magnitude columns but {len(magerr_cols)} mag-error columns."
            )
        mag_t = torch_mod.tensor(mag_cols, dtype=torch_mod.long, device=device)
        magerr_t = torch_mod.tensor(magerr_cols, dtype=torch_mod.long, device=device)
        mask[:, magerr_t] = mask[:, mag_t]

    return mask


def make_curriculum_training_mask(
    batch_shape: tuple[int, int],
    feature_metadata: FeatureMetadata | Mapping[str, Any],
    mask_config: Mapping[str, Any],
    device=None,
):
    """Return a training mask from simple dropout or weighted curriculum modes."""

    torch_mod, _ = _require_torch()
    meta = _coerce_feature_metadata(feature_metadata)
    if device is None:
        device = torch_mod.device("cpu")

    curriculum = mask_config.get("curriculum")
    if not curriculum:
        return make_training_mask(
            batch_shape,
            meta,
            mask_config.get("unknown_fraction", {}),
            device=device,
            known_bands=mask_config.get("known_bands"),
            known_params=mask_config.get("known_params"),
            tie_magerr_to_mag=bool(mask_config.get("tie_magerr_to_mag", True)),
        )

    batch_size, n_features = int(batch_shape[0]), int(batch_shape[1])
    weights = torch_mod.tensor(
        [float(mode.get("weight", 1.0)) for mode in curriculum],
        dtype=torch_mod.float32,
        device=device,
    )
    if torch_mod.any(weights < 0.0) or float(weights.sum().item()) <= 0.0:
        raise ValueError("Mask curriculum weights must be non-negative and sum to a positive value.")

    mode_ids = torch_mod.multinomial(weights / weights.sum(), batch_size, replacement=True)
    mask = torch_mod.empty((batch_size, n_features), dtype=torch_mod.bool, device=device)
    base_unknown = dict(mask_config.get("unknown_fraction", {}))
    for mode_index, mode in enumerate(curriculum):
        rows = mode_ids == mode_index
        n_rows = int(rows.sum().item())
        if n_rows == 0:
            continue
        unknown_fraction = {**base_unknown, **dict(mode.get("unknown_fraction", {}))}
        mask[rows] = make_training_mask(
            (n_rows, n_features),
            meta,
            unknown_fraction,
            device=device,
            known_bands=mode.get("known_bands"),
            known_params=mode.get("known_params"),
            tie_magerr_to_mag=bool(mask_config.get("tie_magerr_to_mag", True)),
        )
    return mask


def make_condition_mask(
    n_rows: int,
    feature_metadata: FeatureMetadata | Mapping[str, Any],
    known_groups: list[str] | tuple[str, ...] = (),
    known_params: list[str] | tuple[str, ...] = (),
    known_features: list[str] | tuple[str, ...] = (),
    device=None,
):
    """Build a condition mask for inference.

    ``known_groups=["mags"]`` means all magnitude columns are clamped.  Use
    ``known_features`` for arbitrary columns and ``known_params`` for names
    inside the ``params`` group.
    """

    torch_mod, _ = _require_torch()
    meta = _coerce_feature_metadata(feature_metadata)
    if device is None:
        device = torch_mod.device("cpu")
    mask = torch_mod.zeros((int(n_rows), meta.n_features), dtype=torch_mod.bool, device=device)

    for group in known_groups:
        if group not in meta.groups:
            raise ValueError(f"Unknown feature group {group!r}. Available groups are {sorted(meta.groups)}.")
        cols = torch_mod.tensor(meta.groups[group], dtype=torch_mod.long, device=device)
        mask[:, cols] = True

    for feature in known_features:
        if feature not in meta.names:
            raise ValueError(f"Unknown feature {feature!r}.")
        mask[:, meta.names.index(feature)] = True

    param_names = meta.group_names.get("params", ())
    param_cols = meta.groups.get("params", ())
    for pname in known_params:
        if pname not in param_names:
            raise ValueError(f"Requested known parameter {pname!r}, but params are {list(param_names)}.")
        mask[:, param_cols[param_names.index(pname)]] = True
    return mask


if torch is not None:

    class SinusoidalPosEmb(nn.Module):
        """Sine/cosine time embedding for diffusion noise level t in [0, 1]."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = int(dim)

        def forward(self, t):
            half = max(self.dim // 2, 1)
            denom = max(half - 1, 1)
            freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / denom)
            args = t.reshape(-1, 1) * freqs.reshape(1, -1)
            emb = torch.cat([args.sin(), args.cos()], dim=-1)
            if emb.shape[1] < self.dim:
                emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
            return emb[:, : self.dim]


    class MLPBlock(nn.Module):
        """Small residual block modulated by a time embedding."""

        def __init__(self, dim: int):
            super().__init__()
            self.lin_h = nn.Linear(dim, dim)
            self.lin_t = nn.Linear(dim, dim)
            self.act = nn.ReLU()

        def forward(self, h, t_emb):
            return h + self.act(self.lin_h(h) + self.lin_t(t_emb))


    class CNN1DBlock(nn.Module):
        """Residual 1D convolution block for ordered band-like features."""

        def __init__(self, channels: int, kernel_size: int = 3):
            super().__init__()
            pad = kernel_size // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
            self.t_proj = nn.Linear(channels, channels)
            self.act = nn.ReLU()

        def forward(self, x, t_emb_branch):
            h = self.conv1(x)
            h = h + self.t_proj(t_emb_branch).unsqueeze(-1)
            h = self.act(h)
            h = self.conv2(h)
            return x + self.act(h)


    class ScoreMLP(nn.Module):
        """Time-conditioned MLP over the full feature vector.

        The network sees three channels concatenated along the feature axis:
        the noisy vector ``x``, the clean known values ``known``, and the
        boolean known mask as floats.
        """

        def __init__(self, feature_metadata: FeatureMetadata | Mapping[str, Any], model_config: Mapping[str, Any]):
            super().__init__()
            meta = _coerce_feature_metadata(feature_metadata)
            self.feature_metadata = meta.to_dict()
            self.n_features = meta.n_features
            emb_dim = int(model_config.get("emb_dim", 128))
            time_hidden = int(model_config.get("time_hidden", 256))
            hidden = int(model_config.get("mlp_hidden", model_config.get("hidden_features", 256)))
            blocks = int(model_config.get("mlp_blocks", model_config.get("num_blocks", 4)))

            self.time_emb = SinusoidalPosEmb(emb_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, time_hidden),
                nn.ReLU(),
            )
            self.input = nn.Linear(3 * self.n_features, hidden)
            self.t_to_hidden = nn.Linear(time_hidden, hidden)
            self.blocks = nn.ModuleList([MLPBlock(hidden) for _ in range(blocks)])
            self.output = nn.Linear(hidden, self.n_features)

        def forward(self, x, known, t, mask=None):
            if mask is None:
                mask = (known.abs() > 1e-12).float()
            else:
                mask = mask.float()
            temb = self.time_mlp(self.time_emb(t))
            h = self.input(torch.cat([x, known, mask], dim=1))
            t_hidden = self.t_to_hidden(temb)
            for block in self.blocks:
                h = block(h, t_hidden)
            return self.output(h)


    class ScoreHybridSED(nn.Module):
        """Group-aware score model for SED-like features.

        Ordered magnitude-like groups use 1D convolutions across bands, while
        parameter-like groups use residual MLP branches.  A small fusion block
        lets the grouped predictions exchange information.
        """

        def __init__(self, feature_metadata: FeatureMetadata | Mapping[str, Any], model_config: Mapping[str, Any]):
            super().__init__()
            meta = _coerce_feature_metadata(feature_metadata)
            self.feature_metadata = meta.to_dict()
            self.n_features = meta.n_features
            self.groups = {
                name: torch.tensor(cols, dtype=torch.long)
                for name, cols in meta.groups.items()
                if len(cols) > 0
            }
            self.group_order = sorted(self.groups.keys(), key=lambda name: int(self.groups[name][0]))

            emb_dim = int(model_config.get("emb_dim", 128))
            time_hidden = int(model_config.get("time_hidden", 256))
            self.fusion_scale = float(model_config.get("fusion_scale", 0.5))

            self.time_emb = SinusoidalPosEmb(emb_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, time_hidden),
                nn.ReLU(),
            )

            self.branch_kind: dict[str, str] = {}
            self.t_to_branch = nn.ModuleDict()
            self.branch_in = nn.ModuleDict()
            self.branch_blocks = nn.ModuleDict()
            self.branch_out = nn.ModuleDict()
            pooled_dims: list[int] = []

            for group in self.group_order:
                n_group = int(len(self.groups[group]))
                if group in {"mags", "magerrs"}:
                    channels = int(model_config.get("cnn_channels", model_config.get("hidden_features", 128)))
                    n_blocks = int(model_config.get("cnn_blocks", model_config.get("num_blocks", 2)))
                    self.branch_kind[group] = "cnn"
                    self.t_to_branch[group] = nn.Linear(time_hidden, channels)
                    self.branch_in[group] = nn.Conv1d(3, channels, kernel_size=1)
                    self.branch_blocks[group] = nn.ModuleList([CNN1DBlock(channels) for _ in range(n_blocks)])
                    self.branch_out[group] = nn.Conv1d(channels, 1, kernel_size=1)
                    pooled_dims.append(channels)
                else:
                    hidden = int(model_config.get("param_hidden", model_config.get("hidden_features", 256)))
                    n_blocks = int(model_config.get("param_blocks", model_config.get("num_blocks", 2)))
                    self.branch_kind[group] = "mlp"
                    self.t_to_branch[group] = nn.Linear(time_hidden, hidden)
                    self.branch_in[group] = nn.Linear(3 * n_group, hidden)
                    self.branch_blocks[group] = nn.ModuleList([MLPBlock(hidden) for _ in range(n_blocks)])
                    self.branch_out[group] = nn.Linear(hidden, n_group)
                    pooled_dims.append(hidden)

            fusion_hidden = int(model_config.get("fusion_hidden", model_config.get("hidden_features", 128)))
            fusion_blocks = int(model_config.get("fusion_blocks", model_config.get("num_blocks", 2)))
            self.t_to_fusion = nn.Linear(time_hidden, fusion_hidden)
            self.fusion_in = nn.Linear(sum(pooled_dims) + self.n_features, fusion_hidden)
            self.fusion_blocks = nn.ModuleList([MLPBlock(fusion_hidden) for _ in range(fusion_blocks)])
            self.fusion_out = nn.Linear(fusion_hidden, self.n_features)
            self.gate = nn.Sequential(
                nn.Linear(fusion_hidden, fusion_hidden),
                nn.ReLU(),
                nn.Linear(fusion_hidden, self.n_features),
                nn.Tanh(),
            )

        def _take_group(self, x, group: str):
            cols = self.groups[group].to(device=x.device)
            return x.index_select(dim=1, index=cols)

        def forward(self, x, known, t, mask=None):
            if mask is None:
                mask = (known.abs() > 1e-12).float()
            else:
                mask = mask.float()

            temb = self.time_mlp(self.time_emb(t))
            direct_by_group: dict[str, Any] = {}
            pooled = []
            for group in self.group_order:
                x_g = self._take_group(x, group)
                known_g = self._take_group(known, group)
                mask_g = self._take_group(mask, group)
                t_g = self.t_to_branch[group](temb)

                if self.branch_kind[group] == "cnn":
                    h = torch.stack([x_g, known_g, mask_g], dim=1)
                    h = self.branch_in[group](h)
                    for block in self.branch_blocks[group]:
                        h = block(h, t_g)
                    pred = self.branch_out[group](h).squeeze(1)
                    pooled.append(h.mean(dim=-1))
                else:
                    h = torch.cat([x_g, known_g, mask_g], dim=1)
                    h = self.branch_in[group](h)
                    for block in self.branch_blocks[group]:
                        h = block(h, t_g)
                    pred = self.branch_out[group](h)
                    pooled.append(h)
                direct_by_group[group] = pred

            pred_direct = torch.empty_like(x)
            for group, pred in direct_by_group.items():
                cols = self.groups[group].to(device=x.device)
                pred_direct.index_copy_(dim=1, index=cols, source=pred)

            t_fusion = self.t_to_fusion(temb)
            h_fusion = self.fusion_in(torch.cat(pooled + [pred_direct], dim=1))
            for block in self.fusion_blocks:
                h_fusion = block(h_fusion, t_fusion)
            correction = self.fusion_out(h_fusion)
            gate = self.gate(h_fusion)
            return pred_direct + self.fusion_scale * gate * correction

else:

    class ScoreMLP:  # pragma: no cover - only used when torch is missing.
        def __init__(self, *args, **kwargs):
            _require_torch()


    class ScoreHybridSED:  # pragma: no cover - only used when torch is missing.
        def __init__(self, *args, **kwargs):
            _require_torch()


class ConditionalDiffusionEstimator:
    """Masked conditional diffusion estimator over a joint feature vector.

    The estimator trains on a two-dimensional array ``x_train``.  At inference,
    ``known`` uses the original physical feature units; entries where
    ``mask=False`` may be arbitrary or NaN.  Samples are returned in original
    physical units with shape ``(n_objects, num_samples, n_features)``.
    """

    def __init__(
        self,
        feature_metadata: FeatureMetadata | Mapping[str, Any],
        model: str = "hybrid_sed",
        hidden_features: int = 128,
        model_config: Mapping[str, Any] | None = None,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        device: str | None = "auto",
        validate_device: bool = True,
        allow_device_fallback: bool = True,
        standardize: bool = True,
    ) -> None:
        torch_mod, _ = _require_torch()
        self.feature_metadata = _coerce_feature_metadata(feature_metadata)
        self.feature_metadata.validate()
        self.model_name = str(model)
        self.model_config = dict(model_config or {})
        self.model_config.setdefault("hidden_features", int(hidden_features))
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if not (self.sigma_min > 0.0 and self.sigma_max > self.sigma_min):
            raise ValueError("Require 0 < sigma_min < sigma_max.")
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.standardize = bool(standardize)
        self.torch_dtype = torch_mod.float32
        self.device = resolve_torch_device(
            device=device,
            validate=bool(validate_device),
            allow_fallback=bool(allow_device_fallback),
        )
        self.score_model = _build_score_model(self.feature_metadata, self.model_name, self.model_config).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.standardizer: Standardizer | None = None
        self.history: dict[str, list[float]] = {"train_loss": []}

    def fit(
        self,
        x_train: np.ndarray,
        mask_config: Mapping[str, Any] | None = None,
        epochs: int = 100,
        batch_size: int = 256,
        seed: int | None = None,
        validation_split: float = 0.0,
        verbose: bool = False,
        clamp_known_in_xt: bool = False,
        loss_on_unknown_only: bool = False,
    ) -> dict[str, list[float]]:
        """Fit the score network to masked noisy feature vectors."""

        torch_mod, _ = _require_torch()
        x_arr = _as_training_array(x_train, self.feature_metadata.n_features)
        epochs = int(epochs)
        batch_size = int(batch_size)
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if seed is not None:
            torch_mod.manual_seed(int(seed))
            if torch_mod.cuda.is_available():
                torch_mod.cuda.manual_seed_all(int(seed))

        n_train = x_arr.shape[0]
        if n_train == 0:
            raise ValueError("x_train must contain at least one row.")
        val_n = int(round(float(validation_split) * n_train))
        if val_n < 0 or val_n >= n_train:
            raise ValueError("validation_split must leave at least one training row.")

        permutation = np.random.default_rng(seed).permutation(n_train)
        validation_indices = permutation[:val_n]
        training_indices = permutation[val_n:]
        if self.standardize:
            self.standardizer = Standardizer.fit(x_arr[training_indices])
            x_fit = self.standardizer.transform(x_arr)
        else:
            self.standardizer = Standardizer(
                np.zeros(self.feature_metadata.n_features),
                np.ones(self.feature_metadata.n_features),
            )
            x_fit = x_arr
        x_fit = np.asarray(x_fit, dtype=np.float32)

        x_t = torch_mod.as_tensor(x_fit, dtype=self.torch_dtype, device=self.device)
        training_index_t = torch_mod.as_tensor(training_indices, dtype=torch_mod.long, device=self.device)
        x_train_t = x_t.index_select(0, training_index_t)
        train_rows = x_train_t.shape[0]
        if val_n > 0:
            validation_index_t = torch_mod.as_tensor(validation_indices, dtype=torch_mod.long, device=self.device)
            x_val_t = x_t.index_select(0, validation_index_t)
        else:
            x_val_t = None

        opt = torch_mod.optim.AdamW(self.score_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        mask_cfg = dict(mask_config or _default_mask_config(self.feature_metadata))
        self.history = {"train_loss": []}
        if x_val_t is not None:
            self.history["val_loss"] = []

        for epoch in range(epochs):
            self.score_model.train()
            perm = torch_mod.randperm(train_rows, device=self.device)
            total_loss = 0.0
            n_seen = 0
            for start in range(0, train_rows, batch_size):
                idx = perm[start : start + batch_size]
                x0 = x_train_t.index_select(0, idx)
                loss = self._training_loss(
                    x0,
                    mask_cfg,
                    clamp_known_in_xt=clamp_known_in_xt,
                    loss_on_unknown_only=loss_on_unknown_only,
                )
                if not bool(torch_mod.isfinite(loss).detach().cpu().item()):
                    raise FloatingPointError(f"Non-finite diffusion training loss at epoch {epoch + 1}.")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                n_batch = x0.shape[0]
                total_loss += float(loss.detach().cpu().item()) * n_batch
                n_seen += n_batch

            train_loss = total_loss / max(n_seen, 1)
            self.history["train_loss"].append(train_loss)
            if x_val_t is not None:
                with torch_mod.no_grad():
                    val_loss = float(
                        self._training_loss(
                            x_val_t,
                            mask_cfg,
                            clamp_known_in_xt=clamp_known_in_xt,
                            loss_on_unknown_only=loss_on_unknown_only,
                        )
                        .detach()
                        .cpu()
                        .item()
                    )
                self.history["val_loss"].append(val_loss)
            if verbose:
                msg = f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6g}"
                if x_val_t is not None:
                    msg += f", val_loss={self.history['val_loss'][-1]:.6g}"
                print(msg)
        return self.history

    def sample(
        self,
        known: np.ndarray,
        mask: np.ndarray,
        num_samples: int = 100,
        steps: int = 100,
        sampler: str = "edm_heun",
        batch_size: int | None = None,
        reimpose_known_each_step: bool = True,
        guidance_fn: Callable[[Any, Any, Any, FeatureMetadata], Any] | None = None,
        guidance_eta: float = 0.0,
        **sampler_kwargs,
    ) -> np.ndarray:
        """Sample full feature vectors while conditioning on ``mask=True`` entries."""

        self._check_fitted()
        torch_mod, _ = _require_torch()
        known_arr, mask_arr = _prepare_known_and_mask(known, mask, self.feature_metadata.n_features)
        known_std = self._standardize_known_values(known_arr, mask_arr)
        known_t = torch_mod.as_tensor(known_std, dtype=self.torch_dtype, device=self.device)
        mask_t = torch_mod.as_tensor(mask_arr, dtype=torch_mod.bool, device=self.device)
        row_batch_size = known_t.shape[0] if batch_size is None else int(batch_size)

        if guidance_fn is not None:
            if str(sampler).lower() not in {"em", "euler_maruyama"}:
                raise ValueError("Guidance is currently implemented for sampler='em' only.")
            samples_std_t = _batch_sample_em_guided_chunked(
                self.score_model,
                known_t,
                mask_t,
                samples_per=int(num_samples),
                guidance_fn=lambda x0_hat, known_big, mask_big: guidance_fn(
                    x0_hat, known_big, mask_big, self.feature_metadata
                ),
                eta=float(guidance_eta),
                steps=int(steps),
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                row_batch_size=row_batch_size,
                reimpose_known_each_step=reimpose_known_each_step,
            )
        else:
            samples_std_t = _batch_sample_chunked(
                self.score_model,
                known_t,
                mask_t,
                samples_per=int(num_samples),
                steps=int(steps),
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                row_batch_size=row_batch_size,
                reimpose_known_each_step=reimpose_known_each_step,
                sampler=sampler,
                **sampler_kwargs,
            )
        samples_std = samples_std_t.detach().cpu().numpy()
        n_obj, n_samp, n_feat = samples_std.shape
        samples = self.standardizer.inverse_transform(samples_std.reshape(n_obj * n_samp, n_feat))
        samples = samples.reshape(n_obj, n_samp, n_feat)
        # The sampler reclamps standardized known values.  Reassigning in
        # physical units removes tiny inverse-transform roundoff.
        for row in range(n_obj):
            samples[row][:, mask_arr[row]] = known_arr[row, mask_arr[row]][None, :]
        return samples

    def save(self, path: str | Path) -> None:
        """Save model weights, metadata, and standardization constants."""

        self._check_fitted()
        torch_mod, _ = _require_torch()
        payload = {
            "feature_metadata": self.feature_metadata.to_dict(),
            "model_name": self.model_name,
            "model_config": self.model_config,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "standardize": self.standardize,
            "standardizer_mean": np.asarray(self.standardizer.mean, dtype=float).tolist(),
            "standardizer_std": np.asarray(self.standardizer.std, dtype=float).tolist(),
            "history": self.history,
            "model_state": self.score_model.state_dict(),
        }
        torch_mod.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | None = "auto",
        *,
        validate_device: bool = True,
        allow_device_fallback: bool = True,
    ) -> "ConditionalDiffusionEstimator":
        """Load a diffusion estimator saved by :meth:`save`."""

        torch_mod, _ = _require_torch()
        target_device = resolve_torch_device(
            device=device,
            validate=bool(validate_device),
            allow_fallback=bool(allow_device_fallback),
        )
        payload = torch_mod.load(Path(path), map_location=target_device)
        estimator = cls(
            FeatureMetadata.from_dict(payload["feature_metadata"]),
            model=payload["model_name"],
            model_config=payload["model_config"],
            sigma_min=payload["sigma_min"],
            sigma_max=payload["sigma_max"],
            learning_rate=payload.get("learning_rate", 1e-4),
            weight_decay=payload.get("weight_decay", 0.0),
            device=str(target_device),
            validate_device=False,
            allow_device_fallback=False,
            standardize=payload.get("standardize", True),
        )
        estimator.score_model.load_state_dict(payload["model_state"])
        estimator.standardizer = Standardizer(
            mean=np.asarray(payload["standardizer_mean"], dtype=float),
            std=np.asarray(payload["standardizer_std"], dtype=float),
        )
        estimator.history = payload.get("history", {"train_loss": []})
        return estimator

    def _training_loss(
        self,
        x0,
        mask_config: Mapping[str, Any],
        *,
        clamp_known_in_xt: bool,
        loss_on_unknown_only: bool,
    ):
        torch_mod, _ = _require_torch()
        batch_n, n_features = x0.shape
        mask = make_curriculum_training_mask((batch_n, n_features), self.feature_metadata, mask_config, self.device)
        known = x0 * mask
        t = torch_mod.rand(batch_n, dtype=x0.dtype, device=self.device) * (1.0 - 1e-5) + 1e-5
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** t).reshape(batch_n, 1)
        eps = torch_mod.randn_like(x0)
        unknown = (~mask).float()
        if clamp_known_in_xt:
            xt = x0 + sigma * eps * unknown
        else:
            xt = x0 + sigma * eps
        pred = self.score_model(xt, known, t, mask)
        target = -eps
        if loss_on_unknown_only:
            return (((pred - target) ** 2) * unknown).sum() / unknown.sum().clamp_min(1.0)
        return ((pred - target) ** 2).mean()

    def _standardize_known_values(self, known: np.ndarray, mask: np.ndarray) -> np.ndarray:
        known_for_transform = np.asarray(known, dtype=float).copy()
        for col in range(known_for_transform.shape[1]):
            known_for_transform[~mask[:, col], col] = self.standardizer.mean[col]
        transformed = self.standardizer.transform(known_for_transform)
        transformed[~mask] = 0.0
        return np.asarray(transformed, dtype=np.float32)

    def _check_fitted(self) -> None:
        if self.standardizer is None:
            raise RuntimeError("ConditionalDiffusionEstimator must be fit or loaded before sampling.")


def _build_score_model(feature_metadata: FeatureMetadata, model: str, model_config: Mapping[str, Any]):
    _require_torch()
    model_key = str(model).lower()
    if model_key in {"mlp", "score_mlp"}:
        return ScoreMLP(feature_metadata, model_config)
    if model_key in {"hybrid", "hybrid_sed", "hybrid_cnn"}:
        return ScoreHybridSED(feature_metadata, model_config)
    raise ValueError("model must be one of 'mlp' or 'hybrid_sed'.")


def _validate_float32_device(torch_mod, device) -> None:
    """Exercise the float32 operations used by the diffusion code."""

    x = torch_mod.randn((4, 3), dtype=torch_mod.float32, device=device, requires_grad=True)
    weight = torch_mod.randn((3, 2), dtype=torch_mod.float32, device=device)
    y = (x @ weight).pow(2).mean()
    y.backward()
    if not bool(torch_mod.isfinite(y.detach()).cpu().item()):
        raise FloatingPointError("float32 smoke test produced a non-finite value.")
    if x.grad is None or not bool(torch_mod.all(torch_mod.isfinite(x.grad)).detach().cpu().item()):
        raise FloatingPointError("float32 smoke test produced non-finite gradients.")
    _synchronize_device(torch_mod, device)


def _synchronize_device(torch_mod, device) -> None:
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type == "cuda" and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize(device)
    elif device_type == "mps" and hasattr(torch_mod, "mps") and hasattr(torch_mod.mps, "synchronize"):
        torch_mod.mps.synchronize()


def _coerce_feature_metadata(feature_metadata: FeatureMetadata | Mapping[str, Any]) -> FeatureMetadata:
    if isinstance(feature_metadata, FeatureMetadata):
        return feature_metadata
    return FeatureMetadata.from_dict(feature_metadata)


def _draw_fraction_by_row(spec, batch_size: int, device):
    torch_mod, _ = _require_torch()
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("Mask fraction lists must not be empty.")
        levels = torch_mod.tensor([float(x) for x in spec], dtype=torch_mod.float32, device=device)
        if torch_mod.any((levels < 0.0) | (levels > 1.0)):
            raise ValueError("Unknown fractions must lie between 0 and 1.")
        choice = torch_mod.randint(0, len(spec), (batch_size,), device=device)
        return levels.index_select(0, choice).reshape(batch_size, 1)
    value = float(spec)
    if value < 0.0 or value > 1.0:
        raise ValueError("Unknown fractions must lie between 0 and 1.")
    return torch_mod.full((batch_size, 1), value, dtype=torch_mod.float32, device=device)


def _force_named_group_mask(mask, meta: FeatureMetadata, group: str, known_names, device) -> None:
    torch_mod, _ = _require_torch()
    if group not in meta.groups:
        raise ValueError(f"Cannot set known names for missing group {group!r}.")
    group_names = meta.group_names[group]
    missing = sorted(set(known_names) - set(group_names))
    if missing:
        raise ValueError(f"Unknown names for group {group!r}: {missing}")
    keep = torch_mod.tensor([name in set(known_names) for name in group_names], dtype=torch_mod.bool, device=device)
    cols = torch_mod.tensor(meta.groups[group], dtype=torch_mod.long, device=device)
    mask[:, cols] = keep.reshape(1, -1)


def _default_mask_config(feature_metadata: FeatureMetadata) -> dict[str, Any]:
    return {"unknown_fraction": {group: 0.5 for group in feature_metadata.groups}}


def _as_training_array(values: np.ndarray, n_features: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != int(n_features):
        raise ValueError(f"x_train must have shape (n, {n_features}); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x_train contains NaN or inf values.")
    return arr


def _prepare_known_and_mask(known: np.ndarray, mask: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    known_arr = np.asarray(known, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)
    if known_arr.ndim == 1:
        known_arr = known_arr[None, :]
    if mask_arr.ndim == 1:
        mask_arr = mask_arr[None, :]
    if known_arr.ndim != 2 or known_arr.shape[1] != int(n_features):
        raise ValueError(f"known must have shape ({n_features},) or (n, {n_features}); got {known_arr.shape}.")
    if mask_arr.ndim != 2 or mask_arr.shape[1] != int(n_features):
        raise ValueError(f"mask must have shape ({n_features},) or (n, {n_features}); got {mask_arr.shape}.")
    if mask_arr.shape[0] == 1 and known_arr.shape[0] > 1:
        mask_arr = np.repeat(mask_arr, known_arr.shape[0], axis=0)
    if known_arr.shape[0] == 1 and mask_arr.shape[0] > 1:
        known_arr = np.repeat(known_arr, mask_arr.shape[0], axis=0)
    if known_arr.shape[0] != mask_arr.shape[0]:
        raise ValueError("known and mask must have compatible row counts.")
    if not np.all(np.isfinite(known_arr[mask_arr])):
        raise ValueError("known contains NaN or inf in entries where mask=True.")
    return known_arr, mask_arr


def _sigma_to_t(sigma, sigma_min: float, sigma_max: float):
    log_range = math.log(sigma_max / sigma_min)
    t = torch.log(sigma / sigma_min) / log_range
    return torch.clamp(t, min=1e-5, max=1.0)


def _score_and_denoised(score_model, x, known, mask, sigma, sigma_min: float, sigma_max: float):
    if not torch.is_tensor(sigma):
        sigma_t = torch.full((x.shape[0], 1), float(sigma), dtype=x.dtype, device=x.device)
    else:
        sigma_t = sigma.to(device=x.device, dtype=x.dtype).reshape(-1, 1)
        if sigma_t.shape[0] == 1:
            sigma_t = sigma_t.expand(x.shape[0], 1)
    t_vec = _sigma_to_t(sigma_t, sigma_min, sigma_max).reshape(-1)
    eps_pred = score_model(x, known, t_vec, mask)
    score = eps_pred / sigma_t
    denoised = x + sigma_t * eps_pred
    return score, denoised


def _prepare_repeated_state(score_model, known, mask, samples_per: int, sigma_max: float):
    device = next(score_model.parameters()).device
    known = known.to(device).float().reshape(known.shape[0], -1)
    mask = mask.to(device).bool().reshape(mask.shape[0], -1)
    mask_f = mask.float()
    batch_size, n_features = known.shape
    known_big = known.unsqueeze(1).expand(batch_size, samples_per, n_features).reshape(batch_size * samples_per, n_features)
    mask_big = mask_f.unsqueeze(1).expand(batch_size, samples_per, n_features).reshape(batch_size * samples_per, n_features)
    unknown_big = 1.0 - mask_big
    x = sigma_max * torch.randn(batch_size * samples_per, n_features, dtype=known.dtype, device=device)
    x = x * unknown_big + known_big * mask_big
    return x, known_big, mask_big, unknown_big


def _reclamp(x, known, mask, unknown):
    return x * unknown + known * mask


def _karras_sigmas(steps: int, sigma_min: float, sigma_max: float, rho: float, device):
    ramp = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    return (max_inv + ramp * (min_inv - max_inv)) ** rho


def _batch_sample(
    score_model,
    known,
    mask,
    samples_per: int,
    steps: int,
    sigma_max: float,
    sigma_min: float,
    reimpose_known_each_step: bool = True,
    sampler: str = "edm_heun",
    corrector_steps: int = 1,
    snr: float = 0.1,
    corrector_step_size_max: float | None = 0.1,
    rho: float = 7.0,
    final_denoise: bool = True,
):
    score_model = score_model.eval()
    batch_size = known.shape[0]
    n_features = known.reshape(known.shape[0], -1).shape[1]
    with torch.no_grad():
        x, known_big, mask_big, unknown_big = _prepare_repeated_state(score_model, known, mask, int(samples_per), sigma_max)
        sampler_key = str(sampler).lower()
        if sampler_key in {"em", "euler_maruyama"}:
            x = _sample_em(score_model, x, known_big, mask_big, unknown_big, steps, sigma_max, sigma_min, reimpose_known_each_step)
        elif sampler_key == "pc":
            x = _sample_pc(
                score_model,
                x,
                known_big,
                mask_big,
                unknown_big,
                steps,
                sigma_max,
                sigma_min,
                reimpose_known_each_step,
                corrector_steps,
                snr,
                corrector_step_size_max,
            )
        elif sampler_key in {"edm", "edm_heun", "heun"}:
            x = _sample_edm_heun(
                score_model,
                x,
                known_big,
                mask_big,
                unknown_big,
                steps,
                sigma_max,
                sigma_min,
                reimpose_known_each_step,
                rho,
                final_denoise,
            )
        elif sampler_key in {"edm_euler", "karras_euler", "ode_euler"}:
            x = _sample_edm_euler(
                score_model,
                x,
                known_big,
                mask_big,
                unknown_big,
                steps,
                sigma_max,
                sigma_min,
                reimpose_known_each_step,
                rho,
                final_denoise,
            )
        else:
            raise ValueError("sampler must be one of 'em', 'pc', 'edm_heun', or 'edm_euler'.")
    return x.reshape(batch_size, int(samples_per), n_features)


def _batch_sample_chunked(score_model, known, mask, row_batch_size: int, **kwargs):
    chunks = []
    row_batch_size = max(1, int(row_batch_size))
    for start in range(0, known.shape[0], row_batch_size):
        end = min(start + row_batch_size, known.shape[0])
        chunk = _batch_sample(score_model, known[start:end], mask[start:end], **kwargs)
        chunks.append(chunk.detach().cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def _sample_em(score_model, x, known, mask, unknown, steps, sigma_max, sigma_min, reimpose_known_each_step):
    sqrt_const = math.sqrt(2.0 * math.log(sigma_max / sigma_min))
    time_seq = torch.linspace(1.0, 1e-5, int(steps), dtype=x.dtype, device=x.device)
    dt = -1.0 / int(steps)
    for t in time_seq:
        sigma_t = sigma_min * ((sigma_max / sigma_min) ** t)
        g_t = sigma_t * sqrt_const
        score, _ = _score_and_denoised(score_model, x, known, mask, sigma_t, sigma_min, sigma_max)
        score = score * unknown
        x = x - (g_t**2) * score * dt + g_t * math.sqrt(-dt) * torch.randn_like(x) * unknown
        if reimpose_known_each_step:
            x = _reclamp(x, known, mask, unknown)
    return x


def _sample_pc(
    score_model,
    x,
    known,
    mask,
    unknown,
    steps,
    sigma_max,
    sigma_min,
    reimpose_known_each_step,
    corrector_steps,
    snr,
    corrector_step_size_max,
):
    sqrt_const = math.sqrt(2.0 * math.log(sigma_max / sigma_min))
    time_seq = torch.linspace(1.0, 1e-5, int(steps), dtype=x.dtype, device=x.device)
    dt = -1.0 / int(steps)
    for t in time_seq:
        sigma_t = sigma_min * ((sigma_max / sigma_min) ** t)
        for _ in range(max(0, int(corrector_steps))):
            score, _ = _score_and_denoised(score_model, x, known, mask, sigma_t, sigma_min, sigma_max)
            grad = score * unknown
            noise = torch.randn_like(x) * unknown
            grad_norm = torch.linalg.vector_norm(grad, dim=1, keepdim=True).clamp_min(1e-12)
            noise_norm = torch.linalg.vector_norm(noise, dim=1, keepdim=True).clamp_min(1e-12)
            step_size = 2.0 * (float(snr) * noise_norm / grad_norm) ** 2
            if corrector_step_size_max is not None:
                step_size = torch.clamp(step_size, max=float(corrector_step_size_max))
            x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise
            if reimpose_known_each_step:
                x = _reclamp(x, known, mask, unknown)
        g_t = sigma_t * sqrt_const
        score, _ = _score_and_denoised(score_model, x, known, mask, sigma_t, sigma_min, sigma_max)
        score = score * unknown
        x = x - (g_t**2) * score * dt + g_t * math.sqrt(-dt) * torch.randn_like(x) * unknown
        if reimpose_known_each_step:
            x = _reclamp(x, known, mask, unknown)
    return x


def _sample_edm_heun(
    score_model,
    x,
    known,
    mask,
    unknown,
    steps,
    sigma_max,
    sigma_min,
    reimpose_known_each_step,
    rho,
    final_denoise,
):
    sigmas = _karras_sigmas(int(steps), sigma_min, sigma_max, float(rho), x.device)
    for i in range(int(steps)):
        sigma_i = float(sigmas[i].item())
        sigma_next = float(sigmas[i + 1].item())
        _, denoised = _score_and_denoised(score_model, x, known, mask, sigma_i, sigma_min, sigma_max)
        d_i = ((x - denoised) / sigma_i) * unknown
        x_euler = x + (sigma_next - sigma_i) * d_i
        if reimpose_known_each_step:
            x_euler = _reclamp(x_euler, known, mask, unknown)
        _, denoised_next = _score_and_denoised(score_model, x_euler, known, mask, sigma_next, sigma_min, sigma_max)
        d_next = ((x_euler - denoised_next) / sigma_next) * unknown
        x = x + (sigma_next - sigma_i) * 0.5 * (d_i + d_next)
        if reimpose_known_each_step:
            x = _reclamp(x, known, mask, unknown)
    if final_denoise:
        _, denoised = _score_and_denoised(score_model, x, known, mask, sigma_min, sigma_min, sigma_max)
        x = denoised * unknown + known * mask
    return x


def _sample_edm_euler(
    score_model,
    x,
    known,
    mask,
    unknown,
    steps,
    sigma_max,
    sigma_min,
    reimpose_known_each_step,
    rho,
    final_denoise,
):
    sigmas = _karras_sigmas(int(steps), sigma_min, sigma_max, float(rho), x.device)
    for i in range(int(steps)):
        sigma_i = float(sigmas[i].item())
        sigma_next = float(sigmas[i + 1].item())
        _, denoised = _score_and_denoised(score_model, x, known, mask, sigma_i, sigma_min, sigma_max)
        d_i = ((x - denoised) / sigma_i) * unknown
        x = x + (sigma_next - sigma_i) * d_i
        if reimpose_known_each_step:
            x = _reclamp(x, known, mask, unknown)
    if final_denoise:
        _, denoised = _score_and_denoised(score_model, x, known, mask, sigma_min, sigma_min, sigma_max)
        x = denoised * unknown + known * mask
    return x


def _batch_sample_em_guided_chunked(
    score_model,
    known,
    mask,
    samples_per: int,
    guidance_fn,
    eta: float,
    steps: int,
    sigma_max: float,
    sigma_min: float,
    row_batch_size: int,
    reimpose_known_each_step: bool,
):
    chunks = []
    row_batch_size = max(1, int(row_batch_size))
    for start in range(0, known.shape[0], row_batch_size):
        end = min(start + row_batch_size, known.shape[0])
        chunk = _batch_sample_em_guided(
            score_model,
            known[start:end],
            mask[start:end],
            samples_per,
            guidance_fn,
            eta,
            steps,
            sigma_max,
            sigma_min,
            reimpose_known_each_step,
        )
        chunks.append(chunk.detach().cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def _batch_sample_em_guided(
    score_model,
    known,
    mask,
    samples_per: int,
    guidance_fn,
    eta: float,
    steps: int,
    sigma_max: float,
    sigma_min: float,
    reimpose_known_each_step: bool,
):
    score_model = score_model.eval()
    for param in score_model.parameters():
        param.requires_grad_(False)
    batch_size = known.shape[0]
    n_features = known.reshape(known.shape[0], -1).shape[1]
    x, known_big, mask_big, unknown_big = _prepare_repeated_state(score_model, known, mask, int(samples_per), sigma_max)
    sqrt_const = math.sqrt(2.0 * math.log(sigma_max / sigma_min))
    time_seq = torch.linspace(1.0, 1e-5, int(steps), dtype=x.dtype, device=x.device)
    dt = -1.0 / int(steps)

    for t in time_seq:
        sigma_t = sigma_min * ((sigma_max / sigma_min) ** t)
        g_t = sigma_t * sqrt_const
        t_vec = t.repeat(batch_size * int(samples_per)).to(x.device)
        with torch.no_grad():
            eps_pred = score_model(x, known_big, t_vec, mask_big)
            simulator_score = eps_pred / sigma_t
        guided_score = simulator_score
        if guidance_fn is not None and float(eta) != 0.0:
            x_for_grad = x.detach().requires_grad_(True)
            x0_hat = x_for_grad + sigma_t * eps_pred.detach()
            cost = guidance_fn(x0_hat, known_big, mask_big)
            total_cost = cost.sum() if cost.ndim > 0 else cost
            grad_cost = torch.autograd.grad(total_cost, x_for_grad, retain_graph=False)[0]
            guided_score = guided_score - float(eta) * grad_cost.detach()
        with torch.no_grad():
            x = x - (g_t**2) * guided_score * dt + g_t * math.sqrt(-dt) * torch.randn_like(x) * unknown_big
            if reimpose_known_each_step:
                x = _reclamp(x, known_big, mask_big, unknown_big)
    return x.detach().reshape(batch_size, int(samples_per), n_features)
