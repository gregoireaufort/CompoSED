from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import multiprocessing as mp
from typing import Any, Callable, Literal
import warnings

import numpy as np


def _require_torch_dependency():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "The MDN posterior estimator requires the optional dependency torch. "
            "Install it with, for example: pip install torch"
        ) from exc


def _require_sbi_dependencies():
    try:
        torch = importlib.import_module("torch")
        nflows = importlib.import_module("nflows")
        return torch, nflows
    except ImportError as exc:
        raise ImportError(
            "inftools.sbi requires optional dependencies torch and nflows. "
            "Install them with, for example: pip install torch nflows"
        ) from exc


def resolve_torch_device(
    torch,
    device: str | None = "auto",
    *,
    validate: bool = True,
    allow_fallback: bool = True,
):
    """Return a usable torch device for SBI neural estimators.

    ``device="auto"`` tries CUDA, then Apple MPS, then CPU.  When validation is
    enabled, a tiny float32 forward/backward smoke test is run before training
    starts.  This catches common accelerator problems early, especially MPS
    float64/default-dtype surprises.
    """

    requested = "auto" if device is None else str(device).lower()
    if requested == "auto":
        candidates: list[str] = []
        if torch.cuda.is_available():
            candidates.append("cuda")
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            candidates.append("mps")
        candidates.append("cpu")
    else:
        candidates = [requested]
        if allow_fallback and requested != "cpu":
            candidates.append("cpu")

    failures: list[str] = []
    for candidate in candidates:
        try:
            candidate_device = torch.device(candidate)
            if validate:
                _validate_float32_device(torch, candidate_device)
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
                    f"Requested torch device {requested!r} is not usable for SBI float32 workloads. "
                    f"Validation failure: {exc!r}"
                ) from exc

    raise RuntimeError("No usable torch device found. Validation failures: " + "; ".join(failures))


@dataclass
class Standardizer:
    """Column-wise affine standardization fitted from a NumPy training table.

    Constant columns use unit scale. ``log_abs_det_inverse`` is the Jacobian
    term required when a density evaluated in standardized coordinates is
    reported in the original coordinates.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        values = np.asarray(values, dtype=float)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean

    @property
    def log_abs_det_inverse(self) -> float:
        return float(np.sum(np.log(self.std)))


def build_maf(theta_dim: int, x_dim: int, hidden_features: int = 128, num_transforms: int = 5, num_blocks: int = 2):
    """Build a conditional MAF q(theta | x) using nflows."""

    torch, _ = _require_sbi_dependencies()
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import ReversePermutation

    transforms = []
    for _ in range(int(num_transforms)):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=int(theta_dim),
                hidden_features=int(hidden_features),
                context_features=int(x_dim),
                num_blocks=int(num_blocks),
                use_residual_blocks=False,
                random_mask=False,
                activation=torch.nn.functional.relu,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
        transforms.append(ReversePermutation(features=int(theta_dim)))
    return Flow(CompositeTransform(transforms), StandardNormal([int(theta_dim)]))


class MAFPosteriorEstimator:
    """NumPy-facing conditional MAF posterior estimator q(theta | x)."""

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        hidden_features: int = 128,
        num_transforms: int = 5,
        num_blocks: int = 2,
        learning_rate: float = 1e-3,
        device: str | None = "auto",
        validate_device: bool = True,
        allow_device_fallback: bool = True,
        standardize: bool = True,
        max_grad_norm: float | None = None,
        restore_best: bool = True,
        initialization_seed: int | None = None,
    ) -> None:
        torch, _ = _require_sbi_dependencies()
        self.torch = torch
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.hidden_features = int(hidden_features)
        self.num_transforms = int(num_transforms)
        self.num_blocks = int(num_blocks)
        self.learning_rate = float(learning_rate)
        self.standardize = bool(standardize)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.restore_best = bool(restore_best)
        self.initialization_seed = None if initialization_seed is None else int(initialization_seed)
        self.device = resolve_torch_device(
            torch,
            device=device,
            validate=bool(validate_device),
            allow_fallback=bool(allow_device_fallback),
        )
        _seed_torch(torch, self.initialization_seed)
        flow = build_maf(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_features=self.hidden_features,
            num_transforms=self.num_transforms,
            num_blocks=self.num_blocks,
        )
        self.flow = _prepare_flow_for_device(flow, torch, self.device)
        self.theta_standardizer: Standardizer | None = None
        self.x_standardizer: Standardizer | None = None
        self.history: dict[str, list[float]] = {"train_loss": []}

    def fit(
        self,
        theta_train: np.ndarray,
        x_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.0,
        patience: int | None = None,
        min_delta: float = 0.0,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        torch = self.torch
        theta_train = _as_2d(theta_train, self.theta_dim, "theta_train")
        x_train = _as_2d(x_train, self.x_dim, "x_train")
        if theta_train.shape[0] != x_train.shape[0]:
            raise ValueError("theta_train and x_train must have the same number of rows.")
        epochs = int(epochs)
        batch_size = int(batch_size)
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        validation_split = float(validation_split)
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must lie in [0, 1).")
        if patience is not None and int(patience) <= 0:
            raise ValueError("patience must be positive or None.")
        patience = None if patience is None else int(patience)
        min_delta = float(min_delta)
        if not np.isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative.")
        n_total = theta_train.shape[0]
        n_validation = int(round(validation_split * n_total))
        if n_validation >= n_total:
            raise ValueError("validation_split must leave at least one training row.")
        permutation = np.random.default_rng(seed).permutation(n_total)
        validation_indices = permutation[:n_validation]
        training_indices = permutation[n_validation:]

        if self.standardize:
            self.theta_standardizer = Standardizer.fit(theta_train[training_indices])
            self.x_standardizer = Standardizer.fit(x_train[training_indices])
            theta_fit = self.theta_standardizer.transform(theta_train)
            x_fit = self.x_standardizer.transform(x_train)
        else:
            self.theta_standardizer = Standardizer(np.zeros(self.theta_dim), np.ones(self.theta_dim))
            self.x_standardizer = Standardizer(np.zeros(self.x_dim), np.ones(self.x_dim))
            theta_fit = theta_train
            x_fit = x_train

        _seed_torch(torch, seed)
        theta_t = torch.as_tensor(theta_fit, dtype=torch.float32, device=self.device)
        x_t = torch.as_tensor(x_fit, dtype=torch.float32, device=self.device)
        training_index_t = torch.as_tensor(training_indices, dtype=torch.long, device=self.device)
        dataset = torch.utils.data.TensorDataset(
            theta_t.index_select(0, training_index_t),
            x_t.index_select(0, training_index_t),
        )
        loader_generator = torch.Generator()
        if seed is not None:
            loader_generator.manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        if n_validation > 0:
            validation_index_t = torch.as_tensor(validation_indices, dtype=torch.long, device=self.device)
            theta_validation = theta_t.index_select(0, validation_index_t)
            x_validation = x_t.index_select(0, validation_index_t)
        else:
            theta_validation = None
            x_validation = None
        opt = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        self.history = {"train_loss": []}
        if n_validation > 0:
            self.history["val_loss"] = []
        self.flow.train()
        best_loss = float("inf")
        best_state = None
        best_epoch = None
        epochs_without_improvement = 0
        for epoch in range(epochs):
            losses = []
            saw_nonfinite_loss = False
            for theta_b, x_b in loader:
                with _temporary_default_dtype(torch, torch.float32):
                    loss = -self.flow.log_prob(inputs=theta_b, context=x_b).mean()
                if not bool(torch.isfinite(loss).detach().cpu().item()):
                    saw_nonfinite_loss = True
                    losses.append(np.nan)
                    break
                opt.zero_grad()
                loss.backward()
                gradients_are_finite = all(
                    parameter.grad is None
                    or bool(torch.all(torch.isfinite(parameter.grad)).detach().cpu().item())
                    for parameter in self.flow.parameters()
                )
                if not gradients_are_finite:
                    saw_nonfinite_loss = True
                    losses.append(np.nan)
                    break
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.max_grad_norm)
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            mean_loss = float(np.mean(losses)) if losses else np.nan
            self.history["train_loss"].append(mean_loss)
            if theta_validation is not None:
                self.flow.eval()
                with torch.no_grad(), _temporary_default_dtype(torch, torch.float32):
                    validation_loss = -self.flow.log_prob(
                        inputs=theta_validation,
                        context=x_validation,
                    ).mean()
                validation_loss_value = float(validation_loss.detach().cpu().item())
                self.history["val_loss"].append(validation_loss_value)
                self.flow.train()
                selection_loss = validation_loss_value
            else:
                selection_loss = mean_loss
            if np.isfinite(selection_loss) and selection_loss < best_loss - min_delta:
                best_loss = selection_loss
                best_state = {key: value.detach().clone() for key, value in self.flow.state_dict().items()}
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if verbose:
                message = f"epoch {epoch + 1}/{epochs}: train_loss={mean_loss:.6g}"
                if theta_validation is not None:
                    message += f", val_loss={validation_loss_value:.6g}"
                print(message)
            if saw_nonfinite_loss:
                self.history["stopped_early_nonfinite_loss"] = [float(epoch + 1)]
                if verbose:
                    print(f"stopping early after non-finite loss at epoch {epoch + 1}")
                break
            if patience is not None and epochs_without_improvement >= patience:
                self.history["stopped_early_patience"] = [float(epoch + 1)]
                if verbose:
                    print(f"stopping early after {patience} epochs without improvement")
                break
        if best_state is None:
            raise FloatingPointError("MAF training produced no finite epoch loss.")
        if self.restore_best and best_state is not None:
            self.flow.load_state_dict(best_state)
            self.history["best_selection_loss"] = [best_loss]
        self.history["best_epoch"] = [float(best_epoch)]
        self.history["epochs_ran"] = [float(len(self.history["train_loss"]))]
        return self.history

    def configuration(self) -> dict[str, Any]:
        """JSON-friendly architecture and training-state configuration."""

        return {
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "hidden_features": self.hidden_features,
            "num_transforms": self.num_transforms,
            "num_blocks": self.num_blocks,
            "learning_rate": self.learning_rate,
            "standardize": self.standardize,
            "max_grad_norm": self.max_grad_norm,
            "restore_best": self.restore_best,
            "initialization_seed": self.initialization_seed,
        }

    def sample(
        self,
        x_obs: np.ndarray,
        num_samples: int = 10000,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        torch = self.torch
        _seed_torch(torch, seed)
        x = _as_context_batch(x_obs, self.x_dim)
        x_std = self.x_standardizer.transform(x)
        context = torch.as_tensor(x_std, dtype=torch.float32, device=self.device)
        self.flow.eval()
        with torch.no_grad(), _temporary_default_dtype(torch, torch.float32):
            samples_std = self.flow.sample(int(num_samples), context=context)
        samples_np = samples_std.detach().cpu().numpy()
        if samples_np.ndim == 3 and samples_np.shape[0] == 1:
            samples_np = samples_np[0]
        elif samples_np.ndim == 2:
            pass
        return self.theta_standardizer.inverse_transform(samples_np)

    def log_prob(self, theta: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        self._check_fitted()
        torch = self.torch
        theta_arr = _as_2d(theta, self.theta_dim, "theta")
        x_arr = _as_context_batch(x_obs, self.x_dim)
        if x_arr.shape[0] == 1 and theta_arr.shape[0] > 1:
            x_arr = np.repeat(x_arr, theta_arr.shape[0], axis=0)
        if x_arr.shape[0] != theta_arr.shape[0]:
            raise ValueError("x_obs must have one row or the same number of rows as theta.")
        theta_std = self.theta_standardizer.transform(theta_arr)
        x_std = self.x_standardizer.transform(x_arr)
        self.flow.eval()
        with torch.no_grad(), _temporary_default_dtype(torch, torch.float32):
            lp_std = self.flow.log_prob(
                inputs=torch.as_tensor(theta_std, dtype=torch.float32, device=self.device),
                context=torch.as_tensor(x_std, dtype=torch.float32, device=self.device),
            )
        lp = lp_std.detach().cpu().numpy() - self.theta_standardizer.log_abs_det_inverse
        return lp[0] if np.asarray(theta).ndim == 1 else lp

    def _check_fitted(self) -> None:
        if self.theta_standardizer is None or self.x_standardizer is None:
            raise RuntimeError("Estimator must be fit before calling sample or log_prob.")


def build_categorical_classifier(
    x_dim: int,
    n_categories: int,
    hidden_features: int = 128,
    num_blocks: int = 2,
):
    """Build logits for the joint discrete state ``q(c | x)``."""

    torch = _require_torch_dependency()
    x_dim = int(x_dim)
    n_categories = int(n_categories)
    hidden_features = int(hidden_features)
    num_blocks = int(num_blocks)
    if x_dim <= 0 or n_categories < 2:
        raise ValueError("Categorical SBI requires x_dim > 0 and at least two categories.")
    if hidden_features <= 0 or num_blocks < 0:
        raise ValueError("hidden_features must be positive and num_blocks non-negative.")

    layers = []
    n_input = x_dim
    for _ in range(num_blocks):
        layers.extend(
            [
                torch.nn.Linear(n_input, hidden_features),
                torch.nn.ReLU(),
            ]
        )
        n_input = hidden_features
    layers.append(torch.nn.Linear(n_input, n_categories))
    return torch.nn.Sequential(*layers)


class HybridMAFPosteriorEstimator:
    """Categorical posterior mass plus a conditional MAF for continuous axes.

    The estimator represents

    ``q(category, theta_continuous | x)
      = q(category | x) q(theta_continuous | x, category)``.

    Category labels are integer indices with no numerical geometry. The
    continuous MAF receives a one-hot category appended to the standardized
    observation context.
    """

    def __init__(
        self,
        continuous_dim: int,
        x_dim: int,
        n_categories: int,
        hidden_features: int = 128,
        num_transforms: int = 5,
        num_blocks: int = 2,
        classifier_hidden_features: int | None = None,
        classifier_num_blocks: int = 2,
        learning_rate: float = 1e-3,
        device: str | None = "auto",
        validate_device: bool = True,
        allow_device_fallback: bool = True,
        standardize: bool = True,
        max_grad_norm: float | None = None,
        restore_best: bool = True,
        initialization_seed: int | None = None,
    ) -> None:
        torch, _ = _require_sbi_dependencies()
        self.torch = torch
        self.continuous_dim = int(continuous_dim)
        self.theta_dim = self.continuous_dim
        self.x_dim = int(x_dim)
        self.n_categories = int(n_categories)
        self.hidden_features = int(hidden_features)
        self.num_transforms = int(num_transforms)
        self.num_blocks = int(num_blocks)
        self.classifier_hidden_features = int(
            self.hidden_features
            if classifier_hidden_features is None
            else classifier_hidden_features
        )
        self.classifier_num_blocks = int(classifier_num_blocks)
        self.learning_rate = float(learning_rate)
        self.standardize = bool(standardize)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.restore_best = bool(restore_best)
        self.initialization_seed = None if initialization_seed is None else int(initialization_seed)
        if self.continuous_dim < 0:
            raise ValueError("continuous_dim must be non-negative.")
        if self.n_categories < 2:
            raise ValueError("Hybrid MAF requires at least two categorical states.")
        self.device = resolve_torch_device(
            torch,
            device=device,
            validate=bool(validate_device),
            allow_fallback=bool(allow_device_fallback),
        )

        _seed_torch(torch, self.initialization_seed)
        classifier = build_categorical_classifier(
            self.x_dim,
            self.n_categories,
            hidden_features=self.classifier_hidden_features,
            num_blocks=self.classifier_num_blocks,
        )
        self.classifier = classifier.to(dtype=torch.float32).to(device=self.device)
        if self.continuous_dim:
            flow = build_maf(
                theta_dim=self.continuous_dim,
                x_dim=self.x_dim + self.n_categories,
                hidden_features=self.hidden_features,
                num_transforms=self.num_transforms,
                num_blocks=self.num_blocks,
            )
            self.flow = _prepare_flow_for_device(flow, torch, self.device)
        else:
            self.flow = None
        self.theta_standardizer: Standardizer | None = None
        self.x_standardizer: Standardizer | None = None
        self.history: dict[str, list[float]] = {"train_loss": []}

    def fit(
        self,
        theta_continuous: np.ndarray,
        category: np.ndarray,
        x_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.0,
        patience: int | None = None,
        min_delta: float = 0.0,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Fit categorical probabilities and the category-conditioned flow."""

        torch = self.torch
        theta = np.asarray(theta_continuous, dtype=float)
        if self.continuous_dim == 0:
            if theta.ndim != 2 or theta.shape[1] != 0:
                raise ValueError("theta_continuous must have shape (n, 0).")
        else:
            theta = _as_2d(theta, self.continuous_dim, "theta_continuous")
        x = _as_2d(x_train, self.x_dim, "x_train")
        labels = np.asarray(category)
        if labels.ndim != 1 or labels.shape[0] != x.shape[0] or theta.shape[0] != x.shape[0]:
            raise ValueError("theta_continuous, category, and x_train must contain the same rows.")
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
            raise ValueError("category must contain finite integer indices.")
        labels = labels.astype(np.int64)
        if np.any((labels < 0) | (labels >= self.n_categories)):
            raise ValueError("category contains an index outside the declared support.")
        observed_categories = np.unique(labels)
        if observed_categories.size != self.n_categories:
            missing = sorted(set(range(self.n_categories)) - set(observed_categories.tolist()))
            raise ValueError(
                "Every declared categorical state needs training rows; missing state indices "
                f"{missing}."
            )

        epochs = int(epochs)
        batch_size = int(batch_size)
        validation_split = float(validation_split)
        patience = None if patience is None else int(patience)
        min_delta = float(min_delta)
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must lie in [0, 1).")
        if patience is not None and patience <= 0:
            raise ValueError("patience must be positive or None.")
        if not np.isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative.")

        rng = np.random.default_rng(seed)
        training_parts = []
        validation_parts = []
        for state in range(self.n_categories):
            state_indices = np.flatnonzero(labels == state)
            state_indices = state_indices[rng.permutation(state_indices.size)]
            n_validation = int(round(validation_split * state_indices.size))
            if validation_split > 0.0 and state_indices.size > 1:
                n_validation = min(max(n_validation, 1), state_indices.size - 1)
            else:
                n_validation = 0
            validation_parts.append(state_indices[:n_validation])
            training_parts.append(state_indices[n_validation:])
        training_indices = np.concatenate(training_parts)
        validation_indices = np.concatenate(validation_parts)
        training_indices = training_indices[rng.permutation(training_indices.size)]
        if validation_indices.size:
            validation_indices = validation_indices[rng.permutation(validation_indices.size)]

        if self.standardize:
            self.x_standardizer = Standardizer.fit(x[training_indices])
            self.theta_standardizer = (
                Standardizer.fit(theta[training_indices])
                if self.continuous_dim
                else Standardizer(np.empty(0), np.empty(0))
            )
            x_fit = self.x_standardizer.transform(x)
            theta_fit = (
                self.theta_standardizer.transform(theta)
                if self.continuous_dim
                else theta
            )
        else:
            self.x_standardizer = Standardizer(np.zeros(self.x_dim), np.ones(self.x_dim))
            self.theta_standardizer = Standardizer(
                np.zeros(self.continuous_dim),
                np.ones(self.continuous_dim),
            )
            x_fit = x
            theta_fit = theta

        _seed_torch(torch, seed)
        x_t = torch.as_tensor(x_fit, dtype=torch.float32, device=self.device)
        theta_t = torch.as_tensor(theta_fit, dtype=torch.float32, device=self.device)
        category_t = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        training_index_t = torch.as_tensor(training_indices, dtype=torch.long, device=self.device)
        dataset = torch.utils.data.TensorDataset(
            theta_t.index_select(0, training_index_t),
            category_t.index_select(0, training_index_t),
            x_t.index_select(0, training_index_t),
        )
        loader_generator = torch.Generator()
        if seed is not None:
            loader_generator.manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        if validation_indices.size:
            validation_index_t = torch.as_tensor(
                validation_indices,
                dtype=torch.long,
                device=self.device,
            )
            theta_validation = theta_t.index_select(0, validation_index_t)
            category_validation = category_t.index_select(0, validation_index_t)
            x_validation = x_t.index_select(0, validation_index_t)
        else:
            theta_validation = None
            category_validation = None
            x_validation = None

        parameters = list(self.classifier.parameters())
        if self.flow is not None:
            parameters.extend(self.flow.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.history = {
            "train_loss": [],
            "train_categorical_loss": [],
            "train_flow_loss": [],
        }
        if validation_indices.size:
            self.history.update(
                {
                    "val_loss": [],
                    "val_categorical_loss": [],
                    "val_flow_loss": [],
                }
            )
        best_loss = float("inf")
        best_classifier_state = None
        best_flow_state = None
        best_epoch = None
        epochs_without_improvement = 0
        self.classifier.train()
        if self.flow is not None:
            self.flow.train()

        for epoch in range(epochs):
            total_losses = []
            categorical_losses = []
            flow_losses = []
            saw_nonfinite_loss = False
            for theta_batch, category_batch, x_batch in loader:
                with _temporary_default_dtype(torch, torch.float32):
                    logits = self.classifier(x_batch)
                    categorical_loss = torch.nn.functional.cross_entropy(
                        logits,
                        category_batch,
                    )
                    if self.flow is None:
                        flow_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    else:
                        category_one_hot = torch.nn.functional.one_hot(
                            category_batch,
                            num_classes=self.n_categories,
                        ).to(dtype=torch.float32)
                        flow_context = torch.cat([x_batch, category_one_hot], dim=1)
                        flow_loss = -self.flow.log_prob(
                            inputs=theta_batch,
                            context=flow_context,
                        ).mean()
                    loss = categorical_loss + flow_loss
                if not bool(torch.isfinite(loss).detach().cpu().item()):
                    saw_nonfinite_loss = True
                    break
                optimizer.zero_grad()
                loss.backward()
                gradients_are_finite = all(
                    parameter.grad is None
                    or bool(torch.all(torch.isfinite(parameter.grad)).detach().cpu().item())
                    for parameter in parameters
                )
                if not gradients_are_finite:
                    saw_nonfinite_loss = True
                    break
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                optimizer.step()
                total_losses.append(float(loss.detach().cpu().item()))
                categorical_losses.append(float(categorical_loss.detach().cpu().item()))
                flow_losses.append(float(flow_loss.detach().cpu().item()))

            train_total = float(np.mean(total_losses)) if total_losses else np.nan
            train_categorical = (
                float(np.mean(categorical_losses)) if categorical_losses else np.nan
            )
            train_flow = float(np.mean(flow_losses)) if flow_losses else np.nan
            self.history["train_loss"].append(train_total)
            self.history["train_categorical_loss"].append(train_categorical)
            self.history["train_flow_loss"].append(train_flow)

            if validation_indices.size:
                self.classifier.eval()
                if self.flow is not None:
                    self.flow.eval()
                with torch.no_grad(), _temporary_default_dtype(torch, torch.float32):
                    validation_logits = self.classifier(x_validation)
                    validation_categorical_loss = torch.nn.functional.cross_entropy(
                        validation_logits,
                        category_validation,
                    )
                    if self.flow is None:
                        validation_flow_loss = torch.zeros(
                            (),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    else:
                        validation_one_hot = torch.nn.functional.one_hot(
                            category_validation,
                            num_classes=self.n_categories,
                        ).to(dtype=torch.float32)
                        validation_context = torch.cat(
                            [x_validation, validation_one_hot],
                            dim=1,
                        )
                        validation_flow_loss = -self.flow.log_prob(
                            inputs=theta_validation,
                            context=validation_context,
                        ).mean()
                    validation_loss = (
                        validation_categorical_loss + validation_flow_loss
                    )
                validation_total = float(validation_loss.detach().cpu().item())
                self.history["val_loss"].append(validation_total)
                self.history["val_categorical_loss"].append(
                    float(validation_categorical_loss.detach().cpu().item())
                )
                self.history["val_flow_loss"].append(
                    float(validation_flow_loss.detach().cpu().item())
                )
                selection_loss = validation_total
                self.classifier.train()
                if self.flow is not None:
                    self.flow.train()
            else:
                selection_loss = train_total

            if np.isfinite(selection_loss) and selection_loss < best_loss - min_delta:
                best_loss = selection_loss
                best_classifier_state = {
                    key: value.detach().clone()
                    for key, value in self.classifier.state_dict().items()
                }
                best_flow_state = (
                    None
                    if self.flow is None
                    else {
                        key: value.detach().clone()
                        for key, value in self.flow.state_dict().items()
                    }
                )
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose:
                message = (
                    f"epoch {epoch + 1}/{epochs}: train_loss={train_total:.6g}, "
                    f"categorical={train_categorical:.6g}, flow={train_flow:.6g}"
                )
                if validation_indices.size:
                    message += f", val_loss={selection_loss:.6g}"
                print(message)
            if saw_nonfinite_loss:
                self.history["stopped_early_nonfinite_loss"] = [float(epoch + 1)]
                break
            if patience is not None and epochs_without_improvement >= patience:
                self.history["stopped_early_patience"] = [float(epoch + 1)]
                break

        if best_classifier_state is None:
            raise FloatingPointError("Hybrid MAF training produced no finite epoch loss.")
        if self.restore_best:
            self.classifier.load_state_dict(best_classifier_state)
            if self.flow is not None:
                self.flow.load_state_dict(best_flow_state)
            self.history["best_selection_loss"] = [best_loss]
        self.history["best_epoch"] = [float(best_epoch)]
        self.history["epochs_ran"] = [float(len(self.history["train_loss"]))]
        self.classifier.eval()
        if self.flow is not None:
            self.flow.eval()
        return self.history

    def configuration(self) -> dict[str, Any]:
        """JSON-friendly architecture and training-state configuration."""

        return {
            "continuous_dim": self.continuous_dim,
            "x_dim": self.x_dim,
            "n_categories": self.n_categories,
            "hidden_features": self.hidden_features,
            "num_transforms": self.num_transforms,
            "num_blocks": self.num_blocks,
            "classifier_hidden_features": self.classifier_hidden_features,
            "classifier_num_blocks": self.classifier_num_blocks,
            "learning_rate": self.learning_rate,
            "standardize": self.standardize,
            "max_grad_norm": self.max_grad_norm,
            "restore_best": self.restore_best,
            "initialization_seed": self.initialization_seed,
        }

    def category_probabilities(self, x_obs: np.ndarray) -> np.ndarray:
        """Return normalized ``q(category | x)`` probabilities."""

        self._check_fitted()
        x = _as_context_batch(x_obs, self.x_dim)
        x_std = self.x_standardizer.transform(x)
        self.classifier.eval()
        with self.torch.no_grad(), _temporary_default_dtype(
            self.torch,
            self.torch.float32,
        ):
            logits = self.classifier(
                self.torch.as_tensor(
                    x_std,
                    dtype=self.torch.float32,
                    device=self.device,
                )
            )
            probabilities = self.torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def sample(
        self,
        x_obs: np.ndarray,
        num_samples: int = 10000,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw continuous coordinates and exact categorical state indices."""

        self._check_fitted()
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        torch = self.torch
        _seed_torch(torch, seed)
        x = _as_context_batch(x_obs, self.x_dim)
        x_std = self.x_standardizer.transform(x)
        x_t = torch.as_tensor(x_std, dtype=torch.float32, device=self.device)
        self.classifier.eval()
        if self.flow is not None:
            self.flow.eval()
        with torch.no_grad(), _temporary_default_dtype(torch, torch.float32):
            probabilities = torch.softmax(self.classifier(x_t), dim=1)
            categories = torch.multinomial(
                probabilities,
                num_samples=num_samples,
                replacement=True,
            )
            if self.flow is None:
                continuous = torch.empty(
                    (x.shape[0], num_samples, 0),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                flat_x = x_t.repeat_interleave(num_samples, dim=0)
                flat_categories = categories.reshape(-1)
                one_hot = torch.nn.functional.one_hot(
                    flat_categories,
                    num_classes=self.n_categories,
                ).to(dtype=torch.float32)
                flow_context = torch.cat([flat_x, one_hot], dim=1)
                continuous = self.flow.sample(1, context=flow_context)
                continuous = continuous.reshape(
                    x.shape[0],
                    num_samples,
                    self.continuous_dim,
                )
        continuous_np = self.theta_standardizer.inverse_transform(
            continuous.detach().cpu().numpy()
        )
        return continuous_np, categories.detach().cpu().numpy()

    def log_prob(
        self,
        theta_continuous: np.ndarray,
        category: np.ndarray | int,
        x_obs: np.ndarray,
    ) -> np.ndarray:
        """Evaluate log categorical mass plus conditional continuous density."""

        self._check_fitted()
        theta = np.asarray(theta_continuous, dtype=float)
        single = theta.ndim == 1
        if self.continuous_dim == 0:
            if theta.ndim == 1 and theta.size == 0:
                theta = theta.reshape(1, 0)
            if theta.ndim != 2 or theta.shape[1] != 0:
                raise ValueError("theta_continuous must have shape (n, 0).")
        else:
            theta = _as_2d(theta, self.continuous_dim, "theta_continuous")
        labels = np.asarray(category)
        if labels.ndim == 0:
            labels = labels.reshape(1)
        if labels.ndim != 1 or labels.size not in {1, theta.shape[0]}:
            raise ValueError("category must be scalar or have one entry per theta row.")
        if labels.size == 1 and theta.shape[0] > 1:
            labels = np.repeat(labels, theta.shape[0])
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
            raise ValueError("category must contain finite integer indices.")
        labels = labels.astype(np.int64)
        if np.any((labels < 0) | (labels >= self.n_categories)):
            raise ValueError("category contains an index outside the declared support.")

        x = _as_context_batch(x_obs, self.x_dim)
        if x.shape[0] == 1 and theta.shape[0] > 1:
            x = np.repeat(x, theta.shape[0], axis=0)
        if x.shape[0] != theta.shape[0]:
            raise ValueError("x_obs must have one row or the same number of rows as theta.")
        x_std = self.x_standardizer.transform(x)
        theta_std = (
            self.theta_standardizer.transform(theta)
            if self.continuous_dim
            else theta
        )
        torch = self.torch
        x_t = torch.as_tensor(x_std, dtype=torch.float32, device=self.device)
        category_t = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        self.classifier.eval()
        if self.flow is not None:
            self.flow.eval()
        with torch.no_grad(), _temporary_default_dtype(torch, torch.float32):
            log_category = torch.log_softmax(self.classifier(x_t), dim=1).gather(
                1,
                category_t[:, None],
            )[:, 0]
            if self.flow is None:
                log_continuous = torch.zeros_like(log_category)
            else:
                one_hot = torch.nn.functional.one_hot(
                    category_t,
                    num_classes=self.n_categories,
                ).to(dtype=torch.float32)
                flow_context = torch.cat([x_t, one_hot], dim=1)
                log_continuous = self.flow.log_prob(
                    inputs=torch.as_tensor(
                        theta_std,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    context=flow_context,
                )
        result = (
            log_category.detach().cpu().numpy()
            + log_continuous.detach().cpu().numpy()
            - self.theta_standardizer.log_abs_det_inverse
        )
        return result[0] if single else result

    def _check_fitted(self) -> None:
        if self.theta_standardizer is None or self.x_standardizer is None:
            raise RuntimeError("Estimator must be fit before calling sample or log_prob.")


def build_mdn(
    theta_dim: int,
    x_dim: int,
    n_components: int = 8,
    hidden_features: int = 128,
    num_blocks: int = 3,
    min_scale: float = 1.0e-3,
):
    """Build a conditional diagonal-Gaussian mixture q(theta | x)."""

    torch = _require_torch_dependency()
    nn = torch.nn
    theta_dim = int(theta_dim)
    x_dim = int(x_dim)
    n_components = int(n_components)
    hidden_features = int(hidden_features)
    num_blocks = int(num_blocks)
    min_scale = float(min_scale)
    if theta_dim <= 0 or x_dim <= 0:
        raise ValueError("theta_dim and x_dim must be positive.")
    if n_components <= 0 or hidden_features <= 0 or num_blocks <= 0:
        raise ValueError("n_components, hidden_features, and num_blocks must be positive.")
    if not np.isfinite(min_scale) or min_scale <= 0.0:
        raise ValueError("min_scale must be positive and finite.")

    class ConditionalDiagonalGaussianMixture(nn.Module):
        """Small context network parameterizing a normalized Gaussian mixture."""

        def __init__(self):
            super().__init__()
            layers = []
            input_features = x_dim
            for _ in range(num_blocks):
                layers.extend(
                    [
                        nn.Linear(input_features, hidden_features),
                        nn.SiLU(),
                    ]
                )
                input_features = hidden_features
            self.conditioner = nn.Sequential(*layers)
            output_features = n_components * (1 + 2 * theta_dim)
            self.output = nn.Linear(hidden_features, output_features)

        def mixture_parameters(self, context):
            raw = self.output(self.conditioner(context))
            logits = raw[:, :n_components]
            remainder = raw[:, n_components:].reshape(
                context.shape[0], n_components, 2 * theta_dim
            )
            means = remainder[:, :, :theta_dim]
            scales = (
                torch.nn.functional.softplus(remainder[:, :, theta_dim:])
                + min_scale
            )
            return logits, means, scales

        def log_prob(self, inputs, context):
            logits, means, scales = self.mixture_parameters(context)
            residual = (inputs[:, None, :] - means) / scales
            component_log_prob = -0.5 * (
                residual.square()
                + 2.0 * torch.log(scales)
                + np.log(2.0 * np.pi)
            ).sum(dim=-1)
            log_weights = torch.log_softmax(logits, dim=-1)
            return torch.logsumexp(log_weights + component_log_prob, dim=-1)

        def sample(self, num_samples, context):
            logits, means, scales = self.mixture_parameters(context)
            component = torch.multinomial(
                torch.softmax(logits, dim=-1),
                int(num_samples),
                replacement=True,
            )
            gather_index = component[:, :, None].expand(
                context.shape[0], int(num_samples), theta_dim
            )
            selected_means = torch.gather(means, dim=1, index=gather_index)
            selected_scales = torch.gather(scales, dim=1, index=gather_index)
            noise = torch.randn(
                selected_means.shape,
                dtype=selected_means.dtype,
                device=selected_means.device,
            )
            return selected_means + selected_scales * noise

    return ConditionalDiagonalGaussianMixture()


class MDNPosteriorEstimator:
    """NumPy-facing conditional Gaussian-mixture posterior q(theta | x).

    The estimator uses a mixture of diagonal Gaussians in standardized target
    coordinates. Mixture weights, means, and positive scales are predicted from
    the observation context by a small multilayer perceptron.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        n_components: int = 8,
        hidden_features: int = 128,
        num_blocks: int = 3,
        min_scale: float = 1.0e-3,
        learning_rate: float = 1.0e-3,
        device: str | None = "auto",
        validate_device: bool = True,
        allow_device_fallback: bool = True,
        standardize: bool = True,
        max_grad_norm: float | None = None,
        restore_best: bool = True,
        initialization_seed: int | None = None,
    ) -> None:
        torch = _require_torch_dependency()
        self.torch = torch
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.n_components = int(n_components)
        self.hidden_features = int(hidden_features)
        self.num_blocks = int(num_blocks)
        self.min_scale = float(min_scale)
        self.learning_rate = float(learning_rate)
        self.standardize = bool(standardize)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.restore_best = bool(restore_best)
        self.initialization_seed = (
            None if initialization_seed is None else int(initialization_seed)
        )
        self.device = resolve_torch_device(
            torch,
            device=device,
            validate=bool(validate_device),
            allow_fallback=bool(allow_device_fallback),
        )
        _seed_torch(torch, self.initialization_seed)
        self.network = build_mdn(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            n_components=self.n_components,
            hidden_features=self.hidden_features,
            num_blocks=self.num_blocks,
            min_scale=self.min_scale,
        ).to(dtype=torch.float32, device=self.device)
        self.theta_standardizer: Standardizer | None = None
        self.x_standardizer: Standardizer | None = None
        self.history: dict[str, list[float]] = {"train_loss": []}

    def fit(
        self,
        theta_train: np.ndarray,
        x_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.0,
        patience: int | None = None,
        min_delta: float = 0.0,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Minimize the exact conditional mixture negative log likelihood."""

        torch = self.torch
        theta_train = _as_2d(theta_train, self.theta_dim, "theta_train")
        x_train = _as_2d(x_train, self.x_dim, "x_train")
        if theta_train.shape[0] != x_train.shape[0]:
            raise ValueError("theta_train and x_train must have the same number of rows.")
        epochs = int(epochs)
        batch_size = int(batch_size)
        validation_split = float(validation_split)
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must lie in [0, 1).")
        if patience is not None and int(patience) <= 0:
            raise ValueError("patience must be positive or None.")
        patience = None if patience is None else int(patience)
        min_delta = float(min_delta)
        if not np.isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative.")

        n_total = theta_train.shape[0]
        n_validation = int(round(validation_split * n_total))
        if n_validation >= n_total:
            raise ValueError("validation_split must leave at least one training row.")
        permutation = np.random.default_rng(seed).permutation(n_total)
        validation_indices = permutation[:n_validation]
        training_indices = permutation[n_validation:]

        if self.standardize:
            self.theta_standardizer = Standardizer.fit(theta_train[training_indices])
            self.x_standardizer = Standardizer.fit(x_train[training_indices])
            theta_fit = self.theta_standardizer.transform(theta_train)
            x_fit = self.x_standardizer.transform(x_train)
        else:
            self.theta_standardizer = Standardizer(
                np.zeros(self.theta_dim), np.ones(self.theta_dim)
            )
            self.x_standardizer = Standardizer(
                np.zeros(self.x_dim), np.ones(self.x_dim)
            )
            theta_fit = theta_train
            x_fit = x_train

        _seed_torch(torch, seed)
        theta_t = torch.as_tensor(
            theta_fit, dtype=torch.float32, device=self.device
        )
        x_t = torch.as_tensor(x_fit, dtype=torch.float32, device=self.device)
        training_index_t = torch.as_tensor(
            training_indices, dtype=torch.long, device=self.device
        )
        dataset = torch.utils.data.TensorDataset(
            theta_t.index_select(0, training_index_t),
            x_t.index_select(0, training_index_t),
        )
        loader_generator = torch.Generator()
        if seed is not None:
            loader_generator.manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        if n_validation > 0:
            validation_index_t = torch.as_tensor(
                validation_indices, dtype=torch.long, device=self.device
            )
            theta_validation = theta_t.index_select(0, validation_index_t)
            x_validation = x_t.index_select(0, validation_index_t)
        else:
            theta_validation = None
            x_validation = None

        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        self.history = {"train_loss": []}
        if n_validation > 0:
            self.history["val_loss"] = []
        best_loss = float("inf")
        best_state = None
        best_epoch = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.network.train()
            losses = []
            saw_nonfinite_loss = False
            for theta_batch, x_batch in loader:
                loss = -self.network.log_prob(theta_batch, x_batch).mean()
                if not bool(torch.isfinite(loss).detach().cpu().item()):
                    saw_nonfinite_loss = True
                    losses.append(np.nan)
                    break
                optimizer.zero_grad()
                loss.backward()
                gradients_are_finite = all(
                    parameter.grad is None
                    or bool(
                        torch.all(torch.isfinite(parameter.grad))
                        .detach()
                        .cpu()
                        .item()
                    )
                    for parameter in self.network.parameters()
                )
                if not gradients_are_finite:
                    saw_nonfinite_loss = True
                    losses.append(np.nan)
                    break
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.max_grad_norm
                    )
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

            mean_loss = float(np.mean(losses)) if losses else np.nan
            self.history["train_loss"].append(mean_loss)
            if theta_validation is not None:
                self.network.eval()
                with torch.no_grad():
                    validation_loss = -self.network.log_prob(
                        theta_validation, x_validation
                    ).mean()
                validation_loss_value = float(
                    validation_loss.detach().cpu().item()
                )
                self.history["val_loss"].append(validation_loss_value)
                selection_loss = validation_loss_value
            else:
                selection_loss = mean_loss

            if np.isfinite(selection_loss) and selection_loss < best_loss - min_delta:
                best_loss = selection_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.network.state_dict().items()
                }
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if verbose:
                message = f"epoch {epoch + 1}/{epochs}: train_loss={mean_loss:.6g}"
                if theta_validation is not None:
                    message += f", val_loss={validation_loss_value:.6g}"
                print(message)
            if saw_nonfinite_loss:
                self.history["stopped_early_nonfinite_loss"] = [
                    float(epoch + 1)
                ]
                break
            if patience is not None and epochs_without_improvement >= patience:
                self.history["stopped_early_patience"] = [float(epoch + 1)]
                break

        if best_state is None:
            raise FloatingPointError("MDN training produced no finite epoch loss.")
        if self.restore_best:
            self.network.load_state_dict(best_state)
            self.history["best_selection_loss"] = [best_loss]
        self.history["best_epoch"] = [float(best_epoch)]
        self.history["epochs_ran"] = [
            float(len(self.history["train_loss"]))
        ]
        return self.history

    def configuration(self) -> dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "n_components": self.n_components,
            "hidden_features": self.hidden_features,
            "num_blocks": self.num_blocks,
            "min_scale": self.min_scale,
            "learning_rate": self.learning_rate,
            "standardize": self.standardize,
            "max_grad_norm": self.max_grad_norm,
            "restore_best": self.restore_best,
            "initialization_seed": self.initialization_seed,
        }

    def sample(
        self,
        x_obs: np.ndarray,
        num_samples: int = 10000,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        torch = self.torch
        _seed_torch(torch, seed)
        x = _as_context_batch(x_obs, self.x_dim)
        x_std = self.x_standardizer.transform(x)
        context = torch.as_tensor(
            x_std, dtype=torch.float32, device=self.device
        )
        self.network.eval()
        with torch.no_grad():
            samples_std = self.network.sample(int(num_samples), context)
        samples = self.theta_standardizer.inverse_transform(
            samples_std.detach().cpu().numpy()
        )
        return samples[0] if samples.shape[0] == 1 else samples

    def log_prob(self, theta: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        self._check_fitted()
        torch = self.torch
        theta_arr = _as_2d(theta, self.theta_dim, "theta")
        x_arr = _as_context_batch(x_obs, self.x_dim)
        if x_arr.shape[0] == 1 and theta_arr.shape[0] > 1:
            x_arr = np.repeat(x_arr, theta_arr.shape[0], axis=0)
        if x_arr.shape[0] != theta_arr.shape[0]:
            raise ValueError(
                "x_obs must have one row or the same number of rows as theta."
            )
        theta_std = self.theta_standardizer.transform(theta_arr)
        x_std = self.x_standardizer.transform(x_arr)
        self.network.eval()
        with torch.no_grad():
            logp_standardized = self.network.log_prob(
                torch.as_tensor(
                    theta_std, dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(x_std, dtype=torch.float32, device=self.device),
            )
        logp = (
            logp_standardized.detach().cpu().numpy()
            - self.theta_standardizer.log_abs_det_inverse
        )
        return logp[0] if np.asarray(theta).ndim == 1 else logp

    def mixture_parameters(self, x_obs: np.ndarray) -> dict[str, np.ndarray]:
        """Return weights, physical means, and physical diagonal scales."""

        self._check_fitted()
        torch = self.torch
        x = _as_context_batch(x_obs, self.x_dim)
        x_std = self.x_standardizer.transform(x)
        self.network.eval()
        with torch.no_grad():
            logits, means_std, scales_std = self.network.mixture_parameters(
                torch.as_tensor(
                    x_std, dtype=torch.float32, device=self.device
                )
            )
        weights = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        means_std = means_std.detach().cpu().numpy()
        scales_std = scales_std.detach().cpu().numpy()
        means = (
            means_std * self.theta_standardizer.std[None, None, :]
            + self.theta_standardizer.mean[None, None, :]
        )
        scales = scales_std * self.theta_standardizer.std[None, None, :]
        return {"weights": weights, "means": means, "scales": scales}

    def _check_fitted(self) -> None:
        if self.theta_standardizer is None or self.x_standardizer is None:
            raise RuntimeError("Estimator must be fit before calling sample or log_prob.")


def simulate_training_set(
    parameter_space,
    simulator,
    n: int,
    noise_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator | None = None,
    max_retries: int = 100,
    failure_policy: Literal["raise", "resample"] = "raise",
    return_metadata: bool = False,
    batch_size: int = 1,
    n_workers: int = 1,
    executor: Literal["process", "thread", "serial"] = "process",
    mp_context: str | None = None,
    return_sigma: bool = False,
):
    """Sample theta from priors and simulate flux-like observations.

    The returned ``x`` rows are the same active-band or active-pixel vectors
    consumed by the likelihood. When ``return_sigma=True``, the simulator must
    expose ``simulate_with_uncertainty`` and return the raw catalog
    uncertainty supplied to the neural context alongside ``x``. A simulator
    may add a separately declared model-discrepancy term to the Gaussian draw
    without adding it to this returned context. For expensive backends such as FSPS, set
    ``n_workers > 1`` and a modest ``batch_size`` so each worker keeps its own
    backend instance alive across many forward-model calls. By default, the
    first failed prior draw raises: silently replacing failed rows would train
    on the declared prior conditioned on simulator success. Set
    ``failure_policy="resample"`` only when that conditioning is scientifically
    intended; the returned metadata records the failures and acceptance
    fraction. Process execution
    requires ``simulator`` and ``noise_fn`` to be pickleable; in notebooks,
    define them as top-level functions/classes or use ``executor="thread"``
    only for thread-safe simulators.
    """

    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative.")
    max_retries = int(max_retries)
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    failure_policy = str(failure_policy).lower()
    if failure_policy not in {"raise", "resample"}:
        raise ValueError("failure_policy must be 'raise' or 'resample'.")
    batch_size = int(batch_size)
    n_workers = int(n_workers)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_workers <= 0:
        raise ValueError("n_workers must be positive.")
    if executor not in {"process", "thread", "serial"}:
        raise ValueError("executor must be one of: process, thread, serial.")
    if n == 0:
        theta_empty = np.empty((0, parameter_space.ndim), dtype=float)
        x_empty = np.empty((0, 0), dtype=float)
        sigma_empty = np.empty((0, 0), dtype=float)
        if return_metadata:
            output = (theta_empty, x_empty, sigma_empty) if return_sigma else (theta_empty, x_empty)
            return (*output, {
                "attempts": 0,
                "failures": [],
                "n_failures": 0,
                "acceptance_fraction": 1.0,
                "failure_policy": failure_policy,
                "returned_prior": "declared_prior",
                "batch_size": batch_size,
                "n_workers": n_workers,
                "executor": executor,
                "mp_context": mp_context,
                "returned_sigma": bool(return_sigma),
            })
        return (theta_empty, x_empty, sigma_empty) if return_sigma else (theta_empty, x_empty)

    if executor == "serial" or n_workers == 1:
        theta_out, x_out, sigma_out, metadata = _simulate_training_set_serial(
            parameter_space,
            simulator,
            n=n,
            noise_fn=noise_fn,
            rng=rng,
            max_retries=max_retries,
            failure_policy=failure_policy,
            return_sigma=bool(return_sigma),
        )
    else:
        theta_out, x_out, sigma_out, metadata = _simulate_training_set_parallel(
            parameter_space,
            simulator,
            n=n,
            noise_fn=noise_fn,
            rng=rng,
            max_retries=max_retries,
            failure_policy=failure_policy,
            batch_size=batch_size,
            n_workers=n_workers,
            executor=executor,
            mp_context=mp_context,
            return_sigma=bool(return_sigma),
        )

    metadata.update(
        {
            "n_failures": len(metadata["failures"]),
            "acceptance_fraction": float(n / metadata["attempts"]),
            "failure_policy": failure_policy,
            "returned_prior": (
                "simulator_success_conditioned"
                if metadata["failures"]
                else "declared_prior"
            ),
            "batch_size": batch_size,
            "n_workers": n_workers,
            "executor": executor,
            "mp_context": mp_context,
            "returned_sigma": bool(return_sigma),
        }
    )
    if failure_policy == "resample" and metadata["failures"]:
        warnings.warn(
            f"Replaced {len(metadata['failures'])} failed simulation(s). Returned theta rows "
            "are draws from the declared prior conditioned on simulator success; inspect "
            "metadata['failures'] before training.",
            RuntimeWarning,
            stacklevel=2,
        )
    if return_metadata:
        if return_sigma:
            return theta_out, x_out, sigma_out, metadata
        return theta_out, x_out, metadata
    if return_sigma:
        return theta_out, x_out, sigma_out
    return theta_out, x_out


def _simulate_training_set_serial(
    parameter_space,
    simulator,
    *,
    n: int,
    noise_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    max_retries: int,
    failure_policy: Literal["raise", "resample"],
    return_sigma: bool,
):
    failures = []
    theta_rows = []
    x_rows = []
    sigma_rows = []
    attempts = 0
    while len(theta_rows) < n:
        if attempts - len(theta_rows) > int(max_retries):
            raise RuntimeError(
                f"Too many failed simulations: {len(failures)} failures while collecting {len(theta_rows)}/{n}."
            )
        attempts += 1
        theta = parameter_space.sample_prior(1, rng=rng)[0]
        try:
            simulated = _simulate_one(simulator, theta, noise_fn, rng, return_sigma=return_sigma)
            if return_sigma:
                x, sigma = simulated
            else:
                x, sigma = simulated, None
            x = np.asarray(x, dtype=float)
            if x.ndim != 1 or not np.all(np.isfinite(x)):
                raise ValueError(f"Simulator returned invalid observation shape/content: shape={x.shape}.")
            if return_sigma:
                sigma = np.asarray(sigma, dtype=float)
                if sigma.shape != x.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
                    raise ValueError("Simulator returned invalid uncertainty values.")
        except Exception as exc:
            failure = {
                "theta": np.asarray(theta, dtype=float),
                "error_type": f"{type(exc).__module__}.{type(exc).__name__}",
                "error": str(exc),
            }
            if failure_policy == "raise":
                raise RuntimeError(
                    "Training simulation failed for a draw from the declared prior. "
                    f"theta={np.asarray(theta, dtype=float).tolist()}, error={type(exc).__name__}: {exc}. "
                    "Fix the prior/parameterization or explicitly set "
                    "failure_policy='resample' to train on the simulator-success-conditioned prior."
                ) from exc
            failures.append(failure)
            continue
        theta_rows.append(theta)
        x_rows.append(x)
        if return_sigma:
            sigma_rows.append(sigma)
    theta_out = np.asarray(theta_rows, dtype=float)
    x_out = np.asarray(x_rows, dtype=float)
    sigma_out = np.asarray(sigma_rows, dtype=float) if return_sigma else None
    return theta_out, x_out, sigma_out, {"attempts": attempts, "failures": failures}


def _simulate_training_set_parallel(
    parameter_space,
    simulator,
    *,
    n: int,
    noise_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    max_retries: int,
    failure_policy: Literal["raise", "resample"],
    batch_size: int,
    n_workers: int,
    executor: Literal["process", "thread"],
    mp_context: str | None,
    return_sigma: bool,
):
    if executor == "thread":
        pool_cls = ThreadPoolExecutor
        pool_kwargs: dict[str, Any] = {
            "max_workers": n_workers,
            "initializer": _init_simulation_worker,
            "initargs": (simulator, noise_fn, return_sigma),
        }
    else:
        context = mp.get_context(mp_context) if mp_context is not None else None
        pool_cls = ProcessPoolExecutor
        pool_kwargs = {
            "max_workers": n_workers,
            "mp_context": context,
            "initializer": _init_simulation_worker,
            "initargs": (simulator, noise_fn, return_sigma),
        }

    theta_rows = []
    x_rows = []
    sigma_rows = []
    failures = []
    attempts = 0
    with pool_cls(**pool_kwargs) as pool:
        while len(theta_rows) < n:
            # Only simulate the rows still required. Failed rows are replaced
            # in a later wave, keeping the expensive backend workers alive
            # without evaluating the unused retry reserve.
            n_candidate = n - len(theta_rows)
            theta_candidates = parameter_space.sample_prior(n_candidate, rng=rng)
            chunks = [
                theta_candidates[start : start + batch_size]
                for start in range(0, n_candidate, batch_size)
            ]
            seeds = rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=len(chunks),
                dtype=np.uint32,
            )
            payloads = [(chunk, int(seed)) for chunk, seed in zip(chunks, seeds)]
            attempts += n_candidate

            for good_theta, good_x, good_sigma, bad in pool.map(
                _simulate_chunk_from_worker,
                payloads,
                chunksize=1,
            ):
                theta_rows.extend(good_theta)
                x_rows.extend(good_x)
                sigma_rows.extend(good_sigma)
                failures.extend(bad)
                if failure_policy == "raise" and bad:
                    first = bad[0]
                    raise RuntimeError(
                        "Training simulation failed for a draw from the declared prior. "
                        f"theta={np.asarray(first['theta'], dtype=float).tolist()}, "
                        f"error={first['error_type']}: {first['error']}. "
                        "Fix the prior/parameterization or explicitly set "
                        "failure_policy='resample' to train on the simulator-success-conditioned prior."
                    )

            if len(theta_rows) < n and len(failures) > max_retries:
                raise RuntimeError(
                    f"Too many failed simulations: {len(failures)} failures "
                    f"while collecting {len(theta_rows)}/{n}."
                )

    theta_out = np.asarray(theta_rows[:n], dtype=float)
    x_out = np.asarray(x_rows[:n], dtype=float)
    sigma_out = np.asarray(sigma_rows[:n], dtype=float) if return_sigma else None
    return theta_out, x_out, sigma_out, {"attempts": attempts, "failures": failures}


def train_maf_posterior_from_dataset(
    theta: np.ndarray,
    x: np.ndarray,
    *,
    theta_names: list[str] | tuple[str, ...] | None = None,
    x_names: list[str] | tuple[str, ...] | None = None,
    source: str = "precomputed_dataset",
    finite: Literal["raise", "drop"] = "raise",
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
    return_metadata: bool = False,
    **kwargs,
):
    """Train a MAF posterior from precomputed paired rows ``(theta, x)``.

    This is the explicit entry point for SBI when the training set already
    exists: a presampled forward model, an external simulation campaign, or an
    empirical catalog with fitted labels.  Row ``i`` of ``theta`` must describe
    the same object/simulation as row ``i`` of ``x``.  No CompoSED backend is
    called here.

    ``x`` should be the exact observation vector that will be supplied at
    inference time: for example active-band fluxes, magnitudes, or
    ``[mags, mag_errors]`` concatenated in a documented order.
    """

    theta_train, x_train, metadata = _prepare_precomputed_training_pairs(
        theta,
        x,
        theta_names=theta_names,
        x_names=x_names,
        source=source,
        finite=finite,
        shuffle=shuffle,
        rng=rng,
    )
    estimator = train_maf_posterior(theta_train, x_train, **kwargs)
    if return_metadata:
        return estimator, metadata
    return estimator


def train_maf_posterior(theta_train: np.ndarray, x_train: np.ndarray, **kwargs) -> MAFPosteriorEstimator:
    """Construct and fit a conditional MAF from paired NumPy arrays.

    Parameters
    ----------
    theta_train
        Physical parameter table with shape ``(n_train, n_theta)``.
    x_train
        Conditioning table with shape ``(n_train, n_context)``. Row ``i`` must
        correspond to row ``i`` of ``theta_train``.
    **kwargs
        Estimator-construction and :meth:`MAFPosteriorEstimator.fit` options.

    Returns
    -------
    MAFPosteriorEstimator
        Fitted NumPy-facing estimator.
    """

    theta_train = np.asarray(theta_train, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    estimator_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key
        in {
            "hidden_features",
            "num_transforms",
            "num_blocks",
            "learning_rate",
            "device",
            "validate_device",
            "allow_device_fallback",
            "standardize",
            "max_grad_norm",
            "restore_best",
            "initialization_seed",
        }
    }
    estimator_kwargs.setdefault("initialization_seed", kwargs.get("seed"))
    estimator = MAFPosteriorEstimator(theta_dim=theta_train.shape[1], x_dim=x_train.shape[1], **estimator_kwargs)
    estimator.fit(theta_train, x_train, **kwargs)
    return estimator


def train_mdn_posterior_from_dataset(
    theta: np.ndarray,
    x: np.ndarray,
    *,
    theta_names: list[str] | tuple[str, ...] | None = None,
    x_names: list[str] | tuple[str, ...] | None = None,
    source: str = "precomputed_dataset",
    finite: Literal["raise", "drop"] = "raise",
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
    return_metadata: bool = False,
    **kwargs,
):
    """Train an MDN posterior from documented paired rows ``(theta, x)``."""

    theta_train, x_train, metadata = _prepare_precomputed_training_pairs(
        theta,
        x,
        theta_names=theta_names,
        x_names=x_names,
        source=source,
        finite=finite,
        shuffle=shuffle,
        rng=rng,
    )
    estimator = train_mdn_posterior(theta_train, x_train, **kwargs)
    if return_metadata:
        return estimator, metadata
    return estimator


def train_mdn_posterior(
    theta_train: np.ndarray,
    x_train: np.ndarray,
    **kwargs,
) -> MDNPosteriorEstimator:
    """Fit a conditional Gaussian-mixture posterior from paired NumPy arrays."""

    theta_train = np.asarray(theta_train, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    estimator_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key
        in {
            "n_components",
            "hidden_features",
            "num_blocks",
            "min_scale",
            "learning_rate",
            "device",
            "validate_device",
            "allow_device_fallback",
            "standardize",
            "max_grad_norm",
            "restore_best",
            "initialization_seed",
        }
    }
    estimator_kwargs.setdefault("initialization_seed", kwargs.get("seed"))
    estimator = MDNPosteriorEstimator(
        theta_dim=theta_train.shape[1],
        x_dim=x_train.shape[1],
        **estimator_kwargs,
    )
    estimator.fit(theta_train, x_train, **kwargs)
    return estimator


def sample_posterior(
    estimator: MAFPosteriorEstimator | MDNPosteriorEstimator,
    x_obs: np.ndarray,
    num_samples: int = 10000,
) -> np.ndarray:
    """Draw posterior samples for one or more conditioning rows."""

    return estimator.sample(x_obs, num_samples=num_samples)


def _prepare_flow_for_device(flow, torch, device):
    """Force nflows modules to float32 before moving to accelerators.

    Some nflows distributions/register buffers as float64 depending on the
    process default dtype. Apple MPS does not support float64 tensors, so moving
    the raw flow directly to MPS can fail even though all training arrays are
    float32. Converting on CPU first keeps construction robust across CPU, CUDA,
    and MPS.
    """

    flow = flow.to(dtype=torch.float32)
    return flow.to(device=device)


def _validate_float32_device(torch, device) -> None:
    """Exercise the float32 operations needed before launching a long SBI run."""

    x = torch.randn((4, 3), dtype=torch.float32, device=device, requires_grad=True)
    weight = torch.randn((3, 2), dtype=torch.float32, device=device)
    y = (x @ weight).pow(2).mean()
    y.backward()
    if not bool(torch.isfinite(y.detach()).cpu().item()):
        raise FloatingPointError("float32 smoke test produced a non-finite value.")
    if x.grad is None or not bool(torch.all(torch.isfinite(x.grad)).detach().cpu().item()):
        raise FloatingPointError("float32 smoke test produced non-finite gradients.")
    _synchronize_device(torch, device)


def _synchronize_device(torch, device) -> None:
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _seed_torch(torch, seed: int | None) -> None:
    """Seed torch initialization or posterior draws when requested."""

    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@contextmanager
def _temporary_default_dtype(torch, dtype):
    old_dtype = torch.get_default_dtype()
    if old_dtype == dtype:
        yield
        return
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _prepare_precomputed_training_pairs(
    theta: np.ndarray,
    x: np.ndarray,
    *,
    theta_names,
    x_names,
    source: str,
    finite: Literal["raise", "drop"],
    shuffle: bool,
    rng: np.random.Generator | None,
):
    theta_arr = np.asarray(theta, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if theta_arr.ndim == 1:
        theta_arr = theta_arr[:, None]
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if theta_arr.ndim != 2:
        raise ValueError(f"theta must be a two-dimensional array; got shape {theta_arr.shape}.")
    if x_arr.ndim != 2:
        raise ValueError(f"x must be a two-dimensional array; got shape {x_arr.shape}.")
    if theta_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("theta and x must have the same number of rows.")
    if theta_arr.shape[0] == 0:
        raise ValueError("theta and x must contain at least one paired row.")

    theta_names_tuple = None if theta_names is None else tuple(str(name) for name in theta_names)
    x_names_tuple = None if x_names is None else tuple(str(name) for name in x_names)
    if theta_names_tuple is not None and len(theta_names_tuple) != theta_arr.shape[1]:
        raise ValueError("theta_names length must match theta.shape[1].")
    if x_names_tuple is not None and len(x_names_tuple) != x_arr.shape[1]:
        raise ValueError("x_names length must match x.shape[1].")

    finite_rows = np.all(np.isfinite(theta_arr), axis=1) & np.all(np.isfinite(x_arr), axis=1)
    dropped_nonfinite = int(np.count_nonzero(~finite_rows))
    if dropped_nonfinite:
        if finite == "raise":
            raise ValueError(
                f"Found {dropped_nonfinite} row(s) with NaN or inf in theta or x. "
                "Pass finite='drop' to remove them before training."
            )
        if finite != "drop":
            raise ValueError("finite must be either 'raise' or 'drop'.")
        theta_arr = theta_arr[finite_rows]
        x_arr = x_arr[finite_rows]
        if theta_arr.shape[0] == 0:
            raise ValueError("All paired rows were removed by finite='drop'.")
    elif finite not in {"raise", "drop"}:
        raise ValueError("finite must be either 'raise' or 'drop'.")

    permutation = None
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        permutation = rng.permutation(theta_arr.shape[0])
        theta_arr = theta_arr[permutation]
        x_arr = x_arr[permutation]

    metadata = {
        "source": str(source),
        "n_input": int(np.asarray(theta).shape[0]),
        "n_train": int(theta_arr.shape[0]),
        "theta_dim": int(theta_arr.shape[1]),
        "x_dim": int(x_arr.shape[1]),
        "theta_names": theta_names_tuple,
        "x_names": x_names_tuple,
        "dropped_nonfinite": dropped_nonfinite,
        "shuffled": bool(shuffle),
        "permutation": permutation,
    }
    return theta_arr, x_arr, metadata


def _simulate_one(
    simulator,
    theta: np.ndarray,
    noise_fn,
    rng: np.random.Generator,
    *,
    return_sigma: bool = False,
):
    if return_sigma:
        if hasattr(simulator, "simulate_with_uncertainty"):
            return simulator.simulate_with_uncertainty(theta, noise_fn=noise_fn, rng=rng)
        raise TypeError(
            "return_sigma=True requires a simulator exposing "
            "simulate_with_uncertainty(theta, noise_fn, rng)."
        )
    if hasattr(simulator, "simulate"):
        return simulator.simulate(theta, noise_fn=noise_fn, rng=rng)
    if hasattr(simulator, "rvs"):
        return simulator.rvs(theta, noise_fn=noise_fn, rng=rng)
    return simulator(theta, noise_fn=noise_fn, rng=rng)


_WORKER_SIMULATOR = None
_WORKER_NOISE_FN = None
_WORKER_RETURN_SIGMA = False


def _init_simulation_worker(simulator, noise_fn, return_sigma=False) -> None:
    global _WORKER_SIMULATOR, _WORKER_NOISE_FN, _WORKER_RETURN_SIGMA
    _WORKER_SIMULATOR = simulator
    _WORKER_NOISE_FN = noise_fn
    _WORKER_RETURN_SIGMA = bool(return_sigma)


def _simulate_chunk_from_worker(payload):
    if _WORKER_SIMULATOR is None or _WORKER_NOISE_FN is None:
        raise RuntimeError("Simulation worker was not initialized.")
    theta_chunk, seed = payload
    rng = np.random.default_rng(int(seed))
    good_theta = []
    good_x = []
    good_sigma = []
    failures = []
    for theta in np.asarray(theta_chunk, dtype=float):
        try:
            simulated = _simulate_one(
                _WORKER_SIMULATOR,
                theta,
                _WORKER_NOISE_FN,
                rng,
                return_sigma=_WORKER_RETURN_SIGMA,
            )
            if _WORKER_RETURN_SIGMA:
                x, sigma = simulated
            else:
                x, sigma = simulated, None
            x = np.asarray(x, dtype=float)
            if x.ndim != 1 or not np.all(np.isfinite(x)):
                raise ValueError(f"Simulator returned invalid observation shape/content: shape={x.shape}.")
            if _WORKER_RETURN_SIGMA:
                sigma = np.asarray(sigma, dtype=float)
                if sigma.shape != x.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
                    raise ValueError("Simulator returned invalid uncertainty values.")
        except Exception as exc:
            failures.append(
                {
                    "theta": np.asarray(theta, dtype=float),
                    "error_type": f"{type(exc).__module__}.{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            continue
        good_theta.append(np.asarray(theta, dtype=float))
        good_x.append(x)
        if _WORKER_RETURN_SIGMA:
            good_sigma.append(sigma)
    return good_theta, good_x, good_sigma, failures


def _as_2d(values: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != int(dim):
        raise ValueError(f"{name} must have shape ({dim},) or (n, {dim}); got {arr.shape}.")
    return arr


def _as_context_batch(values: np.ndarray, dim: int) -> np.ndarray:
    return _as_2d(values, dim, "x_obs")
