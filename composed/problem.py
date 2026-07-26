from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from composed.data import SEDDataset, SpectroPhotometricDataset, SpectrumDataset
from composed.likelihood import GaussianPhotometricLikelihood, GaussianSpectralLikelihood
from composed.parameters import ParameterSpace
from composed.results import InferenceResult, normalize_sampling_result
from composed.units import MASS_CONVENTION_SCHEMA, backend_mass_reference


@dataclass(frozen=True)
class Gaussian:
    """Diagonal Gaussian likelihood configuration for a :class:`Problem`."""

    photometric_sigma_floor: float | None = None
    spectral_sigma_floor: float | None = None


class _TransformedBackend:
    """Apply the scientist-declared parameter transform before backend calls."""

    def __init__(self, backend, transform: Callable[[Mapping[str, float]], Mapping[str, object]]):
        self.backend = backend
        self.transform = transform
        self.mass_normalization = backend.mass_normalization
        self.mass_reference = getattr(backend, "mass_reference", None)

    def _params(self, params):
        transformed = self.transform(dict(params))
        if not isinstance(transformed, Mapping):
            raise TypeError("Problem parameter_transform must return a parameter mapping.")
        return dict(transformed)

    def predict_photometry(self, params, filters):
        return self.backend.predict_photometry(self._params(params), filters)

    def predict_spectrum(self, params, wavelengths=None, wavelength_range=None, resolution=None):
        return self.backend.predict_spectrum(
            self._params(params),
            wavelengths=wavelengths,
            wavelength_range=wavelength_range,
            resolution=resolution,
        )

    def predict_rest_spectrum(self, params, wavelengths=None, wavelength_range=None):
        return self.backend.predict_rest_spectrum(
            self._params(params),
            wavelengths=wavelengths,
            wavelength_range=wavelength_range,
        )


@dataclass
class Problem:
    """Complete statistical definition of one SED inference problem.

    Parameters remain in the ordered :class:`ParameterSpace`. The optional
    ``parameter_transform`` is the explicit scientific step that turns latent
    values into backend inputs, for example converting ``tau_gyr`` and an age
    fraction into FSPS tabular-SFH arrays.
    """

    backend: object
    parameters: ParameterSpace
    data: SEDDataset | SpectrumDataset | SpectroPhotometricDataset
    likelihood: Gaussian = field(default_factory=Gaussian)
    filters: object | None = None
    parameter_transform: Callable[[Mapping[str, float]], Mapping[str, object]] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    _components: tuple[object, ...] = field(init=False, repr=False)
    _evaluation_backend: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, ParameterSpace):
            raise TypeError("Problem.parameters must be a ParameterSpace.")
        if not isinstance(self.likelihood, Gaussian):
            raise TypeError("Problem.likelihood must currently be composed.Gaussian().")
        self.metadata = dict(self.metadata)
        evaluation_backend = self.backend
        if self.parameter_transform is not None:
            evaluation_backend = _TransformedBackend(self.backend, self.parameter_transform)
        self._evaluation_backend = evaluation_backend
        backend_mass_reference(evaluation_backend)

        if isinstance(self.data, SpectroPhotometricDataset):
            photometry = self.data.photometry
            spectrum = self.data.spectrum
        elif isinstance(self.data, SEDDataset):
            photometry, spectrum = self.data, None
        elif isinstance(self.data, SpectrumDataset):
            photometry, spectrum = None, self.data
        else:
            raise TypeError("Problem.data must be SEDDataset, SpectrumDataset, or SpectroPhotometricDataset.")

        components = []
        if photometry is not None:
            components.append(
                GaussianPhotometricLikelihood(
                    evaluation_backend,
                    photometry,
                    self.parameters,
                    filters=self.filters,
                    sigma_floor=self.likelihood.photometric_sigma_floor,
                )
            )
        if spectrum is not None:
            components.append(
                GaussianSpectralLikelihood(
                    evaluation_backend,
                    spectrum,
                    self.parameters,
                    sigma_floor=self.likelihood.spectral_sigma_floor,
                )
            )
        self._components = tuple(components)

    @property
    def parameter_space(self) -> ParameterSpace:
        """Compatibility name used by existing sampler adapters."""

        return self.parameters

    def log_prior(self, theta: Sequence[float]) -> float:
        return self.parameters.log_prior(np.asarray(theta, dtype=float))

    def log_likelihood(self, theta: Sequence[float]) -> float:
        theta = np.asarray(theta, dtype=float)
        values = [component.log_likelihood(theta) for component in self._components]
        return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf

    def log_posterior(self, theta: Sequence[float]) -> float:
        theta = np.asarray(theta, dtype=float)
        log_prior = self.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf
        log_likelihood = self.log_likelihood(theta)
        return float(log_prior + log_likelihood) if np.isfinite(log_likelihood) else -np.inf

    log_prob = log_posterior

    def simulate(self, theta, noise_fn, rng: np.random.Generator | None = None):
        """Simulate the same active observation vectors consumed by the problem."""

        if len(self._components) == 1:
            return self._components[0].simulate(theta, noise_fn=noise_fn, rng=rng)
        if not isinstance(noise_fn, Mapping):
            raise TypeError("Joint simulation requires noise_fn={'photometry': ..., 'spectrum': ...}.")
        return {
            "photometry": self._components[0].simulate(theta, noise_fn=noise_fn["photometry"], rng=rng),
            "spectrum": self._components[1].simulate(theta, noise_fn=noise_fn["spectrum"], rng=rng),
        }

    def simulate_with_uncertainty(self, theta, noise_fn, rng: np.random.Generator | None = None):
        """Simulate photometry and return ``(active_flux, active_sigma)``.

        The uncertainty is the exact value returned by ``noise_fn`` for the
        same noiseless model and random realization. This first stable SBI
        contract is photometry-only; joint spectrophotometric SBI requires a
        separate explicit context schema.
        """

        if len(self._components) != 1 or not isinstance(self.data, SEDDataset):
            raise NotImplementedError(
                "simulate_with_uncertainty currently supports a photometric SEDDataset Problem only."
            )
        return self._components[0].simulate_with_uncertainty(theta, noise_fn=noise_fn, rng=rng)

    def to_inftools_posterior(self):
        from inftools.core import Posterior

        return Posterior(
            log_prob_fn=self.log_posterior,
            log_likelihood_fn=self.log_likelihood,
            log_prior_fn=self.log_prior,
            dim=self.parameters.ndim,
            theta_names=self.parameters.names,
            extra={"parameter_space": self.parameters, "problem": self},
        )

    def specification(self) -> dict[str, object]:
        """Return a JSON-friendly scientific summary for saved provenance."""

        transform_specification = _callable_specification(self.parameter_transform)
        transform_name = None
        if self.parameter_transform is not None:
            transform_name = getattr(self.parameter_transform, "__name__", type(self.parameter_transform).__name__)
        mass_normalization = getattr(self.backend, "mass_normalization", None)
        mass_reference = backend_mass_reference(self.backend)
        return {
            "backend": f"{type(self.backend).__module__}.{type(self.backend).__name__}",
            "backend_configuration": _backend_configuration(self.backend),
            "mass_normalization": getattr(mass_normalization, "value", mass_normalization),
            "mass_reference": getattr(mass_reference, "value", mass_reference),
            "mass_convention": MASS_CONVENTION_SCHEMA,
            "parameters": tuple(self.parameters.names),
            "priors": {name: repr(self.parameters.priors[name]) for name in self.parameters.names},
            "data": type(self.data).__name__,
            "data_configuration": _dataset_specification(self.data),
            "filters": _filter_specification(self.filters),
            "likelihood": repr(self.likelihood),
            "parameter_transform": transform_name,
            "parameter_transform_configuration": transform_specification,
            "metadata": _stable_value(self.metadata),
        }


@dataclass(frozen=True)
class SamplerCapabilities:
    """Parameter types accepted by one sampler implementation."""

    continuous: bool
    discrete: bool
    fixed: bool
    gradients: bool = False
    simulation: bool = False


_SAMPLER_CAPABILITIES = {
    "emcee": SamplerCapabilities(True, False, True),
    "random_walk": SamplerCapabilities(True, False, True),
    "rw_metropolis": SamplerCapabilities(True, False, True),
    "grid": SamplerCapabilities(False, True, True),
    "mixed_gibbs": SamplerCapabilities(True, True, True),
    "mixed_tamis": SamplerCapabilities(True, True, True),
    "laplace": SamplerCapabilities(True, False, False),
    "tamis": SamplerCapabilities(True, False, False),
    "pocomc": SamplerCapabilities(True, False, False),
}


@dataclass(frozen=True)
class Sampler:
    """Named inference runner plus its explicit options."""

    name: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).lower()
        if name not in _SAMPLER_CAPABILITIES:
            raise ValueError(f"Unknown sampler {name!r}.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "options", dict(self.options))

    @property
    def capabilities(self) -> SamplerCapabilities:
        return _SAMPLER_CAPABILITIES[self.name]


def Emcee(**options) -> Sampler:
    return Sampler("emcee", options)


def RandomWalk(**options) -> Sampler:
    return Sampler("random_walk", options)


def Grid(**options) -> Sampler:
    return Sampler("grid", options)


def MixedGibbs(**options) -> Sampler:
    return Sampler("mixed_gibbs", options)


def MixedTAMIS(**options) -> Sampler:
    return Sampler("mixed_tamis", options)


def Laplace(**options) -> Sampler:
    return Sampler("laplace", options)


def TAMIS(**options) -> Sampler:
    return Sampler("tamis", options)


def PocoMC(**options) -> Sampler:
    return Sampler("pocomc", options)


def fit(
    problem: Problem,
    method: object | str | None = None,
    *,
    sampler: Sampler | str | None = None,
    training: object | None = None,
    conditions: Mapping[str, float] | None = None,
    x0: Sequence[float] | None = None,
    seed: int | None = None,
) -> InferenceResult:
    """Run one inference method and return a normalized inference result.

    ``sampler=`` remains a compatibility spelling for traditional samplers.
    Problem-driven SBI uses ``method=MAF(...)``, ``method=MDN(...)``, or
    ``method=Diffusion(...)`` together with ``training=Simulate(...)``.
    Existing paired SBI datasets use ``train_sbi`` directly and do not declare
    a scientifically unrelated Problem. ``conditions`` fixes named model
    parameters for the observed object. Traditional samplers operate only on
    the remaining free axes; neural SBI appends the same named values to its
    observation context.
    """

    if not isinstance(problem, Problem):
        raise TypeError("fit requires a composed.Problem.")
    if method is not None and sampler is not None:
        raise TypeError("Pass either method= or the compatibility sampler= argument, not both.")
    if method is None:
        method = sampler
    if method is None:
        raise TypeError("fit requires an inference method.")

    from composed.sbi import Diffusion, MAF, MDN, Simulate, fit_sbi_problem

    if isinstance(method, (MAF, MDN, Diffusion)):
        if x0 is not None:
            raise TypeError("x0 is not used by SBI inference methods.")
        if not isinstance(training, Simulate):
            raise TypeError(
                "Problem-based SBI requires training=Simulate(...). "
                "For pre-existing pairs use train_sbi(SBITrainingSet.from_arrays(...), method)."
            )
        return fit_sbi_problem(
            problem,
            method,
            training,
            conditions=conditions,
            seed=seed,
        )

    if training is not None:
        raise TypeError("training= is only used by Problem-driven SBI methods.")
    sampler_config = Sampler(method) if isinstance(method, str) else method
    if not isinstance(sampler_config, Sampler):
        raise TypeError("sampler must be a composed.Sampler or sampler name.")

    rng = np.random.default_rng(seed)
    conditioning = _build_conditioning_reduction(problem.parameters, conditions)
    inference_space = (
        problem.parameters if conditioning is None else conditioning.free_space
    )
    if inference_space.ndim == 0:
        return _fully_conditioned_result(
            problem,
            sampler_config,
            conditioning,
            seed=seed,
        )
    _validate_sampler_capabilities(inference_space, sampler_config)
    posterior = (
        problem.to_inftools_posterior()
        if conditioning is None
        else _conditioned_posterior(problem, conditioning)
    )
    options = dict(sampler_config.options)
    if x0 is None and sampler_config.name not in {"grid"}:
        x0 = inference_space.sample_prior(1, rng=rng)[0]
    x0_arr = None if x0 is None else np.asarray(x0, dtype=float)
    if conditioning is not None and x0_arr is not None:
        x0_arr = conditioning.reduce_initial_position(x0_arr)

    if sampler_config.name == "emcee":
        from inftools.mcmc import run_emcee

        raw = run_emcee(posterior, x0_arr, seed=seed, **options)
    elif sampler_config.name in {"random_walk", "rw_metropolis"}:
        from inftools.mcmc import run_rw_metropolis

        raw = run_rw_metropolis(posterior, x0_arr, rng=rng, **options)
    elif sampler_config.name == "grid":
        from inftools.grid import run_grid_sampler

        raw = run_grid_sampler(posterior, inference_space, **options)
    elif sampler_config.name == "mixed_gibbs":
        from inftools.grid import run_mixed_gibbs

        raw = run_mixed_gibbs(posterior, inference_space, x0_arr, rng=rng, **options)
    elif sampler_config.name == "mixed_tamis":
        from inftools.mixed_tamis import run_mixed_tamis

        raw = run_mixed_tamis(posterior, inference_space, x0=x0_arr, seed=seed, **options)
    elif sampler_config.name == "laplace":
        from inftools.laplace import run_laplace

        _reject_noncontinuous_for_sampler(inference_space, "Laplace")
        raw = run_laplace(posterior, x0_arr, **options)
    elif sampler_config.name == "tamis":
        from inftools.core import SamplingResult
        from inftools.tamis_adapter import run_tamis

        _reject_noncontinuous_for_sampler(inference_space, "TAMIS")
        tamis = run_tamis(posterior, x0_arr, seed=seed, **options)
        logp = np.asarray(
            [posterior.log_prob_fn(theta) for theta in tamis.samples],
            dtype=float,
        )
        raw = SamplingResult(
            samples=tamis.samples,
            logp=logp,
            map_estimate=tamis.samples[int(np.nanargmax(logp))],
            cov=tamis.cov,
            meta={**tamis.meta, "weights_norm": tamis.weights},
        )
    elif sampler_config.name == "pocomc":
        from inftools.pocomc_adapter import pocomc_prior_from_parameter_space, run_pocomc

        _reject_noncontinuous_for_sampler(inference_space, "PocoMC")
        if "prior" not in options and "bounds" not in options:
            options["prior"] = pocomc_prior_from_parameter_space(inference_space)
        options.setdefault("random_state", seed)
        raw = run_pocomc(posterior, **options)
    else:
        raise ValueError(f"Unknown sampler {sampler_config.name!r}.")

    if conditioning is not None:
        raw = _expand_conditioned_sampling_result(raw, conditioning)
    chain = raw.meta.get("raw_chain") if hasattr(raw, "meta") else None
    condition_metadata = (
        {}
        if conditioning is None
        else {
            "conditions": dict(conditioning.conditions),
            "conditioned_parameter_names": conditioning.conditioned_names,
            "free_parameter_names": conditioning.free_space.names,
        }
    )
    return normalize_sampling_result(
        raw,
        problem.parameters,
        sampler_name=sampler_config.name,
        chain=chain,
        metadata={
            "problem": problem.specification(),
            "sampler_options": options,
            "sampler_capabilities": sampler_config.capabilities.__dict__,
            "seed": seed,
            **condition_metadata,
        },
    )


@dataclass(frozen=True)
class _ConditioningReduction:
    """Map between a sampler's free vector and the full model vector."""

    full_space: ParameterSpace
    free_space: ParameterSpace
    conditions: Mapping[str, float]
    free_indices: tuple[int, ...]
    conditioned_indices: tuple[int, ...]

    @property
    def conditioned_names(self) -> tuple[str, ...]:
        return tuple(self.full_space.names[index] for index in self.conditioned_indices)

    def expand(self, theta_free: Sequence[float]) -> np.ndarray:
        theta_free = np.asarray(theta_free, dtype=float)
        if theta_free.shape != (self.free_space.ndim,):
            raise ValueError(
                f"Expected free theta shape {(self.free_space.ndim,)}, got {theta_free.shape}."
            )
        theta_full = np.empty(self.full_space.ndim, dtype=float)
        theta_full[np.asarray(self.free_indices, dtype=int)] = theta_free
        for index in self.conditioned_indices:
            theta_full[index] = self.conditions[self.full_space.names[index]]
        return theta_full

    def expand_rows(self, theta_free: np.ndarray) -> np.ndarray:
        theta_free = np.asarray(theta_free, dtype=float)
        if theta_free.ndim != 2 or theta_free.shape[1] != self.free_space.ndim:
            raise ValueError(
                "Free samples must have shape "
                f"(n_sample, {self.free_space.ndim}), got {theta_free.shape}."
            )
        rows = np.empty((theta_free.shape[0], self.full_space.ndim), dtype=float)
        rows[:, np.asarray(self.free_indices, dtype=int)] = theta_free
        for index in self.conditioned_indices:
            rows[:, index] = self.conditions[self.full_space.names[index]]
        return rows

    def reduce_initial_position(self, x0: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape == (self.free_space.ndim,):
            return x0
        if x0.shape != (self.full_space.ndim,):
            raise ValueError(
                "x0 must follow either the free or full parameter order; got "
                f"shape {x0.shape}, expected {(self.free_space.ndim,)} or "
                f"{(self.full_space.ndim,)}."
            )
        for index in self.conditioned_indices:
            name = self.full_space.names[index]
            if not np.isclose(
                x0[index],
                self.conditions[name],
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"x0 value for conditioned parameter {name!r} does not match "
                    f"conditions[{name!r}]."
                )
        return x0[np.asarray(self.free_indices, dtype=int)]


def _build_conditioning_reduction(
    parameter_space: ParameterSpace,
    conditions: Mapping[str, float] | None,
) -> _ConditioningReduction | None:
    """Validate point conditions and construct a deterministic reduced space."""

    if conditions is None:
        return None
    if not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping from parameter names to values.")
    if not conditions:
        return None
    unknown = sorted(set(conditions) - set(parameter_space.names))
    if unknown:
        raise ValueError("Unknown conditioned parameter(s): " + ", ".join(unknown))

    canonical_conditions: dict[str, float] = {}
    for name in parameter_space.names:
        if name not in conditions:
            continue
        value = float(conditions[name])
        if not np.isfinite(value):
            raise ValueError(f"Conditioned parameter {name!r} must be finite.")
        if not np.isfinite(parameter_space.priors[name].logpdf(value)):
            raise ValueError(
                f"Conditioned value {value:.8g} for {name!r} lies outside its declared prior support."
            )
        canonical_conditions[name] = value

    free_names = tuple(
        name for name in parameter_space.names if name not in canonical_conditions
    )
    free_space = ParameterSpace(
        names=free_names,
        priors={name: parameter_space.priors[name] for name in free_names},
    )
    free_indices = tuple(
        index
        for index, name in enumerate(parameter_space.names)
        if name not in canonical_conditions
    )
    conditioned_indices = tuple(
        index
        for index, name in enumerate(parameter_space.names)
        if name in canonical_conditions
    )
    return _ConditioningReduction(
        full_space=parameter_space,
        free_space=free_space,
        conditions=canonical_conditions,
        free_indices=free_indices,
        conditioned_indices=conditioned_indices,
    )


def _conditioned_posterior(problem: Problem, reduction: _ConditioningReduction):
    """Return a posterior over free axes with full-vector model evaluation."""

    from inftools.core import Posterior

    def log_prior(theta_free):
        return reduction.free_space.log_prior(theta_free)

    def log_likelihood(theta_free):
        return problem.log_likelihood(reduction.expand(theta_free))

    def log_posterior(theta_free):
        prior = log_prior(theta_free)
        if not np.isfinite(prior):
            return -np.inf
        likelihood = log_likelihood(theta_free)
        return float(prior + likelihood) if np.isfinite(likelihood) else -np.inf

    return Posterior(
        log_prob_fn=log_posterior,
        log_likelihood_fn=log_likelihood,
        log_prior_fn=log_prior,
        dim=reduction.free_space.ndim,
        theta_names=reduction.free_space.names,
        extra={
            "parameter_space": reduction.free_space,
            "problem": problem,
            "conditions": dict(reduction.conditions),
            "full_parameter_space": reduction.full_space,
        },
    )


def _expand_conditioned_sampling_result(raw, reduction: _ConditioningReduction):
    """Restore conditioned columns in a sampler result and its raw chain."""

    from inftools.core import SamplingResult

    samples = reduction.expand_rows(raw.samples)
    map_estimate = (
        None
        if raw.map_estimate is None
        else reduction.expand(np.asarray(raw.map_estimate, dtype=float))
    )
    cov = None
    if raw.cov is not None:
        free_cov = np.asarray(raw.cov, dtype=float)
        if free_cov.shape == (reduction.free_space.ndim, reduction.free_space.ndim):
            cov = np.zeros(
                (reduction.full_space.ndim, reduction.full_space.ndim),
                dtype=float,
            )
            free = np.asarray(reduction.free_indices, dtype=int)
            cov[np.ix_(free, free)] = free_cov
        else:
            cov = free_cov

    meta = dict(raw.meta)
    raw_chain = meta.get("raw_chain")
    if raw_chain is not None:
        chain = np.asarray(raw_chain, dtype=float)
        if chain.shape[-1] == reduction.free_space.ndim:
            flat = chain.reshape(-1, reduction.free_space.ndim)
            meta["raw_chain"] = reduction.expand_rows(flat).reshape(
                *chain.shape[:-1],
                reduction.full_space.ndim,
            )
    meta["conditions"] = dict(reduction.conditions)
    meta["free_parameter_names"] = reduction.free_space.names
    return SamplingResult(
        samples=samples,
        logp=np.asarray(raw.logp, dtype=float),
        map_estimate=map_estimate,
        cov=cov,
        meta=meta,
    )


def _fully_conditioned_result(
    problem: Problem,
    sampler: Sampler,
    reduction: _ConditioningReduction,
    *,
    seed: int | None,
) -> InferenceResult:
    """Return the deterministic posterior when every model parameter is fixed."""

    theta = reduction.expand(np.empty(0, dtype=float))
    log_likelihood = problem.log_likelihood(theta)
    if not np.isfinite(log_likelihood):
        raise ValueError("The fully conditioned model has non-finite likelihood.")
    return InferenceResult(
        samples=theta[None, :],
        logp=np.asarray([log_likelihood], dtype=float),
        weights=np.ones(1, dtype=float),
        parameter_names=problem.parameters.names,
        sampler_name=sampler.name,
        metadata={
            "problem": problem.specification(),
            "sampler_options": dict(sampler.options),
            "sampler_capabilities": sampler.capabilities.__dict__,
            "seed": seed,
            "conditions": dict(reduction.conditions),
            "conditioned_parameter_names": reduction.conditioned_names,
            "free_parameter_names": (),
        },
    )


def _reject_noncontinuous_for_sampler(parameter_space: ParameterSpace, sampler_name: str) -> None:
    unsupported = [
        name
        for name in parameter_space.names
        if type(parameter_space.priors[name]).__name__ in {"DeltaPrior", "ChoicePrior", "IntegerUniformPrior"}
    ]
    if unsupported:
        raise ValueError(
            f"{sampler_name} facade currently requires continuous parameters; unsupported: {', '.join(unsupported)}."
        )


def _validate_sampler_capabilities(parameter_space: ParameterSpace, sampler: Sampler) -> None:
    kinds = {"continuous": [], "discrete": [], "fixed": []}
    for name in parameter_space.names:
        prior_name = type(parameter_space.priors[name]).__name__
        if prior_name == "DeltaPrior":
            kinds["fixed"].append(name)
        elif prior_name in {"ChoicePrior", "IntegerUniformPrior"}:
            kinds["discrete"].append(name)
        else:
            kinds["continuous"].append(name)
    unsupported = [kind for kind, names in kinds.items() if names and not getattr(sampler.capabilities, kind)]
    if unsupported:
        details = "; ".join(f"{kind}: {', '.join(kinds[kind])}" for kind in unsupported)
        raise ValueError(f"Sampler {sampler.name!r} does not support this ParameterSpace ({details}).")


def _uniform_bounds(parameter_space: ParameterSpace, sampler_name: str):
    bounds = []
    for name in parameter_space.names:
        prior = parameter_space.priors[name]
        if type(prior).__name__ != "UniformPrior":
            raise ValueError(
                f"{sampler_name} needs an explicit sampler prior for non-uniform parameter {name!r}."
            )
        bounds.append((float(prior.low), float(prior.high)))
    return bounds


def _array_specification(values: object) -> dict[str, object]:
    """Describe an array by shape, dtype, and content hash."""

    array = np.ascontiguousarray(np.asarray(values))
    if array.dtype.hasobject:
        payload = repr(_stable_value(array.tolist())).encode("utf-8")
    else:
        payload = array.tobytes(order="C")
    digest = sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(array.shape)).encode("utf-8"))
    digest.update(payload)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


def _dataset_specification(data: object) -> dict[str, object]:
    """Record observed arrays and masks without embedding them in metadata."""

    if isinstance(data, SEDDataset):
        return {
            "type": "SEDDataset",
            "band_names": list(data.band_names),
            "flux_unit": data.flux_unit,
            "flux": _array_specification(data.flux),
            "sigma": _array_specification(data.sigma),
            "mask": None if data.mask is None else _array_specification(data.mask),
            "upper_limit": _array_specification(data.upper_limit),
            "upper_limit_mask": _array_specification(data.upper_limit_mask),
            "metadata": _stable_value(data.metadata),
        }
    if isinstance(data, SpectrumDataset):
        return {
            "type": "SpectrumDataset",
            "wavelength_unit": data.wavelength_unit,
            "flux_unit": data.flux_unit,
            "wavelength": _array_specification(data.wavelength),
            "flux": _array_specification(data.flux),
            "sigma": _array_specification(data.sigma),
            "mask": None if data.mask is None else _array_specification(data.mask),
            "metadata": _stable_value(data.metadata),
        }
    if isinstance(data, SpectroPhotometricDataset):
        return {
            "type": "SpectroPhotometricDataset",
            "photometry": None if data.photometry is None else _dataset_specification(data.photometry),
            "spectrum": None if data.spectrum is None else _dataset_specification(data.spectrum),
            "metadata": _stable_value(data.metadata),
        }
    raise TypeError(f"Unsupported Problem data type {type(data).__name__!r}.")


def _filter_specification(filters: object | None) -> dict[str, object] | None:
    """Record filter order and transmission curves when available."""

    if filters is None:
        return None
    filter_objects = getattr(filters, "filters", filters)
    if isinstance(filter_objects, (str, bytes)) or not isinstance(filter_objects, Sequence):
        return {"value": _stable_value(filters)}
    filter_objects = tuple(filter_objects)
    declared_names = getattr(filters, "names", None)
    if declared_names is None:
        declared_names = tuple(
            item if isinstance(item, str) else getattr(item, "name", str(index))
            for index, item in enumerate(filter_objects)
        )

    curves = []
    for name, item in zip(declared_names, filter_objects):
        entry: dict[str, object] = {"name": str(name)}
        if not isinstance(item, str):
            wavelength = getattr(item, "wavelength", getattr(item, "wave", None))
            transmission = getattr(item, "transmission", None)
            if wavelength is not None:
                entry["wavelength"] = _array_specification(wavelength)
            if transmission is not None:
                entry["transmission"] = _array_specification(transmission)
        curves.append(entry)
    return {"names": [str(name) for name in declared_names], "curves": curves}


def _backend_configuration(backend: object) -> dict[str, object]:
    """Extract constructor-level backend configuration, excluding live state."""

    type_name = f"{type(backend).__module__}.{type(backend).__name__}"
    if is_dataclass(backend):
        configuration = {
            item.name: _stable_value(getattr(backend, item.name))
            for item in fields(backend)
            if item.init and not item.name.startswith("_")
        }
    else:
        backend_state = getattr(backend, "__dict__", {})
        configuration = {
            name: _stable_value(value)
            for name, value in backend_state.items()
            if not name.startswith("_")
        }
    return {"type": type_name, "configuration": configuration}


def _callable_specification(function: object | None) -> dict[str, object] | None:
    """Identify a parameter transform and hash its Python operations."""

    if function is None:
        return None
    name = f"{getattr(function, '__module__', type(function).__module__)}."
    name += getattr(function, "__qualname__", type(function).__qualname__)
    code = getattr(function, "__code__", None)
    if code is None:
        return {"name": name, "configuration": _stable_value(function)}

    digest = sha256()
    digest.update(code.co_code)
    digest.update(repr(code.co_consts).encode("utf-8"))
    digest.update(repr(code.co_names).encode("utf-8"))
    digest.update(repr(code.co_varnames).encode("utf-8"))
    digest.update(repr(getattr(function, "__defaults__", None)).encode("utf-8"))
    closure = getattr(function, "__closure__", None)
    if closure:
        digest.update(repr([_stable_value(cell.cell_contents) for cell in closure]).encode("utf-8"))
    return {"name": name, "code_sha256": digest.hexdigest()}


def _stable_value(value: object) -> object:
    """Convert scientific configuration values to deterministic JSON data."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _array_specification(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_value(item) for item in value), key=repr)
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "configuration": {
                item.name: _stable_value(getattr(value, item.name))
                for item in fields(value)
                if item.init and not item.name.startswith("_")
            },
        }
    if callable(value):
        return _callable_specification(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    representation = repr(value)
    if " at 0x" in representation:
        representation = f"<{type(value).__module__}.{type(value).__name__}>"
    return {
        "type": f"{type(value).__module__}.{type(value).__name__}",
        "repr": representation,
    }


__all__ = [
    "Emcee",
    "Gaussian",
    "Grid",
    "Laplace",
    "MixedGibbs",
    "MixedTAMIS",
    "PocoMC",
    "Problem",
    "RandomWalk",
    "Sampler",
    "SamplerCapabilities",
    "TAMIS",
    "fit",
]
