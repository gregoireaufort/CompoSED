from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from composed.data import SEDDataset, SpectroPhotometricDataset, SpectrumDataset
from composed.likelihood import GaussianPhotometricLikelihood, GaussianSpectralLikelihood
from composed.parameters import ParameterSpace
from composed.results import InferenceResult, normalize_sampling_result


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

        transform_name = None
        if self.parameter_transform is not None:
            transform_name = getattr(self.parameter_transform, "__name__", type(self.parameter_transform).__name__)
        return {
            "backend": f"{type(self.backend).__module__}.{type(self.backend).__name__}",
            "parameters": tuple(self.parameters.names),
            "priors": {name: repr(self.parameters.priors[name]) for name in self.parameters.names},
            "data": type(self.data).__name__,
            "likelihood": repr(self.likelihood),
            "parameter_transform": transform_name,
            "metadata": dict(self.metadata),
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
    x0: Sequence[float] | None = None,
    seed: int | None = None,
) -> InferenceResult:
    """Run one inference method and return a normalized inference result.

    ``sampler=`` remains a compatibility spelling for traditional samplers.
    Problem-driven SBI uses ``method=MAF(...)`` or ``method=Diffusion(...)``
    together with ``training=Simulate(...)``. Existing paired SBI datasets use
    ``train_sbi`` directly and do not declare a scientifically unrelated
    Problem.
    """

    if not isinstance(problem, Problem):
        raise TypeError("fit requires a composed.Problem.")
    if method is not None and sampler is not None:
        raise TypeError("Pass either method= or the compatibility sampler= argument, not both.")
    if method is None:
        method = sampler
    if method is None:
        raise TypeError("fit requires an inference method.")

    from composed.sbi import Diffusion, MAF, Simulate, fit_sbi_problem

    if isinstance(method, (MAF, Diffusion)):
        if x0 is not None:
            raise TypeError("x0 is not used by SBI inference methods.")
        if not isinstance(training, Simulate):
            raise TypeError(
                "Problem-based SBI requires training=Simulate(...). "
                "For pre-existing pairs use train_sbi(SBITrainingSet.from_arrays(...), method)."
            )
        return fit_sbi_problem(problem, method, training, seed=seed)

    if training is not None:
        raise TypeError("training= is only used by Problem-driven SBI methods.")
    sampler_config = Sampler(method) if isinstance(method, str) else method
    if not isinstance(sampler_config, Sampler):
        raise TypeError("sampler must be a composed.Sampler or sampler name.")

    rng = np.random.default_rng(seed)
    _validate_sampler_capabilities(problem.parameters, sampler_config)
    posterior = problem.to_inftools_posterior()
    options = dict(sampler_config.options)
    if x0 is None and sampler_config.name not in {"grid"}:
        x0 = problem.parameters.sample_prior(1, rng=rng)[0]
    x0_arr = None if x0 is None else np.asarray(x0, dtype=float)

    if sampler_config.name == "emcee":
        from inftools.mcmc import run_emcee

        raw = run_emcee(posterior, x0_arr, seed=seed, **options)
    elif sampler_config.name in {"random_walk", "rw_metropolis"}:
        from inftools.mcmc import run_rw_metropolis

        raw = run_rw_metropolis(posterior, x0_arr, rng=rng, **options)
    elif sampler_config.name == "grid":
        from inftools.grid import run_grid_sampler

        raw = run_grid_sampler(posterior, problem.parameters, **options)
    elif sampler_config.name == "mixed_gibbs":
        from inftools.grid import run_mixed_gibbs

        raw = run_mixed_gibbs(posterior, problem.parameters, x0_arr, rng=rng, **options)
    elif sampler_config.name == "mixed_tamis":
        from inftools.mixed_tamis import run_mixed_tamis

        raw = run_mixed_tamis(posterior, problem.parameters, x0=x0_arr, seed=seed, **options)
    elif sampler_config.name == "laplace":
        from inftools.laplace import run_laplace

        _reject_noncontinuous_for_sampler(problem.parameters, "Laplace")
        raw = run_laplace(posterior, x0_arr, **options)
    elif sampler_config.name == "tamis":
        from inftools.core import SamplingResult
        from inftools.tamis_adapter import run_tamis

        _reject_noncontinuous_for_sampler(problem.parameters, "TAMIS")
        tamis = run_tamis(posterior, x0_arr, seed=seed, **options)
        logp = np.asarray([problem.log_posterior(theta) for theta in tamis.samples], dtype=float)
        raw = SamplingResult(
            samples=tamis.samples,
            logp=logp,
            map_estimate=tamis.samples[int(np.nanargmax(logp))],
            cov=tamis.cov,
            meta={**tamis.meta, "weights_norm": tamis.weights},
        )
    elif sampler_config.name == "pocomc":
        from inftools.pocomc_adapter import run_pocomc

        _reject_noncontinuous_for_sampler(problem.parameters, "PocoMC")
        if "prior" not in options and "bounds" not in options:
            options["bounds"] = _uniform_bounds(problem.parameters, "PocoMC")
        options.setdefault("random_state", seed)
        raw = run_pocomc(posterior, **options)
    else:
        raise ValueError(f"Unknown sampler {sampler_config.name!r}.")

    chain = raw.meta.get("raw_chain") if hasattr(raw, "meta") else None
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
