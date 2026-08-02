from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from inftools.core import Posterior
from inftools.grid import run_grid_sampler
from composed.catalog import (
    build_photometric_model_grid,
    evaluate_catalog_model_grid_likelihood,
    load_photometric_model_grid,
    run_photometric_grid_catalog,
)
from composed.data import SEDDataset
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.errors import ModelDomainError
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, NormalPrior, UniformPrior
from composed.units import MassNormalization, MassReference
from composed.provenance import provenance_path_for


@dataclass
class TemplateBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        template = int(round(float(params["template"])))
        fluxes = {
            0: np.asarray([1.0, 2.0, 3.0]),
            1: np.asarray([2.0, 1.0, 3.5]),
        }
        return ModelPhotometry(band_names=("u", "g", "r"), flux=fluxes[template])


@dataclass
class PerMassBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    mass_reference: MassReference = MassReference.SURVIVING_STELLAR_MASS

    def predict_photometry(self, params, filters):
        del params, filters
        return ModelPhotometry(band_names=("u", "g"), flux=np.asarray([1.0, 2.0]))


@dataclass
class PerMassTemplateBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    mass_reference: MassReference = MassReference.SURVIVING_STELLAR_MASS

    def predict_photometry(self, params, filters):
        del filters
        template = int(round(float(params["template"])))
        fluxes = {
            0: np.asarray([1.0, 2.0]),
            1: np.asarray([2.0, 1.0]),
        }
        return ModelPhotometry(band_names=("u", "g"), flux=fluxes[template])


@dataclass
class ZeroPerMassBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    mass_reference: MassReference = MassReference.SURVIVING_STELLAR_MASS

    def predict_photometry(self, params, filters):
        del params, filters
        return ModelPhotometry(band_names=("g",), flux=np.asarray([0.0]))


@dataclass
class OneBandBackend(SEDBackend):
    flux: float
    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del params, filters
        return ModelPhotometry(band_names=("fuv",), flux=np.asarray([self.flux], dtype=float))


@dataclass
class TwoBandBackend(SEDBackend):
    flux: tuple[float, float]
    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del params, filters
        return ModelPhotometry(band_names=("g", "fuv"), flux=np.asarray(self.flux, dtype=float))


@dataclass
class PartiallyInvalidBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        template = int(round(float(params["template"])))
        fluxes = {
            0: np.asarray([1.0, np.nan]),
            1: np.asarray([2.0, 3.0]),
        }
        return ModelPhotometry(("g", "r"), fluxes[template])


@dataclass
class PartiallyInvalidPerMassBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    mass_reference: MassReference = MassReference.SURVIVING_STELLAR_MASS

    def predict_photometry(self, params, filters):
        del filters
        template = int(round(float(params["template"])))
        fluxes = {
            0: np.asarray([1.0, np.nan]),
            1: np.asarray([2.0, 3.0]),
        }
        return ModelPhotometry(("g", "r"), fluxes[template])


@dataclass
class DomainRejectingBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        template = int(round(float(params["template"])))
        if template == 0:
            raise ModelDomainError("template is outside the toy physical domain")
        return ModelPhotometry(("g",), np.asarray([1.0]))


@dataclass
class DomainRejectingPerMassBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    mass_reference: MassReference = MassReference.SURVIVING_STELLAR_MASS

    def predict_photometry(self, params, filters):
        del filters
        template = int(round(float(params["template"])))
        if template == 0:
            raise ModelDomainError("template is outside the toy physical domain")
        return ModelPhotometry(("g",), np.asarray([1.0]))


def test_photometric_grid_catalog_matches_single_object_grid_likelihoods():
    backend = TemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0, 1.0])})
    datasets = [
        SEDDataset(
            band_names=("u", "g", "r"),
            flux=np.asarray([1.0, 100.0, 3.0]),
            sigma=np.asarray([0.1, 0.1, 0.1]),
            mask=np.asarray([True, False, True]),
        ),
        SEDDataset(
            band_names=("u", "g", "r"),
            flux=np.asarray([2.0, 1.0, 3.5]),
            sigma=np.asarray([0.1, 0.1, 0.1]),
        ),
    ]

    catalog = run_photometric_grid_catalog(
        backend,
        datasets,
        space,
        filters=("u", "g", "r"),
        model_chunk_size=1,
        object_chunk_size=1,
    )

    assert catalog.logp.shape == (2, 2)
    assert catalog.samples.shape == (2, 1)
    assert np.allclose(catalog.map_estimates[:, 0], [0.0, 1.0])
    assert np.allclose(np.sum(catalog.weights_norm, axis=1), 1.0)

    for i, dataset in enumerate(datasets):
        likelihood = GaussianPhotometricLikelihood(backend, dataset, space, filters=("u", "g", "r"))
        posterior = Posterior(likelihood.log_prob, dim=space.ndim, theta_names=space.names)
        single = run_grid_sampler(posterior, space)
        assert np.allclose(catalog.samples, single.samples)
        assert np.allclose(catalog.logp[i], single.logp)
        assert np.allclose(catalog.weights_norm[i], single.meta["weights_norm"])


def test_photometric_grid_catalog_applies_per_solar_mass_scaling_once():
    backend = PerMassBackend()
    space = ParameterSpace(
        names=("log10_mass",),
        priors={"log10_mass": ChoicePrior([0.0, 1.0])},
    )
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([10.0, 20.0]),
        sigma=np.asarray([0.1, 0.1]),
    )

    catalog = run_photometric_grid_catalog(backend, [dataset], space, filters=("u", "g"))

    assert catalog.map_estimates.shape == (1, 1)
    assert catalog.map_estimates[0, 0] == 1.0
    assert catalog.logp[0, 1] > catalog.logp[0, 0]


def test_photometric_grid_catalog_handles_one_band_upper_limit_like_scalar_likelihood():
    backend = OneBandBackend(flux=1.0)
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(
        band_names=("fuv",),
        flux=np.asarray([np.nan]),
        sigma=np.asarray([0.2]),
        upper_limit=np.asarray([1.0]),
        upper_limit_mask=np.asarray([True]),
    )

    catalog = run_photometric_grid_catalog(backend, [dataset], space)
    scalar = GaussianPhotometricLikelihood(backend, dataset, space).log_prob([0.0])

    assert np.isclose(scalar, np.log(0.5))
    assert np.allclose(catalog.logp[0], [scalar])
    assert np.allclose(catalog.weights_norm[0], [1.0])


def test_photometric_grid_catalog_handles_mixed_detection_and_upper_limit_like_scalar_likelihood():
    backend = TwoBandBackend(flux=(2.0, 1.0))
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(
        band_names=("g", "fuv"),
        flux=np.asarray([2.0, np.nan]),
        sigma=np.asarray([0.1, 0.5]),
        upper_limit=np.asarray([0.0, 1.0]),
        upper_limit_mask=np.asarray([False, True]),
    )

    catalog = run_photometric_grid_catalog(backend, [dataset], space, model_chunk_size=1, object_chunk_size=1)
    scalar = GaussianPhotometricLikelihood(backend, dataset, space).log_prob([0.0])

    assert np.allclose(catalog.logp[0], [scalar])
    assert np.all(np.isfinite(catalog.logp[0]))


def test_catalog_grid_model_discrepancy_matches_scalar_with_detection_and_upper_limit():
    backend = TwoBandBackend(flux=(1.4, 1.2))
    space = ParameterSpace(
        names=("template",),
        priors={"template": ChoicePrior([0.0, 1.0])},
    )
    dataset = SEDDataset(
        band_names=("g", "fuv"),
        flux=np.asarray([1.8, np.nan]),
        sigma=np.asarray([0.2, 0.3]),
        upper_limit=np.asarray([np.nan, 1.0]),
        upper_limit_mask=np.asarray([False, True]),
    )
    eta = 0.25

    catalog = run_photometric_grid_catalog(
        backend,
        [dataset],
        space,
        filters=("g", "fuv"),
        model_discrepancy=eta,
    )
    scalar = GaussianPhotometricLikelihood(
        backend,
        dataset,
        space,
        filters=("g", "fuv"),
        model_discrepancy=eta,
    )
    expected = np.asarray(
        [scalar.log_posterior(theta) for theta in catalog.samples],
        dtype=float,
    )
    assert np.allclose(catalog.logp[0], expected)


def test_catalog_model_finiteness_is_checked_only_in_each_objects_active_bands():
    backend = PartiallyInvalidBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0, 1.0])})
    masked = SEDDataset(
        band_names=("g", "r"),
        flux=np.asarray([1.0, 999.0]),
        sigma=np.asarray([0.2, 0.2]),
        mask=np.asarray([True, False]),
    )
    unmasked = SEDDataset(
        band_names=("g", "r"),
        flux=np.asarray([2.0, 3.0]),
        sigma=np.asarray([0.2, 0.2]),
    )

    catalog = run_photometric_grid_catalog(
        backend,
        [masked, unmasked],
        space,
        filters=("g", "r"),
        model_chunk_size=1,
        object_chunk_size=1,
    )

    for object_index, dataset in enumerate((masked, unmasked)):
        scalar = GaussianPhotometricLikelihood(backend, dataset, space, filters=("g", "r"))
        expected = np.asarray([scalar.log_prob([template]) for template in (0.0, 1.0)])
        assert np.allclose(catalog.logp[object_index], expected)
    assert np.isfinite(catalog.logp[0, 0])
    assert np.isneginf(catalog.logp[1, 0])


def test_cached_mass_grid_uses_the_same_per_object_finite_band_rule():
    backend = PartiallyInvalidPerMassBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0, 1.0])})
    masked = SEDDataset(
        band_names=("g", "r"),
        flux=np.asarray([1.0, 999.0]),
        sigma=np.asarray([0.2, 0.2]),
        mask=np.asarray([True, False]),
    )
    unmasked = SEDDataset(
        band_names=("g", "r"),
        flux=np.asarray([2.0, 3.0]),
        sigma=np.asarray([0.2, 0.2]),
    )
    grid = build_photometric_model_grid(
        backend,
        space,
        filters=("g", "r"),
        band_names=("g", "r"),
    )

    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [masked, unmasked],
        log10_mass_grid=np.linspace(-1.0, 1.0, 41),
        log10_mass_prior=UniformPrior(-1.0, 1.0),
        model_chunk_size=1,
        object_chunk_size=1,
        mass_chunk_size=7,
    )

    assert np.array_equal(grid.valid, [True, True])
    assert np.isfinite(result.profile_logp[0, 0])
    assert np.isfinite(result.marginal_logp[0, 0])
    assert np.isneginf(result.profile_logp[1, 0])
    assert np.isneginf(result.marginal_logp[1, 0])
    assert np.isnan(result.log10_mass_profile[1, 0])


def test_model_domain_rejection_has_zero_weight_in_scalar_direct_and_cached_grids():
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0, 1.0])})
    dataset = SEDDataset(("g",), np.asarray([1.0]), np.asarray([0.2]))

    direct_backend = DomainRejectingBackend()
    scalar = GaussianPhotometricLikelihood(direct_backend, dataset, space, filters=("g",))
    direct = run_photometric_grid_catalog(
        direct_backend,
        [dataset],
        space,
        filters=("g",),
        model_chunk_size=1,
        object_chunk_size=1,
    )
    assert np.isneginf(scalar.log_prob([0.0]))
    assert np.isneginf(direct.logp[0, 0])
    assert np.isfinite(direct.logp[0, 1])
    assert direct.weights_norm[0, 0] == 0.0

    cached_backend = DomainRejectingPerMassBackend()
    grid = build_photometric_model_grid(
        cached_backend,
        space,
        filters=("g",),
        band_names=("g",),
    )
    cached = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        log10_mass_bounds=(-1.0, 1.0),
    )
    assert np.array_equal(grid.valid, [False, True])
    assert np.isneginf(cached.profile_logp[0, 0])
    assert np.isfinite(cached.profile_logp[0, 1])


def test_photometric_grid_catalog_rejects_mismatched_band_order():
    backend = TemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    datasets = [
        SEDDataset(("u", "g", "r"), np.ones(3), np.ones(3)),
        SEDDataset(("g", "u", "r"), np.ones(3), np.ones(3)),
    ]

    try:
        run_photometric_grid_catalog(backend, datasets, space, filters=("u", "g", "r"))
    except ValueError as exc:
        assert "same band order" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected mismatched band order to raise")


def test_build_model_grid_excludes_mass_and_profiles_mass_for_catalog():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(0.0, 3.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([20.0, 10.0]),
        sigma=np.asarray([0.1, 0.1]),
    )

    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))
    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        log10_mass_bounds=(0.0, 3.0),
        log10_mass_grid=np.linspace(0.0, 3.0, 301),
        log10_mass_prior=space.priors["log10_mass"],
        model_chunk_size=1,
        object_chunk_size=1,
        mass_chunk_size=64,
    )

    assert grid.parameter_names == ("template",)
    assert grid.samples.shape == (2, 1)
    assert np.allclose(grid.flux, [[1.0, 2.0], [2.0, 1.0]])
    assert result.profile_map_estimates[0, 0] == 1.0
    assert np.isclose(result.log10_mass_profile[0, result.profile_map_indices[0]], 1.0)
    assert result.marginal_map_estimates[0, 0] == 1.0
    assert np.isclose(result.log10_mass_quantiles[0, 1], 1.0, atol=0.02)


def test_cached_grid_uses_declared_uniform_mass_prior_as_profile_bounds():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(0.0, 1.0),
            "template": ChoicePrior([0.0]),
        },
    )
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([100.0, 200.0]),
        sigma=np.asarray([0.1, 0.1]),
    )

    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))
    result = evaluate_catalog_model_grid_likelihood(grid, [dataset])

    assert result.meta["log10_mass_bounds"] == (0.0, 1.0)
    assert result.meta["mass_prior"]["type"] == "UniformPrior"
    assert grid.meta["excluded_parameter_priors"]["log10_mass"]["type"].endswith("UniformPrior")
    assert np.allclose(result.log10_mass_profile, 1.0)
    assert np.all(result.mass_profile_at_boundary)


def test_cached_grid_uses_declared_mass_prior_for_numerical_marginalization():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(0.0, 2.0),
            "template": ChoicePrior([0.0]),
        },
    )
    dataset = SEDDataset(("u", "g"), np.asarray([10.0, 20.0]), np.asarray([0.1, 0.1]))
    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))

    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        log10_mass_grid=np.linspace(0.0, 2.0, 41),
    )

    assert result.marginal_logp is not None
    assert result.meta["mass_prior"]["type"] == "UniformPrior"
    assert result.meta["mass_prior"]["integration_bounds"] == (0.0, 2.0)


def test_cached_grid_refuses_to_ignore_declared_informative_mass_prior():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": NormalPrior(9.0, 1.0),
            "template": ChoicePrior([0.0]),
        },
    )
    dataset = SEDDataset(("u", "g"), np.asarray([1.0, 2.0]), np.asarray([0.1, 0.1]))
    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))

    with pytest.raises(ValueError, match="Analytic cached-grid mass profiling"):
        evaluate_catalog_model_grid_likelihood(grid, [dataset])


def test_cached_grid_rejects_profile_bounds_that_disagree_with_declared_prior():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(8.0, 11.0),
            "template": ChoicePrior([0.0]),
        },
    )
    dataset = SEDDataset(("u", "g"), np.asarray([1.0, 2.0]), np.asarray([0.1, 0.1]))
    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))

    with pytest.raises(ValueError, match="do not match the UniformPrior"):
        evaluate_catalog_model_grid_likelihood(
            grid,
            [dataset],
            log10_mass_bounds=(7.0, 12.0),
        )


def test_unbounded_analytic_mass_profile_rejects_nonpositive_amplitude():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([-1.0, -2.0]),
        sigma=np.asarray([0.1, 0.1]),
    )
    grid = build_photometric_model_grid(
        backend,
        space,
        filters=("u", "g"),
        band_names=("u", "g"),
    )

    with pytest.raises(RuntimeError, match="No finite positive analytic mass normalization"):
        evaluate_catalog_model_grid_likelihood(grid, [dataset])


def test_bounded_analytic_mass_profile_flags_clipped_boundary_solution():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([-1.0, -2.0]),
        sigma=np.asarray([0.1, 0.1]),
    )
    grid = build_photometric_model_grid(
        backend,
        space,
        filters=("u", "g"),
        band_names=("u", "g"),
    )

    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        log10_mass_bounds=(-2.0, 3.0),
    )

    assert np.allclose(result.log10_mass_profile, -2.0)
    assert np.all(result.mass_profile_at_boundary)
    assert not np.any(result.log10_mass_profile < -2.0)


def test_model_grid_save_load_roundtrip(tmp_path):
    backend = PerMassTemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0, 1.0])})

    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))
    path = tmp_path / "grid.npz"
    grid.save(path)
    loaded = load_photometric_model_grid(path)

    assert loaded.parameter_names == grid.parameter_names
    assert loaded.band_names == grid.band_names
    assert loaded.mass_normalization == MassNormalization.PER_SOLAR_MASS
    assert loaded.mass_reference == MassReference.SURVIVING_STELLAR_MASS
    assert np.allclose(loaded.samples, grid.samples)
    assert np.allclose(loaded.flux, grid.flux)
    assert np.allclose(loaded.log_prior, grid.log_prior)
    assert np.array_equal(loaded.valid, grid.valid)
    assert loaded.meta["schema"] == "composed.photometric_model_grid.v3"
    assert loaded.meta["excluded_parameter_priors"] == {}
    specification = loaded.meta["scientific_specification"]
    assert specification["backend"]["type"].endswith("PerMassTemplateBackend")
    assert specification["parameters"] == ["template"]
    assert specification["band_names"] == ["u", "g"]
    assert provenance_path_for(path).exists()
    locked = load_photometric_model_grid(path, require_provenance_sidecar=True)
    assert locked.meta["provenance"]["schema"] == "composed.provenance.v1"
    assert locked.meta["provenance"]["extra"]["mass_convention"] == "composed.mass.surviving_stellar.v1"


def test_legacy_formed_mass_photometric_grid_is_rejected(tmp_path):
    path = tmp_path / "legacy_grid.npz"
    np.savez(
        path,
        samples=np.zeros((1, 1)),
        flux=np.ones((1, 1)),
        log_prior=np.zeros(1),
        valid=np.ones(1, dtype=bool),
        parameter_names=np.asarray(["template"], dtype=object),
        band_names=np.asarray(["g"], dtype=object),
        mass_normalization=np.asarray(MassNormalization.PER_SOLAR_MASS.value, dtype=object),
        flux_unit=np.asarray("maggies", dtype=object),
        meta=np.asarray("{}", dtype=object),
    )

    with pytest.raises(ValueError, match="Legacy photometric model grid"):
        load_photometric_model_grid(path, require_provenance_sidecar=False)


def test_v2_model_grid_is_rejected_because_mask_validity_semantics_changed(tmp_path):
    path = tmp_path / "v2_grid.npz"
    np.savez(
        path,
        samples=np.zeros((1, 1)),
        flux=np.asarray([[1.0, np.nan]]),
        log_prior=np.zeros(1),
        valid=np.zeros(1, dtype=bool),
        parameter_names=np.asarray(["template"], dtype=object),
        band_names=np.asarray(["g", "r"], dtype=object),
        mass_normalization=np.asarray(MassNormalization.PER_SOLAR_MASS.value, dtype=object),
        mass_reference=np.asarray(MassReference.SURVIVING_STELLAR_MASS.value, dtype=object),
        flux_unit=np.asarray("maggies", dtype=object),
        meta=np.asarray(
            '{"schema": "composed.photometric_model_grid.v2"}',
            dtype=object,
        ),
    )

    with pytest.raises(ValueError, match="per-object mask semantics"):
        load_photometric_model_grid(path, require_provenance_sidecar=False)


def test_mass_grid_profiles_complete_upper_limit_likelihood():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0, 1.0])})
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([np.nan, np.nan]),
        sigma=np.asarray([0.2, 0.2]),
        upper_limit=np.asarray([1.0, 1.0]),
        upper_limit_mask=np.asarray([True, True]),
    )
    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))

    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        log10_mass_grid=np.linspace(-3.0, 0.0, 61),
        log10_mass_prior=UniformPrior(-3.0, 0.0),
        model_chunk_size=1,
        object_chunk_size=1,
        mass_chunk_size=16,
    )

    assert np.all(np.isfinite(result.profile_logp[0]))
    assert np.all(np.isfinite(result.marginal_logp[0]))
    assert np.isclose(np.sum(result.marginal_weights_norm[0]), 1.0)
    assert np.allclose(result.log10_mass_profile[0], -3.0)


def test_profile_mass_with_upper_limits_requires_explicit_mass_grid():
    backend = PerMassTemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([1.0, np.nan]),
        sigma=np.asarray([0.1, 0.2]),
        upper_limit=np.asarray([0.0, 0.5]),
        upper_limit_mask=np.asarray([False, True]),
    )
    grid = build_photometric_model_grid(backend, space, filters=("u", "g"), band_names=("u", "g"))

    with pytest.raises(ValueError, match="requires an explicit log10_mass_grid"):
        evaluate_catalog_model_grid_likelihood(grid, [dataset], log10_mass_bounds=(-3.0, 1.0))


def test_censored_profile_matches_brute_force_scalar_likelihood_on_mass_grid():
    backend = PerMassTemplateBackend()
    template_space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([0.7, np.nan]),
        sigma=np.asarray([0.15, 0.1]),
        upper_limit=np.asarray([0.0, 0.5]),
        upper_limit_mask=np.asarray([False, True]),
    )
    mass_grid = np.linspace(-2.0, 0.5, 251)
    grid = build_photometric_model_grid(
        backend,
        template_space,
        filters=("u", "g"),
        band_names=("u", "g"),
    )
    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        log10_mass_grid=mass_grid,
        log10_mass_prior=UniformPrior(-2.0, 0.5),
    )

    full_space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={"log10_mass": UniformPrior(-2.0, 0.5), "template": ChoicePrior([0.0])},
    )
    scalar = GaussianPhotometricLikelihood(backend, dataset, full_space, filters=("u", "g"))
    scalar_log_like = np.asarray([scalar.log_likelihood([mass, 0.0]) for mass in mass_grid])
    expected_index = int(np.argmax(scalar_log_like))

    assert result.log10_mass_profile[0, 0] == pytest.approx(mass_grid[expected_index])
    # The cached profile includes only the non-mass template prior, while this
    # comparison uses the likelihood term only; both are zero constants here.
    assert result.profile_logp[0, 0] == pytest.approx(scalar_log_like[expected_index])


def test_profile_grid_requires_per_solar_mass_models():
    backend = TemplateBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    grid = build_photometric_model_grid(backend, space, filters=("u", "g", "r"), band_names=("u", "g", "r"))
    dataset = SEDDataset(("u", "g", "r"), np.ones(3), np.ones(3))

    with pytest.raises(ValueError, match="PER_SOLAR_MASS"):
        evaluate_catalog_model_grid_likelihood(grid, [dataset], log10_mass_grid=np.linspace(0.0, 1.0, 3))


def test_mass_marginalization_uses_declared_prior_and_irregular_grid_cell_widths():
    backend = ZeroPerMassBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(("g",), flux=np.asarray([0.0]), sigma=np.asarray([1.0]))
    grid = build_photometric_model_grid(backend, space, filters=("g",), band_names=("g",))

    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        log10_mass_grid=np.asarray([8.0, 9.0, 11.0]),
        log10_mass_prior=UniformPrior(8.0, 11.0),
        store_mass_posterior=True,
    )

    expected_weights = np.asarray([0.5, 1.5, 1.0]) / 3.0
    assert np.allclose(result.mass_posterior_norm[0, 0], expected_weights)
    assert result.log10_mass_quantiles[0, 1] == pytest.approx(9.5)
    assert result.meta["mass_prior"]["type"] == "UniformPrior"
    assert result.meta["mass_prior"]["quadrature"] == "prior density times midpoint-cell width"


def test_mass_grid_requires_a_prior_object_instead_of_implicit_or_array_weights():
    backend = ZeroPerMassBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})
    dataset = SEDDataset(("g",), flux=np.asarray([0.0]), sigma=np.asarray([1.0]))
    grid = build_photometric_model_grid(backend, space, filters=("g",), band_names=("g",))
    mass_grid = np.asarray([8.0, 9.0, 11.0])

    with pytest.raises(ValueError, match="numerical grid does not define a prior"):
        evaluate_catalog_model_grid_likelihood(grid, [dataset], log10_mass_grid=mass_grid)
    with pytest.raises(TypeError, match="Prior instance"):
        evaluate_catalog_model_grid_likelihood(
            grid,
            [dataset],
            log10_mass_grid=mass_grid,
            log10_mass_prior=np.ones(3),
        )


def test_cached_mass_profile_with_model_discrepancy_requires_explicit_mass_grid():
    backend = PerMassBackend()
    model_space = ParameterSpace(
        names=("template",),
        priors={"template": ChoicePrior([0.0])},
    )
    grid = build_photometric_model_grid(
        backend,
        model_space,
        filters=("u", "g"),
        band_names=("u", "g"),
        excluded_parameters=(),
    )
    dataset = SEDDataset(
        ("u", "g"),
        flux=np.asarray([10.0, 20.0]),
        sigma=np.asarray([1.0, 1.0]),
    )

    with pytest.raises(ValueError, match="model_discrepancy.*log10_mass_grid"):
        evaluate_catalog_model_grid_likelihood(
            grid,
            [dataset],
            model_discrepancy=0.1,
        )

    mass_grid = np.linspace(0.0, 1.2, 121)
    result = evaluate_catalog_model_grid_likelihood(
        grid,
        [dataset],
        model_discrepancy=0.1,
        log10_mass_grid=mass_grid,
        log10_mass_prior=UniformPrior(0.0, 1.2),
    )
    full_space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(0.0, 1.2),
            "template": ChoicePrior([0.0]),
        },
    )
    scalar = GaussianPhotometricLikelihood(
        backend,
        dataset,
        full_space,
        filters=("u", "g"),
        model_discrepancy=0.1,
    )
    scalar_log_like = np.asarray(
        [scalar.log_likelihood([mass, 0.0]) for mass in mass_grid],
        dtype=float,
    )
    best = int(np.argmax(scalar_log_like))
    assert result.log10_mass_profile[0, 0] == pytest.approx(mass_grid[best])
    assert result.profile_logp[0, 0] == pytest.approx(scalar_log_like[best])
