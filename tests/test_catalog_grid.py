from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inftools.core import Posterior
from inftools.grid import run_grid_sampler
from composed.catalog import run_photometric_grid_catalog
from composed.data import SEDDataset
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior
from composed.units import MassNormalization


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

    def predict_photometry(self, params, filters):
        del params, filters
        return ModelPhotometry(band_names=("u", "g"), flux=np.asarray([1.0, 2.0]))


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
