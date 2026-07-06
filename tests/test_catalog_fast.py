from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from composed.backends.base import ModelSpectrum, SEDBackend
from composed.catalog_fast import (
    PARSEC_M,
    RestFrameSpectralGrid,
    build_redshift_filter_operator,
    build_restframe_spectral_grid,
    fit_catalog_with_restframe_grid,
    load_restframe_spectral_grid,
    project_rest_grid_to_photometric_grid,
    save_restframe_spectral_grid,
)
from composed.data import SEDDataset
from composed.filters import FilterSet
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, UniformPrior
from composed.units import MassNormalization


@dataclass(frozen=True)
class TabulatedFilter:
    name: str
    wavelength_nm: np.ndarray
    transmission: np.ndarray


@dataclass
class ToyRestBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS

    def predict_rest_spectrum(self, params, wavelengths=None, wavelength_range=None):
        template = int(round(float(params["template"])))
        wave = np.asarray([400.0, 500.0, 600.0], dtype=float)
        luminosities = {
            0: np.asarray([1.0, 1.0, 1.0], dtype=float),
            1: np.asarray([1.0, 3.0, 2.0], dtype=float),
        }
        lum = luminosities[template]
        if wavelengths is not None:
            requested = np.asarray(wavelengths, dtype=float)
            lum = np.interp(requested, wave, lum)
            wave = requested
        if wavelength_range is not None:
            lo, hi = wavelength_range
            keep = (wave >= lo) & (wave <= hi)
            wave = wave[keep]
            lum = lum[keep]
        return ModelSpectrum(
            wavelength=wave,
            flux=lum,
            wavelength_unit="nm",
            flux_unit="W/nm",
            metadata={"spectrum_frame": "rest"},
        )


def test_redshift_filter_operator_matches_manual_trapezoid_at_z0():
    wavelength_nm = np.asarray([400.0, 500.0, 600.0])
    filt = TabulatedFilter(
        name="box",
        wavelength_nm=np.asarray([400.0, 600.0]),
        transmission=np.asarray([1.0, 1.0]),
    )

    operator = build_redshift_filter_operator(
        wavelength_nm,
        FilterSet([filt]),
        redshift=0.0,
        igm_model=None,
        luminosity_distance_m=10.0 * PARSEC_M,
    )

    luminosity = np.asarray([2.0, 2.0, 2.0])
    flux = luminosity @ operator.matrix[0]
    expected_integral_w = 2.0 * (600.0 - 400.0)
    expected_maggies = expected_integral_w / (4.0 * np.pi * (10.0 * PARSEC_M) ** 2) / (3631.0e3)

    assert operator.band_names == ("box",)
    assert np.array_equal(operator.valid_bands, [True])
    assert np.isclose(flux, expected_maggies)


def test_build_restframe_grid_and_project_to_photometry():
    backend = ToyRestBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(0.0, 3.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )
    filters = FilterSet(
        [
            TabulatedFilter("blue", np.asarray([400.0, 500.0]), np.asarray([1.0, 1.0])),
            TabulatedFilter("red", np.asarray([500.0, 600.0]), np.asarray([1.0, 1.0])),
        ]
    )

    rest_grid = build_restframe_spectral_grid(backend, space)
    operator = build_redshift_filter_operator(
        rest_grid.wavelength_nm,
        filters,
        redshift=0.0,
        igm_model=None,
        luminosity_distance_m=10.0 * PARSEC_M,
    )
    phot_grid = project_rest_grid_to_photometric_grid(rest_grid, operator)

    assert rest_grid.parameter_names == ("template",)
    assert rest_grid.luminosity_w_per_nm.shape == (2, 3)
    assert phot_grid.band_names == ("blue", "red")
    assert phot_grid.flux.shape == (2, 2)
    assert np.all(phot_grid.valid)


def test_restframe_grid_save_load_roundtrip(tmp_path):
    backend = ToyRestBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(0.0, 3.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )
    rest_grid = build_restframe_spectral_grid(backend, space)
    rest_grid.meta["purpose"] = "roundtrip test"
    path = tmp_path / "rest_grid.npz"

    save_restframe_spectral_grid(rest_grid, path)
    loaded = load_restframe_spectral_grid(path)

    assert loaded.parameter_names == rest_grid.parameter_names
    assert loaded.mass_normalization == rest_grid.mass_normalization
    assert loaded.meta["purpose"] == "roundtrip test"
    assert np.allclose(loaded.wavelength_nm, rest_grid.wavelength_nm)
    assert np.allclose(loaded.luminosity_w_per_nm, rest_grid.luminosity_w_per_nm)
    assert np.allclose(loaded.samples, rest_grid.samples)
    assert np.allclose(loaded.log_prior, rest_grid.log_prior)
    assert np.array_equal(loaded.valid, rest_grid.valid)


@pytest.mark.cigale
def test_cached_restframe_catalog_fit_with_real_cigale_backend(tmp_path):
    pytest.importorskip("pcigale")

    from composed.backends.cigale import CIGALEBackend

    modules = ("sfhdelayed", "bc03", "redshifting")
    module_parameters = {
        "sfhdelayed": {
            "tau_main": 1000.0,
            "age_main": 1000,
            "tau_burst": 50.0,
            "age_burst": 20,
            "f_burst": 0.0,
            "sfr_A": 1.0,
            "normalise": True,
        },
        "bc03": {
            "imf": 1,
            "metallicity": {"values": [0.008, 0.02]},
        },
        "redshifting": {
            "redshift": {"name": "z", "range": [0.01, 0.2]},
        },
    }
    backend = CIGALEBackend(modules=modules, module_parameters=module_parameters)
    parameter_space = ParameterSpace(
        names=("log10_mass", "metallicity", "z"),
        priors={
            "log10_mass": UniformPrior(8.0, 10.0),
            "metallicity": ChoicePrior([0.008, 0.02]),
            "z": UniformPrior(0.01, 0.2),
        },
    )
    filters = FilterSet(
        [
            TabulatedFilter("blue", np.asarray([420.0, 500.0, 580.0]), np.asarray([0.0, 1.0, 0.0])),
            TabulatedFilter("red", np.asarray([650.0, 760.0, 870.0]), np.asarray([0.0, 1.0, 0.0])),
        ]
    )

    rest_grid = build_restframe_spectral_grid(
        backend,
        parameter_space,
        wavelengths_nm=np.linspace(250.0, 1000.0, 400),
    )
    cache_path = tmp_path / "cigale_rest_grid.npz"
    save_restframe_spectral_grid(rest_grid, cache_path)
    loaded_grid = load_restframe_spectral_grid(cache_path)

    operator = build_redshift_filter_operator(
        loaded_grid.wavelength_nm,
        filters,
        redshift=0.05,
        igm_model=None,
    )
    phot_grid = project_rest_grid_to_photometric_grid(
        loaded_grid,
        operator,
        age_parameter=None,
    )
    true_model = int(np.where(np.isclose(loaded_grid.samples[:, 0], 0.02))[0][0])
    observed_flux = 10.0**9.0 * phot_grid.flux[true_model]
    dataset = SEDDataset(
        phot_grid.band_names,
        flux=observed_flux,
        sigma=np.maximum(0.01 * observed_flux, 1.0e-40),
    )

    result = fit_catalog_with_restframe_grid(
        loaded_grid,
        [dataset],
        redshifts=[0.05],
        filters=filters,
        redshift_decimals=None,
        igm_model=None,
        log10_mass_bounds=(8.0, 10.0),
        age_parameter=None,
    )

    best = result.profile_map_indices[0]
    assert loaded_grid.parameter_names == ("metallicity",)
    assert np.all(loaded_grid.valid)
    assert result.profile_map_estimates[0, 0] == pytest.approx(0.02)
    assert result.log10_mass_profile[0, best] == pytest.approx(9.0, abs=1.0e-10)


def test_native_catalog_fit_recovers_template_and_profile_mass():
    backend = ToyRestBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={
            "log10_mass": UniformPrior(0.0, 3.0),
            "template": ChoicePrior([0.0, 1.0]),
        },
    )
    filters = FilterSet(
        [
            TabulatedFilter("blue", np.asarray([400.0, 500.0]), np.asarray([1.0, 1.0])),
            TabulatedFilter("red", np.asarray([500.0, 600.0]), np.asarray([1.0, 1.0])),
        ]
    )
    rest_grid = build_restframe_spectral_grid(backend, space)
    operator = build_redshift_filter_operator(
        rest_grid.wavelength_nm,
        filters,
        redshift=0.0,
        igm_model=None,
        luminosity_distance_m=10.0 * PARSEC_M,
    )
    phot_grid = project_rest_grid_to_photometric_grid(rest_grid, operator)
    true_model = 1
    true_mass_scale = 10.0
    observed_flux = true_mass_scale * phot_grid.flux[true_model]
    dataset = SEDDataset(
        band_names=("blue", "red"),
        flux=observed_flux,
        sigma=np.full(2, np.max(observed_flux) * 1.0e-3),
    )

    result = fit_catalog_with_restframe_grid(
        rest_grid,
        [dataset],
        redshifts=[0.0],
        filters=filters,
        igm_model=None,
        log10_mass_bounds=(0.0, 3.0),
    )

    assert result.profile_map_estimates.shape == (1, 1)
    assert result.profile_map_estimates[0, 0] == 1.0
    best = result.profile_map_indices[0]
    assert np.isclose(result.log10_mass_profile[0, best], 1.0, atol=1.0e-10)


def test_projection_rejects_models_older_than_universe():
    rest_grid = RestFrameSpectralGrid(
        wavelength_nm=np.asarray([400.0, 500.0, 600.0]),
        luminosity_w_per_nm=np.ones((2, 3)),
        samples=np.asarray([[100.0], [20_000.0]]),
        log_prior=np.zeros(2),
        valid=np.ones(2, dtype=bool),
        parameter_names=("age",),
        mass_normalization=MassNormalization.PER_SOLAR_MASS,
    )
    filt = FilterSet([TabulatedFilter("box", np.asarray([400.0, 600.0]), np.asarray([1.0, 1.0]))])
    operator = build_redshift_filter_operator(
        rest_grid.wavelength_nm,
        filt,
        redshift=0.0,
        igm_model=None,
        luminosity_distance_m=10.0 * PARSEC_M,
    )

    phot_grid = project_rest_grid_to_photometric_grid(rest_grid, operator, age_parameter="age")

    assert np.array_equal(phot_grid.valid, [True, False])
