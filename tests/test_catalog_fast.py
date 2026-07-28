from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from composed.backends.base import ModelSpectrum, SEDBackend
import composed.catalog_fast as catalog_fast
from composed.catalog_fast import (
    AB_ZERO_FNU_W_M2_HZ,
    C_NM_PER_S,
    ExperimentalFastCatalogWarning,
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
from composed.provenance import provenance_path_for
from composed.units import MassNormalization, MassReference


@dataclass(frozen=True)
class TabulatedFilter:
    name: str
    wavelength_nm: np.ndarray
    transmission: np.ndarray


@dataclass(frozen=True)
class AngstromFilter:
    name: str
    wavelength: np.ndarray
    transmission: np.ndarray
    wavelength_unit: str = "angstrom"


class FixedDistanceCosmology:
    name = "fixed-distance-test"

    def __init__(self, distance_m, age_myr=14_000.0):
        self.distance_m = float(distance_m)
        self.age_myr = float(age_myr)

    def luminosity_distance(self, redshift):
        del redshift
        return self.distance_m

    def age(self, redshift):
        del redshift
        return self.age_myr


@dataclass
class ToyRestBackend(SEDBackend):
    mass_normalization: MassNormalization = MassNormalization.PER_SOLAR_MASS
    mass_reference: MassReference = MassReference.SURVIVING_STELLAR_MASS
    supports_fast_catalog_restframe: ClassVar[bool] = True

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


class UnsupportedRestBackend(ToyRestBackend):
    supports_fast_catalog_restframe: ClassVar[bool] = False


def test_raw_filter_operator_returns_one_maggie_for_flat_ab_standard():
    wavelength_nm = np.linspace(400.0, 600.0, 2001)
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

    # A 3631 Jy source is one maggie by definition. Convert flat f_nu to
    # f_lambda [W m^-2 nm^-1], then to luminosity density at 10 pc.
    f_lambda = AB_ZERO_FNU_W_M2_HZ * C_NM_PER_S / wavelength_nm**2
    luminosity = f_lambda * 4.0 * np.pi * (10.0 * PARSEC_M) ** 2
    flux = luminosity @ operator.matrix[0]

    assert operator.band_names == ("box",)
    assert np.array_equal(operator.valid_bands, [True])
    assert np.isclose(flux, 1.0, rtol=2.0e-6)


def test_angstrom_filter_curve_is_converted_before_ab_integration():
    wavelength_nm = np.linspace(400.0, 600.0, 2001)
    filt = AngstromFilter(
        name="box",
        wavelength=np.asarray([4000.0, 6000.0]),
        transmission=np.asarray([1.0, 1.0]),
    )
    operator = build_redshift_filter_operator(
        wavelength_nm,
        FilterSet([filt]),
        redshift=0.0,
        igm_model=None,
        luminosity_distance_m=10.0 * PARSEC_M,
    )
    f_lambda = AB_ZERO_FNU_W_M2_HZ * C_NM_PER_S / wavelength_nm**2
    luminosity = f_lambda * 4.0 * np.pi * (10.0 * PARSEC_M) ** 2

    assert np.isclose(luminosity @ operator.matrix[0], 1.0, rtol=2.0e-6)


def test_default_cosmology_does_not_depend_on_cigale_import(monkeypatch):
    import composed.catalog_fast as catalog_fast

    distance = 123.0 * PARSEC_M
    cosmology = FixedDistanceCosmology(distance)
    operator = build_redshift_filter_operator(
        np.linspace(400.0, 600.0, 101),
        FilterSet([TabulatedFilter("box", np.asarray([400.0, 600.0]), np.ones(2))]),
        redshift=0.1,
        igm_model=None,
        cosmology=cosmology,
    )

    assert operator.meta["luminosity_distance_m"] == pytest.approx(distance)
    assert operator.meta["cosmology"] == "fixed-distance-test"


def test_rest_spectrum_angstrom_coordinate_does_not_rescale_w_per_nm_luminosity():
    model = ModelSpectrum(
        wavelength=np.asarray([4000.0, 5000.0]),
        flux=np.asarray([2.0, 3.0]),
        wavelength_unit="angstrom",
        flux_unit="W/nm",
    )

    wavelength_nm, luminosity_w_per_nm = catalog_fast._coerce_rest_spectrum_to_w_per_nm(model)

    assert np.allclose(wavelength_nm, [400.0, 500.0])
    assert np.allclose(luminosity_w_per_nm, [2.0, 3.0])


def test_fast_rest_grid_rejects_backend_without_explicit_capability():
    backend = UnsupportedRestBackend()
    space = ParameterSpace(names=("template",), priors={"template": ChoicePrior([0.0])})

    with pytest.warns(ExperimentalFastCatalogWarning):
        with pytest.raises(NotImplementedError, match="does not declare support"):
            build_restframe_spectral_grid(backend, space)


@pytest.mark.cigale
def test_fast_rest_grid_rejects_redshift_aware_cigale_sfh():
    pytest.importorskip("pcigale")

    from composed import DelayedTauSFH
    from composed.backends.cigale import CIGALEBackend

    backend = CIGALEBackend(
        modules=("bc03", "redshifting"),
        module_parameters={
            "bc03": {"imf": 1, "metallicity": 0.02, "separation_age": 10},
            "redshifting": {"redshift": {"range": [0.05, 1.0]}},
        },
        sfh=DelayedTauSFH(
            age="age_fraction",
            age_kind="fraction_of_universe",
            tau="tau_gyr",
        ),
    )
    space = ParameterSpace(
        names=("age_fraction", "tau_gyr", "redshift"),
        priors={
            "age_fraction": UniformPrior(0.3, 0.95),
            "tau_gyr": UniformPrior(0.1, 5.0),
            "redshift": UniformPrior(0.05, 1.0),
        },
    )

    assert backend.supports_fast_catalog_restframe is False
    with pytest.warns(ExperimentalFastCatalogWarning):
        with pytest.raises(NotImplementedError, match="does not declare support"):
            build_restframe_spectral_grid(backend, space)


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


def test_fast_projection_rejects_filter_outside_rest_wavelength_grid():
    rest_grid = RestFrameSpectralGrid(
        wavelength_nm=np.asarray([400.0, 500.0, 600.0]),
        luminosity_w_per_nm=np.ones((1, 3)),
        samples=np.asarray([[0.0]]),
        log_prior=np.zeros(1),
        valid=np.ones(1, dtype=bool),
        parameter_names=("template",),
        mass_normalization=MassNormalization.PER_SOLAR_MASS,
        mass_reference=MassReference.SURVIVING_STELLAR_MASS,
    )
    filters = FilterSet(
        [TabulatedFilter("outside", np.asarray([250.0, 300.0]), np.ones(2))]
    )
    operator = build_redshift_filter_operator(
        rest_grid.wavelength_nm,
        filters,
        redshift=0.0,
        igm_model=None,
        luminosity_distance_m=10.0 * PARSEC_M,
    )

    with pytest.raises(ValueError, match=r"Unavailable band\(s\): outside"):
        project_rest_grid_to_photometric_grid(rest_grid, operator, age_parameter=None)


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
    assert loaded.mass_reference == MassReference.SURVIVING_STELLAR_MASS
    assert loaded.meta["purpose"] == "roundtrip test"
    assert np.allclose(loaded.wavelength_nm, rest_grid.wavelength_nm)
    assert np.allclose(loaded.luminosity_w_per_nm, rest_grid.luminosity_w_per_nm)
    assert np.allclose(loaded.samples, rest_grid.samples)
    assert np.allclose(loaded.log_prior, rest_grid.log_prior)
    assert np.array_equal(loaded.valid, rest_grid.valid)
    assert provenance_path_for(path).exists()
    locked = load_restframe_spectral_grid(path, require_provenance_sidecar=True)
    assert locked.meta["provenance"]["schema"] == "composed.provenance.v1"
    assert locked.meta["provenance"]["extra"]["mass_convention"] == "composed.mass.surviving_stellar.v1"


def test_legacy_formed_mass_rest_grid_is_rejected(tmp_path):
    path = tmp_path / "legacy_rest_grid.npz"
    np.savez(
        path,
        wavelength_nm=np.asarray([400.0, 500.0]),
        luminosity_w_per_nm=np.ones((1, 2)),
        samples=np.zeros((1, 1)),
        log_prior=np.zeros(1),
        valid=np.ones(1, dtype=bool),
        parameter_names=np.asarray(["template"], dtype=object),
        mass_normalization=np.asarray(MassNormalization.PER_SOLAR_MASS.value, dtype=object),
        meta=np.asarray("{}", dtype=object),
    )

    with pytest.raises(ValueError, match="Legacy rest-frame spectral grid"):
        load_restframe_spectral_grid(path)


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

    native_filters = FilterSet(["sdss.gp", "sdss.rp"])
    from astropy.cosmology import WMAP7

    native_operator = build_redshift_filter_operator(
        loaded_grid.wavelength_nm,
        native_filters,
        redshift=0.05,
        igm_model="cigale",
        cosmology=WMAP7,
    )
    native_grid = project_rest_grid_to_photometric_grid(
        loaded_grid,
        native_operator,
        age_parameter=None,
        cosmology=WMAP7,
    )
    direct = backend.predict_photometry({"metallicity": 0.02, "z": 0.05}, native_filters)
    assert np.allclose(native_grid.flux[true_model], direct.flux, rtol=2.0e-5, atol=0.0)


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


def test_catalog_keeps_input_redshift_and_does_not_round_by_default():
    backend = ToyRestBackend()
    space = ParameterSpace(
        names=("log10_mass", "template"),
        priors={"log10_mass": UniformPrior(0.0, 3.0), "template": ChoicePrior([0.0])},
    )
    rest_grid = build_restframe_spectral_grid(backend, space)
    filters = FilterSet([TabulatedFilter("box", np.asarray([405.0, 595.0]), np.ones(2))])
    z = 0.004
    operator = build_redshift_filter_operator(
        rest_grid.wavelength_nm,
        filters,
        z,
        igm_model=None,
        cosmology=FixedDistanceCosmology(1.0e24),
    )
    phot_grid = project_rest_grid_to_photometric_grid(
        rest_grid,
        operator,
        age_parameter=None,
        cosmology=FixedDistanceCosmology(1.0e24),
    )
    dataset = SEDDataset(("box",), 10.0 * phot_grid.flux[0], np.maximum(phot_grid.flux[0] * 0.01, 1e-40))

    result = fit_catalog_with_restframe_grid(
        rest_grid,
        [dataset],
        [z],
        filters,
        igm_model=None,
        cosmology=FixedDistanceCosmology(1.0e24),
        log10_mass_bounds=(0.0, 3.0),
        age_parameter=None,
    )

    assert result.redshifts[0] == pytest.approx(z)
    assert result.evaluated_redshifts[0] == pytest.approx(z)


def test_catalog_rejects_rounding_positive_redshift_to_zero():
    rest_grid = RestFrameSpectralGrid(
        wavelength_nm=np.asarray([400.0, 500.0, 600.0]),
        luminosity_w_per_nm=np.ones((1, 3)),
        samples=np.asarray([[0.0]]),
        log_prior=np.zeros(1),
        valid=np.ones(1, dtype=bool),
        parameter_names=("template",),
        mass_normalization=MassNormalization.PER_SOLAR_MASS,
    )
    dataset = SEDDataset(("box",), np.asarray([1.0]), np.asarray([0.1]))
    filters = FilterSet([TabulatedFilter("box", np.asarray([400.0, 600.0]), np.ones(2))])

    with pytest.raises(ValueError, match="mapped positive observed redshift.*to zero"):
        fit_catalog_with_restframe_grid(
            rest_grid,
            [dataset],
            [0.004],
            filters,
            redshift_decimals=2,
            igm_model=None,
            cosmology=FixedDistanceCosmology(1.0e24),
            age_parameter=None,
        )


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
