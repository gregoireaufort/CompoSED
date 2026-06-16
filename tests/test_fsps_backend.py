import os
import sys
import types

import numpy as np
import pytest

from composed.backends.fsps import FSPSBackend
from composed.filters import FilterSet
from composed.units import MassNormalization


def test_fsps_backend_module_imports_without_fsps_installed():
    import composed.backends.fsps as fsps_backend

    assert hasattr(fsps_backend, "FSPSBackend")


def test_constructing_fsps_backend_raises_helpful_error_if_fsps_missing(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: False)
    with pytest.raises(ImportError, match="python-fsps"):
        FSPSBackend()


def test_invalid_sfh_time_grid_raises_clear_error(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend()
    with pytest.raises(ValueError, match="strictly increasing"):
        backend.predict_photometry(
            {
                "z": 0.0,
                "tabular_time_gyr": [0.0, 1.0, 0.5],
                "tabular_sfr_msun_per_yr": [1.0, 1.0, 1.0],
            },
            FilterSet([], names=[]),
        )


def test_negative_sfr_raises_clear_error(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend()
    with pytest.raises(ValueError, match="non-negative"):
        backend.predict_photometry(
            {
                "zred": 0.0,
                "tabular_time_gyr": [0.0, 1.0, 2.0],
                "tabular_sfr_msun_per_yr": [1.0, -1.0, 1.0],
            },
            FilterSet([], names=[]),
        )


def test_sfh_age_exceeding_universe_age_raises_clear_error(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(cosmology=FakeCosmology(age_gyr=1.0))
    with pytest.raises(ValueError, match="age of the Universe"):
        backend.predict_photometry(
            {
                "redshift": 8.0,
                "tabular_time_gyr": [0.1, 10.0],
                "tabular_sfr_msun_per_yr": [1.0, 1.0],
            },
            FilterSet([], names=[]),
        )


def test_missing_redshift_raises_clear_error(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend()
    with pytest.raises(ValueError, match="Missing redshift"):
        backend.predict_photometry(
            {
                "tabular_time_gyr": [0.0, 1.0],
                "tabular_sfr_msun_per_yr": [1.0, 1.0],
            },
            FilterSet([], names=[]),
        )


def test_photometry_output_shape_matches_number_of_filters(monkeypatch):
    install_fake_sedpy(monkeypatch, magnitudes=[20.0, 21.0, 22.0])

    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    backend._sp = FakeStellarPopulation()
    filters = FilterSet([object(), object(), object()], names=["u", "g", "r"])

    phot = backend.predict_photometry(
        {
            "z": 0.0,
            "logzsol": -0.2,
            "dust2": 0.1,
            "tabular_time_gyr": [0.0, 1.0, 2.0],
            "tabular_sfr_msun_per_yr": [1.0, 1.0, 1.0],
        },
        filters,
    )

    assert phot.band_names == ("u", "g", "r")
    assert phot.flux.shape == (3,)
    assert np.all(np.isfinite(phot.flux))


def test_spectrum_output_matches_requested_wavelengths(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    backend._sp = FakeStellarPopulation()
    requested = np.array([1000.0, 1500.0, 2000.0])

    spectrum = backend.predict_spectrum(
        {
            "z": 0.0,
            "tabular_time_gyr": [0.0, 1.0, 2.0],
            "tabular_sfr_msun_per_yr": [1.0, 1.0, 1.0],
        },
        wavelengths=requested,
    )

    assert spectrum.wavelength_unit == "angstrom"
    assert spectrum.flux_unit == "erg/s/cm^2/angstrom"
    assert np.allclose(spectrum.wavelength, requested)
    assert spectrum.flux.shape == requested.shape
    assert np.all(np.isfinite(spectrum.flux))


def test_spectrum_wavelength_range_clips_native_grid(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    backend._sp = FakeStellarPopulation()

    spectrum = backend.predict_spectrum(
        {
            "z": 0.0,
            "tabular_time_gyr": [0.0, 1.0, 2.0],
            "tabular_sfr_msun_per_yr": [1.0, 1.0, 1.0],
        },
        wavelength_range=(1500.0, 2500.0),
    )

    assert np.all(spectrum.wavelength >= 1500.0)
    assert np.all(spectrum.wavelength <= 2500.0)
    assert spectrum.wavelength.shape == spectrum.flux.shape


def test_sedpy_shape_mismatch_raises_clear_error(monkeypatch):
    install_fake_sedpy(monkeypatch, magnitudes=[20.0])

    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    backend._sp = FakeStellarPopulation()
    filters = FilterSet([object(), object()], names=["g", "r"])

    with pytest.raises(ValueError, match="sedpy returned photometry shape"):
        backend.predict_photometry(
            {
                "z": 0.0,
                "tabular_time_gyr": [0.0, 1.0],
                "tabular_sfr_msun_per_yr": [1.0, 1.0],
            },
            filters,
        )


@pytest.mark.fsps
def test_real_fsps_integration_smoke():
    pytest.importorskip("fsps")
    pytest.importorskip("sedpy")
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")

    from sedpy.observate import load_filters

    backend = FSPSBackend()
    filters = load_filters(["sdss_g0", "sdss_r0"])
    phot = backend.predict_photometry(
        {
            "zred": 0.1,
            "tabular_time_gyr": [0.01, 1.0, 5.0],
            "tabular_sfr_msun_per_yr": [1.0, 1.0, 0.2],
        },
        FilterSet(filters, names=["sdss_g0", "sdss_r0"]),
    )

    assert phot.flux.shape == (2,)
    assert np.all(np.isfinite(phot.flux))


class FakeStellarPopulation:
    def __init__(self):
        self.params = {}
        self.tabular_sfh = None

    def set_tabular_sfh(self, time_gyr, sfr):
        self.tabular_sfh = (np.asarray(time_gyr), np.asarray(sfr))

    def get_spectrum(self, tage, peraa=True):
        assert peraa is True
        assert tage > 0.0
        return np.array([1000.0, 2000.0, 3000.0]), np.array([1.0, 2.0, 1.0])


class FakeQuantity:
    def __init__(self, value):
        self.value = float(value)

    def to(self, unit):
        return self


class FakeCosmology:
    def __init__(self, age_gyr):
        self.age_gyr = float(age_gyr)

    def age(self, z):
        return FakeQuantity(self.age_gyr)

    def luminosity_distance(self, z):
        return FakeQuantity(1.0e27)


def install_fake_sedpy(monkeypatch, magnitudes):
    sedpy = types.ModuleType("sedpy")
    observate = types.ModuleType("sedpy.observate")

    def get_sed(wave, flam, filters, linear_flux=False):
        assert linear_flux is False
        assert len(wave) == len(flam)
        assert len(filters) >= 0
        return np.asarray(magnitudes, dtype=float)

    observate.getSED = get_sed
    sedpy.observate = observate
    monkeypatch.setitem(sys.modules, "sedpy", sedpy)
    monkeypatch.setitem(sys.modules, "sedpy.observate", observate)
