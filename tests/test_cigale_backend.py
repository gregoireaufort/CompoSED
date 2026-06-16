import sys
import types

import numpy as np
import pytest

from composed.backends.cigale import (
    C_A_PER_S,
    CIGALEBackend,
    MJY_PER_MAGGIE,
    build_cigale_backend_and_parameter_space,
    build_cigale_parameter_space,
)
from composed.data import SEDDataset
from composed.filters import FilterSet
from composed.likelihood import GaussianPhotometricLikelihood
from composed.priors import DeltaPrior, UniformPrior
from composed.units import MassNormalization


def test_cigale_backend_module_imports_without_pcigale_installed():
    import composed.backends.cigale as cigale_backend

    assert hasattr(cigale_backend, "CIGALEBackend")


def test_constructing_cigale_backend_raises_helpful_error_if_pcigale_missing(monkeypatch):
    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: False)
    with pytest.raises(ImportError, match="pcigale"):
        CIGALEBackend(modules=["sfhdelayed", "redshifting"])


def test_cigale_backend_rejects_absolute_mass_normalization(monkeypatch):
    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    with pytest.raises(ValueError, match="PER_SOLAR_MASS"):
        CIGALEBackend(modules=["sfhdelayed", "redshifting"], mass_normalization=MassNormalization.ABSOLUTE)


def test_cigale_parameter_space_from_ranges_and_choices():
    modules = ["sfhdelayed", "bc03", "redshifting"]
    module_parameters = {
        "sfhdelayed": {
            "tau_main": {"range": [100.0, 5000.0], "scale": "linear"},
            "age_main": {"values": [1000, 3000], "dtype": "int"},
        },
        "bc03": {
            "imf": 1,
            "metallicity": [0.008, 0.02],
        },
        "redshifting": {
            "redshift": {"name": "z", "range": [0.0, 2.0]},
        },
    }

    space = build_cigale_parameter_space(
        modules,
        module_parameters,
        additional_priors={"log10_mass": UniformPrior(8.0, 12.0)},
    )

    assert space.names == ("log10_mass", "tau_main", "age_main", "metallicity", "z")
    sample = space.sample_prior(32, rng=np.random.default_rng(4))
    assert np.all(np.isfinite([space.log_prior(row) for row in sample]))
    assert set(np.unique(sample[:, 2])).issubset({1000.0, 3000.0})
    assert set(np.unique(sample[:, 3])).issubset({0.008, 0.02})


def test_cigale_native_photometry_maps_params_and_enforces_sfh_normalise(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    modules = ["sfhdelayed", "bc03", "redshifting"]
    module_parameters = {
        "sfhdelayed": {
            "tau_main": {"range": [100.0, 5000.0]},
            "age_main": {"values": [1000, 3000], "dtype": "int"},
        },
        "bc03": {
            "imf": 1,
            "metallicity": [0.008, 0.02],
        },
        "redshifting": {
            "redshift": {"name": "z", "range": [0.0, 2.0]},
        },
    }
    backend = CIGALEBackend(modules=modules, module_parameters=module_parameters)

    phot = backend.predict_photometry(
        {"tau_main": 500.0, "age_main": 3000.0, "metallicity": 0.02, "z": 0.3},
        FilterSet(["g", "r"]),
    )

    assert phot.band_names == ("g", "r")
    assert np.allclose(phot.flux, [1.0, 0.5])

    call = FakeSedWarehouse.calls[-1]
    assert call["module_list"] == modules
    sfh_params, bc03_params, redshift_params = call["parameter_list"]
    assert sfh_params["normalise"] is True
    assert sfh_params["tau_main"] == 500.0
    assert sfh_params["age_main"] == 3000
    assert bc03_params == {"imf": 1, "metallicity": 0.02}
    assert redshift_params["redshift"] == 0.3


def test_cigale_likelihood_mass_scaling_is_centralized(monkeypatch):
    install_fake_pcigale(monkeypatch, flux_by_filter={"g": 2.0 * MJY_PER_MAGGIE})

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend, space = build_cigale_backend_and_parameter_space(
        modules=["sfhdelayed", "redshifting"],
        module_parameters={
            "redshifting": {"redshift": {"name": "z", "range": [0.0, 1.0]}},
        },
        additional_priors={"log10_mass": DeltaPrior(1.0)},
    )
    data = SEDDataset(["g"], flux=np.array([20.0]), sigma=np.array([1.0]))
    like = GaussianPhotometricLikelihood(backend, data, space, filters=FilterSet(["g"]))

    logp = like.log_prob([1.0, 0.2])
    expected = -0.5 * np.log(2.0 * np.pi)
    assert np.isclose(logp, expected)


def test_cigale_missing_redshift_raises_clear_error(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "redshifting"])
    with pytest.raises(ValueError, match="Missing redshift"):
        backend.predict_photometry({}, FilterSet(["g"]))


def test_cigale_unknown_parameter_raises_clear_error(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "redshifting"])
    with pytest.raises(KeyError, match="Unexpected parameter"):
        backend.predict_photometry({"redshift": 0.1, "dust2": 0.3}, FilterSet(["g"]))


def test_cigale_sfh_normalise_false_raises_clear_error(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(
        modules=["sfhdelayed", "redshifting"],
        module_parameters={"sfhdelayed": {"normalise": False}},
    )
    with pytest.raises(ValueError, match="normalise=True"):
        backend.predict_photometry({"redshift": 0.1}, FilterSet(["g"]))


def test_cigale_nonfinite_flux_raises_controlled_error(monkeypatch):
    install_fake_pcigale(monkeypatch, flux_by_filter={"g": np.nan})

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "redshifting"])
    with pytest.raises(FloatingPointError, match="non-finite"):
        backend.predict_photometry({"redshift": 0.1}, FilterSet(["g"]))


def test_cigale_sedpy_mode_integrates_via_sedpy(monkeypatch):
    install_fake_pcigale(monkeypatch)
    install_fake_sedpy(monkeypatch, magnitudes=[20.0, 21.0])

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "redshifting"], photometry_mode="sedpy")

    filters = FilterSet([object(), object()], names=["g", "r"])
    phot = backend.predict_photometry({"redshift": 0.1}, filters)

    assert phot.band_names == ("g", "r")
    assert np.allclose(phot.flux, 10.0 ** (-0.4 * np.array([20.0, 21.0])))


def test_cigale_predict_spectrum_returns_observed_flambda(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "redshifting"])
    requested = np.array([1000.0, 2000.0, 3000.0])

    spectrum = backend.predict_spectrum({"redshift": 0.1}, wavelengths=requested)

    expected = (np.array([1.0, 1.0, 1.0]) * 1e-26) * C_A_PER_S / requested**2
    assert spectrum.wavelength_unit == "angstrom"
    assert spectrum.flux_unit == "erg/s/cm^2/angstrom"
    assert np.allclose(spectrum.wavelength, requested)
    assert np.allclose(spectrum.flux, expected)


class FakeCigaleSED:
    def __init__(self, flux_by_filter=None):
        self.info = {"sfh.integrated": 1.0}
        self.wavelength_grid = np.array([100.0, 200.0, 300.0])
        self.fnu = np.array([1.0, 1.0, 1.0])
        self.flux_by_filter = flux_by_filter or {
            "g": MJY_PER_MAGGIE,
            "r": 0.5 * MJY_PER_MAGGIE,
        }

    def compute_fnu(self, filter_name):
        return self.flux_by_filter[filter_name]


class FakeSedWarehouse:
    calls = []
    flux_by_filter = None

    def __init__(self, nocache=None):
        self.nocache = nocache

    def get_sed(self, module_list, parameter_list):
        call = {
            "module_list": list(module_list),
            "parameter_list": [dict(params) for params in parameter_list],
            "nocache": self.nocache,
        }
        type(self).calls.append(call)
        return FakeCigaleSED(type(self).flux_by_filter)


def install_fake_pcigale(monkeypatch, flux_by_filter=None):
    FakeSedWarehouse.calls = []
    FakeSedWarehouse.flux_by_filter = flux_by_filter

    pcigale = types.ModuleType("pcigale")
    warehouse = types.ModuleType("pcigale.warehouse")
    warehouse.SedWarehouse = FakeSedWarehouse
    pcigale.warehouse = warehouse

    monkeypatch.setitem(sys.modules, "pcigale", pcigale)
    monkeypatch.setitem(sys.modules, "pcigale.warehouse", warehouse)
    return FakeSedWarehouse


def install_fake_sedpy(monkeypatch, magnitudes):
    sedpy = types.ModuleType("sedpy")
    observate = types.ModuleType("sedpy.observate")

    def get_sed(wave, flam, filters, linear_flux=False):
        assert linear_flux is False
        assert wave.shape == flam.shape
        assert len(filters) == len(magnitudes)
        return np.asarray(magnitudes, dtype=float)

    observate.getSED = get_sed
    sedpy.observate = observate
    monkeypatch.setitem(sys.modules, "sedpy", sedpy)
    monkeypatch.setitem(sys.modules, "sedpy.observate", observate)
