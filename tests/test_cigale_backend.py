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
from composed.sfh import ContinuitySFH, DelayedTauSFH
from composed.units import MassNormalization, MassReference


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


def test_named_delayed_tau_sfh_maps_to_native_cigale_module(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(
        modules=["bc03", "redshifting"],
        sfh=DelayedTauSFH(),
        module_parameters={
            "bc03": {"imf": 1, "metallicity": 0.02},
            "redshifting": {"redshift": {"name": "z", "range": [0.0, 2.0]}},
        },
    )
    backend.predict_photometry(
        {"tage_gyr": 3.0, "tau_gyr": 0.5, "z": 0.3},
        FilterSet(["g"]),
    )

    call = FakeSedWarehouse.calls[-1]
    assert call["module_list"] == ["sfhdelayed", "bc03", "redshifting"]
    sfh_params = call["parameter_list"][0]
    assert sfh_params["age_main"] == 3000
    assert sfh_params["tau_main"] == 500.0
    assert sfh_params["f_burst"] == 0.0
    assert sfh_params["normalise"] is True


def test_named_cigale_sfh_rejects_native_sfh_module_and_unsupported_model(monkeypatch):
    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    with pytest.raises(ValueError, match="Use one SFH declaration only"):
        CIGALEBackend(modules=["sfhdelayed", "redshifting"], sfh="delayed_tau")
    with pytest.raises(ValueError, match="does not support backend 'cigale'"):
        CIGALEBackend(modules=["bc03", "redshifting"], sfh=ContinuitySFH())


def test_named_constant_cigale_sfh_reports_upstream_numpy_incompatibility(monkeypatch):
    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    monkeypatch.delattr(np, "float", raising=False)
    with pytest.raises(ImportError, match="NumPy 1.23.5"):
        CIGALEBackend(modules=["bc03", "redshifting"], sfh="constant")


def test_named_cigale_builder_requires_sfh_priors(monkeypatch):
    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    with pytest.raises(ValueError, match="missing: tau_gyr"):
        build_cigale_backend_and_parameter_space(
            modules=["bc03", "redshifting"],
            module_parameters={"redshifting": {"redshift": {"name": "z", "range": [0.0, 1.0]}}},
            additional_priors={"tage_gyr": UniformPrior(0.1, 5.0)},
            sfh="delayed_tau",
        )


def test_named_cigale_builder_keeps_parameter_order_deterministic(monkeypatch):
    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend, space = build_cigale_backend_and_parameter_space(
        modules=["bc03", "redshifting"],
        module_parameters={"redshifting": {"redshift": {"name": "z", "range": [0.0, 1.0]}}},
        additional_priors={
            "log10_mass": UniformPrior(8.0, 12.0),
            "tage_gyr": UniformPrior(0.1, 5.0),
            "tau_gyr": UniformPrior(0.1, 5.0),
        },
        sfh="delayed_tau",
    )

    assert isinstance(backend.sfh, DelayedTauSFH)
    assert space.names == ("log10_mass", "tage_gyr", "tau_gyr", "z")


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
    assert np.allclose(phot.flux, [2.0, 1.0])
    assert phot.metadata["formed_mass_msun"] == pytest.approx(1.0)
    assert phot.metadata["surviving_stellar_mass_msun"] == pytest.approx(0.5)
    assert phot.metadata["mass_reference"] == MassReference.SURVIVING_STELLAR_MASS.value

    call = FakeSedWarehouse.calls[-1]
    assert call["module_list"] == modules
    sfh_params, bc03_params, redshift_params = call["parameter_list"]
    assert sfh_params["normalise"] is True
    assert sfh_params["tau_main"] == 500.0
    assert sfh_params["age_main"] == 3000
    assert bc03_params == {"imf": 1, "metallicity": 0.02}
    assert redshift_params["redshift"] == 0.3


def test_cigale_likelihood_scales_surviving_stellar_mass_once(monkeypatch):
    install_fake_pcigale(monkeypatch, flux_by_filter={"g": 1.0 * MJY_PER_MAGGIE})

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
    assert np.allclose(phot.flux, 2.0 * 10.0 ** (-0.4 * np.array([20.0, 21.0])))


def test_cigale_predict_spectrum_returns_observed_flambda(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "redshifting"])
    requested = np.array([1000.0, 2000.0, 3000.0])

    spectrum = backend.predict_spectrum({"redshift": 0.1}, wavelengths=requested)

    expected = (2.0 * np.array([1.0, 1.0, 1.0]) * 1e-26) * C_A_PER_S / requested**2
    assert spectrum.wavelength_unit == "angstrom"
    assert spectrum.flux_unit == "erg/s/cm^2/angstrom"
    assert np.allclose(spectrum.wavelength, requested)
    assert np.allclose(spectrum.flux, expected)


def test_cigale_rest_spectrum_is_per_surviving_stellar_mass(monkeypatch):
    install_fake_pcigale(monkeypatch)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "bc03", "redshifting"])
    spectrum = backend.predict_rest_spectrum({"redshift": 0.1})

    assert spectrum.wavelength_unit == "nm"
    assert spectrum.flux_unit == "w/nm"
    assert np.allclose(spectrum.flux, [20.0, 40.0, 20.0])
    assert spectrum.metadata["surviving_stellar_mass_fraction"] == pytest.approx(0.5)


@pytest.mark.cigale
def test_real_cigale_photometry_is_per_surviving_stellar_mass():
    pytest.importorskip("pcigale")
    from pcigale.warehouse import SedWarehouse

    modules = ["sfhdelayed", "bc03", "redshifting"]
    module_parameters = {
        "sfhdelayed": {
            "age_main": 1000,
            "tau_main": 3000.0,
            "normalise": True,
        },
        "bc03": {
            "imf": 1,
            "metallicity": 0.02,
            "separation_age": 10,
        },
        "redshifting": {"redshift": 0.1},
    }
    filter_name = "sdss.gp"
    try:
        direct = SedWarehouse().get_sed(modules, [module_parameters[name] for name in modules])
        raw_maggies = float(direct.compute_fnu(filter_name)) / MJY_PER_MAGGIE
    except Exception as exc:
        pytest.skip(f"CIGALE v2022.0 database or {filter_name!r} filter is unavailable: {exc}")

    backend = CIGALEBackend(modules=modules, module_parameters=module_parameters)
    phot = backend.predict_photometry({}, FilterSet([filter_name]))
    stellar_mass = float(direct.info["stellar.m_star"])

    assert phot.flux[0] == pytest.approx(raw_maggies / stellar_mass, rel=1e-12)
    assert phot.metadata["surviving_stellar_mass_msun"] == pytest.approx(stellar_mass)
    assert phot.metadata["mass_reference"] == MassReference.SURVIVING_STELLAR_MASS.value


@pytest.mark.cigale
def test_real_named_cigale_delayed_tau_matches_direct_native_module():
    pytest.importorskip("pcigale")
    from pcigale.warehouse import SedWarehouse

    sfh = DelayedTauSFH()
    scalar_params = {"tage_gyr": 1.0, "tau_gyr": 3.0, "z": 0.1}
    modules = ["bc03", "redshifting"]
    module_parameters = {
        "bc03": {"imf": 1, "metallicity": 0.02, "separation_age": 10},
        "redshifting": {"redshift": {"name": "z", "range": [0.05, 0.2]}},
    }
    filter_name = "sdss.gp"

    direct_modules = ["sfhdelayed", "bc03", "redshifting"]
    direct_parameters = [
        sfh.cigale_parameters(scalar_params),
        module_parameters["bc03"],
        {"redshift": scalar_params["z"]},
    ]
    try:
        direct = SedWarehouse().get_sed(direct_modules, direct_parameters)
        raw_maggies = float(direct.compute_fnu(filter_name)) / MJY_PER_MAGGIE
    except Exception as exc:
        pytest.skip(f"CIGALE v2022.0 database or {filter_name!r} filter is unavailable: {exc}")

    backend = CIGALEBackend(modules=modules, module_parameters=module_parameters, sfh=sfh)
    phot = backend.predict_photometry(scalar_params, FilterSet([filter_name]))

    assert phot.flux[0] == pytest.approx(raw_maggies / direct.info["stellar.m_star"], rel=1.0e-12)


def test_cigale_missing_surviving_stellar_mass_raises(monkeypatch):
    install_fake_pcigale(monkeypatch, stellar_mass=None)

    import composed.backends.cigale as cigale_backend

    monkeypatch.setattr(cigale_backend, "_module_available", lambda name: True)
    backend = CIGALEBackend(modules=["sfhdelayed", "redshifting"])
    with pytest.raises(ValueError, match="stellar.m_star"):
        backend.predict_photometry({"redshift": 0.1}, FilterSet(["g"]))


class FakeCigaleSED:
    def __init__(self, flux_by_filter=None, stellar_mass=0.5):
        self.info = {"sfh.integrated": 1.0}
        if stellar_mass is not None:
            self.info["stellar.m_star"] = float(stellar_mass)
        self.wavelength_grid = np.array([100.0, 200.0, 300.0])
        self.fnu = np.array([1.0, 1.0, 1.0])
        self.luminosity = np.array([10.0, 20.0, 10.0])
        self.flux_by_filter = flux_by_filter or {
            "g": MJY_PER_MAGGIE,
            "r": 0.5 * MJY_PER_MAGGIE,
        }

    def compute_fnu(self, filter_name):
        return self.flux_by_filter[filter_name]


class FakeSedWarehouse:
    calls = []
    flux_by_filter = None
    stellar_mass = 0.5

    def __init__(self, nocache=None):
        self.nocache = nocache

    def get_sed(self, module_list, parameter_list):
        call = {
            "module_list": list(module_list),
            "parameter_list": [dict(params) for params in parameter_list],
            "nocache": self.nocache,
        }
        type(self).calls.append(call)
        return FakeCigaleSED(type(self).flux_by_filter, type(self).stellar_mass)


def install_fake_pcigale(monkeypatch, flux_by_filter=None, stellar_mass=0.5):
    FakeSedWarehouse.calls = []
    FakeSedWarehouse.flux_by_filter = flux_by_filter
    FakeSedWarehouse.stellar_mass = stellar_mass

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
