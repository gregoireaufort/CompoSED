import os
import sys
import types

import numpy as np
import pytest

from composed.backends.fsps import FSPSBackend
from composed.errors import ModelDomainError
from composed.filters import FilterSet
from composed.sfh import ContinuitySFH, DelayedTauSFH, TabularSFH
from composed.units import MassNormalization, MassReference
from composed._numerics import trapezoid


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
    with pytest.raises(ModelDomainError, match="strictly increasing"):
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
    with pytest.raises(ModelDomainError, match="non-negative"):
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
    with pytest.raises(ModelDomainError, match="age of the Universe"):
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


def test_invalid_sampled_redshift_is_a_model_domain_error(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend()

    with pytest.raises(ModelDomainError, match="finite and non-negative"):
        backend.predict_photometry(
            {
                "z": -0.1,
                "tabular_time_gyr": [0.0, 1.0],
                "tabular_sfr_msun_per_yr": [1.0, 1.0],
            },
            FilterSet([], names=[]),
        )


def test_zero_tabular_sfh_is_a_model_domain_error(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend()

    with pytest.raises(ModelDomainError, match="positive finite stellar mass"):
        backend.predict_photometry(
            {
                "z": 0.0,
                "tabular_time_gyr": [0.0, 1.0],
                "tabular_sfr_msun_per_yr": [0.0, 0.0],
            },
            FilterSet([], names=[]),
        )


def test_named_delayed_tau_sfh_is_evaluated_and_not_forwarded_to_fsps(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(
        sfh=DelayedTauSFH(age="age_fraction", age_kind="fraction_of_universe", n_time=41),
        cosmology=FakeCosmology(age_gyr=10.0),
    )
    backend._sp = FakeStellarPopulation()

    spectrum = backend.predict_rest_spectrum(
        {
            "z": 1.0,
            "age_fraction": 0.4,
            "tau_gyr": 1.0,
            "dust2": 0.2,
        }
    )

    time_gyr, sfr = backend._sp.tabular_sfh
    assert time_gyr.shape == sfr.shape == (41,)
    assert time_gyr[-1] == pytest.approx(4.0)
    assert np.all(np.diff(time_gyr) > 0.0)
    assert np.all(sfr >= 0.0)
    assert backend._sp.params["dust2"] == pytest.approx(0.2)
    assert spectrum.metadata["sfh_model"]["name"] == "delayed_tau"
    assert spectrum.metadata["sfh_model"]["age_kind"] == "fraction_of_universe"


def test_named_fsps_sfh_rejects_raw_tabular_parameters(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(sfh="constant", cosmology=FakeCosmology(age_gyr=10.0))
    backend._sp = FakeStellarPopulation()

    with pytest.raises(ValueError, match="Use one SFH declaration only"):
        backend.predict_rest_spectrum(
            {
                "z": 0.2,
                "tage_gyr": 2.0,
                "tabular_time_gyr": [0.0, 1.0],
                "tabular_sfr_msun_per_yr": [1.0, 1.0],
            }
        )


def test_parametric_named_fsps_sfh_requires_per_solar_mass(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    with pytest.raises(ValueError, match="PER_SOLAR_MASS"):
        FSPSBackend(sfh="exponential", mass_normalization=MassNormalization.ABSOLUTE)


def test_tabular_sfh_object_allows_absolute_amplitude(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(
        sfh=TabularSFH(time="time", sfr="sfr"),
        mass_normalization=MassNormalization.ABSOLUTE,
        cosmology=FakeCosmology(age_gyr=10.0),
    )
    backend._sp = FakeStellarPopulation()
    backend.predict_rest_spectrum(
        {
            "z": 0.2,
            "time": [0.0, 1.0],
            "sfr": [2.0e-9, 2.0e-9],
        }
    )

    _, sfr = backend._sp.tabular_sfh
    assert np.allclose(sfr, [2.0e-9, 2.0e-9])
    assert backend._sp.formed_mass == pytest.approx(2.0)


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


def test_per_solar_mass_spectrum_is_normalized_by_surviving_stellar_mass(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.PER_SOLAR_MASS, cosmology=FakeCosmology(20.0))
    backend._sp = FakeStellarPopulation()

    spectrum = backend.predict_rest_spectrum(
        {
            "z": 0.0,
            "tabular_time_gyr": [0.0, 1.0, 2.0],
            "tabular_sfr_msun_per_yr": [1.0, 1.0, 1.0],
        }
    )

    assert spectrum.metadata["formed_mass_msun"] == pytest.approx(1.0)
    assert spectrum.metadata["surviving_stellar_mass_msun"] == pytest.approx(0.65)
    assert spectrum.metadata["surviving_stellar_mass_fraction"] == pytest.approx(0.65)
    assert spectrum.metadata["mass_reference"] == MassReference.SURVIVING_STELLAR_MASS.value
    expected_luminosity_w_per_nm = np.array([1.0, 2.0, 1.0]) * fsps_backend.LSUN_CGS * 1e-6 / 0.65
    assert np.allclose(spectrum.flux, expected_luminosity_w_per_nm)


def test_per_solar_mass_requires_fsps_stellar_mass(monkeypatch):
    import composed.backends.fsps as fsps_backend

    class MissingStellarMassPopulation(FakeStellarPopulation):
        def set_tabular_sfh(self, time_gyr, sfr):
            super().set_tabular_sfh(time_gyr, sfr)
            del self.stellar_mass

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.PER_SOLAR_MASS, cosmology=FakeCosmology(20.0))
    backend._sp = MissingStellarMassPopulation()

    with pytest.raises(FloatingPointError, match="stellar_mass"):
        backend.predict_rest_spectrum(
            {
                "z": 0.0,
                "tabular_time_gyr": [0.0, 1.0, 2.0],
                "tabular_sfr_msun_per_yr": [1.0, 1.0, 1.0],
            }
        )


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


def test_per_call_fsps_parameters_reset_to_constructor_baseline(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    reused_population = FakeStellarPopulation()
    backend = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    backend._sp = reused_population
    common = {
        "z": 0.0,
        "tabular_time_gyr": [0.0, 1.0],
        "tabular_sfr_msun_per_yr": [1.0, 1.0],
    }

    backend.predict_rest_spectrum({**common, "dust2": 2.0})
    after_omission = backend.predict_rest_spectrum(common)

    fresh = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    fresh._sp = FakeStellarPopulation()
    fresh_result = fresh.predict_rest_spectrum(common)

    assert np.allclose(after_omission.flux, fresh_result.flux)
    assert reused_population.params["dust2"] == 0.0


def test_any_valid_python_fsps_parameter_is_forwarded(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    backend._sp = FakeStellarPopulation()
    backend.predict_rest_spectrum(
        {
            "z": 0.0,
            "imf_type": 1,
            "dust_type": 2,
            "tabular_time_gyr": [0.0, 1.0],
            "tabular_sfr_msun_per_yr": [1.0, 1.0],
        }
    )

    assert backend._sp.params["imf_type"] == 1
    assert backend._sp.params["dust_type"] == 2


def test_unknown_fsps_parameter_raises_instead_of_being_ignored(monkeypatch):
    import composed.backends.fsps as fsps_backend

    monkeypatch.setattr(fsps_backend, "_module_available", lambda name: True)
    backend = FSPSBackend(mass_normalization=MassNormalization.ABSOLUTE, cosmology=FakeCosmology(age_gyr=20.0))
    backend._sp = FakeStellarPopulation()

    with pytest.raises(ValueError, match="valid python-fsps parameters"):
        backend.predict_rest_spectrum(
            {
                "z": 0.0,
                "parameter_that_fsps_does_not_have": 1.0,
                "tabular_time_gyr": [0.0, 1.0],
                "tabular_sfr_msun_per_yr": [1.0, 1.0],
            }
        )


@pytest.mark.fsps
def test_real_fsps_integration_smoke():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")
    pytest.importorskip("fsps")
    pytest.importorskip("sedpy")

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


@pytest.mark.fsps
def test_real_fsps_reused_backend_matches_fresh_backend_after_parameter_omission():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")
    pytest.importorskip("fsps")
    pytest.importorskip("sedpy")
    from sedpy.observate import load_filters

    filters = FilterSet(load_filters(["sdss_g0", "sdss_r0"]), names=["sdss_g0", "sdss_r0"])
    common = {
        "zred": 0.1,
        "tabular_time_gyr": [0.01, 1.0, 5.0],
        "tabular_sfr_msun_per_yr": [1.0, 1.0, 0.2],
    }
    reused = FSPSBackend(sp_kwargs={"add_neb_emission": False})
    reused.predict_photometry({**common, "dust2": 1.5}, filters)
    after_omission = reused.predict_photometry(common, filters)
    fresh = FSPSBackend(sp_kwargs={"add_neb_emission": False}).predict_photometry(common, filters)

    assert np.allclose(after_omission.flux, fresh.flux, rtol=1.0e-12, atol=0.0)


@pytest.mark.fsps
def test_real_fsps_named_delayed_tau_matches_its_explicit_tabular_history():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")
    pytest.importorskip("fsps")
    pytest.importorskip("sedpy")
    from sedpy.observate import load_filters

    filters = FilterSet(load_filters(["sdss_g0", "sdss_r0"]), names=["sdss_g0", "sdss_r0"])
    sfh = DelayedTauSFH(n_time=64)
    scalar_params = {"zred": 0.1, "tage_gyr": 5.0, "tau_gyr": 1.5, "logzsol": -0.3}
    named = FSPSBackend(sfh=sfh).predict_photometry(scalar_params, filters)

    history = sfh.evaluate(scalar_params, redshift=0.1)
    tabular_params = {
        "zred": 0.1,
        "logzsol": -0.3,
        "tabular_time_gyr": history.time_gyr,
        "tabular_sfr_msun_per_yr": history.sfr_msun_per_yr,
    }
    explicit = FSPSBackend().predict_photometry(tabular_params, filters)

    assert np.allclose(named.flux, explicit.flux, rtol=1.0e-12, atol=0.0)
    assert named.metadata["sfh_model"]["name"] == "delayed_tau"


@pytest.mark.fsps
def test_real_fsps_high_redshift_continuity_history_is_numerically_valid():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")
    pytest.importorskip("fsps")
    pytest.importorskip("sedpy")
    from sedpy.observate import load_filters

    sfh = ContinuitySFH(
        age="age_fraction",
        age_kind="fraction_of_universe",
        lookback_edges_gyr=(0.0, 0.01, 0.03, 0.1, 0.3),
    )
    params = {
        "zred": 4.8,
        "logzsol": -0.2,
        "dust2": 0.3,
        "age_fraction": 0.65,
        **{name: 0.0 for name in sfh.ratio_names},
    }
    filters = FilterSet(load_filters(["sdss_g0", "sdss_r0"]), names=["sdss_g0", "sdss_r0"])
    phot = FSPSBackend(sfh=sfh).predict_photometry(params, filters)

    assert np.all(np.isfinite(phot.flux))
    assert np.all(phot.flux >= 0.0)
    assert phot.metadata["formed_mass_msun"] > 0.0
    assert phot.metadata["surviving_stellar_mass_msun"] > 0.0


class FakeStellarPopulation:
    def __init__(self):
        self.params = FakeParameterSet(
            {
                "zred": 0.0,
                "logzsol": 0.0,
                "dust2": 0.0,
                "dust1": 0.0,
                "dust_index": 0.0,
                "gas_logz": 0.0,
                "gas_logu": -2.0,
                "fagn": 0.0,
                "agn_tau": 10.0,
                "imf_type": 2,
                "dust_type": 0,
            }
        )
        self.tabular_sfh = None
        self.formed_mass = 1.0
        self.stellar_mass = 0.65

    def set_tabular_sfh(self, time_gyr, sfr):
        time_gyr = np.asarray(time_gyr)
        sfr = np.asarray(sfr)
        self.tabular_sfh = (time_gyr, sfr)
        self.formed_mass = float(trapezoid(sfr, time_gyr) * 1.0e9)
        self.stellar_mass = 0.65 * self.formed_mass

    def get_spectrum(self, tage, peraa=True):
        assert peraa is True
        assert tage > 0.0
        amplitude = 1.0 + float(self.params["dust2"])
        return np.array([1000.0, 2000.0, 3000.0]), amplitude * np.array([1.0, 2.0, 1.0])


class FakeParameterSet(dict):
    @property
    def all_params(self):
        return list(self.keys())


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
