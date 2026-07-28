import warnings

import numpy as np
import pytest

from composed import (
    ConstantSFH,
    ContinuitySFH,
    DelayedTauSFH,
    ExponentialSFH,
    TabularSFH,
    available_sfh_models,
    make_sfh,
)
from composed.backends.base import ModelPhotometry, SEDBackend
from composed.data import SEDDataset
from composed.errors import ModelDomainError
from composed.likelihood import GaussianPhotometricLikelihood
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.sfh import coerce_sfh_model
from composed.transforms.sfh import normalize_sfh_to_formed_mass
from composed.units import MassNormalization


@pytest.mark.parametrize(
    "model",
    [ConstantSFH(n_time=33), ExponentialSFH(n_time=33), DelayedTauSFH(n_time=33)],
)
def test_parametric_sfh_histories_are_finite_increasing_and_unit_formed_mass(model):
    history = model.evaluate({"tage_gyr": 4.0, "tau_gyr": 1.5})

    assert history.time_gyr.shape == (33,)
    assert np.all(np.diff(history.time_gyr) > 0.0)
    assert np.all(np.isfinite(history.sfr_msun_per_yr))
    assert np.all(history.sfr_msun_per_yr >= 0.0)
    assert history.formed_mass_msun == pytest.approx(1.0, rel=1.0e-10)


def test_parametric_sfh_shapes_follow_the_documented_equations():
    constant = ConstantSFH(n_time=101).evaluate({"tage_gyr": 4.0})
    exponential = ExponentialSFH(n_time=101).evaluate({"tage_gyr": 4.0, "tau_gyr": 2.0})
    delayed = DelayedTauSFH(n_time=101).evaluate({"tage_gyr": 4.0, "tau_gyr": 2.0})

    assert np.allclose(constant.sfr_msun_per_yr, constant.sfr_msun_per_yr[0])
    assert exponential.sfr_msun_per_yr[-1] / exponential.sfr_msun_per_yr[0] == pytest.approx(
        np.exp(-2.0)
    )
    assert delayed.sfr_msun_per_yr[0] == 0.0
    peak_time = delayed.time_gyr[np.argmax(delayed.sfr_msun_per_yr)]
    assert peak_time == pytest.approx(2.0, abs=0.05)


def test_nonpositive_tau_is_a_model_domain_rejection_and_likelihood_returns_minus_infinity():
    model = DelayedTauSFH()
    with pytest.raises(ModelDomainError, match="must be positive"):
        model.evaluate({"tage_gyr": 4.0, "tau_gyr": -0.5})

    backend = SFHToyBackend(model)
    space = ParameterSpace(
        names=("tau_gyr",),
        priors={"tau_gyr": UniformPrior(-1.0, 1.0)},
    )
    dataset = SEDDataset(("g",), np.asarray([1.0]), np.asarray([0.1]))
    likelihood = GaussianPhotometricLikelihood(backend, dataset, space, filters=("g",))

    assert likelihood.log_prob([-0.5]) == -np.inf


def test_zero_formed_mass_normalization_is_a_model_domain_rejection():
    with pytest.raises(ModelDomainError, match="Invalid formed mass"):
        normalize_sfh_to_formed_mass(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 0.0]),
        )


def test_age_fraction_uses_universe_age_and_rejects_unphysical_absolute_age():
    fraction_model = ConstantSFH(age="age_fraction", age_kind="fraction_of_universe")
    history = fraction_model.evaluate(
        {"age_fraction": 0.4},
        redshift=2.0,
        cosmology=FakeCosmology(age_gyr=5.0),
    )
    assert history.time_gyr[-1] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="requires a redshift"):
        fraction_model.evaluate({"age_fraction": 0.4}, cosmology=FakeCosmology(5.0))

    with pytest.raises(ValueError, match="exceeds the Universe age"):
        ConstantSFH().evaluate(
            {"tage_gyr": 5.1},
            redshift=2.0,
            cosmology=FakeCosmology(age_gyr=5.0),
        )


def test_continuity_sfh_ratio_order_and_piecewise_grid_are_explicit():
    model = ContinuitySFH(
        lookback_edges_gyr=(0.0, 0.1, 0.5),
        samples_per_bin=4,
    )
    history = model.evaluate(
        {
            "tage_gyr": 2.0,
            "logsfr_ratio_0": 1.0,
            "logsfr_ratio_1": -0.5,
        }
    )

    assert model.required_parameters == ("tage_gyr", "logsfr_ratio_0", "logsfr_ratio_1")
    assert np.all(np.diff(history.time_gyr) > 0.0)
    assert np.all(np.diff(history.time_gyr.astype(np.float32)) > 0.0)
    assert history.formed_mass_msun == pytest.approx(1.0, rel=1.0e-10)
    bin_sfr = np.asarray(history.metadata["bin_sfr_recent_to_old"])
    assert bin_sfr[0] / bin_sfr[1] == pytest.approx(10.0)
    assert bin_sfr[1] / bin_sfr[2] == pytest.approx(10.0**-0.5)
    assert np.unique(history.sfr_msun_per_yr).size == 3


def test_continuity_sfh_requires_age_beyond_fixed_lookback_edges():
    model = ContinuitySFH(lookback_edges_gyr=(0.0, 0.1, 1.0))
    with pytest.raises(ValueError, match="must exceed the oldest fixed lookback edge"):
        model.evaluate(
            {
                "tage_gyr": 1.0,
                "logsfr_ratio_0": 0.0,
                "logsfr_ratio_1": 0.0,
            }
        )


def test_continuity_sfh_rejects_extreme_ratios_without_runtime_warnings():
    model = ContinuitySFH(lookback_edges_gyr=(0.0, 0.1, 1.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="non-finite or non-positive SFR"):
            model.evaluate(
                {
                    "tage_gyr": 2.0,
                    "logsfr_ratio_0": 400.0,
                    "logsfr_ratio_1": 0.0,
                }
            )

    assert caught == []


def test_tabular_sfh_preserves_input_amplitude_and_validates_arrays():
    model = TabularSFH(time="time", sfr="sfr")
    history = model.evaluate({"time": [0.0, 1.0], "sfr": [2.0e-9, 2.0e-9]})

    assert history.formed_mass_msun == pytest.approx(2.0)
    assert np.allclose(history.sfr_msun_per_yr, [2.0e-9, 2.0e-9])
    assert history.metadata["input_formed_mass_msun"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        model.evaluate({"time": [0.0, 0.0], "sfr": [1.0, 1.0]})


def test_named_sfh_registry_and_backend_support_are_deterministic():
    assert available_sfh_models() == (
        "constant",
        "exponential",
        "delayed_tau",
        "continuity",
        "tabular",
    )
    assert available_sfh_models("cigale") == ("constant", "exponential", "delayed_tau")
    assert isinstance(make_sfh("delayed"), DelayedTauSFH)
    assert isinstance(make_sfh("exp"), ExponentialSFH)

    with pytest.raises(ValueError, match="Unknown SFH model"):
        make_sfh("not-a-history")
    with pytest.raises(ValueError, match="does not support backend 'cigale'"):
        coerce_sfh_model(ContinuitySFH(), backend="cigale")


def test_cigale_adapters_use_native_v2022_modules_and_myr_units():
    delayed = DelayedTauSFH().cigale_parameters({"tage_gyr": 3.0, "tau_gyr": 0.5})
    exponential = ExponentialSFH().cigale_parameters({"tage_gyr": 3.0, "tau_gyr": 0.5})
    constant = ConstantSFH().cigale_parameters({"tage_gyr": 3.0})

    assert delayed["age_main"] == 3000
    assert delayed["tau_main"] == 500.0
    assert delayed["f_burst"] == 0.0
    assert delayed["normalise"] is True
    assert exponential["age"] == 3000
    assert exponential["tau_main"] == 500.0
    assert exponential["f_burst"] == 0.0
    assert constant == {
        "type_bursts": 2,
        "delta_bursts": 3001,
        "tau_bursts": 3000.0,
        "age": 3000,
        "sfr_A": 1.0,
        "normalise": True,
    }


class FakeCosmology:
    def __init__(self, age_gyr):
        self.age_gyr = float(age_gyr)

    def age(self, redshift):
        del redshift
        return self.age_gyr


class SFHToyBackend(SEDBackend):
    mass_normalization = MassNormalization.ABSOLUTE

    def __init__(self, sfh):
        self.sfh = sfh

    def predict_photometry(self, params, filters):
        del filters
        self.sfh.evaluate({"tage_gyr": 4.0, **params})
        return ModelPhotometry(("g",), np.asarray([1.0]))
