import pickle

import numpy as np
import pytest

from composed.backends.base import ModelPhotometry
from composed.derived import derive_sfh_quantities
from composed.noise import EmpiricalPhotometricNoise
from composed.sfh import ConstantSFH
from composed.units import MassNormalization, MassReference


def test_empirical_noise_preserves_rows_and_is_pickleable():
    sigma_rows = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    noise = EmpiricalPhotometricNoise(
        sigma_rows,
        fractional_error=0.1,
        band_names=["g", "r"],
        flux_unit="maggies",
    )
    restored = pickle.loads(pickle.dumps(noise))
    sigma = restored(np.asarray([1.0, 2.0]), rng=np.random.default_rng(2))

    possible = [np.hypot(row, [0.1, 0.2]) for row in sigma_rows]
    assert any(np.allclose(sigma, candidate) for candidate in possible)
    assert restored.specification()["band_names"] == ("g", "r")


def test_empirical_noise_validates_shape_and_uncertainties():
    with pytest.raises(ValueError, match="strictly positive"):
        EmpiricalPhotometricNoise([[0.1, 0.0]])
    noise = EmpiricalPhotometricNoise([[0.1, 0.2]])
    with pytest.raises(ValueError, match="expected flux shape"):
        noise([1.0])


class SurvivingMassBackend:
    mass_normalization = MassNormalization.PER_SOLAR_MASS
    mass_reference = MassReference.SURVIVING_STELLAR_MASS
    default_z_key = "zred"
    cosmology = None
    sfh = ConstantSFH(age="tage_gyr")

    def predict_photometry(self, params, filters):
        del filters
        assert params == {"zred": 0.2, "tage_gyr": 1.0}
        return ModelPhotometry(
            ["g"],
            [1.0],
            metadata={"surviving_stellar_mass_fraction": 0.5},
        )


def test_derived_sfh_uses_surviving_mass_and_returns_physical_sfr():
    derived = derive_sfh_quantities(
        SurvivingMassBackend(),
        {"zred": 0.2, "tage_gyr": 1.0, "log10_mass": 10.0},
        filters=[object()],
    )

    assert np.isclose(derived.surviving_stellar_mass_msun, 1.0e10)
    assert np.isclose(derived.formed_mass_msun, 2.0e10)
    assert np.isclose(derived.current_sfr_msun_per_yr, 20.0)
    assert np.isclose(derived.log10_ssfr_per_yr, np.log10(2.0e-9))
    assert np.isclose(derived.mass_weighted_age_gyr, 0.5)
