import numpy as np

from composed.transforms.popcosmos import PARAM_NAMES, popcosmos_theta_to_tabular_sfh


class FakeQuantity:
    def __init__(self, value):
        self.value = float(value)


class FakeCosmology:
    def age(self, z):
        return FakeQuantity(8.0)

    def distmod(self, z):
        return FakeQuantity(42.0)


def test_popcosmos_sfh_grid_increases_and_sfr_is_valid():
    values = {
        "N": 20.0,
        "log10Z": -0.2,
        "logsfr_ratio1": 0.0,
        "logsfr_ratio2": 0.1,
        "logsfr_ratio3": -0.1,
        "logsfr_ratio4": 0.2,
        "logsfr_ratio5": -0.2,
        "logsfr_ratio6": 0.0,
        "dust2": 0.3,
        "dust_index": -0.7,
        "dust1_fraction": 0.5,
        "lnfagn": -5.0,
        "lnagntau": 1.0,
        "gaslog10Z": -0.2,
        "gaslog10U": -2.0,
        "z": 0.7,
    }
    theta = np.array([values[name] for name in PARAM_NAMES], dtype=float)
    t_gyr, sfr, log10_mass = popcosmos_theta_to_tabular_sfh(theta, cosmology=FakeCosmology())
    assert np.all(np.diff(t_gyr) > 0.0)
    assert np.all(np.isfinite(sfr))
    assert np.all(sfr >= 0.0)
    assert np.isfinite(log10_mass)
