import os

import numpy as np
import pytest

from composed.experimental.cigale_fsps_stellar import (
    mixed_grid_nocache_modules,
    module_directory,
    register_cigale_fsps_stellar_module,
)


def test_experimental_cigale_fsps_module_files_are_packaged():
    path = module_directory()
    assert path.exists()
    assert (path / "fsps_stellar.py").exists()


def test_mixed_grid_nocache_modules_include_grid_sensitive_modules():
    nocache = mixed_grid_nocache_modules()
    assert "fsps_stellar" in nocache
    assert "nebular" in nocache
    assert "redshifting" in nocache


@pytest.mark.cigale_fsps
def test_experimental_fsps_stellar_cigale_chain_smoke():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")

    pytest.importorskip("pcigale")
    pytest.importorskip("fsps")

    register_cigale_fsps_stellar_module()

    from pcigale.warehouse import SedWarehouse

    warehouse = SedWarehouse(nocache=["fsps_stellar"])
    sed = warehouse.get_sed(
        ["sfhdelayed", "fsps_stellar", "redshifting"],
        [
            {"age_main": 100, "tau_main": 50.0, "normalise": True},
            {"logzsol": 0.0, "separation_age": 10},
            {"redshift": 0.1},
        ],
    )

    assert "stellar.old" in sed.luminosities
    assert "stellar.young" in sed.luminosities
    assert np.all(np.isfinite(sed.luminosity))
    assert np.any(sed.luminosity > 0.0)
    for key in [
        "stellar.m_star",
        "stellar.m_star_young",
        "stellar.m_star_old",
        "stellar.n_ly",
        "stellar.lum",
        "stellar.n_ly_young",
        "stellar.n_ly_old",
        "stellar.lum_young",
        "stellar.lum_old",
        "sfh.integrated",
        "stellar.fsps.logzsol",
        "stellar.fsps.equivalent_metallicity",
        "stellar.fsps.z_sun",
    ]:
        assert key in sed.info
        assert np.isfinite(sed.info[key])
    assert np.isclose(sed.info["sfh.integrated"], 1.0)
    assert sed.info["stellar.fsps.imf_label"] == "chabrier"
    assert sed.info["stellar.bc03_equivalent_imf"] == 1
    assert np.isclose(sed.info["stellar.metallicity"], 0.02)
    assert np.isclose(sed.info["stellar.fsps.equivalent_metallicity"], 0.02)
    assert np.isclose(
        sed.info["stellar.n_ly"],
        sed.info["stellar.n_ly_young"] + sed.info["stellar.n_ly_old"],
    )
    assert np.isclose(
        sed.info["stellar.lum"],
        sed.info["stellar.lum_young"] + sed.info["stellar.lum_old"],
    )
    assert np.isclose(
        sed.info["stellar.m_star"],
        sed.info["stellar.m_star_young"] + sed.info["stellar.m_star_old"],
    )


@pytest.mark.cigale_fsps
def test_experimental_fsps_stellar_works_before_nebular_smoke():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")

    pytest.importorskip("pcigale")
    pytest.importorskip("fsps")

    register_cigale_fsps_stellar_module()

    from pcigale.warehouse import SedWarehouse

    warehouse = SedWarehouse(nocache=["fsps_stellar"])
    sed = warehouse.get_sed(
        ["sfhdelayed", "fsps_stellar", "nebular", "redshifting"],
        [
            {"age_main": 50, "tau_main": 30.0, "normalise": True},
            {"logzsol": -0.3, "separation_age": 10},
            {"logU": -2.0, "zgas": 0.014, "emission": True},
            {"redshift": 0.1},
        ],
    )

    # CIGALE v2022 names these lines/continuum, while newer versions may use
    # emission labels. The key point is that the downstream nebular module ran.
    assert any(name.startswith("nebular.") for name in sed.luminosities)
    assert any(name.endswith("_young") for name in sed.luminosities)
    assert any(name.endswith("_old") for name in sed.luminosities)
    assert np.all(np.isfinite(sed.luminosity))
    assert np.any(sed.luminosity > 0.0)


@pytest.mark.cigale_fsps
def test_experimental_fsps_stellar_zero_young_component_has_zero_young_bookkeeping():
    if not os.environ.get("SPS_HOME"):
        pytest.skip("SPS_HOME is not configured.")

    pytest.importorskip("pcigale")
    pytest.importorskip("fsps")

    register_cigale_fsps_stellar_module()

    from pcigale.warehouse import SedWarehouse

    warehouse = SedWarehouse(nocache=["fsps_stellar"])
    sed = warehouse.get_sed(
        ["sfhdelayed", "fsps_stellar"],
        [
            {"age_main": 100, "tau_main": 50.0, "normalise": True},
            {"logzsol": 0.0, "separation_age": 0},
        ],
    )

    assert sed.info["stellar.n_ly_young"] == pytest.approx(0.0)
    assert sed.info["stellar.lum_young"] == pytest.approx(0.0)
    assert sed.info["stellar.m_star_young"] == pytest.approx(0.0)
    assert np.allclose(sed.luminosities["stellar.young"], 0.0)
