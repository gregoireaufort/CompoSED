import sys
import types
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pytest

from composed.backends._cigale_tabular import (
    CIGALE_TABULAR_IMPORT,
    CIGALE_TABULAR_MODULE,
    project_sfh_to_cigale_1myr,
    register_cigale_tabular_module,
    registered_cigale_sfh,
    registry_size,
    sfh_content_hash,
)
from composed.sfh import ConstantSFH, DelayedTauSFH, SFHHistory


def test_constant_sfh_projection_is_constant_and_unit_formed_mass():
    history = ConstantSFH(n_time=11).evaluate({"tage_gyr": 0.010})
    projected = project_sfh_to_cigale_1myr(history)

    assert projected.cigale_age_myr == 10
    assert projected.sfr_msun_per_yr.shape == (10,)
    assert np.allclose(projected.sfr_msun_per_yr, projected.sfr_msun_per_yr[0])
    assert np.sum(projected.sfr_msun_per_yr) * 1.0e6 == pytest.approx(1.0)


def test_projection_preserves_chronological_delayed_tau_shape():
    history = DelayedTauSFH(n_time=1001).evaluate(
        {"tage_gyr": 1.0, "tau_gyr": 0.2}
    )
    projected = project_sfh_to_cigale_1myr(history)

    peak_bin = int(np.argmax(projected.sfr_msun_per_yr))
    assert peak_bin == pytest.approx(200, abs=2)
    assert projected.sfr_msun_per_yr[0] < projected.sfr_msun_per_yr[peak_bin]
    assert projected.sfr_msun_per_yr[-1] < projected.sfr_msun_per_yr[peak_bin]


def test_cigale_projection_converges_as_named_sfh_grid_is_refined():
    params = {"tage_gyr": 5.0, "tau_gyr": 0.07}
    reference = project_sfh_to_cigale_1myr(
        DelayedTauSFH(n_time=32769).evaluate(params)
    ).sfr_msun_per_yr
    errors = []
    for n_time in (65, 257, 2049):
        projected = project_sfh_to_cigale_1myr(
            DelayedTauSFH(n_time=n_time).evaluate(params)
        ).sfr_msun_per_yr
        errors.append(float(np.sum(np.abs(projected - reference)) * 1.0e6))

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 2.0e-4


def test_projection_records_sub_myr_age_quantization():
    history = SFHHistory(
        np.asarray([0.0, 0.0104]),
        np.asarray([1.0, 2.0]),
    )
    projected = project_sfh_to_cigale_1myr(history)

    assert projected.cigale_age_myr == 10
    assert projected.time_scale == pytest.approx(0.010 / 0.0104)
    assert projected.metadata()["source_age_gyr"] == pytest.approx(0.0104)


def test_projection_rejects_history_that_does_not_start_at_onset():
    history = SFHHistory(
        np.asarray([0.001, 0.010]),
        np.asarray([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="start at time 0"):
        project_sfh_to_cigale_1myr(history)


def test_registry_is_content_addressed_and_released():
    sfr = np.asarray([2.0e-7, 3.0e-7, 5.0e-7])
    key = sfh_content_hash(sfr)

    assert registry_size() == 0
    with registered_cigale_sfh(sfr) as registered_key:
        assert registered_key == key
        assert registry_size() == 1
        with registered_cigale_sfh(sfr) as nested_key:
            assert nested_key == key
            assert registry_size() == 1
        assert registry_size() == 1
    assert registry_size() == 0


def test_registered_cigale_module_sets_sfh_and_standard_info(monkeypatch):
    sed_modules = types.ModuleType("pcigale.sed_modules")

    class FakeSedModule:
        def __init__(self, name=None, **kwargs):
            self.name = name
            self.parameters = kwargs
            self._init_code()

    sed_modules.SedModule = FakeSedModule
    pcigale = types.ModuleType("pcigale")
    pcigale.sed_modules = sed_modules
    monkeypatch.setitem(sys.modules, "pcigale", pcigale)
    monkeypatch.setitem(sys.modules, "pcigale.sed_modules", sed_modules)
    monkeypatch.delitem(sys.modules, CIGALE_TABULAR_IMPORT, raising=False)

    class FakeSED:
        def __init__(self):
            self.modules = []
            self.info = {}

        def add_module(self, name, parameters):
            self.modules.append((name, parameters))

        def add_info(self, name, value, *args, **kwargs):
            del args, kwargs
            self.info[name] = value

    register_cigale_tabular_module()
    module = sys.modules[CIGALE_TABULAR_IMPORT]
    sfr = np.asarray([2.0e-7, 3.0e-7, 5.0e-7])
    with registered_cigale_sfh(sfr) as history_hash:
        instance = module.Module(
            name=CIGALE_TABULAR_MODULE,
            history_hash=history_hash,
            normalise=True,
        )
        sed = FakeSED()
        instance.process(sed)

    assert np.allclose(sed.sfh, sfr)
    assert sed.info["sfh.integrated"] == pytest.approx(1.0)
    assert sed.modules[0][0] == CIGALE_TABULAR_MODULE


@pytest.mark.cigale
def test_real_bridge_matches_native_sfhfromfile_array_and_info(tmp_path, monkeypatch):
    pytest.importorskip("pcigale")
    table_module = pytest.importorskip("astropy.table")
    from pcigale.warehouse import SedWarehouse

    # CIGALE v2022.0 predates NumPy's removal of these aliases. They are used
    # only by the trusted sfhfromfile reference path, not by the CompoSED bridge.
    monkeypatch.setattr(np, "float", float, raising=False)
    monkeypatch.setattr(np, "int", int, raising=False)

    sfr = np.asarray([1.0, 2.0, 4.0, 3.0, 2.0], dtype=float)
    sfh_path = tmp_path / "reference_sfh.fits"
    table_module.Table(
        {"time_myr": np.arange(sfr.size, dtype=int), "sfr": sfr}
    ).write(sfh_path)

    reference = SedWarehouse(nocache=["sfhfromfile"]).get_sed(
        ["sfhfromfile"],
        [
            {
                "filename": str(sfh_path),
                "sfr_column": 1,
                "age": sfr.size - 1,
                "normalise": True,
            }
        ],
    )

    normalized_sfr = sfr / (np.sum(sfr) * 1.0e6)
    register_cigale_tabular_module()
    with registered_cigale_sfh(normalized_sfr) as history_hash:
        bridged = SedWarehouse(nocache=[CIGALE_TABULAR_MODULE]).get_sed(
            [CIGALE_TABULAR_MODULE],
            [{"history_hash": history_hash, "normalise": True}],
        )

    assert np.array_equal(bridged.sfh, reference.sfh)
    for name in (
        "sfh.integrated",
        "sfh.sfr",
        "sfh.sfr10Myrs",
        "sfh.sfr100Myrs",
        "sfh.age",
    ):
        assert bridged.info[name] == pytest.approx(reference.info[name], rel=0.0, abs=0.0)


def _evaluate_bridge_in_spawned_process(sfr):
    from pcigale.warehouse import SedWarehouse
    from composed.backends._cigale_tabular import (
        CIGALE_TABULAR_MODULE,
        register_cigale_tabular_module,
        registered_cigale_sfh,
    )

    register_cigale_tabular_module()
    with registered_cigale_sfh(np.asarray(sfr, dtype=float)) as history_hash:
        sed = SedWarehouse(nocache=[CIGALE_TABULAR_MODULE]).get_sed(
            [CIGALE_TABULAR_MODULE],
            [{"history_hash": history_hash, "normalise": True}],
        )
    return sed.sfh, sed.info["sfh.integrated"]


@pytest.mark.cigale
def test_real_bridge_registers_inside_spawned_worker():
    pytest.importorskip("pcigale")
    sfr = np.asarray([2.0e-7, 3.0e-7, 5.0e-7])
    try:
        pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"Spawned-process semaphores are unavailable: {exc}")
    with pool:
        worker_sfr, formed_mass = pool.submit(
            _evaluate_bridge_in_spawned_process,
            sfr,
        ).result(timeout=30.0)

    assert np.array_equal(worker_sfr, sfr)
    assert formed_mass == pytest.approx(1.0)
