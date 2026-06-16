import math

import pytest

from composed.experimental.cigale_fsps_stellar_conventions import (
    FSPS_SOLAR_METALLICITY,
    cigale_bc03_imf_to_fsps_imf_type,
    cigale_metallicity_to_fsps_logzsol,
    fsps_imf_type_to_cigale_bc03_imf,
    fsps_logzsol_to_cigale_metallicity,
    fsps_parameters_from_cigale_bc03,
    nearest_cigale_nebular_zgas,
)


def test_bc03_imf_mapping_is_explicit():
    assert cigale_bc03_imf_to_fsps_imf_type(0) == 0
    assert cigale_bc03_imf_to_fsps_imf_type(1) == 1
    assert fsps_imf_type_to_cigale_bc03_imf(0) == 0
    assert fsps_imf_type_to_cigale_bc03_imf(1) == 1


@pytest.mark.parametrize("bad_imf", [-1, 2, 99])
def test_unsupported_bc03_imf_raises(bad_imf):
    with pytest.raises(ValueError, match="no explicit FSPS mapping"):
        cigale_bc03_imf_to_fsps_imf_type(bad_imf)


def test_fsps_imf_without_bc03_counterpart_raises():
    with pytest.raises(ValueError, match="no direct CIGALE BC03 IMF mapping"):
        fsps_imf_type_to_cigale_bc03_imf(2)


@pytest.mark.parametrize("metallicity", [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])
def test_metallicity_roundtrip(metallicity):
    logzsol = cigale_metallicity_to_fsps_logzsol(metallicity)
    assert fsps_logzsol_to_cigale_metallicity(logzsol) == pytest.approx(metallicity)


def test_metallicity_mapping_uses_explicit_solar_convention():
    assert FSPS_SOLAR_METALLICITY == 0.02
    assert cigale_metallicity_to_fsps_logzsol(0.02, z_sun=0.02) == pytest.approx(0.0)
    assert cigale_metallicity_to_fsps_logzsol(0.004, z_sun=0.02) == pytest.approx(math.log10(0.2))
    assert fsps_logzsol_to_cigale_metallicity(-1.0, z_sun=0.0142) == pytest.approx(0.00142)


@pytest.mark.parametrize("bad_value", [0.0, -0.01, float("nan"), float("inf")])
def test_bad_metallicity_values_raise(bad_value):
    with pytest.raises(ValueError):
        cigale_metallicity_to_fsps_logzsol(bad_value)
    with pytest.raises(ValueError):
        nearest_cigale_nebular_zgas(bad_value)


def test_nearest_nebular_zgas_mapping():
    assert nearest_cigale_nebular_zgas(0.02) == pytest.approx(0.019)
    assert nearest_cigale_nebular_zgas(0.004) == pytest.approx(0.004)
    assert nearest_cigale_nebular_zgas(0.05) == pytest.approx(0.051)


def test_fsps_parameter_mapping_from_cigale_bc03():
    mapping = fsps_parameters_from_cigale_bc03(imf=1, metallicity=0.008)
    assert mapping.imf_type == 1
    assert mapping.logzsol == pytest.approx(math.log10(0.008 / 0.02))
    assert mapping.metallicity == pytest.approx(0.008)
    assert mapping.z_sun == pytest.approx(0.02)
    assert mapping.zgas == pytest.approx(0.008)
