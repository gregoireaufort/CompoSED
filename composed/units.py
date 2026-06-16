from __future__ import annotations

from enum import Enum


class MassNormalization(str, Enum):
    PER_SOLAR_MASS = "per_solar_mass"
    ABSOLUTE = "absolute"


LSUN_CGS = 3.828e33
PARSEC_CM = 3.085677581491367e18
