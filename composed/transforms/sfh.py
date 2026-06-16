from __future__ import annotations

import numpy as np


def normalize_sfh_to_formed_mass(time_gyr: np.ndarray, sfr_msun_per_yr: np.ndarray) -> np.ndarray:
    time_gyr = np.asarray(time_gyr, dtype=float)
    sfr_msun_per_yr = np.asarray(sfr_msun_per_yr, dtype=float)
    formed_mass = np.trapz(sfr_msun_per_yr, time_gyr) * 1e9
    if not np.isfinite(formed_mass) or formed_mass <= 0.0:
        raise ValueError(f"Invalid formed mass from SFH integral: {formed_mass}")
    return sfr_msun_per_yr / formed_mass
