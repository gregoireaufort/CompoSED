from __future__ import annotations

from typing import Sequence

from composed.experimental.jaxcigale.modules import (
    analytic_stellar_module,
    calzetti_attenuation_module,
    delayed_sfh_cosmic_time_module,
    delayed_sfh_module,
    dsps_stellar_module,
    madau_igm_module,
    modified_starburst_attenuation_module,
    modified_blackbody_dust_module,
    no_nebular_module,
    redshift_module,
)


def restricted_cigale_modules_to_jaxcigale(
    modules: Sequence[str],
    *,
    age_grid_gyr,
    ssp_data=None,
    include_igm: bool = True,
) -> list[object]:
    """Translate one restricted CIGALE-like chain to JAX-CIGALE module specs.

    This is intentionally conservative. It preserves the order and spirit of a
    CIGALE chain, but it does not call pcigale and it does not claim exact
    parameter-name parity. The first supported SFH parameter names are
    ``tau_gyr`` and ``tage_gyr``.
    """

    out = []
    for module in modules:
        name = str(module).lower()
        if name.startswith("sfhdelayed"):
            if ssp_data is None:
                out.append(delayed_sfh_module(age_grid_gyr=age_grid_gyr))
            else:
                out.append(delayed_sfh_cosmic_time_module(n_time=len(age_grid_gyr)))
        elif name in {"dsps", "fsps_stellar", "bc03", "stellar"}:
            if ssp_data is None:
                out.append(analytic_stellar_module())
            else:
                out.append(dsps_stellar_module(ssp_data))
        elif name == "nebular":
            out.append(no_nebular_module())
        elif name.startswith("dustatt"):
            if "modified_starburst" in name:
                out.append(modified_starburst_attenuation_module())
            else:
                out.append(calzetti_attenuation_module())
        elif name.startswith("dust") and "att" not in name:
            out.append(modified_blackbody_dust_module())
        elif name in {"igm", "madau"}:
            out.append(madau_igm_module())
        elif name == "redshifting":
            if include_igm and not any(getattr(spec, "name", "") == "madau_igm" for spec in out):
                out.append(madau_igm_module())
            out.append(redshift_module())
        else:
            raise ValueError(f"Unsupported restricted JAX-CIGALE module: {module!r}")
    return out
