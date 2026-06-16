from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

PARAM_NAMES = (
    "N",
    "log10Z",
    "logsfr_ratio1",
    "logsfr_ratio2",
    "logsfr_ratio3",
    "logsfr_ratio4",
    "logsfr_ratio5",
    "logsfr_ratio6",
    "dust2",
    "dust_index",
    "dust1_fraction",
    "lnfagn",
    "lnagntau",
    "gaslog10Z",
    "gaslog10U",
    "z",
)

DEFAULT_AGEBINS_TEMPLATE = np.array(
    [
        [0.0, 7.0],
        [7.0, 8.0],
        [8.0, 8.7],
        [8.7, 9.2],
        [9.2, 9.6],
        [9.6, 9.9],
        [9.9, 10.14],
    ],
    dtype=float,
)


def theta_row_to_dict(theta_row: Sequence[float], names: Sequence[str] = PARAM_NAMES) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, theta_row)}


def n_to_log10m_formed(n_value: float, z: float, cosmology=None) -> float:
    if cosmology is None:
        from astropy.cosmology import Planck18 as cosmo
    else:
        cosmo = cosmology
    return float(-0.4 * (float(n_value) - cosmo.distmod(float(z)).value))


def zred_to_agebins(zred: float, agebins_template: np.ndarray | None = None, cosmology=None) -> np.ndarray:
    if cosmology is None:
        from astropy.cosmology import Planck18 as cosmo
    else:
        cosmo = cosmology
    if agebins_template is None:
        agebins_template = DEFAULT_AGEBINS_TEMPLATE
    agebins_template = np.asarray(agebins_template, dtype=float)
    tuniv = cosmo.age(float(zred)).value * 1e9
    tbinmax = 0.85 * tuniv
    ncomp = len(agebins_template)
    agelims = (
        list(agebins_template[0])
        + np.linspace(agebins_template[1, 1], np.log10(tbinmax), ncomp - 2).tolist()
        + [np.log10(tuniv)]
    )
    return np.array([agelims[:-1], agelims[1:]], dtype=float).T


def logsfr_ratios_to_masses(logmass: float, logsfr_ratios: Sequence[float], agebins: np.ndarray) -> np.ndarray:
    logsfr_ratios = np.asarray(logsfr_ratios, dtype=float)
    agebins = np.asarray(agebins, dtype=float)
    nbins = agebins.shape[0]
    if logsfr_ratios.shape != (nbins - 1,):
        raise ValueError(f"Expected {nbins - 1} logsfr ratios, got {logsfr_ratios.shape}.")
    sratios = 10.0 ** np.clip(logsfr_ratios, -10.0, 10.0)
    dt = 10.0 ** agebins[:, 1] - 10.0 ** agebins[:, 0]
    coeffs = np.array(
        [
            (1.0 / np.prod(sratios[:i])) * (np.prod(dt[1 : i + 1]) / np.prod(dt[:i]))
            for i in range(nbins)
        ],
        dtype=float,
    )
    return (10.0**float(logmass) / coeffs.sum()) * coeffs


def logsfr_ratios_to_sfrs(logmass: float, logsfr_ratios: Sequence[float], agebins: np.ndarray) -> np.ndarray:
    masses = logsfr_ratios_to_masses(logmass, logsfr_ratios, agebins)
    dt = 10.0 ** agebins[:, 1] - 10.0 ** agebins[:, 0]
    return masses / dt


def popcosmos_theta_to_tabular_sfh(
    theta_row: Sequence[float] | Mapping[str, float],
    agebins_template: np.ndarray | None = None,
    cosmology=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    p = dict(theta_row) if isinstance(theta_row, Mapping) else theta_row_to_dict(theta_row)
    z = float(p["z"])
    log10_mass = n_to_log10m_formed(p["N"], z, cosmology=cosmology)
    logsfr_ratios = np.asarray([p[f"logsfr_ratio{i}"] for i in range(1, 7)], dtype=float)
    agebins = zred_to_agebins(z, agebins_template=agebins_template, cosmology=cosmology)
    sfr_bins = logsfr_ratios_to_sfrs(log10_mass, logsfr_ratios, agebins)

    if cosmology is None:
        from astropy.cosmology import Planck18 as cosmo
    else:
        cosmo = cosmology
    tuniv_yr = cosmo.age(z).value * 1e9

    t_samples = []
    sfr_samples = []
    for i in range(len(sfr_bins) - 1, -1, -1):
        age_lo = 10.0 ** agebins[i, 0]
        age_hi = 10.0 ** agebins[i, 1]
        t_start = max(0.0, min(tuniv_yr - age_hi, tuniv_yr))
        t_end = max(0.0, min(tuniv_yr - age_lo, tuniv_yr))
        if t_end <= t_start:
            continue
        tt = np.linspace(t_start, t_end, 8, endpoint=(i == 0))
        t_samples.append(tt)
        sfr_samples.append(np.full_like(tt, sfr_bins[i], dtype=float))

    if not t_samples:
        raise ValueError("No valid SFH bins were produced.")
    t_yr = np.concatenate(t_samples)
    sfr = np.concatenate(sfr_samples)
    order = np.argsort(t_yr)
    t_yr = t_yr[order]
    sfr = sfr[order]

    keep = np.ones_like(t_yr, dtype=bool)
    keep[1:] = np.diff(t_yr) > 0.0
    t_yr = t_yr[keep]
    sfr = sfr[keep]

    if t_yr.size < 2:
        raise ValueError("Tabular SFH has fewer than 2 distinct time points.")
    if not np.all(np.isfinite(sfr)) or np.any(sfr < 0.0):
        raise ValueError("Tabular SFH contains invalid SFR values.")
    return t_yr / 1e9, sfr, log10_mass


def popcosmos_theta_to_fsps_params(
    theta_row: Sequence[float] | Mapping[str, float],
    use_agn: bool = False,
    agebins_template: np.ndarray | None = None,
    cosmology=None,
) -> dict[str, object]:
    p = dict(theta_row) if isinstance(theta_row, Mapping) else theta_row_to_dict(theta_row)
    t_gyr, sfr_msun_per_yr, log10_mass = popcosmos_theta_to_tabular_sfh(
        p, agebins_template=agebins_template, cosmology=cosmology
    )
    return {
        "zred": float(p["z"]),
        "logzsol": float(p["log10Z"]),
        "dust2": float(p["dust2"]),
        "dust_index": float(p["dust_index"]),
        "dust1": float(p["dust1_fraction"]) * float(p["dust2"]),
        "gas_logz": float(p["gaslog10Z"]),
        "gas_logu": float(p["gaslog10U"]),
        "fagn": float(np.exp(p["lnfagn"])) if use_agn else 0.0,
        "agn_tau": float(np.exp(p["lnagntau"])) if use_agn else 10.0,
        "tabular_time_gyr": t_gyr,
        "tabular_sfr_msun_per_yr": sfr_msun_per_yr,
        "log10_mass_formed": log10_mass,
    }
