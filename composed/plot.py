from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from composed.data import SEDDataset, SpectrumDataset
from composed.likelihood import _backend_params_and_mass_scale
from composed.results import InferenceResult


def plot_corner_hexbin(
    result: InferenceResult,
    *,
    parameters: Sequence[str] | None = None,
    true_values: Mapping[str, float] | Sequence[float] | None = None,
    max_points: int = 50_000,
    gridsize: int = 35,
    bins: int = 40,
    seed: int | None = 0,
):
    """Corner-style posterior plot using histograms and hexbins."""

    plt = _require_matplotlib()
    indices = _parameter_indices(result, parameters)
    names = [result.parameter_names[i] for i in indices]
    samples = result.samples[:, indices]
    weights = result.weights
    draw = _resample_indices(weights, min(int(max_points), samples.shape[0]), seed=seed)
    shown = samples[draw]

    ndim = shown.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.4 * ndim, 2.4 * ndim), squeeze=False)
    truths = _truth_vector(true_values, result.parameter_names, indices)

    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(shown[:, col], bins=bins, color="0.25", alpha=0.85)
                if truths is not None:
                    ax.axvline(truths[col], color="tab:red", lw=1.5)
            else:
                hb = ax.hexbin(shown[:, col], shown[:, row], gridsize=gridsize, mincnt=1, cmap="viridis")
                if truths is not None:
                    ax.plot(truths[col], truths[row], marker="*", color="tab:red", ms=9)
                if row == ndim - 1 and col == 0:
                    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="resampled count")
            if row == ndim - 1:
                ax.set_xlabel(names[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(names[row])
            elif col > 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig, axes


def plot_traces(result: InferenceResult, *, parameters: Sequence[str] | None = None):
    """Plot MCMC traces when available, otherwise the sample sequence."""

    plt = _require_matplotlib()
    indices = _parameter_indices(result, parameters)
    names = [result.parameter_names[i] for i in indices]
    if result.chain is not None:
        chain = np.asarray(result.chain, dtype=float)
        if chain.ndim == 3:
            y = chain[:, :, indices]
            nstep = y.shape[0]
            fig, axes = plt.subplots(len(indices), 1, figsize=(8, 2.2 * len(indices)), sharex=True)
            axes = np.atleast_1d(axes)
            x = np.arange(nstep)
            for j, ax in enumerate(axes):
                for walker in range(y.shape[1]):
                    ax.plot(x, y[:, walker, j], color="0.2", alpha=0.25, lw=0.8)
                ax.set_ylabel(names[j])
            axes[-1].set_xlabel("step")
            fig.tight_layout()
            return fig, axes

    samples = result.samples[:, indices]
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 2.2 * len(indices)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(samples.shape[0])
    for j, ax in enumerate(axes):
        ax.plot(x, samples[:, j], color="0.2", alpha=0.8, lw=0.8)
        ax.set_ylabel(names[j])
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    return fig, axes


def plot_posterior_predictive_sed(
    result: InferenceResult,
    backend,
    parameter_space,
    *,
    photometry: SEDDataset | None = None,
    filters=None,
    spectrum: SpectrumDataset | None = None,
    wavelengths: Sequence[float] | None = None,
    photometry_wavelengths: Sequence[float] | None = None,
    n_draw: int = 200,
    seed: int | None = 0,
):
    """Plot posterior predictive spectra and/or photometry.

    Spectra and photometry are shown in separate panels because their native
    units are usually different. Photometric model points are the posterior
    predictive fluxes in the same units as the photometric likelihood.
    """

    if photometry is None and spectrum is None and wavelengths is None:
        raise ValueError("Provide photometry, spectrum, or explicit wavelengths.")
    plt = _require_matplotlib()
    rng = np.random.default_rng(seed)
    draw = _resample_indices(result.weights, min(int(n_draw), result.samples.shape[0]), rng=rng)
    theta_draws = result.samples[draw]

    want_spectrum = spectrum is not None or wavelengths is not None
    want_photometry = photometry is not None or filters is not None
    n_panel = int(want_spectrum) + int(want_photometry)
    fig, axes = plt.subplots(n_panel, 1, figsize=(8, 3.3 * n_panel), squeeze=False)
    axes = axes[:, 0]
    panel = 0

    if want_spectrum:
        wave = np.asarray(wavelengths if wavelengths is not None else spectrum.wavelength, dtype=float)
        spectra = _posterior_predictive_spectra(backend, parameter_space, theta_draws, wave)
        median, lo, hi = _central_band(spectra)
        ax = axes[panel]
        ax.fill_between(wave, lo, hi, color="tab:blue", alpha=0.2, label="model 16-84%")
        ax.plot(wave, median, color="tab:blue", lw=1.5, label="model median")
        if spectrum is not None:
            active = spectrum.active_mask
            ax.plot(spectrum.wavelength[active], spectrum.flux[active], color="0.25", lw=0.8, alpha=0.8, label="observed")
        ax.set_xlabel(f"wavelength [{spectrum.wavelength_unit if spectrum is not None else 'angstrom'}]")
        ax.set_ylabel(spectrum.flux_unit if spectrum is not None else "model flux")
        ax.legend()
        panel += 1

    if want_photometry:
        if filters is None and photometry is not None:
            filters = photometry.metadata.get("filters")
        phot = _posterior_predictive_photometry(backend, parameter_space, theta_draws, filters, photometry)
        median, lo, hi = _central_band(phot)
        band_names = tuple(photometry.band_names) if photometry is not None else tuple(str(i) for i in range(phot.shape[1]))
        x, xlabel = _photometry_x(filters, band_names, photometry_wavelengths)
        ax = axes[panel]
        yerr = np.vstack([median - lo, hi - median])
        ax.errorbar(x, median, yerr=yerr, fmt="o", color="tab:blue", label="model photometry")
        if photometry is not None:
            active = photometry.active_mask
            upper_mask = active & np.asarray(photometry.upper_limit_mask, dtype=bool)
            detection_mask = active & ~upper_mask
            if np.any(detection_mask):
                ax.errorbar(
                    x[detection_mask],
                    photometry.flux[detection_mask],
                    yerr=photometry.sigma[detection_mask],
                    fmt="s",
                    color="0.2",
                    label="observed detection",
                )
            if np.any(upper_mask):
                ax.errorbar(
                    x[upper_mask],
                    photometry.upper_limit[upper_mask],
                    yerr=photometry.sigma[upper_mask],
                    uplims=True,
                    fmt="v",
                    color="0.35",
                    label="upper limit",
                )
            if np.any(~active):
                ax.plot(x[~active], photometry.flux[~active], "x", color="0.6", label="masked")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("flux [photometry units]")
        if xlabel == "band":
            ax.set_xticks(x)
            ax.set_xticklabels(band_names, rotation=35, ha="right")
        ax.legend()

    fig.tight_layout()
    return fig, axes


def _posterior_predictive_photometry(backend, parameter_space, theta_draws, filters, photometry):
    rows = []
    band_names = tuple(photometry.band_names) if photometry is not None else None
    for theta in theta_draws:
        params = parameter_space.to_dict(theta)
        backend_params, mass_scale = _backend_params_and_mass_scale(params, backend, quantity_name="photometry")
        try:
            model = backend.predict_photometry(backend_params, filters)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        flux = np.asarray(model.flux, dtype=float)
        if band_names is not None:
            flux = _align_model_flux(model, band_names)
        flux = mass_scale * flux
        if np.all(np.isfinite(flux)):
            rows.append(flux)
    if not rows:
        raise RuntimeError("No finite posterior predictive photometry draws.")
    return np.asarray(rows, dtype=float)


def _posterior_predictive_spectra(backend, parameter_space, theta_draws, wavelengths):
    rows = []
    for theta in theta_draws:
        params = parameter_space.to_dict(theta)
        backend_params, mass_scale = _backend_params_and_mass_scale(params, backend, quantity_name="spectrum")
        try:
            model = backend.predict_spectrum(backend_params, wavelengths=wavelengths)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        flux = mass_scale * np.asarray(model.flux, dtype=float)
        if flux.shape != wavelengths.shape:
            raise ValueError(f"Model spectrum shape {flux.shape}; expected {wavelengths.shape}.")
        if np.all(np.isfinite(flux)):
            rows.append(flux)
    if not rows:
        raise RuntimeError("No finite posterior predictive spectrum draws.")
    return np.asarray(rows, dtype=float)


def _align_model_flux(model, band_names):
    flux = np.asarray(model.flux, dtype=float)
    names = tuple(str(name) for name in getattr(model, "band_names", ()))
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in band_names if name not in lookup]
    if missing:
        raise ValueError(f"Model photometry is missing band(s): {', '.join(missing)}")
    return np.asarray([flux[lookup[name]] for name in band_names], dtype=float)


def _central_band(draws):
    q16, q50, q84 = np.percentile(np.asarray(draws, dtype=float), [16, 50, 84], axis=0)
    return q50, q16, q84


def _parameter_indices(result, parameters):
    if parameters is None:
        return list(range(len(result.parameter_names)))
    lookup = {name: i for i, name in enumerate(result.parameter_names)}
    missing = [name for name in parameters if name not in lookup]
    if missing:
        raise KeyError(f"Unknown parameter(s): {', '.join(missing)}")
    return [lookup[name] for name in parameters]


def _truth_vector(true_values, parameter_names, indices):
    if true_values is None:
        return None
    if isinstance(true_values, Mapping):
        return np.asarray([true_values.get(parameter_names[i], np.nan) for i in indices], dtype=float)
    values = np.asarray(true_values, dtype=float)
    return values[indices]


def _resample_indices(weights, n, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    replace = int(n) > weights.size
    return rng.choice(np.arange(weights.size), size=int(n), replace=replace, p=weights)


def _photometry_x(filters, band_names, photometry_wavelengths):
    if photometry_wavelengths is not None:
        return np.asarray(photometry_wavelengths, dtype=float), "wavelength"
    filter_objects = getattr(filters, "filters", None)
    if filter_objects is not None:
        centers = []
        for filt in filter_objects:
            center = _filter_center(filt)
            if center is None:
                break
            centers.append(center)
        if len(centers) == len(band_names):
            return np.asarray(centers, dtype=float), "wavelength"
    return np.arange(len(band_names), dtype=float), "band"


def _filter_center(filt):
    for name in ("wave_effective", "effective_wavelength", "wave_mean", "pivot", "lambda_eff"):
        if hasattr(filt, name):
            value = getattr(filt, name)
            try:
                return float(value() if callable(value) else value)
            except TypeError:
                continue
    return None


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError("Plotting requires matplotlib. Install matplotlib to use composed.plot.") from exc
    return plt
