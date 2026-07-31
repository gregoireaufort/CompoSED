from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from composed.data import SEDDataset, SpectroPhotometricDataset, SpectrumDataset
from composed.likelihood import _backend_params_and_mass_scale
from composed.problem import Problem
from composed.results import InferenceResult, require_result_matches_problem
from composed.units import convert_photometric_flux


def plot_corner_hexbin(
    result: InferenceResult,
    *,
    parameters: Sequence[str] | None = None,
    true_values: Mapping[str, float] | Sequence[float] | None = None,
    comparison_result: InferenceResult | None = None,
    result_label: str = "posterior",
    comparison_label: str = "comparison",
    comparison_color: str = "tab:orange",
    max_points: int = 50_000,
    gridsize: int = 35,
    bins: int = 40,
    seed: int | None = 0,
):
    """Corner-style posterior plot using histograms and hexbins.

    When ``comparison_result`` is supplied, both posteriors use the same
    parameter limits. The primary posterior remains a filled histogram/hexbin;
    the comparison is overlaid as a density histogram and weighted
    highest-density contours. This is useful for comparing an amortized
    posterior against a reference Monte Carlo run.
    """

    plt = _require_matplotlib()
    indices = _parameter_indices(result, parameters)
    names = [result.parameter_names[i] for i in indices]
    samples = result.samples[:, indices]
    weights = result.weights
    draw = _resample_indices(weights, min(int(max_points), samples.shape[0]), seed=seed)
    shown = samples[draw]

    comparison_samples = None
    comparison_weights = None
    limits = None
    if comparison_result is not None:
        comparison_indices = _parameter_indices(comparison_result, names)
        comparison_samples = comparison_result.samples[:, comparison_indices]
        comparison_weights = comparison_result.weights
        limits = _shared_corner_limits(samples, comparison_samples)

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
                histogram_kwargs = {}
                if limits is not None:
                    histogram_kwargs = {"range": limits[col], "density": True}
                ax.hist(
                    shown[:, col],
                    bins=bins,
                    color="0.25",
                    alpha=0.75 if comparison_result is not None else 0.85,
                    label=result_label if comparison_result is not None else None,
                    **histogram_kwargs,
                )
                if comparison_result is not None:
                    ax.hist(
                        comparison_samples[:, col],
                        bins=bins,
                        range=limits[col],
                        weights=comparison_weights,
                        density=True,
                        histtype="step",
                        color=comparison_color,
                        linewidth=1.8,
                        label=comparison_label,
                    )
                    ax.set_xlim(*limits[col])
                if truths is not None:
                    ax.axvline(truths[col], color="tab:red", lw=1.5)
            else:
                hexbin_kwargs = {}
                if limits is not None:
                    hexbin_kwargs["extent"] = (
                        limits[col][0],
                        limits[col][1],
                        limits[row][0],
                        limits[row][1],
                    )
                hb = ax.hexbin(
                    shown[:, col],
                    shown[:, row],
                    gridsize=gridsize,
                    mincnt=1,
                    cmap="viridis",
                    **hexbin_kwargs,
                )
                if comparison_result is not None:
                    _plot_weighted_credible_contours(
                        ax,
                        comparison_samples[:, col],
                        comparison_samples[:, row],
                        comparison_weights,
                        xlim=limits[col],
                        ylim=limits[row],
                        gridsize=gridsize,
                        color=comparison_color,
                    )
                    ax.set_xlim(*limits[col])
                    ax.set_ylim(*limits[row])
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

    if comparison_result is not None:
        axes[0, 0].legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return fig, axes


def plot_effective_prior(
    training_set,
    parameter_space,
    *,
    parameters: Sequence[str] | None = None,
    reference_size: int = 50_000,
    max_points: int = 50_000,
    seed: int | None = 0,
):
    """Compare accepted simulator parameters with the declared prior.

    This plot is specifically for simulator-generated SBI training sets.
    Rejected forward-model rows are absent from ``training_set.theta_full``;
    their replacement can therefore introduce correlations or truncate
    marginal support even when the declared :class:`ParameterSpace` factors.
    The accepted table is shown as the filled distribution and fresh declared
    prior draws as orange contours.
    """

    from composed.parameters import ParameterSpace
    from composed.sbi import SBITrainingSet

    if not isinstance(training_set, SBITrainingSet):
        raise TypeError("plot_effective_prior requires an SBITrainingSet.")
    if not isinstance(parameter_space, ParameterSpace):
        raise TypeError("plot_effective_prior requires a ParameterSpace.")
    full_names = tuple(training_set.full_parameter_names)
    if full_names != tuple(parameter_space.names):
        raise ValueError(
            "SBITrainingSet.full_parameter_names must exactly match the "
            "declared ParameterSpace order."
        )
    accepted = np.asarray(training_set.theta_full, dtype=float)
    if accepted.ndim != 2 or accepted.shape[1] != len(full_names):
        raise ValueError("SBITrainingSet.theta_full does not match its parameter names.")
    reference_size = int(reference_size)
    if reference_size <= 0:
        raise ValueError("reference_size must be positive.")

    rng = np.random.default_rng(seed)
    declared = parameter_space.sample_prior(reference_size, rng=rng)
    accepted_result = InferenceResult(
        samples=accepted,
        logp=None,
        weights=np.ones(accepted.shape[0], dtype=float),
        parameter_names=full_names,
        sampler_name="effective_sbi_prior",
    )
    declared_result = InferenceResult(
        samples=declared,
        logp=None,
        weights=np.ones(declared.shape[0], dtype=float),
        parameter_names=full_names,
        sampler_name="declared_prior",
    )
    fig, axes = plot_corner_hexbin(
        accepted_result,
        parameters=parameters,
        comparison_result=declared_result,
        result_label="accepted simulations",
        comparison_label="declared prior",
        max_points=max_points,
        seed=seed,
    )

    simulation_metadata = training_set.metadata.get("simulate_training_set", {})
    acceptance = simulation_metadata.get("acceptance_fraction")
    failures = simulation_metadata.get(
        "n_failures",
        len(simulation_metadata.get("failures", ())),
    )
    if acceptance is None:
        title = "Effective SBI training prior"
    else:
        title = (
            "Effective SBI training prior "
            f"(acceptance {float(acceptance):.1%}, failures {int(failures)})"
        )
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
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
    """Low-level posterior-predictive plot from explicitly supplied pieces.

    Spectra and photometry are shown in separate panels because their native
    units are usually different. Photometric model points are the posterior
    predictive fluxes in the same units as the photometric likelihood. Prefer
    :func:`plot_posterior_predictive` for fitted results because that public
    path validates and reuses the complete ``Problem``.
    """

    if photometry is None and spectrum is None and wavelengths is None:
        raise ValueError("Provide photometry, spectrum, or explicit wavelengths.")
    if int(n_draw) <= 0:
        raise ValueError("n_draw must be positive.")
    plt = _require_matplotlib()
    rng = np.random.default_rng(seed)
    draw = _resample_indices(result.weights, int(n_draw), rng=rng)
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
        flux_unit = photometry.flux_unit if photometry is not None else "backend units"
        ax.set_ylabel(f"flux [{flux_unit}]")
        if xlabel == "band":
            ax.set_xticks(x)
            ax.set_xticklabels(band_names, rotation=35, ha="right")
        ax.legend()

    fig.tight_layout()
    return fig, axes


def plot_posterior_predictive(
    result: InferenceResult,
    problem: Problem,
    *,
    wavelengths: Sequence[float] | None = None,
    photometry_wavelengths: Sequence[float] | None = None,
    n_draw: int = 200,
    seed: int | None = 0,
):
    """Plot posterior predictions from the exact fitted :class:`Problem`.

    The result fingerprint is checked before any backend call. Data, filters,
    parameter order, parameter transforms, mass normalization, and
    photometric units therefore come from the same scientific specification
    that produced the posterior.
    """

    if not isinstance(problem, Problem):
        raise TypeError("problem must be a composed.Problem.")
    require_result_matches_problem(result, problem)
    if tuple(result.parameter_names) != tuple(problem.parameters.names):
        missing = [
            name for name in problem.parameters.names if name not in result.parameter_names
        ]
        raise ValueError(
            "Posterior-predictive SEDs require samples for every Problem parameter. "
            "This result marginalizes or omits: "
            + ", ".join(missing)
        )

    if isinstance(problem.data, SpectroPhotometricDataset):
        photometry = problem.data.photometry
        spectrum = problem.data.spectrum
    elif isinstance(problem.data, SEDDataset):
        photometry, spectrum = problem.data, None
    elif isinstance(problem.data, SpectrumDataset):
        photometry, spectrum = None, problem.data
    else:  # pragma: no cover - guarded by Problem construction
        raise TypeError("Unsupported Problem data type.")

    filters = problem.filters
    if filters is None and photometry is not None:
        filters = photometry.metadata.get("filters")
    return plot_posterior_predictive_sed(
        result,
        problem.evaluation_backend,
        problem.parameters,
        photometry=photometry,
        filters=filters,
        spectrum=spectrum,
        wavelengths=wavelengths,
        photometry_wavelengths=photometry_wavelengths,
        n_draw=n_draw,
        seed=seed,
    )


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
            flux = convert_photometric_flux(
                flux,
                getattr(model, "flux_unit", "maggies"),
                photometry.flux_unit,
            )
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


def _shared_corner_limits(primary_samples, comparison_samples, quantiles=(0.001, 0.999)):
    """Return robust shared plotting limits for two posterior sample arrays."""

    primary = np.asarray(primary_samples, dtype=float)
    comparison = np.asarray(comparison_samples, dtype=float)
    if primary.ndim != 2 or comparison.ndim != 2 or primary.shape[1] != comparison.shape[1]:
        raise ValueError("Corner comparison samples must be two-dimensional with matching columns.")

    limits = []
    for column in range(primary.shape[1]):
        values = np.concatenate([primary[:, column], comparison[:, column]])
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Corner comparison contains a parameter with no finite samples.")
        low, high = np.quantile(values, quantiles)
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("Corner comparison limits are non-finite.")
        if high <= low:
            scale = max(abs(float(low)), 1.0)
            low, high = float(low) - 0.05 * scale, float(high) + 0.05 * scale
        padding = 0.03 * (high - low)
        limits.append((float(low - padding), float(high + padding)))
    return tuple(limits)


def _plot_weighted_credible_contours(
    ax,
    x,
    y,
    weights,
    *,
    xlim,
    ylim,
    gridsize,
    color,
    credible_masses=(0.50, 0.80, 0.95),
):
    """Overlay weighted highest-density contours from one reference posterior."""

    histogram, x_edges, y_edges = np.histogram2d(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        bins=int(gridsize),
        range=(xlim, ylim),
        weights=np.asarray(weights, dtype=float),
    )
    total = float(np.sum(histogram))
    maximum = float(np.max(histogram))
    if not np.isfinite(total) or total <= 0.0 or maximum <= 0.0:
        return

    ordered = np.sort(histogram.ravel())[::-1]
    cumulative = np.cumsum(ordered) / total
    thresholds = []
    for mass in credible_masses:
        index = min(int(np.searchsorted(cumulative, float(mass), side="left")), ordered.size - 1)
        thresholds.append(float(ordered[index]))
    levels = np.unique(np.asarray(thresholds, dtype=float))
    levels = levels[(levels > 0.0) & (levels < maximum)]
    if levels.size == 0:
        return

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ax.contour(
        x_centers,
        y_centers,
        histogram.T,
        levels=levels,
        colors=color,
        linewidths=1.2,
    )


def _resample_indices(weights, n, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have positive total mass.")
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    weights = weights / np.sum(weights)
    # These are independent draws from the weighted empirical posterior.
    # Sampling without replacement would flatten unequal weights whenever all
    # support points fit in the requested plotting sample.
    return rng.choice(np.arange(weights.size), size=int(n), replace=True, p=weights)


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
