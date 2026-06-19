from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from composed.backends.mock import MockBackend
from composed.data import SEDDataset, SpectrumDataset
from composed.parameters import ParameterSpace
from composed.priors import UniformPrior
from composed.results import InferenceResult


pytestmark = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed")


def _toy_result():
    return InferenceResult(
        samples=np.asarray([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0], [1.5, 2.5]]),
        logp=np.asarray([-2.0, -1.0, 0.0, -0.5]),
        weights=np.ones(4),
        parameter_names=("z", "log10_mass"),
        sampler_name="toy_mcmc",
        chain=np.asarray([[[0.0, 1.0]], [[0.5, 1.5]], [[1.0, 2.0]]]),
    )


def test_corner_hexbin_and_trace_plots_return_figures():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from composed.plot import plot_corner_hexbin, plot_traces

    result = _toy_result()
    fig1, axes1 = plot_corner_hexbin(result, true_values={"z": 1.0, "log10_mass": 2.0})
    fig2, axes2 = plot_traces(result)

    assert axes1.shape == (2, 2)
    assert len(np.atleast_1d(axes2)) == 2
    fig1.canvas.draw()
    fig2.canvas.draw()


def test_posterior_predictive_sed_plots_photometry_and_spectrum():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from composed.plot import plot_posterior_predictive_sed

    result = _toy_result()
    space = ParameterSpace(
        names=("z", "log10_mass"),
        priors={"z": UniformPrior(0.0, 3.0), "log10_mass": UniformPrior(0.0, 4.0)},
    )
    backend = MockBackend(
        flux=[1.0, 2.0],
        band_names=("u", "g"),
        spectrum_wavelength=[4000.0, 5000.0, 6000.0],
        spectrum_flux=[1e-17, 2e-17, 1.5e-17],
    )
    photometry = SEDDataset(
        band_names=("u", "g"),
        flux=np.asarray([1.1, 1.9]),
        sigma=np.asarray([0.1, 0.2]),
    )
    spectrum = SpectrumDataset(
        wavelength=np.asarray([4000.0, 5000.0, 6000.0]),
        flux=np.asarray([1.2e-17, 2.1e-17, 1.4e-17]),
        sigma=np.full(3, 0.2e-17),
    )

    fig, axes = plot_posterior_predictive_sed(
        result,
        backend,
        space,
        photometry=photometry,
        filters=("u", "g"),
        spectrum=spectrum,
        n_draw=3,
    )

    assert len(axes) == 2
    fig.canvas.draw()


def test_posterior_predictive_sed_plots_upper_limits_separately_from_detections():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from composed.plot import plot_posterior_predictive_sed

    result = _toy_result()
    space = ParameterSpace(
        names=("z", "log10_mass"),
        priors={"z": UniformPrior(0.0, 3.0), "log10_mass": UniformPrior(0.0, 4.0)},
    )
    backend = MockBackend(flux=[1.0, 0.5], band_names=("g", "fuv"))
    photometry = SEDDataset(
        band_names=("g", "fuv"),
        flux=np.asarray([1.1, np.nan]),
        sigma=np.asarray([0.1, 0.2]),
        upper_limit=np.asarray([0.0, 1.0]),
        upper_limit_mask=np.asarray([False, True]),
    )

    fig, axes = plot_posterior_predictive_sed(
        result,
        backend,
        space,
        photometry=photometry,
        filters=("g", "fuv"),
        n_draw=3,
    )

    ax = np.atleast_1d(axes)[0]
    labels = ax.get_legend_handles_labels()[1]
    assert "observed detection" in labels
    assert "upper limit" in labels
    detection = [container for container in ax.containers if container.get_label() == "observed detection"]
    upper = [container for container in ax.containers if container.get_label() == "upper limit"]
    assert len(detection) == 1
    assert len(upper) == 1
    assert len(detection[0].lines[0].get_xdata()) == 1
    assert len(upper[0].lines[0].get_xdata()) == 1
    fig.canvas.draw()
