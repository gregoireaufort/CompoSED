import importlib.util

import numpy as np
import pytest


def test_importing_inftools_diagnostics_is_lightweight():
    import inftools
    import inftools.diagnostics as diagnostics

    assert hasattr(inftools, "run_sbi_diagnostics")
    assert hasattr(diagnostics, "PosteriorSampleSet")


def test_sample_posterior_dataset_normalizes_shapes():
    from inftools.diagnostics import sample_posterior_dataset

    class FakeEstimator:
        def sample(self, x_obs, num_samples):
            x_obs = np.asarray(x_obs)
            if x_obs.ndim == 1 or x_obs.shape[0] == 1:
                return np.zeros((num_samples, 2))
            return np.zeros((x_obs.shape[0], num_samples, 2))

    x = np.ones((3, 4))
    samples = sample_posterior_dataset(FakeEstimator(), x, num_samples=5, batch_size=1)
    assert samples.shape == (3, 5, 2)


def test_rank_statistics_and_coverage_have_controlled_toy_values():
    from inftools.diagnostics import marginal_coverage, rank_statistics

    samples = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ]
    )
    truth = np.array([[1.5], [12.5]])

    ranks = rank_statistics(samples, truth)
    assert ranks["ranks"].shape == (2, 1)
    assert np.all(ranks["ranks"] == np.array([[2], [3]]))
    assert np.allclose(ranks["rank_percentiles"], np.array([[0.5], [0.75]]))

    coverage = marginal_coverage(samples, truth, levels=[0.5, 0.9])
    assert coverage["coverage"].shape == (2, 1)
    assert np.all((coverage["coverage"] >= 0.0) & (coverage["coverage"] <= 1.0))


def test_prediction_summary_residuals():
    from inftools.diagnostics import prediction_summary

    samples = np.array([[[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]]])
    truth = np.array([[1.0, 11.0]])
    summary = prediction_summary(samples, truth)
    assert summary["median"].shape == (1, 2)
    assert np.allclose(summary["median"], [[2.0, 12.0]])
    assert np.allclose(summary["residual_median"], [[1.0, 1.0]])


def test_run_sbi_diagnostics_accepts_precomputed_samples():
    from inftools.diagnostics import run_sbi_diagnostics

    samples = np.random.default_rng(1).normal(size=(4, 20, 2))
    truth = np.zeros((4, 2))
    result = run_sbi_diagnostics(
        posterior_samples=samples,
        theta_true=truth,
        theta_names=["z", "log10_mass"],
        make_plots=False,
    )
    assert result["sample_set"].samples.shape == (4, 20, 2)
    assert "ranks" in result
    assert "coverage" in result


def test_tarp_path_is_optional_and_helpful_when_missing():
    if importlib.util.find_spec("tarp") is not None:
        pytest.skip("tarp is installed in this environment.")

    from inftools.diagnostics import tarp_coverage

    samples = np.zeros((1, 4, 1))
    truth = np.zeros((1, 1))
    with pytest.raises(ImportError, match="TARP diagnostics"):
        tarp_coverage(samples, truth)
