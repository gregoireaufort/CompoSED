import numpy as np

from composed.parameters import ParameterSpace
from composed.priors import DeltaPrior, LogUniformPrior, NormalPrior, UniformPrior


def test_prior_sampling_and_logpdf():
    rng = np.random.default_rng(123)

    u = UniformPrior(-1.0, 2.0)
    us = u.sample(rng, size=1000)
    assert np.all((us >= -1.0) & (us <= 2.0))
    assert np.isclose(u.logpdf(0.0), -np.log(3.0))
    assert not np.isfinite(u.logpdf(3.0))

    n = NormalPrior(1.0, 2.0)
    assert n.sample(rng, size=8).shape == (8,)
    assert np.isclose(n.logpdf(1.0), -np.log(2.0) - 0.5 * np.log(2.0 * np.pi))

    lu = LogUniformPrior(1.0, 100.0)
    lus = lu.sample(rng, size=1000)
    assert np.all((lus >= 1.0) & (lus <= 100.0))
    assert np.isclose(lu.logpdf(10.0), -np.log(np.log(100.0)) - np.log(10.0))
    assert not np.isfinite(lu.logpdf(0.0))

    d = DeltaPrior(4.0)
    assert d.sample(rng) == 4.0
    assert np.all(d.sample(rng, size=3) == 4.0)
    assert d.logpdf(4.0) == 0.0
    assert not np.isfinite(d.logpdf(4.1))


def test_prior_boundaries_are_inclusive_and_nonfinite_inputs_rejected():
    uniform = UniformPrior(0.0, 1.0)
    assert np.isfinite(uniform.logpdf(0.0))
    assert np.isfinite(uniform.logpdf(1.0))
    assert not np.isfinite(uniform.logpdf(np.nan))

    log_uniform = LogUniformPrior(1.0, 10.0)
    assert np.isfinite(log_uniform.logpdf(1.0))
    assert np.isfinite(log_uniform.logpdf(10.0))
    assert not np.isfinite(log_uniform.logpdf(np.inf))

    normal = NormalPrior(0.0, 1.0)
    assert not np.isfinite(normal.logpdf(np.nan))

    delta = DeltaPrior(2.0)
    assert delta.logpdf(2.0) == 0.0
    assert not np.isfinite(delta.logpdf(np.inf))


def test_parameter_dict_vector_roundtrip_and_prior_sampling():
    ps = ParameterSpace(
        names=["z", "log10_mass"],
        priors={"z": UniformPrior(0.0, 2.0), "log10_mass": NormalPrior(10.0, 0.5)},
    )
    theta = np.array([0.7, 10.2])
    params = ps.to_dict(theta)
    assert params == {"z": 0.7, "log10_mass": 10.2}
    assert np.allclose(ps.from_dict(params), theta)
    assert np.isfinite(ps.log_prior(theta))
    samples = ps.sample_prior(5, np.random.default_rng(7))
    assert samples.shape == (5, 2)


def test_prior_samples_have_finite_log_prior_where_expected():
    ps = ParameterSpace(
        names=["a", "b", "c", "d"],
        priors={
            "a": UniformPrior(-2.0, 2.0),
            "b": NormalPrior(0.0, 1.0),
            "c": LogUniformPrior(0.1, 10.0),
            "d": DeltaPrior(3.0),
        },
    )
    samples = ps.sample_prior(128, np.random.default_rng(1))
    assert np.all([np.isfinite(ps.log_prior(row)) for row in samples])


def test_parameter_space_ordering_is_deterministic():
    ps = ParameterSpace(
        names=["z", "dust2", "log10_mass"],
        priors={"log10_mass": DeltaPrior(10.0), "z": DeltaPrior(0.5), "dust2": DeltaPrior(0.1)},
    )
    params = {"log10_mass": 10.0, "dust2": 0.1, "z": 0.5}
    theta = ps.from_dict(params)
    assert np.allclose(theta, [0.5, 0.1, 10.0])
    assert list(ps.to_dict(theta)) == ["z", "dust2", "log10_mass"]


def test_parameter_space_rejects_duplicate_names():
    try:
        ParameterSpace(names=["z", "z"], priors={"z": DeltaPrior(0.1)})
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate names to be rejected.")
