import numpy as np
import pytest

from inftools.transforms import BoxLogitTransform, box_logit_transform_from_parameter_space
from composed.parameters import ParameterSpace
from composed.priors import ChoicePrior, NormalPrior, UniformPrior


def test_box_logit_transform_roundtrip_and_rowwise_jacobian():
    transform = BoxLogitTransform(low=[0.0, 10.0], high=[1.0, 20.0])
    theta = np.asarray([[0.25, 12.0], [0.75, 18.0]])

    y = transform.theta_to_y(theta)
    recovered = transform.y_to_theta(y)
    log_jac = transform.log_abs_det_jac_each(y)

    assert np.allclose(recovered, theta)
    assert log_jac.shape == (2,)
    assert np.all(np.isfinite(log_jac))
    assert np.isclose(transform.log_abs_det_jac(y[0]), log_jac[0])


def test_box_logit_transform_from_parameter_space_uses_named_bounds():
    space = ParameterSpace(
        names=("z", "template", "dust"),
        priors={
            "z": UniformPrior(0.0, 5.0),
            "template": ChoicePrior([0.0, 1.0]),
            "dust": UniformPrior(0.0, 2.0),
        },
    )

    transform = box_logit_transform_from_parameter_space(space, names=("z", "dust"))

    assert transform.names == ("z", "dust")
    assert np.allclose(transform.low, [0.0, 0.0])
    assert np.allclose(transform.high, [5.0, 2.0])


def test_box_logit_transform_from_parameter_space_rejects_unbounded_prior():
    space = ParameterSpace(names=("x",), priors={"x": NormalPrior(0.0, 1.0)})

    with pytest.raises(ValueError, match="finite low/high bounds"):
        box_logit_transform_from_parameter_space(space)
