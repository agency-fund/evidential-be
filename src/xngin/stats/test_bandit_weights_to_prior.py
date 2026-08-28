import numpy as np
import pytest

from xngin.apiserver.routers.common_api_types import PriorTypes
from xngin.stats.bandit_weights_to_prior import (
    bandit_weights_to_beta_prior,
    bandit_weights_to_normal_prior,
    convert_arm_weights_to_prior_params,
)


@pytest.mark.parametrize(
    ("expected_probabilities", "expected_alphas"),
    [
        ([12.5, 12.5, 25, 50], [0.249, 0.249, 0.7999, 1.386]),
        ([25, 75], [0.414, 1.2]),
        ([25, 15, 15, 15, 15, 15], [1.65, 0.688, 0.688, 0.688, 0.688, 0.688]),
        ([33.3, 33.3, 33.4], [1.0, 1.0, 1.0]),
    ],
)
def test_bandit_weights_to_beta_prior(expected_probabilities: list[float], expected_alphas: list[float]):
    alpha, beta = bandit_weights_to_beta_prior(np.array(expected_probabilities))

    assert len(alpha) == len(expected_probabilities)
    assert len(beta) == len(expected_probabilities)
    assert alpha.tolist() == pytest.approx(expected_alphas, rel=1e-2)
    assert beta.tolist() == [1.0] * len(expected_probabilities)


@pytest.mark.parametrize(
    ("expected_probabilities", "expected_mus"),
    [
        ([12.5, 12.5, 25, 50], [-0.194, -0.194, -0.00965, 0.398]),
        ([25, 75], [-0.337, 0.337]),
        ([33.33, 33.33, 33.34], [0.0, 0.0, 0.0]),
        ([10, 20, 30, 40], [-0.226, -0.084, 0.073, 0.237]),
        ([25, 15, 15, 15, 15, 15], [0.135, -0.027, -0.027, -0.027, -0.027, -0.027]),
    ],
)
def test_bandit_weights_to_normal_prior(expected_probabilities: list[float], expected_mus: list[float]):
    mu, sigma = bandit_weights_to_normal_prior(np.array(expected_probabilities))

    assert len(mu) == len(expected_probabilities)
    assert len(sigma) == len(expected_probabilities)
    assert mu.tolist() == pytest.approx(expected_mus, rel=1e-2)
    assert sigma.tolist() == [1.0] * len(expected_probabilities)


@pytest.mark.parametrize(
    ("prior_type", "expected_params"), [(PriorTypes.BETA, [0.414, 1.201]), (PriorTypes.NORMAL, [-0.337, 0.337])]
)
def test_convert_bandit_weights_to_prior_params(prior_type: PriorTypes, expected_params: list[float]):
    arm_weights = [25.0, 75.0]
    params = convert_arm_weights_to_prior_params(arm_weights, prior_type=prior_type)

    assert len(params) == 2
    assert params[0] == pytest.approx(expected_params, rel=1e-2)
    assert params[1] == [1.0] * len(expected_params)
