import numpy as np
import pytest

from xngin.apiserver.routers.common_enums import (
    ContextLinkFunctions,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)
from xngin.stats.bandit_analysis import _analyze_normal_binary, analyze_experiment  # noqa: PLC2701
from xngin.stats.test_bandit_sampling import make_experiment_table


@pytest.mark.parametrize(
    "prior_type,reward_type",
    [
        (PriorTypes.BETA, LikelihoodTypes.BERNOULLI),
        (PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI),
        (PriorTypes.NORMAL, LikelihoodTypes.NORMAL),
    ],
)
def test_mab_analysis(prior_type, reward_type):
    mab_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE, prior_type=prior_type, reward_type=reward_type
    )
    # Update posterior parameters for arms to simulate some draws
    for i, arm in enumerate(mab_experiment.arms):
        if prior_type == PriorTypes.NORMAL:
            arm.mu_init = 0.0 + i
            arm.sigma_init = 1.0 * (i + 1)
            arm.mu = [arm.mu_init + (i + 2)]
            arm.covariance = [[arm.sigma_init * (i + 2)]]
        elif prior_type == PriorTypes.BETA:
            arm.alpha_init = 1.0 + i
            arm.beta_init = 1.0 + i
            arm.alpha = arm.alpha_init + (i + 1) * 10
            arm.beta = arm.beta_init + (2 - i) * 10

    arm_analyses = analyze_experiment(mab_experiment, random_state=66)
    assert len(arm_analyses) == len(mab_experiment.arms)
    for arm_analysis in arm_analyses:
        assert arm_analysis.prior_pred_mean != arm_analysis.post_pred_mean
        assert arm_analysis.prior_pred_stdev != arm_analysis.post_pred_stdev
        assert arm_analysis.prior_pred_ci_upper != arm_analysis.post_pred_ci_upper
        assert arm_analysis.prior_pred_ci_lower != arm_analysis.post_pred_ci_lower
        assert arm_analysis.post_pred_ci_upper > arm_analysis.prior_pred_mean
        assert arm_analysis.post_pred_ci_lower < arm_analysis.prior_pred_mean


@pytest.mark.parametrize(
    "prior_type,reward_type",
    [
        (PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI),
        (PriorTypes.NORMAL, LikelihoodTypes.NORMAL),
    ],
)
def test_cmab_analysis(prior_type, reward_type):
    cmab_experiment = make_experiment_table(
        experiment_type=ExperimentsType.CMAB_ONLINE, prior_type=prior_type, reward_type=reward_type
    )
    # Update posterior parameters for arms to simulate some draws
    for i, arm in enumerate(cmab_experiment.arms):
        arm.mu_init = 0.0 + i
        arm.sigma_init = 1.0 * (i + 1)
        arm.mu = [arm.mu_init + (i + 2)] * len(cmab_experiment.contexts)
        arm.covariance = np.diag([arm.sigma_init * (i + 2)] * len(cmab_experiment.contexts)).tolist()

    contexts = [1.0] * len(cmab_experiment.contexts)
    arm_analyses = analyze_experiment(cmab_experiment, context_vals=contexts, random_state=66)
    assert len(arm_analyses) == len(cmab_experiment.arms)
    for arm_analysis in arm_analyses:
        assert arm_analysis.prior_pred_mean != arm_analysis.post_pred_mean
        assert arm_analysis.prior_pred_stdev != arm_analysis.post_pred_stdev
        assert arm_analysis.prior_pred_ci_upper != arm_analysis.post_pred_ci_upper
        assert arm_analysis.prior_pred_ci_lower != arm_analysis.post_pred_ci_lower
        assert arm_analysis.post_pred_ci_upper > arm_analysis.post_pred_mean
        assert arm_analysis.post_pred_ci_lower < arm_analysis.post_pred_mean


def test_analyze_normal_binary_ci_correctness():
    """Regression test for bug where link function was applied twice to CI calculation."""
    # Setup a case where mean logit is high (e.g. 2.2 -> p ~ 0.9)
    # If our (logistic) link function is applied twice, CI upper would be ~0.71, which is < mean 0.9
    mu = np.array([2.2])
    cov = np.array([[0.01]])

    mean, _, ci_upper, ci_lower = _analyze_normal_binary(mu, cov, ContextLinkFunctions.LOGISTIC, random_state=42)

    assert 0.8 < mean < 1.0, f"Mean {mean} should be around 0.9"
    # If the original bug was present, ci_upper would be sigmoid(0.9) approx 0.71
    assert ci_upper > mean, f"CI upper {ci_upper} should be > mean"
    assert ci_lower < mean < ci_upper, f"CI [{ci_lower}, {ci_upper}] should contain mean {mean}"
