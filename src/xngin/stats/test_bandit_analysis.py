import numpy as np
import pytest

from xngin.apiserver.routers.common_enums import ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.stats.bandit_analysis import analyze_experiment
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
    arm_analyses = analyze_experiment(cmab_experiment, contexts=contexts, random_state=66)
    assert len(arm_analyses) == len(cmab_experiment.arms)
    for arm_analysis in arm_analyses:
        assert arm_analysis.prior_pred_mean != arm_analysis.post_pred_mean
        assert arm_analysis.prior_pred_stdev != arm_analysis.post_pred_stdev
