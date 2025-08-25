from xngin.apiserver.conftest import fixture_testing_datasource, fixture_xngin_db_session
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.apiserver.routers.experiments.test_experiments_common import make_insertable_experiment
from xngin.stats.bandit_sampling import choose_arm, update_arm

xngin_session = fixture_xngin_db_session
testing_datasource = fixture_testing_datasource


def test_check_arm_drawn_correctly(testing_datasource):
    # Test for Beta prior
    beta_binom_experiment, _ = make_insertable_experiment(
        datasource=testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
    )
    sorted_beta_binom_arms = sorted(beta_binom_experiment.arms, key=lambda arm: arm.id)
    arm = choose_arm(experiment=beta_binom_experiment, random_state=42)
    assert arm.id == sorted_beta_binom_arms[0].id

    # Test for Normal prior
    normal_experiment, _ = make_insertable_experiment(
        datasource=testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.BERNOULLI,
    )
    sorted_normal_arms = sorted(normal_experiment.arms, key=lambda arm: arm.id)
    arm = choose_arm(experiment=normal_experiment, random_state=42)
    assert arm.id == sorted_normal_arms[0].id


def test_update_arm(testing_datasource):
    # Test for Beta-binom experiments
    beta_binom_experiment, _ = make_insertable_experiment(
        datasource=testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
    )
    arm = choose_arm(experiment=beta_binom_experiment, random_state=42)
    alpha, beta = update_arm(experiment=beta_binom_experiment, arm_to_update=arm, outcomes=[0.0])

    assert arm.alpha is not None and arm.beta is not None
    assert float(arm.alpha) == alpha
    assert float(arm.beta + 1) == beta

    # Test for Normal prior with binary outcomes
    normal_binary_experiment, _ = make_insertable_experiment(
        datasource=testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.BERNOULLI,
    )
    arm = choose_arm(experiment=normal_binary_experiment, random_state=42)
    mu, covariance = update_arm(experiment=normal_binary_experiment, arm_to_update=arm, outcomes=[0.0])
    assert arm.mu != mu
    assert arm.covariance != covariance

    # Test for Normal prior with real-valued outcomes
    normal_experiment, _ = make_insertable_experiment(
        datasource=testing_datasource.ds,
        state=ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
    )
    arm = choose_arm(experiment=normal_experiment, random_state=42)
    mu, covariance = update_arm(experiment=normal_experiment, arm_to_update=arm, outcomes=[0.0])
    assert arm.mu != mu
    assert arm.covariance != covariance
