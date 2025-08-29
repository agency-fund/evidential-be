from pydantic import TypeAdapter

from xngin.apiserver.routers.common_api_types import DesignSpec
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.apiserver.routers.experiments.test_experiments_common import make_createexperimentrequest_json
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.stats.bandit_sampling import choose_arm, update_arm


def make_experiment_table(
    experiment_type: ExperimentsType, prior_type: PriorTypes, reward_type: LikelihoodTypes
) -> tables.Experiment:
    request = make_createexperimentrequest_json(
        experiment_type=experiment_type,
        prior_type=prior_type,
        reward_type=reward_type,
    )
    design_spec: DesignSpec = TypeAdapter(DesignSpec).validate_python(request["design_spec"])
    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id="ds_id",
        organization_id="org_id",
        experiment_type=experiment_type,
        design_spec=design_spec,
        state=ExperimentState.COMMITTED,
        stopped_assignments_at=None,
        stopped_assignments_reason=None,
    )
    fake_experiment = experiment_converter.get_experiment()
    # Add fake ids to the arms
    for i, arm in enumerate(fake_experiment.arms):
        arm.id = f"arm_{i}"

    return fake_experiment


def test_check_arm_draw_is_reproducible():
    tables.Datasource()

    beta_binom_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE, prior_type=PriorTypes.BETA, reward_type=LikelihoodTypes.BERNOULLI
    )
    arm1 = choose_arm(experiment=beta_binom_experiment, random_state=0)
    arm2 = choose_arm(experiment=beta_binom_experiment, random_state=0)
    assert arm1.id == arm2.id

    normal_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE, prior_type=PriorTypes.NORMAL, reward_type=LikelihoodTypes.BERNOULLI
    )
    arm1 = choose_arm(experiment=normal_experiment, random_state=66)
    arm2 = choose_arm(experiment=normal_experiment, random_state=66)
    assert arm1.id == arm2.id


def test_check_arm_drawn_correctly():
    # Test for Beta prior
    beta_binom_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE, prior_type=PriorTypes.BETA, reward_type=LikelihoodTypes.BERNOULLI
    )
    beta_binom_experiment.arms[0].id = "abcd"
    beta_binom_experiment.arms[1].id = "bcde"
    sorted_beta_binom_arms = sorted(beta_binom_experiment.arms, key=lambda arm: arm.id)
    arm = choose_arm(experiment=beta_binom_experiment, random_state=0)
    assert arm.id == sorted_beta_binom_arms[0].id

    # Test for Normal prior
    normal_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE, prior_type=PriorTypes.NORMAL, reward_type=LikelihoodTypes.BERNOULLI
    )
    normal_experiment.arms[0].id = "abcd"
    normal_experiment.arms[1].id = "bcde"
    sorted_normal_arms = sorted(normal_experiment.arms, key=lambda arm: arm.id)
    arm = choose_arm(experiment=normal_experiment, random_state=66)
    assert arm.id == sorted_normal_arms[0].id


def test_update_arm():
    # Test for Beta-binom experiments
    beta_binom_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE, prior_type=PriorTypes.BETA, reward_type=LikelihoodTypes.BERNOULLI
    )
    arm = choose_arm(experiment=beta_binom_experiment, random_state=42)
    alpha, beta = update_arm(experiment=beta_binom_experiment, arm_to_update=arm, outcomes=[0.0])

    assert arm.alpha is not None and arm.beta is not None
    assert float(arm.alpha) == alpha
    assert float(arm.beta + 1) == beta

    # Test for Normal prior with binary outcomes
    normal_binary_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE, prior_type=PriorTypes.NORMAL, reward_type=LikelihoodTypes.BERNOULLI
    )
    arm = choose_arm(experiment=normal_binary_experiment, random_state=42)
    mu, covariance = update_arm(experiment=normal_binary_experiment, arm_to_update=arm, outcomes=[0.0])
    assert arm.mu != mu
    assert arm.covariance != covariance

    # Test for Normal prior with real-valued outcomes
    normal_experiment = make_experiment_table(
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
    )
    arm = choose_arm(experiment=normal_experiment, random_state=42)
    mu, covariance = update_arm(experiment=normal_experiment, arm_to_update=arm, outcomes=[0.0])
    assert arm.mu != mu
    assert arm.covariance != covariance
