from pydantic import TypeAdapter

from xngin.apiserver.routers.common_api_types import DesignSpec
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.apiserver.routers.experiments.test_experiments_common import make_createexperimentrequest_json
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.stats.bandit_sampling import TOP_TWO_MIN_ARMS, choose_arm, update_arm


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


# --- Top-Two Thompson Sampling tests ---


def _make_many_arm_experiment(n_arms: int, prior_type: PriorTypes, reward_type: LikelihoodTypes) -> tables.Experiment:
    """Create an experiment with N arms for testing Top-Two TS.

    Builds the design spec JSON with N arms, then uses the standard converter.
    """
    is_beta = prior_type == PriorTypes.BETA
    arm_spec = {
        "arm_name": "arm",
        "arm_description": "arm",
        "alpha_init": 1.0 if is_beta else None,
        "beta_init": 1.0 if is_beta else None,
        "mu_init": 0.0 if not is_beta else None,
        "sigma_init": 1.0 if not is_beta else None,
    }
    request = {
        "design_spec": {
            "participant_type": "test",
            "experiment_name": "test_top_two",
            "description": "test",
            "start_date": "2024-01-01T00:00:00+00:00",
            "end_date": "2030-01-01T00:00:00+00:00",
            "experiment_type": "mab_online",
            "prior_type": prior_type,
            "reward_type": reward_type,
            "arms": [arm_spec for _ in range(n_arms)],
        }
    }
    design_spec: DesignSpec = TypeAdapter(DesignSpec).validate_python(request["design_spec"])
    converter = ExperimentStorageConverter.init_from_components(
        datasource_id="ds_id",
        organization_id="org_id",
        experiment_type=ExperimentsType.MAB_ONLINE,
        design_spec=design_spec,
        state=ExperimentState.COMMITTED,
        stopped_assignments_at=None,
        stopped_assignments_reason=None,
    )
    experiment = converter.get_experiment()
    for i, arm in enumerate(experiment.arms):
        arm.id = f"arm_{i}"
    return experiment


def test_top_two_returns_valid_arm_for_all_prior_types():
    """With ≥5 arms, Top-Two TS returns an arm that belongs to the experiment."""
    for prior, reward in [
        (PriorTypes.BETA, LikelihoodTypes.BERNOULLI),
        (PriorTypes.NORMAL, LikelihoodTypes.NORMAL),
        (PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI),
    ]:
        experiment = _make_many_arm_experiment(TOP_TWO_MIN_ARMS, prior, reward)
        arm = choose_arm(experiment=experiment, random_state=42)
        assert arm.id in {a.id for a in experiment.arms}


def test_top_two_sometimes_picks_challenger():
    """Top-Two TS should explore challengers even when one arm dominates.

    With 6 arms (≥5, so Top-Two is active) and arm_0 having alpha=1000,
    Top-Two with β=0.9 should still pick a challenger ~10% of the time.
    Regular TS would pick the dominant arm 98%+ of the time.
    We check that at least 10/200 selections are non-dominant — this would
    fail under regular TS (which gets ~3/200 empirically).
    """
    experiment = _make_many_arm_experiment(6, PriorTypes.BETA, LikelihoodTypes.BERNOULLI)
    experiment.arms[0].alpha = 1000.0
    experiment.arms[0].beta = 1.0

    selections = [choose_arm(experiment=experiment, random_state=s).id for s in range(200)]
    non_dominant = sum(1 for s in selections if s != "arm_0")
    assert non_dominant >= 10, f"Expected Top-Two to explore challengers, but got {non_dominant}/200"


def test_below_threshold_barely_explores_with_dominant_arm():
    """With <5 arms, regular TS is used — a dominant arm should win almost always.

    Same dominant prior as above, but with 4 arms (below Top-Two threshold).
    Regular TS with alpha=1000 empirically picks the dominant arm 98%+ of the
    time. We check that non-dominant selections are ≤8/200 — this would fail
    under Top-Two (which gets ~26/200 empirically).
    """
    experiment = _make_many_arm_experiment(TOP_TWO_MIN_ARMS - 1, PriorTypes.BETA, LikelihoodTypes.BERNOULLI)
    experiment.arms[0].alpha = 1000.0
    experiment.arms[0].beta = 1.0

    selections = [choose_arm(experiment=experiment, random_state=s).id for s in range(200)]
    non_dominant = sum(1 for s in selections if s != "arm_0")
    assert non_dominant <= 8, f"Expected regular TS to exploit dominant arm, but got {non_dominant}/200 non-dominant"
