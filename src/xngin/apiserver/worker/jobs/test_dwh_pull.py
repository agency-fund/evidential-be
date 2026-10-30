import pytest
from sqlalchemy import select

from xngin.apiserver import database
from xngin.apiserver.routers.admin.test_admin_api import make_bandit_online_experiment
from xngin.apiserver.routers.common_api_types import (
    BanditExperimentAnalysisResponse,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)
from xngin.apiserver.routers.experiments.test_experiments_common import insert_experiment_and_arms
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient
from xngin.apiserver.testing.worker_helpers import read_assignments
from xngin.apiserver.worker.jobs.bandit_outcomes import OutcomeReport
from xngin.apiserver.worker.jobs.dwh_pull import pull_one_experiment


def create_mab_dwh_experiment(
    aclient: AdminAPIClient,
    testing_datasource,
    *,
    target_field_name: str = "is_onboarded",
    prior_type: PriorTypes = PriorTypes.BETA,
    reward_type: LikelihoodTypes = LikelihoodTypes.BERNOULLI,
) -> str:
    """Create and commit a MAB-DWH experiment through the Admin API, returning its id."""
    experiment = make_bandit_online_experiment(
        aclient,
        testing_datasource.datasource_id,
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        prior_type=prior_type,
        reward_type=reward_type,
        target_field_name=target_field_name,
    )
    return experiment.config.experiment_id


def assign(eclient: ExperimentsAPIClient, testing_datasource, experiment_id: str, *participant_ids: str) -> None:
    for participant_id in participant_ids:
        eclient.get_assignment(
            api_key=testing_datasource.key, experiment_id=experiment_id, participant_id=participant_id
        )


def outcomes_of(
    eclient: ExperimentsAPIClient, testing_datasource, experiment_id: str, *participant_ids: str
) -> dict[str, float | None]:
    assignments = read_assignments(eclient, testing_datasource, experiment_id, participant_ids)
    return {assignment.participant_id: assignment.outcome for assignment in assignments}


def analyze(aclient: AdminAPIClient, testing_datasource, experiment_id: str) -> BanditExperimentAnalysisResponse:
    analysis = aclient.analyze_experiment(
        datasource_id=testing_datasource.datasource_id, experiment_id=experiment_id
    ).data
    assert isinstance(analysis, BanditExperimentAnalysisResponse)
    return analysis


def pull(experiment_id: str, *, dwh_timeout: float = 90) -> OutcomeReport:
    with database.get_session() as session, session.begin():
        return pull_one_experiment(session, session.get_one(tables.Experiment, experiment_id), dwh_timeout=dwh_timeout)


@pytest.mark.parametrize(
    ("target_field_name", "prior_type", "reward_type", "outcomes", "report"),
    [
        # Testing DWH: id=1 has is_onboarded=false, id=2 has is_onboarded=true. 99999999999 is not present.
        pytest.param(
            "is_onboarded",
            PriorTypes.BETA,
            LikelihoodTypes.BERNOULLI,
            {"1": 0.0, "2": 1.0, "99999999999": None},
            OutcomeReport(ingested=2, pending=1),
            id="boolean",
        ),
        # A numeric target under a Normal reward takes arbitrary float values, not just 0/1. Moved here from the
        # outcome API, which no longer accepts pushes for this type.
        pytest.param(
            "current_income",
            PriorTypes.NORMAL,
            LikelihoodTypes.NORMAL,
            {"1": 29912.0},
            OutcomeReport(ingested=1),
            id="numeric",
        ),
        # is_onboarded_onetime is NULL for low ids in the testing DWH, so the draw stays pending for a later run.
        pytest.param(
            "is_onboarded_onetime",
            PriorTypes.BETA,
            LikelihoodTypes.BERNOULLI,
            {"1": None},
            OutcomeReport(pending=1),
            id="null",
        ),
        # The common case: no draws without outcomes, so there is nothing to read.
        pytest.param("is_onboarded", PriorTypes.BETA, LikelihoodTypes.BERNOULLI, {}, OutcomeReport(), id="no-draws"),
    ],
)
def test_pull_one_experiment_applies_warehouse_values(
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    target_field_name,
    prior_type,
    reward_type,
    outcomes,
    report,
):
    experiment_id = create_mab_dwh_experiment(
        aclient,
        testing_datasource,
        target_field_name=target_field_name,
        prior_type=prior_type,
        reward_type=reward_type,
    )
    assign(eclient, testing_datasource, experiment_id, *outcomes)

    assert pull(experiment_id) == report
    assert outcomes_of(eclient, testing_datasource, experiment_id, *outcomes) == outcomes
    assert analyze(aclient, testing_datasource, experiment_id).n_outcomes == report.ingested
    # Re-running ingests nothing new: filled draws no longer match the outcome-less query.
    assert pull(experiment_id) == OutcomeReport(pending=report.pending)


def test_pull_one_experiment_invalid_values_skipped_and_left_null(
    xngin_session, testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """A pulled value that fails validation is skipped without stopping the participants after it. The
    draw stays NULL, so a corrected warehouse value is picked up on a later run.

    This happens when a warehouse column's type changes after the experiment was created. The API rejects a
    target column whose type does not match the reward type, so the experiment is inserted directly.
    """
    # No current_income in the testing DWH is 0 or 1, so every value fails the Bernoulli guard.
    experiment_id = insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
        target_field_name="current_income",
    ).id
    assign(eclient, testing_datasource, experiment_id, "1", "3")

    report = pull(experiment_id)
    assert (report.ingested, report.invalid, report.pending) == (0, 2, 0)
    assert outcomes_of(eclient, testing_datasource, experiment_id, "1", "3") == {"1": None, "3": None}
    analysis = analyze(aclient, testing_datasource, experiment_id)
    assert analysis.n_outcomes == 0
    assert all(arm.post_pred_mean == arm.prior_pred_mean for arm in analysis.arm_analyses)


def test_a_pull_skips_draws_another_transaction_holds(
    xngin_session, testing_datasource, aclient: AdminAPIClient, eclient: ExperimentsAPIClient
):
    """A pull never waits for another writer's draws, such as an experiment delete, which would deadlock."""
    experiment_id = create_mab_dwh_experiment(aclient, testing_datasource)
    assign(eclient, testing_datasource, experiment_id, "1", "2")
    with database.get_session() as deleter, deleter.begin():
        deleter.execute(
            select(tables.Draw.participant_id)
            .where(tables.Draw.experiment_id == experiment_id, tables.Draw.participant_id == "1")
            .with_for_update()
        )
        assert pull(experiment_id) == OutcomeReport(ingested=1)
    assert outcomes_of(eclient, testing_datasource, experiment_id, "1", "2") == {"1": None, "2": 1.0}
