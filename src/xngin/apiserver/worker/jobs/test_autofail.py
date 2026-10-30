from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

from xngin.apiserver import database
from xngin.apiserver.routers.admin.test_admin_api import make_cmab_context_inputs
from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    CMABContextInputRequest,
    CMABExperimentSpec,
    Context,
    ContextType,
    CreateExperimentRequest,
    ExperimentsType,
    LikelihoodTypes,
    MABExperimentSpec,
    PriorTypes,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient
from xngin.apiserver.testing.worker_helpers import BACKOFF, read_assignments
from xngin.apiserver.worker.jobs.autofail import AutofailJob
from xngin.apiserver.worker.jobs.base import JobDeferred, JobResult, JobSucceeded


def create_autofail_experiment(
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    testing_datasource,
    *,
    experiment_type: ExperimentsType = ExperimentsType.MAB_ONLINE,
    participants: Sequence[str] = ("0", "1"),
    name: str = "autofail test",
) -> str:
    """Create a committed bandit experiment with a one-hour autofail window, and assign each participant."""
    spec_fields = {
        "experiment_name": name,
        "description": name,
        "start_date": datetime(2024, 1, 1, tzinfo=UTC),
        "end_date": datetime.now(UTC) + timedelta(days=1),
        "arms": [
            ArmBandit(arm_name=arm_name, arm_description="", mu_init=0, sigma_init=1)
            for arm_name in ("control", "treatment")
        ],
        "prior_type": PriorTypes.NORMAL,
        "reward_type": LikelihoodTypes.BERNOULLI,
        "enable_autofail": True,
        "autofail_window": 1,
        "autofail_outcome_value": 0.0,
    }
    design_spec: MABExperimentSpec | CMABExperimentSpec
    match experiment_type:
        case ExperimentsType.MAB_ONLINE:
            design_spec = MABExperimentSpec(experiment_type=experiment_type, **spec_fields)
        case ExperimentsType.CMAB_ONLINE:
            design_spec = CMABExperimentSpec(
                experiment_type=experiment_type,
                contexts=[
                    Context(context_name="context1", context_description="", value_type=ContextType.BINARY),
                    Context(context_name="context2", context_description="", value_type=ContextType.REAL_VALUED),
                ],
                **spec_fields,
            )
        case _:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    datasource_id = testing_datasource.datasource_id
    experiment_id = aclient.create_experiment(
        datasource_id=datasource_id, body=CreateExperimentRequest(design_spec=design_spec), random_state=42
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)

    if experiment_type == ExperimentsType.CMAB_ONLINE:
        experiment = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data
        context_inputs = make_cmab_context_inputs(experiment, [1.0, 1.0])
        for participant_id in participants:
            eclient.get_assignment_cmab(
                api_key=testing_datasource.key,
                body=CMABContextInputRequest(context_inputs=context_inputs),
                experiment_id=experiment_id,
                participant_id=participant_id,
            )
    else:
        for participant_id in participants:
            eclient.get_assignment(
                api_key=testing_datasource.key, experiment_id=experiment_id, participant_id=participant_id
            )
    return experiment_id


def age_draws(
    xngin_session: Session,
    experiment_id: str,
    hours: float,
    participant_ids: Sequence[str] | None = None,
) -> None:
    """Set draw creation times relative to their autofail eligibility window."""
    stmt = update(tables.Draw).where(tables.Draw.experiment_id == experiment_id)
    if participant_ids is not None:
        stmt = stmt.where(tables.Draw.participant_id.in_(participant_ids))
    xngin_session.execute(stmt.values(created_at=datetime.now(UTC) - timedelta(hours=hours)))
    xngin_session.commit()


def commit_autofail_batch(experiment_id: str, batch_size: int = 500) -> JobResult:
    with database.get_session() as session, session.begin():
        return AutofailJob(batch_size=batch_size, backoff=BACKOFF).execute(
            session, session.get_one(tables.Experiment, experiment_id)
        )


# What a batch reports when it found no overdue draws to process.
NOTHING_TO_PROCESS = JobDeferred(retry_after=timedelta(seconds=1))


def test_autofail_eligibility_uses_window_boundary(
    xngin_session: Session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = create_autofail_experiment(aclient, eclient, testing_datasource, name="window boundary")
    age_draws(xngin_session, experiment_id, hours=0.9, participant_ids=["0"])
    age_draws(xngin_session, experiment_id, hours=1.1, participant_ids=["1"])

    assert commit_autofail_batch(experiment_id) == JobSucceeded()

    assignments = read_assignments(eclient, testing_datasource, experiment_id, ("0", "1"))
    assert [(assignment.participant_id, assignment.outcome) for assignment in assignments] == [("0", None), ("1", 0.0)]


def test_autofail_skips_draws_without_autofail_flag_or_with_outcomes(
    xngin_session: Session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = create_autofail_experiment(
        aclient, eclient, testing_datasource, participants=("0", "1", "2"), name="skip draws"
    )
    xngin_session.execute(
        update(tables.Draw)
        .where(tables.Draw.experiment_id == experiment_id, tables.Draw.participant_id == "0")
        .values(enable_autofail=False)
    )
    xngin_session.commit()
    eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=experiment_id,
        participant_id="1",
    )
    age_draws(xngin_session, experiment_id, hours=100)

    assert commit_autofail_batch(experiment_id) == JobSucceeded()
    assert commit_autofail_batch(experiment_id) == NOTHING_TO_PROCESS

    assignments = read_assignments(eclient, testing_datasource, experiment_id, ("0", "1", "2"))
    assert [
        (assignment.participant_id, assignment.outcome, assignment.autofailed_outcome) for assignment in assignments
    ] == [
        ("0", None, None),
        ("1", 1.0, False),
        ("2", 0.0, True),
    ]


@pytest.mark.parametrize("experiment_type", [ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE])
def test_autofail_records_outcomes_for_supported_bandits(
    xngin_session: Session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    experiment_type: ExperimentsType,
):
    experiment_id = create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        experiment_type=experiment_type,
        name=f"process updates {experiment_type}",
    )
    age_draws(xngin_session, experiment_id, hours=2)

    assert commit_autofail_batch(experiment_id) == JobSucceeded()

    assignments = read_assignments(eclient, testing_datasource, experiment_id, ("0", "1"))
    assert all(assignment.outcome == 0.0 for assignment in assignments)
    assert all(assignment.autofailed_outcome is True for assignment in assignments)
    assert all(assignment.observed_at is not None for assignment in assignments)
