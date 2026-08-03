import asyncio
import contextlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    CMABContextInputRequest,
    CMABExperimentSpec,
    Context,
    ContextInput,
    ContextType,
    CreateExperimentRequest,
    ExperimentsType,
    LikelihoodTypes,
    MABExperimentSpec,
    PriorTypes,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.snapshots import cli
from xngin.apiserver.snapshots.autofail import (
    AUTOFAIL_TIMEOUT_SECS,
    create_pending_autofail_updates,
    process_pending_autofail_updates,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient

UPDATE_OUTCOME_IMPL = "xngin.apiserver.routers.experiments.experiments_common.update_bandit_arm_with_outcome_impl"


async def create_autofail_experiment(
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    testing_datasource,
    *,
    experiment_type: ExperimentsType = ExperimentsType.MAB_ONLINE,
    enable_autofail: bool = True,
    autofail_window: int = 24,
    autofail_outcome_value: float = 0.0,
    reward_type: LikelihoodTypes = LikelihoodTypes.BERNOULLI,
    prior_type: PriorTypes = PriorTypes.NORMAL,
    participants: Sequence[str] = ("0", "1"),
    name: str = "autofail test",
) -> str:
    """
    Creates a committed bandit experiment and draws an assignment for each participant.
    """
    autofail_config = {
        "enable_autofail": enable_autofail,
        "autofail_window": autofail_window,
        "autofail_outcome_value": autofail_outcome_value,
    }
    design_spec: MABExperimentSpec | CMABExperimentSpec
    match experiment_type:
        case ExperimentsType.MAB_ONLINE:
            design_spec = MABExperimentSpec(
                experiment_type=experiment_type,
                experiment_name=name,
                description=name,
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    ArmBandit(
                        arm_name=arm_name,
                        arm_description="",
                        alpha_init=1 if prior_type == PriorTypes.BETA else None,
                        beta_init=1 if prior_type == PriorTypes.BETA else None,
                        mu_init=0 if prior_type == PriorTypes.NORMAL else None,
                        sigma_init=1 if prior_type == PriorTypes.NORMAL else None,
                    )
                    for arm_name in ("control", "treatment")
                ],
                prior_type=prior_type,
                reward_type=reward_type,
                **autofail_config,
            )
        case ExperimentsType.CMAB_ONLINE:
            design_spec = CMABExperimentSpec(
                experiment_type=experiment_type,
                experiment_name=name,
                description=name,
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    ArmBandit(arm_name="control", arm_description="", mu_init=0, sigma_init=1),
                    ArmBandit(arm_name="treatment", arm_description="", mu_init=0, sigma_init=1),
                ],
                contexts=[
                    Context(context_name="context1", context_description="", value_type=ContextType.BINARY),
                    Context(context_name="context2", context_description="", value_type=ContextType.REAL_VALUED),
                ],
                prior_type=PriorTypes.NORMAL,
                reward_type=reward_type,
                **autofail_config,
            )
        case _:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.datasource_id,
        body=CreateExperimentRequest(design_spec=design_spec),
        random_state=42,
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.datasource_id, experiment_id=experiment_id)

    if experiment_type == ExperimentsType.CMAB_ONLINE:
        context_inputs = get_sorted_context_inputs(aclient, testing_datasource, experiment_id)
        for participant_id in participants:
            _ = eclient.get_assignment_cmab(
                api_key=testing_datasource.key,
                body=CMABContextInputRequest(context_inputs=context_inputs),
                experiment_id=experiment_id,
                participant_id=participant_id,
            )
    else:
        for participant_id in participants:
            _ = eclient.get_assignment(
                api_key=testing_datasource.key,
                experiment_id=experiment_id,
                participant_id=participant_id,
            )
    return experiment_id


def get_sorted_context_inputs(
    aclient: AdminAPIClient,
    testing_datasource,
    experiment_id: str,
    value: float = 1.0,
) -> list[ContextInput]:
    """Returns context inputs for a CMAB, sorted by context id as the assignment path expects."""
    config = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment_id,
    ).data.config
    assert isinstance(config.design_spec, CMABExperimentSpec)
    assert config.design_spec.contexts is not None
    return [
        ContextInput(context_id=context.context_id or "", context_value=value)
        for context in sorted(config.design_spec.contexts, key=lambda c: c.context_id or "")
    ]


async def age_draws(
    xngin_session,
    experiment_id: str,
    hours: float,
    participant_ids: Sequence[str] | None = None,
) -> None:
    """Updates the created_at timestamp of draws to be older than the autofail window."""
    stmt = update(tables.Draw).where(tables.Draw.experiment_id == experiment_id)
    if participant_ids is not None:
        stmt = stmt.where(tables.Draw.participant_id.in_(participant_ids))
    await xngin_session.execute(stmt.values(created_at=datetime.now(UTC) - timedelta(hours=hours)))
    await xngin_session.commit()


async def get_autofail_updates(xngin_session, experiment_id: str | None = None) -> list[tables.AutofailUpdate]:
    xngin_session.expire_all()
    stmt = select(tables.AutofailUpdate).order_by(
        tables.AutofailUpdate.experiment_id, tables.AutofailUpdate.participant_id
    )
    if experiment_id is not None:
        stmt = stmt.where(tables.AutofailUpdate.experiment_id == experiment_id)
    return list((await xngin_session.execute(stmt)).scalars().all())


async def test_create_pending_autofail_updates_boundary(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="window boundary"
    )
    await age_draws(xngin_session, experiment_id, hours=0.9, participant_ids=["0"])
    await age_draws(xngin_session, experiment_id, hours=1.1, participant_ids=["1"])

    await create_pending_autofail_updates()

    updates = await get_autofail_updates(xngin_session)
    assert [u.participant_id for u in updates] == ["1"]


async def test_create_pending_autofail_updates_skips_experiments_without_autofail(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        enable_autofail=False,
        autofail_window=1,
        name="autofail disabled",
    )
    await age_draws(xngin_session, experiment_id, hours=100)

    await create_pending_autofail_updates()

    assert await get_autofail_updates(xngin_session) == []


async def test_create_pending_autofail_updates_skips_draws_with_outcomes(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="outcome already reported"
    )
    eclient.update_bandit_arm_with_participant_outcome(
        api_key=testing_datasource.key,
        body=UpdateBanditArmOutcomeRequest(outcome=1.0),
        experiment_id=experiment_id,
        participant_id="0",
    )
    await age_draws(xngin_session, experiment_id, hours=100)

    await create_pending_autofail_updates()

    updates = await get_autofail_updates(xngin_session)
    assert [u.participant_id for u in updates] == ["1"]


@pytest.mark.parametrize("experiment_type", [ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE])
async def test_create_pending_autofail_updates_covers_mab_and_cmab(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    experiment_type: ExperimentsType,
):
    experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        experiment_type=experiment_type,
        autofail_window=1,
        name=f"pending updates {experiment_type}",
    )
    await age_draws(xngin_session, experiment_id, hours=2)

    await create_pending_autofail_updates()

    updates = await get_autofail_updates(xngin_session)
    assert [(u.experiment_id, u.participant_id, u.status) for u in updates] == [
        (experiment_id, "0", "pending"),
        (experiment_id, "1", "pending"),
    ]


@pytest.mark.parametrize("experiment_type", [ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE])
async def test_process_pending_autofail_updates_records_outcome(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    experiment_type: ExperimentsType,
):
    experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        experiment_type=experiment_type,
        autofail_window=1,
        autofail_outcome_value=0.0,
        name=f"process updates {experiment_type}",
    )
    await age_draws(xngin_session, experiment_id, hours=2)

    await create_pending_autofail_updates()
    await process_pending_autofail_updates(AUTOFAIL_TIMEOUT_SECS, max_jitter_secs=0)

    updates = await get_autofail_updates(xngin_session)
    assert [u.status for u in updates] == ["success", "success"]
    assert all(u.message for u in updates)


async def test_process_pending_autofail_updates_processes_until_empty(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    first_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        autofail_outcome_value=0.0,
        name="until empty a",
    )
    second_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        autofail_outcome_value=1.0,
        name="until empty b",
    )
    for experiment_id in (first_id, second_id):
        await age_draws(xngin_session, experiment_id, hours=2)

    await create_pending_autofail_updates()
    assert len(await get_autofail_updates(xngin_session)) == 4

    await process_pending_autofail_updates(AUTOFAIL_TIMEOUT_SECS, max_jitter_secs=0)

    updates = await get_autofail_updates(xngin_session)
    assert [u.status for u in updates] == ["success"] * 4


async def test_process_pending_autofail_updates_is_noop_when_none_pending(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="nothing pending"
    )
    await age_draws(xngin_session, experiment_id, hours=2)

    # Nothing in the table at all.
    await process_pending_autofail_updates(AUTOFAIL_TIMEOUT_SECS, max_jitter_secs=0)
    assert await get_autofail_updates(xngin_session) == []

    # A terminal row is not reprocessed.
    xngin_session.add(
        tables.AutofailUpdate(
            experiment_id=experiment_id,
            participant_id="0",
            status="success",
            message="already done",
        )
    )
    await xngin_session.commit()

    await process_pending_autofail_updates(AUTOFAIL_TIMEOUT_SECS, max_jitter_secs=0)

    updates = await get_autofail_updates(xngin_session)
    assert [(u.status, u.message) for u in updates] == [("success", "already done")]


async def test_process_pending_autofail_updates_marks_failed_on_exception(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    mocker,
):
    experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        participants=["0"],
        name="failure recorded",
    )
    await age_draws(xngin_session, experiment_id, hours=2)
    await create_pending_autofail_updates()

    mocker.patch(UPDATE_OUTCOME_IMPL, side_effect=RuntimeError("boom"))

    await process_pending_autofail_updates(AUTOFAIL_TIMEOUT_SECS, max_jitter_secs=0)

    updates = await get_autofail_updates(xngin_session)
    assert [(u.status, u.message) for u in updates] == [("failed", "RuntimeError: boom")]


async def test_process_pending_autofail_updates_marks_failed_on_timeout(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    mocker,
):
    experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        participants=["0"],
        name="timeout recorded",
    )
    await age_draws(xngin_session, experiment_id, hours=2)
    await create_pending_autofail_updates()

    async def slow_update(*args, **kwargs):
        await asyncio.sleep(0.01)

    mocker.patch(UPDATE_OUTCOME_IMPL, side_effect=slow_update)

    await process_pending_autofail_updates(0, max_jitter_secs=0)

    updates = await get_autofail_updates(xngin_session)
    assert len(updates) == 1
    assert updates[0].status == "failed"
    assert updates[0].message is not None
    assert "TimeoutError" in updates[0].message


async def test_process_pending_autofail_updates_survives_a_failure(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    mocker,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="drain past failure"
    )
    await age_draws(xngin_session, experiment_id, hours=2)
    await create_pending_autofail_updates()

    original = experiments_common.update_bandit_arm_with_outcome_impl

    async def fail_first_participant(*args, **kwargs):
        if kwargs["participant_id"] == "0":
            raise RuntimeError("boom")
        return await original(*args, **kwargs)

    mocker.patch(UPDATE_OUTCOME_IMPL, side_effect=fail_first_participant)

    await process_pending_autofail_updates(AUTOFAIL_TIMEOUT_SECS, max_jitter_secs=0)

    updates = await get_autofail_updates(xngin_session)
    assert [(u.participant_id, u.status) for u in updates] == [("0", "failed"), ("1", "success")]


async def test_autofail_acollect_creates_then_processes(mocker):

    @contextlib.asynccontextmanager
    async def noop_setup():
        yield

    mocker.patch("xngin.apiserver.snapshots.cli.database.setup", noop_setup)
    create_mock = mocker.patch("xngin.apiserver.snapshots.cli.autofail.create_pending_autofail_updates")
    process_mock = mocker.patch("xngin.apiserver.snapshots.cli.autofail.process_pending_autofail_updates")

    await cli.autofail_acollect(autofail_timeout=42, parallelism=3)

    create_mock.assert_awaited_once_with()
    assert process_mock.await_count == 3
    assert all(call.args == (42,) for call in process_mock.await_args_list)
