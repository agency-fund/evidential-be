from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    Assignment,
    CMABContextInputRequest,
    CMABExperimentSpec,
    Context,
    ContextInput,
    ContextType,
    CreateExperimentRequest,
    ExperimentsType,
    LikelihoodTypes,
    MABDwhExperimentSpec,
    MABExperimentSpec,
    PriorTypes,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.snapshots import cli
from xngin.apiserver.snapshots.autofail import (
    DEFAULT_AUTOFAIL_TIMEOUT_SECS,
    _autofail_experiment_ids_query,  # noqa: PLC2701
    process_autofails,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF


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
    """Create a committed bandit experiment and draw an assignment for each participant."""
    autofail_config = {
        "enable_autofail": enable_autofail,
        "autofail_window": autofail_window,
        "autofail_outcome_value": autofail_outcome_value,
    }
    design_spec: MABExperimentSpec | CMABExperimentSpec | MABDwhExperimentSpec
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
        case ExperimentsType.MAB_ONLINE_DWH:
            design_spec = MABDwhExperimentSpec(
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
                table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
                primary_key="id",
                target_field_name="is_onboarded",
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
    """Return context inputs sorted as the assignment path expects."""
    config = aclient.get_experiment_for_ui(
        datasource_id=testing_datasource.datasource_id,
        experiment_id=experiment_id,
    ).data.config
    assert isinstance(config.design_spec, CMABExperimentSpec)
    assert config.design_spec.contexts is not None
    return [
        ContextInput(context_id=context.context_id or "", context_value=value)
        for context in sorted(config.design_spec.contexts, key=lambda context: context.context_id or "")
    ]


async def age_draws(
    xngin_session: AsyncSession,
    experiment_id: str,
    hours: float,
    participant_ids: Sequence[str] | None = None,
) -> None:
    """Set draw creation times relative to their autofail eligibility window."""
    stmt = update(tables.Draw).where(tables.Draw.experiment_id == experiment_id)
    if participant_ids is not None:
        stmt = stmt.where(tables.Draw.participant_id.in_(participant_ids))
    await xngin_session.execute(stmt.values(created_at=datetime.now(UTC) - timedelta(hours=hours)))
    await xngin_session.commit()


async def get_draws(xngin_session: AsyncSession, experiment_id: str) -> list[tables.Draw]:
    xngin_session.expire_all()
    return list(
        (
            await xngin_session.execute(
                select(tables.Draw)
                .where(tables.Draw.experiment_id == experiment_id)
                .order_by(tables.Draw.participant_id)
            )
        )
        .scalars()
        .all()
    )


def get_assignments_via_api(
    eclient: ExperimentsAPIClient,
    testing_datasource,
    experiment_id: str,
    participant_ids: Sequence[str] = ("0", "1"),
) -> list[Assignment]:
    assignments: list[Assignment] = []
    for participant_id in participant_ids:
        assignment = eclient.get_assignment(
            api_key=testing_datasource.key,
            experiment_id=experiment_id,
            participant_id=participant_id,
            create_if_none=False,
        ).data.assignment
        assert assignment is not None
        assignments.append(assignment)
    return assignments


@dataclass(slots=True)
class ManualClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


async def test_autofail_eligibility_uses_window_boundary(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="window boundary"
    )
    await age_draws(xngin_session, experiment_id, hours=0.9, participant_ids=["0"])
    await age_draws(xngin_session, experiment_id, hours=1.1, participant_ids=["1"])

    await process_autofails(DEFAULT_AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    draws = await get_draws(xngin_session, experiment_id)
    assert [(draw.participant_id, draw.outcome) for draw in draws] == [("0", None), ("1", 0.0)]


async def test_autofail_skips_experiments_without_autofail(
    xngin_session: AsyncSession,
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

    await process_autofails(DEFAULT_AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    draws = await get_draws(xngin_session, experiment_id)
    assert all(draw.enable_autofail is False for draw in draws)
    assert all(draw.outcome is None for draw in draws)


async def test_autofail_skips_draws_without_autofail_flag(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="draw autofail disabled"
    )
    await xngin_session.execute(
        update(tables.Draw)
        .where(tables.Draw.experiment_id == experiment_id, tables.Draw.participant_id == "0")
        .values(enable_autofail=False)
    )
    await xngin_session.commit()
    await age_draws(xngin_session, experiment_id, hours=2)

    await process_autofails(DEFAULT_AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    draws = await get_draws(xngin_session, experiment_id)
    assert [(draw.participant_id, draw.outcome) for draw in draws] == [("0", None), ("1", 0.0)]


async def test_autofail_skips_draws_with_outcomes(
    xngin_session: AsyncSession,
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

    await process_autofails(DEFAULT_AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    assignments = get_assignments_via_api(eclient, testing_datasource, experiment_id)
    assert [
        (assignment.participant_id, assignment.outcome, assignment.autofailed_outcome) for assignment in assignments
    ] == [
        ("0", 1.0, False),
        ("1", 0.0, True),
    ]


@pytest.mark.parametrize(
    "experiment_type", [ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE, ExperimentsType.MAB_ONLINE_DWH]
)
async def test_autofail_records_outcomes_for_supported_bandits(
    xngin_session: AsyncSession,
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

    await process_autofails(DEFAULT_AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    assignments = get_assignments_via_api(eclient, testing_datasource, experiment_id)
    assert all(assignment.outcome == 0.0 for assignment in assignments)
    assert all(assignment.autofailed_outcome is True for assignment in assignments)
    assert all(assignment.observed_at is not None for assignment in assignments)


async def test_autofail_processes_in_bounded_batches_and_sleeps_between_them(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        participants=["0", "1", "2", "3"],
        name="bounded batches",
    )
    await age_draws(xngin_session, experiment_id, hours=2)
    clock = ManualClock()

    await process_autofails(
        autofail_timeout=100,
        batch_sleep=3,
        batch_size=2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.sleeps == [3, 3]
    assert all(draw.outcome == 0.0 for draw in await get_draws(xngin_session, experiment_id))


async def test_autofail_reloads_experiment_config_for_each_batch(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        autofail_outcome_value=0.0,
        participants=["0", "1"],
        name="reload configuration",
    )
    await age_draws(xngin_session, experiment_id, hours=2)
    config_updated = False

    async def update_config_after_first_batch(_duration: float) -> None:
        nonlocal config_updated
        if config_updated:
            return
        await xngin_session.execute(
            update(tables.Experiment).where(tables.Experiment.id == experiment_id).values(autofail_outcome_value=1.0)
        )
        await xngin_session.commit()
        config_updated = True

    await process_autofails(
        autofail_timeout=DEFAULT_AUTOFAIL_TIMEOUT_SECS,
        batch_sleep=0,
        batch_size=1,
        sleep=update_config_after_first_batch,
    )

    draws = await get_draws(xngin_session, experiment_id)
    assert sorted(draw.outcome for draw in draws if draw.outcome is not None) == [0.0, 1.0]


async def test_autofail_processes_one_batch_per_experiment_before_repeating(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_ids = [
        await create_autofail_experiment(
            aclient,
            eclient,
            testing_datasource,
            autofail_window=1,
            participants=["0", "1"],
            name=f"round robin {index}",
        )
        for index in range(2)
    ]
    for experiment_id in experiment_ids:
        await age_draws(xngin_session, experiment_id, hours=2)
    clock = ManualClock()
    processed_experiment_ids: list[str] = []

    async def update_two_experiments_then_reach_deadline(
        xngin_session: AsyncSession,
        experiment: tables.Experiment,
        participant_id: str,
        outcome: float,
        autofailed_outcome: bool = False,
    ) -> tables.Arm:
        arm = await experiments_common.update_bandit_arm_with_outcome_impl(
            xngin_session=xngin_session,
            experiment=experiment,
            participant_id=participant_id,
            outcome=outcome,
            autofailed_outcome=autofailed_outcome,
        )
        processed_experiment_ids.append(experiment.id)
        if len(processed_experiment_ids) == 2:
            clock.now = 10
        return arm

    await process_autofails(
        autofail_timeout=10,
        batch_sleep=0,
        batch_size=1,
        update_outcome=update_two_experiments_then_reach_deadline,
        monotonic=clock.monotonic,
    )

    assert set(processed_experiment_ids) == set(experiment_ids)
    for experiment_id in experiment_ids:
        draws = await get_draws(xngin_session, experiment_id)
        assert sum(draw.outcome is not None for draw in draws) == 1


async def test_autofail_experiment_discovery_excludes_completed_experiments(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    completed_experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        participants=["0"],
        name="completed autofail",
    )
    await age_draws(xngin_session, completed_experiment_id, hours=2)
    await process_autofails(DEFAULT_AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    pending_experiment_id = await create_autofail_experiment(
        aclient,
        eclient,
        testing_datasource,
        autofail_window=1,
        participants=["0"],
        name="pending autofail",
    )

    discovered_experiment_ids = set((await xngin_session.scalars(_autofail_experiment_ids_query())).all())

    assert completed_experiment_id not in discovered_experiment_ids
    assert pending_experiment_id in discovered_experiment_ids


async def test_autofail_rolls_back_failed_experiment_and_continues_others(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    failing_experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="failing experiment"
    )
    healthy_experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="healthy experiment"
    )
    await age_draws(xngin_session, failing_experiment_id, hours=2)
    await age_draws(xngin_session, healthy_experiment_id, hours=2)
    failing_update_count = 0

    async def fail_after_second_update(
        xngin_session: AsyncSession,
        experiment: tables.Experiment,
        participant_id: str,
        outcome: float,
        autofailed_outcome: bool = False,
    ) -> tables.Arm:
        nonlocal failing_update_count
        arm = await experiments_common.update_bandit_arm_with_outcome_impl(
            xngin_session=xngin_session,
            experiment=experiment,
            participant_id=participant_id,
            outcome=outcome,
            autofailed_outcome=autofailed_outcome,
        )
        if experiment.id != failing_experiment_id:
            return arm
        failing_update_count += 1
        if failing_update_count == 2:
            raise RuntimeError(f"failed after updating {participant_id}")
        return arm

    await process_autofails(
        DEFAULT_AUTOFAIL_TIMEOUT_SECS,
        batch_sleep=0,
        update_outcome=fail_after_second_update,
    )

    assert failing_update_count == 2
    assert all(draw.outcome is None for draw in await get_draws(xngin_session, failing_experiment_id))
    assert all(draw.outcome == 0.0 for draw in await get_draws(xngin_session, healthy_experiment_id))


async def test_autofail_keeps_prior_batches_when_a_later_batch_fails(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="later batch failure"
    )
    await age_draws(xngin_session, experiment_id, hours=2)

    update_count = 0

    async def fail_second_update(
        xngin_session: AsyncSession,
        experiment: tables.Experiment,
        participant_id: str,
        outcome: float,
        autofailed_outcome: bool = False,
    ) -> tables.Arm:
        nonlocal update_count
        arm = await experiments_common.update_bandit_arm_with_outcome_impl(
            xngin_session=xngin_session,
            experiment=experiment,
            participant_id=participant_id,
            outcome=outcome,
            autofailed_outcome=autofailed_outcome,
        )
        update_count += 1
        if update_count == 2:
            raise RuntimeError("second batch failed")
        return arm

    await process_autofails(
        DEFAULT_AUTOFAIL_TIMEOUT_SECS,
        batch_sleep=0,
        batch_size=1,
        update_outcome=fail_second_update,
    )

    draws = await get_draws(xngin_session, experiment_id)
    assert sum(draw.outcome is not None for draw in draws) == 1


async def test_autofail_deadline_is_checked_before_the_next_batch(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="deadline between batches"
    )
    await age_draws(xngin_session, experiment_id, hours=2)
    clock = ManualClock()

    async def update_then_reach_deadline(
        xngin_session: AsyncSession,
        experiment: tables.Experiment,
        participant_id: str,
        outcome: float,
        autofailed_outcome: bool = False,
    ) -> tables.Arm:
        arm = await experiments_common.update_bandit_arm_with_outcome_impl(
            xngin_session=xngin_session,
            experiment=experiment,
            participant_id=participant_id,
            outcome=outcome,
            autofailed_outcome=autofailed_outcome,
        )
        clock.now = 10
        return arm

    await process_autofails(
        autofail_timeout=10,
        batch_sleep=0,
        batch_size=1,
        update_outcome=update_then_reach_deadline,
        monotonic=clock.monotonic,
    )

    draws = await get_draws(xngin_session, experiment_id)
    assert sum(draw.outcome is not None for draw in draws) == 1


async def test_autofail_acollect_forwards_timing_options():
    calls: list[tuple[str, float, float, int] | tuple[str]] = []

    @asynccontextmanager
    async def fake_database_setup():
        calls.append(("setup",))
        yield

    async def fake_process_autofails(autofail_timeout: float, batch_sleep: float, batch_size: int) -> None:
        calls.append(("process", autofail_timeout, batch_sleep, batch_size))

    await cli.autofail_acollect(
        autofail_timeout=42,
        autofail_batch_sleep=1.5,
        autofail_batch_size=123,
        database_setup=fake_database_setup,
        process_autofails=fake_process_autofails,
    )

    assert calls == [("setup",), ("process", 42, 1.5, 123)]
