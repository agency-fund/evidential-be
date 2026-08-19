from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

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
    MABDwhExperimentSpec,
    MABExperimentSpec,
    PriorTypes,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.snapshots import cli
from xngin.apiserver.snapshots.autofail import (
    AUTOFAIL_BATCH_SIZE,
    AUTOFAIL_TIMEOUT_SECS,
    process_autofail_updates,
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

    await process_autofail_updates(AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

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

    await process_autofail_updates(AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    assert all(draw.outcome is None for draw in await get_draws(xngin_session, experiment_id))


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

    await process_autofail_updates(AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    draws = await get_draws(xngin_session, experiment_id)
    assert [(draw.participant_id, draw.outcome, draw.autofailed_outcome) for draw in draws] == [
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

    await process_autofail_updates(AUTOFAIL_TIMEOUT_SECS, batch_sleep=0)

    draws = await get_draws(xngin_session, experiment_id)
    assert all(draw.outcome == 0.0 for draw in draws)
    assert all(draw.autofailed_outcome is True for draw in draws)
    assert all(draw.observed_at is not None for draw in draws)


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

    await process_autofail_updates(
        autofail_timeout=100,
        batch_sleep=3,
        batch_size=2,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert AUTOFAIL_BATCH_SIZE == 500
    assert clock.sleeps == [3, 3]
    assert all(draw.outcome == 0.0 for draw in await get_draws(xngin_session, experiment_id))


async def test_autofail_is_noop_when_no_draws_are_eligible(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="nothing eligible"
    )
    await age_draws(xngin_session, experiment_id, hours=0.5)
    clock = ManualClock()

    await process_autofail_updates(
        autofail_timeout=100,
        batch_sleep=3,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert clock.sleeps == []


async def test_autofail_rolls_back_the_batch_on_failure(
    xngin_session: AsyncSession,
    testing_datasource,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
):
    experiment_id = await create_autofail_experiment(
        aclient, eclient, testing_datasource, autofail_window=1, name="atomic batch"
    )
    await age_draws(xngin_session, experiment_id, hours=2)

    async def fail_after_update(
        xngin_session: AsyncSession,
        experiment: tables.Experiment,
        participant_id: str,
        outcome: float,
        autofailed_outcome: bool = False,
    ) -> tables.Arm:
        await experiments_common.update_bandit_arm_with_outcome_impl(
            xngin_session=xngin_session,
            experiment=experiment,
            participant_id=participant_id,
            outcome=outcome,
            autofailed_outcome=autofailed_outcome,
        )
        raise RuntimeError(f"failed after updating {participant_id}")

    with pytest.raises(RuntimeError, match="failed after updating 0"):
        await process_autofail_updates(
            AUTOFAIL_TIMEOUT_SECS,
            batch_sleep=0,
            update_outcome=fail_after_update,
        )

    assert all(draw.outcome is None for draw in await get_draws(xngin_session, experiment_id))


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

    async def fail_second_participant(
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
        if participant_id == "1":
            raise RuntimeError("second batch failed")
        return arm

    with pytest.raises(RuntimeError, match="second batch failed"):
        await process_autofail_updates(
            AUTOFAIL_TIMEOUT_SECS,
            batch_sleep=0,
            batch_size=1,
            update_outcome=fail_second_participant,
        )

    draws = await get_draws(xngin_session, experiment_id)
    assert [(draw.participant_id, draw.outcome) for draw in draws] == [("0", 0.0), ("1", None)]


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

    await process_autofail_updates(
        autofail_timeout=10,
        batch_sleep=0,
        batch_size=1,
        update_outcome=update_then_reach_deadline,
        monotonic=clock.monotonic,
    )

    draws = await get_draws(xngin_session, experiment_id)
    assert [(draw.participant_id, draw.outcome) for draw in draws] == [("0", 0.0), ("1", None)]


async def test_autofail_acollect_forwards_timing_options():
    calls: list[tuple[str, float, float, int] | tuple[str]] = []

    @asynccontextmanager
    async def fake_database_setup():
        calls.append(("setup",))
        yield

    async def fake_process_updates(autofail_timeout: float, batch_sleep: float, batch_size: int) -> None:
        calls.append(("process", autofail_timeout, batch_sleep, batch_size))

    await cli.autofail_acollect(
        autofail_timeout=42,
        autofail_batch_sleep=1.5,
        autofail_batch_size=123,
        database_setup=fake_database_setup,
        process_updates=fake_process_updates,
    )

    assert calls == [("setup",), ("process", 42, 1.5, 123)]


def test_collect_help_exposes_autofail_timing_options():
    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "--autofail-timeout" in result.output
    assert "--autofail-batch-sleep" in result.output
    assert "--autofail-batch-size" in result.output
