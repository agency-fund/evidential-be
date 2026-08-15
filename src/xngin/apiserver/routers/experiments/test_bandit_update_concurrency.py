"""Vibed test to validate that two concurrent update_bandit_arm_with_participant_outcome calls don't lose updates."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
import sqlalchemy
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from xngin.apiserver import database
from xngin.apiserver.conftest import DatasourceMetadata
from xngin.apiserver.routers.common_api_types import ArmBandit, CreateExperimentRequest, DesignSpec, MABExperimentSpec
from xngin.apiserver.routers.common_enums import ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.routers.experiments.experiments_common import ExperimentsAssignmentError
from xngin.apiserver.sqla import tables

if TYPE_CHECKING:
    from xngin.apiserver.testing.admin_api_client import AdminAPIClient
    from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient


@dataclass(frozen=True, slots=True)
class _ContendedBandit:
    experiment_id: str
    design_spec: DesignSpec
    arm_id: str
    participant_ids: tuple[str, str]


@dataclass(frozen=True, slots=True)
class _PendingOutcomeUpdate:
    task: asyncio.Task[tables.Arm | ExperimentsAssignmentError]
    backend_pid: asyncio.Future[int]

    async def expect_blocked(self) -> None:
        """Wait until PostgreSQL reports that this update is blocked acquiring a lock."""
        backend_pid = await self.backend_pid
        async with database.async_session() as monitor_session:
            while True:
                waiting_on_lock = await monitor_session.scalar(
                    sqlalchemy.text("SELECT wait_event_type = 'Lock' FROM pg_stat_activity WHERE pid = :pid"),
                    {"pid": backend_pid},
                )
                if waiting_on_lock:
                    return
                await asyncio.sleep(0.01)

    async def result(self) -> tables.Arm:
        result = await self.task
        if isinstance(result, ExperimentsAssignmentError):
            raise result
        return result


async def _create_contended_bandit(
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    datasource: DatasourceMetadata,
) -> _ContendedBandit:
    design_spec = MABExperimentSpec(
        experiment_name="concurrent outcome updates",
        description="",
        start_date=datetime.now(UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
        arms=[
            ArmBandit(arm_name="control", arm_description="", alpha_init=1, beta_init=1),
            ArmBandit(arm_name="treatment", arm_description="", alpha_init=1, beta_init=1),
        ],
    )
    experiment_id = aclient.create_experiment(
        datasource_id=datasource.datasource_id,
        body=CreateExperimentRequest(design_spec=design_spec),
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=datasource.datasource_id, experiment_id=experiment_id)

    participants_by_arm: dict[str, list[str]] = {}
    for participant_id in ("p1", "p2", "p3"):
        assignment = eclient.get_assignment(
            api_key=datasource.key,
            experiment_id=experiment_id,
            participant_id=participant_id,
            create_if_none=True,
        ).data.assignment
        assert assignment is not None
        participants = participants_by_arm.setdefault(assignment.arm_id, [])
        participants.append(participant_id)
        if len(participants) == 2:
            return _ContendedBandit(
                experiment_id=experiment_id,
                design_spec=design_spec,
                arm_id=assignment.arm_id,
                participant_ids=(participants[0], participants[1]),
            )

    raise AssertionError("three participants assigned across two arms must include a shared arm")


async def _load_experiment(session: AsyncSession, experiment_id: str) -> tables.Experiment:
    experiment = await session.scalar(
        select(tables.Experiment)
        .where(tables.Experiment.id == experiment_id)
        .options(selectinload(tables.Experiment.arms))
    )
    assert experiment is not None
    return experiment


def _get_arm(aclient: AdminAPIClient, datasource_id: str, bandit: _ContendedBandit) -> ArmBandit:
    design_spec = aclient.get_experiment_for_ui(
        datasource_id=datasource_id,
        experiment_id=bandit.experiment_id,
    ).data.config.design_spec
    assert isinstance(design_spec, MABExperimentSpec)
    return next(arm for arm in design_spec.arms if arm.arm_id == bandit.arm_id)


@asynccontextmanager
async def _hold_outcome_update(
    experiment_id: str,
    design_spec: DesignSpec,
    participant_id: str,
    outcome: float,
) -> AsyncIterator[None]:
    """Hold an outcome's row locks, then apply and commit it when released."""
    async with database.async_session() as session:
        experiment = await _load_experiment(session, experiment_id)
        state = await experiments_common._read_arm_draw_state(session, experiment_id, participant_id)
        state = experiments_common._validate_outcome_update(
            experiment,
            design_spec,
            state,
            participant_id,
            outcome,
        )
        plan = experiments_common._compute_update(experiment, state, outcome)
        yield
        experiments_common._apply_update_plan(plan)
        await session.commit()


def _start_outcome_update(
    task_group: asyncio.TaskGroup,
    experiment_id: str,
    participant_id: str,
    outcome: float,
) -> _PendingOutcomeUpdate:
    backend_pid: asyncio.Future[int] = asyncio.get_running_loop().create_future()

    async def record() -> tables.Arm | ExperimentsAssignmentError:
        async with database.async_session() as session:
            experiment = await _load_experiment(session, experiment_id)
            pid = await session.scalar(sqlalchemy.text("SELECT pg_backend_pid()"))
            assert pid is not None
            backend_pid.set_result(pid)
            try:
                arm = await experiments_common.update_bandit_with_outcome_impl(
                    session,
                    experiment,
                    participant_id,
                    outcome,
                )
            except ExperimentsAssignmentError as exc:
                return exc
            await session.commit()
            return arm

    return _PendingOutcomeUpdate(task=task_group.create_task(record()), backend_pid=backend_pid)


async def test_concurrent_outcomes_on_one_arm_do_not_lose_contributions(
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    testing_datasource: DatasourceMetadata,
):
    bandit = await _create_contended_bandit(aclient, eclient, testing_datasource)
    alpha_before = _get_arm(aclient, testing_datasource.datasource_id, bandit).alpha
    assert alpha_before is not None
    first_participant, second_participant = bandit.participant_ids

    async with (
        asyncio.timeout(30),
        asyncio.TaskGroup() as task_group,
        _hold_outcome_update(bandit.experiment_id, bandit.design_spec, first_participant, 1),
    ):
        second = _start_outcome_update(task_group, bandit.experiment_id, second_participant, 1)
        await second.expect_blocked()

    _ = await second.result()
    alpha_after = _get_arm(aclient, testing_datasource.datasource_id, bandit).alpha
    assert alpha_after is not None
    assert alpha_after - alpha_before == 2


async def test_concurrent_outcomes_for_one_participant_refuse_the_second(
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    testing_datasource: DatasourceMetadata,
):
    bandit = await _create_contended_bandit(aclient, eclient, testing_datasource)
    alpha_before = _get_arm(aclient, testing_datasource.datasource_id, bandit).alpha
    assert alpha_before is not None
    participant_id = bandit.participant_ids[0]

    async with (
        asyncio.timeout(30),
        asyncio.TaskGroup() as task_group,
        _hold_outcome_update(bandit.experiment_id, bandit.design_spec, participant_id, 1),
    ):
        duplicate = _start_outcome_update(task_group, bandit.experiment_id, participant_id, 1)
        await duplicate.expect_blocked()

    with pytest.raises(ExperimentsAssignmentError, match="already has an outcome recorded"):
        await duplicate.result()

    alpha_after = _get_arm(aclient, testing_datasource.datasource_id, bandit).alpha
    assert alpha_after is not None
    assert alpha_after - alpha_before == 1
