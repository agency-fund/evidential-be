import asyncio
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import sentry_sdk
from loguru import logger
from sqlalchemy import and_, func, select
from sqlalchemy.orm import selectinload

from xngin.apiserver import database
from xngin.apiserver.routers.common_api_types import ExperimentsType
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.sqla import tables

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# The amount of time the API server will wait for an autofail update to complete.
AUTOFAIL_TIMEOUT_SECS = 90


async def create_pending_autofail_updates() -> None:
    """
    Identify experiments and corresponding assignments for which we should create pending autofail updates.
    For each experiment with autofail enabled, we check the assignments that don't have an outcome recorded.
    Of these assignments, those that have exceeded the set autofail window (e.g., 1 hour) we create a new entry
    in the PendingAutofailUpdate table to indicate that an autofailed outcome update should be processed.
    """
    async with database.async_session() as session:
        # Query for experiments that are running and have autofail enabled.
        stmt = (
            select(tables.Draw)
            .join(
                tables.Experiment,
                tables.Draw.experiment_id == tables.Experiment.id,
            )
            .where(
                tables.Experiment.enable_autofail.is_(True),
                tables.Experiment.experiment_type.in_([
                    ExperimentsType.MAB_ONLINE,
                    ExperimentsType.MAB_ONLINE_DWH,
                    ExperimentsType.CMAB_ONLINE,
                ]),
                tables.Draw.outcome.is_(None),
            )
        )

        result = await session.execute(stmt)
        draws = result.scalars().all()
        n_autofail_updates = 0
        for draw in draws:
            experiment = draw.experiment
            # Check if the draw has no outcome and has exceeded the threshold time.
            now = datetime.now(UTC)
            if ((now - draw.created_at).total_seconds() / 3600) > experiment.autofail_window:
                # Create a pending autofail update for this draw.
                pending_update = tables.AutofailUpdate(
                    participant_id=draw.participant_id,
                    experiment_id=experiment.id,
                    created_at=func.now(),
                    status="pending",
                )
                n_autofail_updates += 1
                session.add(pending_update)
        await session.commit()
        logger.info(f"Created {n_autofail_updates} pending autofail updates.")


async def _make_one_autofail_update(
    update: tables.AutofailUpdate,
    draw: tables.Draw,
    session: AsyncSession,
    autofail_update_timeout: int,
) -> None:
    """
    Process one pending autofail update. This function is intended to be called repeatedly,
    e.g., in a loop or by a scheduler. It checks for any pending autofail updates and processes
    one of them by updating the corresponding draw's outcome to "autofailed".
    """
    experiment = draw.experiment
    experiment_id = draw.experiment_id
    participant_id = draw.participant_id
    outcome = experiment.autofail_outcome_value

    try:
        async with asyncio.timeout(autofail_update_timeout):
            async with session.begin_nested():
                await experiments_common.update_bandit_arm_with_outcome_impl(
                    xngin_session=session,
                    experiment=experiment,
                    participant_id=participant_id,
                    outcome=outcome,
                    autofailed_outcome=True,
                )
            update.status = "success"
            update.message = (
                f"Autofail processed successfully. Participant {participant_id} recorded outcome {outcome}."
            )

    except Exception as exc:
        logger.opt(exception=exc).error(
            f"Failed to process autofail update for experiment {experiment_id} and participant {participant_id}: "
            f"{exc!s}"
        )
        sentry_sdk.capture_exception(exc)
        sentry_sdk.metrics.count(
            "autofail_update.failed",
            1,
            attributes={"experiment_id": experiment_id, "participant_id": participant_id},
        )
        update.status = "failed"
        update.message = f"{type(exc).__name__}: {exc}"

    sentry_sdk.metrics.count(
        "autofail_update.finished",
        1,
        attributes={"experiment_id": experiment_id, "participant_id": participant_id},
    )
    logger.info(f"Autofail update for experiment {experiment_id} and participant {participant_id}: done")


async def process_pending_autofail_updates(autofail_update_timeout: int, *, max_jitter_secs: float = 2):
    """
    Process pending autofail updates. For each pending update, we check if the corresponding draw still has no outcome.
    If so, we update the draw's outcome to indicate that it has been autofailed. We also mark the pending update
    as processed.
    """
    autofail_update = (
        select(tables.AutofailUpdate)
        .join(
            tables.Draw,
            and_(
                tables.AutofailUpdate.experiment_id == tables.Draw.experiment_id,
                tables.AutofailUpdate.participant_id == tables.Draw.participant_id,
            ),
        )
        .join(tables.Experiment, tables.Draw.experiment_id == tables.Experiment.id)
        .where(tables.AutofailUpdate.status == "pending")
        .limit(1)
        .with_for_update(skip_locked=True)
        .options(selectinload(tables.AutofailUpdate.draw).joinedload(tables.Draw.experiment))
    )
    while True:
        await asyncio.sleep(random.uniform(0, max_jitter_secs))  # jitter  # noqa: S311
        async with database.async_session() as session, session.begin():
            one_update = (await session.execute(autofail_update)).scalar_one_or_none()
            if one_update is None:
                logger.info("No pending autofail updates found.")
                return
            draw = one_update.draw
            await _make_one_autofail_update(
                update=one_update,
                draw=draw,
                session=session,
                autofail_update_timeout=autofail_update_timeout,
            )
