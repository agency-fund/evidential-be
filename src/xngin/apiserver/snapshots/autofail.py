import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

import sentry_sdk
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select

from xngin.apiserver import database
from xngin.apiserver.routers.common_api_types import ExperimentsType
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.sqla import tables

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_AUTOFAIL_BATCH_SIZE = 500
DEFAULT_AUTOFAIL_TIMEOUT_SECS = timedelta(minutes=45).seconds
DEFAULT_AUTOFAIL_BATCH_SLEEP_SECS = 1.0

AUTOFAIL_EXPERIMENT_TYPES = (
    ExperimentsType.MAB_ONLINE,
    ExperimentsType.MAB_ONLINE_DWH,
    ExperimentsType.CMAB_ONLINE,
)


class AutofailOutcomeUpdater(Protocol):
    async def __call__(
        self,
        xngin_session: AsyncSession,
        experiment: tables.Experiment,
        participant_id: str,
        outcome: float,
        autofailed_outcome: bool = False,
    ) -> tables.Arm: ...


def _autofail_experiment_ids_query() -> Select[tuple[str]]:
    """Collect eligible autofailing experiment IDs."""

    # Uses the partial index to efficiently skip experiments without any unresolved outcomes.
    has_pending_draw = (
        select(tables.Draw.experiment_id)
        .where(
            tables.Draw.experiment_id == tables.Experiment.id,
            tables.Draw.enable_autofail.is_(True),
            tables.Draw.outcome.is_(None),
        )
        .exists()
    )
    return (
        select(tables.Experiment.id)
        .where(
            tables.Experiment.enable_autofail.is_(True),
            tables.Experiment.experiment_type.in_(AUTOFAIL_EXPERIMENT_TYPES),
            has_pending_draw,
        )
        .order_by(func.random())  # return experiment IDs in a random order
    )


def _eligible_draw_participant_ids_query(
    experiment_id: str,
    autofail_window: int,
    batch_size: int,
) -> Select[tuple[str]]:
    """Query for participant IDs in an experiment that should be autofailed.

    tables.Draw has a partial index of pending autofail candidates ordered by created_at. The autofail window predicate
    narrows those candidates to the next `batch_size` items eligible for processing.
    """
    return (
        select(tables.Draw.participant_id)
        .where(
            tables.Draw.experiment_id == experiment_id,
            tables.Draw.enable_autofail.is_(True),
            tables.Draw.outcome.is_(None),
            tables.Draw.created_at < func.now() - timedelta(hours=autofail_window),
        )
        .order_by(tables.Draw.created_at)
        .limit(batch_size)
        .with_for_update(of=tables.Draw, skip_locked=True)
    )


def _autofail_experiment_query(experiment_id: str) -> Select[tuple[tables.Experiment]]:
    """Reads the full Experiment ORM object.

    We force some relations to be loaded because we are re-using the API endpoint's implementation so we need to match
    its expectations.
    """
    return (
        select(tables.Experiment)
        .where(
            tables.Experiment.id == experiment_id,
            tables.Experiment.enable_autofail.is_(True),
            tables.Experiment.experiment_type.in_(AUTOFAIL_EXPERIMENT_TYPES),
        )
        .options(
            selectinload(tables.Experiment.arms),
            selectinload(tables.Experiment.contexts),
            selectinload(tables.Experiment.experiment_fields),
        )
    )


async def _process_autofail_batch_for_experiment(
    experiment_id: str,
    batch_size: int,
    update_outcome: AutofailOutcomeUpdater,
) -> int | None:
    """Processes one batch of autofails for a single experiment in a single transaction.

    Returns the number of participant outcomes committed, or None when the experiment is no longer eligible for
    autofailing or has no draws currently eligible for processing.
    """
    async with database.async_session() as session, session.begin():
        experiment = await session.scalar(_autofail_experiment_query(experiment_id))
        if experiment is None:
            logger.warning(f"Autofail tried to process an experiment that is no longer eligible: {experiment_id}")
            return None

        participant_ids: list[str] = list(
            (
                await session.scalars(
                    _eligible_draw_participant_ids_query(
                        experiment.id,
                        experiment.autofail_window,
                        batch_size,
                    )
                )
            ).all()
        )
        if not participant_ids:
            logger.info(f"Autofail processing finished for experiment {experiment_id}.")
            return None

        for participant_id in participant_ids:
            with logger.contextualize(participant_id=participant_id):
                await update_outcome(
                    xngin_session=session,
                    experiment=experiment,
                    participant_id=participant_id,
                    outcome=experiment.autofail_outcome_value,
                    autofailed_outcome=True,
                )

    return len(participant_ids)


async def process_autofails(
    autofail_timeout: float,
    batch_sleep: float,
    batch_size: int = DEFAULT_AUTOFAIL_BATCH_SIZE,
    *,
    update_outcome: AutofailOutcomeUpdater = experiments_common.update_bandit_arm_with_outcome_impl,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Autofail eligible draws in atomic batches until no work remains or the run deadline is reached.

    We process the experiments in random order, one batch per experiment, in a round-robin fashion. As deadline permits,
    we process subsequent batches.
    """

    started_at = monotonic()
    deadline = started_at + autofail_timeout

    processed = 0
    batches = 0
    logger.info(
        f"Autofail run started with batch_size={batch_size}, timeout={autofail_timeout}s, batch_sleep={batch_sleep}s."
    )

    async with database.async_session() as discovery_session:
        active_experiment_ids = deque(await discovery_session.scalars(_autofail_experiment_ids_query()))
    logger.info(f"Autofail run found {len(active_experiment_ids)} eligible experiments.")

    while active_experiment_ids and monotonic() < deadline:
        experiment_id = active_experiment_ids.popleft()
        with logger.contextualize(experiment_id=experiment_id):
            batch_number = batches + 1
            try:
                processed_in_batch = await _process_autofail_batch_for_experiment(
                    experiment_id,
                    batch_size,
                    update_outcome,
                )
            except Exception as exc:
                logger.opt(exception=exc).error(f"Autofail batch {batch_number} failed and was rolled back.")
                sentry_sdk.metrics.count(
                    "autofail.batches.failed",
                    1,
                    attributes={"batch": batch_number, "experiment_id": experiment_id},
                )
                raise
            if processed_in_batch is None:
                continue
            processed += processed_in_batch
            batches += 1
            active_experiment_ids.append(experiment_id)

            logger.info(
                f"Autofail batch {batches} committed {processed_in_batch} updates for experiment {experiment_id}; "
                f"total committed updates={processed}."
            )
            sentry_sdk.metrics.count(
                "autofail.batches.finished",
                1,
                attributes={"experiment_id": experiment_id},
            )
            await sleep(batch_sleep)

    elapsed = monotonic() - started_at
    if active_experiment_ids:
        logger.warning(
            f"Autofail did not finish processing all eligible experiments within the deadline: elapsed={elapsed:.2f}s; "
            f"experiments remaining={len(active_experiment_ids)}; "
            f"committed {processed} updates in {batches} batches."
        )
    else:
        logger.info(
            f"Autofail finished processing all eligible experiments in {elapsed:.2f}s; committed {processed} "
            f"updates in {batches} batches."
        )
