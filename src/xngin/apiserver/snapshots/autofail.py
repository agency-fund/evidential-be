import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

import sentry_sdk
from loguru import logger
from sqlalchemy import func, select, text
from sqlalchemy.orm import contains_eager
from sqlalchemy.sql import Select

from xngin.apiserver import database
from xngin.apiserver.routers.common_api_types import ExperimentsType
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.sqla import tables

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

AUTOFAIL_BATCH_SIZE = 500
AUTOFAIL_TIMEOUT_SECS = 45 * 60
AUTOFAIL_BATCH_SLEEP_SECS = 1.0


class AutofailOutcomeUpdater(Protocol):
    async def __call__(
        self,
        xngin_session: AsyncSession,
        experiment: tables.Experiment,
        participant_id: str,
        outcome: float,
        autofailed_outcome: bool = False,
    ) -> tables.Arm: ...


def _eligible_draws_query(batch_size: int) -> Select[tuple[tables.Draw]]:
    experiment_loader = contains_eager(tables.Draw.experiment)
    return (
        select(tables.Draw)
        .join(tables.Draw.experiment)
        .where(
            tables.Experiment.enable_autofail.is_(True),
            tables.Experiment.experiment_type.in_([
                ExperimentsType.MAB_ONLINE,
                ExperimentsType.MAB_ONLINE_DWH,
                ExperimentsType.CMAB_ONLINE,
            ]),
            tables.Draw.outcome.is_(None),
            tables.Draw.created_at < func.now() - tables.Experiment.autofail_window * text("interval '1 hour'"),
        )
        .order_by(tables.Draw.created_at, tables.Draw.experiment_id, tables.Draw.participant_id)
        .limit(batch_size)
        .with_for_update(of=tables.Draw, skip_locked=True)
        .options(
            experiment_loader.selectinload(tables.Experiment.arms),
            experiment_loader.selectinload(tables.Experiment.contexts),
            experiment_loader.selectinload(tables.Experiment.experiment_fields),
        )
    )


async def _update_one_draw(
    session: AsyncSession,
    draw: tables.Draw,
    update_outcome: AutofailOutcomeUpdater,
) -> None:
    experiment = draw.experiment
    with logger.contextualize(experiment_id=experiment.id, participant_id=draw.participant_id):
        await update_outcome(
            xngin_session=session,
            experiment=experiment,
            participant_id=draw.participant_id,
            outcome=experiment.autofail_outcome_value,
            autofailed_outcome=True,
        )


async def process_autofail_updates(
    autofail_timeout: float,
    batch_sleep: float,
    batch_size: int = AUTOFAIL_BATCH_SIZE,
    *,
    update_outcome: AutofailOutcomeUpdater = experiments_common.update_bandit_arm_with_outcome_impl,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Autofail eligible draws in atomic batches until no work remains or the run deadline is reached."""
    if autofail_timeout < 0:
        raise ValueError("autofail_timeout must be non-negative")
    if batch_sleep < 0:
        raise ValueError("batch_sleep must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")

    started_at = monotonic()
    deadline = started_at + autofail_timeout
    processed = 0
    batches = 0
    logger.info(
        f"Autofail run started with batch_size={batch_size}, timeout={autofail_timeout}s, batch_sleep={batch_sleep}s."
    )

    async with database.async_session() as session:
        while monotonic() < deadline:
            batch_number = batches + 1
            current_experiment_id: str | None = None
            attempted_in_batch = 0
            try:
                async with session.begin():
                    draws: list[tables.Draw] = list(
                        (await session.execute(_eligible_draws_query(batch_size))).scalars().all()
                    )
                    if not draws:
                        elapsed = monotonic() - started_at
                        logger.info(
                            f"Autofail run finished after {elapsed:.2f}s; "
                            f"committed {processed} updates in {batches} batches."
                        )
                        return

                    logger.info(f"Autofail batch {batch_number} selected {len(draws)} eligible draws.")
                    for draw in draws:
                        current_experiment_id = draw.experiment_id
                        await _update_one_draw(session, draw, update_outcome)
                        attempted_in_batch += 1
            except Exception as exc:
                attributes: dict[str, str | int] = {"batch": batch_number}
                if current_experiment_id is not None:
                    attributes.update({"experiment_id": current_experiment_id})
                logger.opt(exception=exc).error(
                    f"Autofail batch {batch_number} failed after {attempted_in_batch} attempted updates and was rolled "
                    f"back; processed={processed}, attributes={attributes}."
                )
                sentry_sdk.metrics.count("autofail_update.failed", 1, attributes=attributes)
                raise

            processed += len(draws)
            batches += 1
            sentry_sdk.metrics.count(
                "autofail_update.finished",
                len(draws),
                attributes={"experiment_id": draw.experiment_id},
            )
            logger.info(
                f"Autofail batch {batches} committed {len(draws)} updates; total committed updates={processed}."
            )

            await sleep(batch_sleep)

    elapsed = monotonic() - started_at
    logger.info(
        f"Autofail run reached its deadline after {elapsed:.2f}s; committed {processed} updates in {batches} batches."
    )
    return
