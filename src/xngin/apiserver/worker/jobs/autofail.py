"""Autofailing bandit draws at their individual expiry deadlines."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

import sentry_sdk
from loguru import logger
from sqlalchemy import ColumnElement, and_, case, func, select
from sqlalchemy.orm import InstrumentedAttribute, Session
from sqlalchemy.sql import Select

from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.bandit_outcomes import Outcome, apply_outcomes, is_pending_draw
from xngin.apiserver.worker.jobs.base import Backoff, JobDeferred, JobResult, JobSucceeded

AUTOFAIL_EXPERIMENT_TYPES = frozenset({ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE})


def is_pending_autofail_draw(experiment_id: str | InstrumentedAttribute[str]) -> ColumnElement[bool]:
    """Match the experiment's unresolved autofail draws, as indexed by ix_draws_pending_autofail."""
    return and_(is_pending_draw(experiment_id), tables.Draw.enable_autofail.is_(True))


def select_pending_autofail_draws[T](
    column: InstrumentedAttribute[T], experiment_id: str | InstrumentedAttribute[str]
) -> Select[tuple[T]]:
    """Select a column of the experiment's unresolved autofail draws, oldest first."""
    return select(column).where(is_pending_autofail_draw(experiment_id)).order_by(tables.Draw.created_at)


def _eligible_draw_participant_ids_query(
    experiment_id: str,
    autofail_window: int,
    batch_size: int,
) -> Select[tuple[str]]:
    """Query for the next `batch_size` participant IDs whose draws have outlived the autofail window.

    Draws already locked by an in-progress write, such as an API outcome report, are skipped so
    the batch never waits on the API. AutofailJob.next_run_at deliberately counts those draws, so a batch
    can come back empty while the job is due; AutofailJob.execute handles that.
    """
    return (
        select_pending_autofail_draws(tables.Draw.participant_id, experiment_id)
        # A fresh statement timestamp also lets the index use a constant cutoff for this batch.
        .where(tables.Draw.created_at <= func.statement_timestamp() - timedelta(hours=autofail_window))
        .limit(batch_size)
        # key_share without read renders FOR NO KEY UPDATE, which conflicts with an API outcome UPDATE.
        .with_for_update(of=tables.Draw, skip_locked=True, key_share=True)
    )


@dataclass(slots=True)
class AutofailJob:
    name: ClassVar[str] = "autofail"
    batch_size: int
    backoff: Backoff

    def next_run_at(self, now: ColumnElement[datetime]) -> ColumnElement[datetime | None]:
        experiment = tables.Experiment
        # Read without locking, so draws locked by an in-progress API write still count, unlike in
        # _eligible_draw_participant_ids_query. Skipping them here could leave a draw unscheduled if that write
        # rolled back. Unresolved draws still count after the experiment ends, so there is deliberately no end date
        # check: draws can expire long after the end date.
        first_pending_at = (
            select_pending_autofail_draws(tables.Draw.created_at, experiment.id).limit(1).scalar_subquery()
        )
        return case(
            (
                and_(
                    experiment.enable_autofail.is_(True),
                    experiment.experiment_type.in_(AUTOFAIL_EXPERIMENT_TYPES),
                ),
                first_pending_at + func.make_interval(0, 0, 0, 0, experiment.autofail_window),
            ),
            else_=None,
        )

    def execute(self, session: Session, experiment: tables.Experiment) -> JobResult:
        """Autofail one batch of overdue draws.

        The caller's claim holds the experiment row, so it cannot be deleted while the batch runs.
        """
        participant_ids = list(
            session.scalars(
                _eligible_draw_participant_ids_query(experiment.id, experiment.autofail_window, self.batch_size)
            )
        )
        apply_outcomes(
            session,
            experiment,
            (Outcome(participant_id, experiment.autofail_outcome_value, True) for participant_id in participant_ids),
        )
        logger.info(f"Autofailed {len(participant_ids)} draws.")
        sentry_sdk.metrics.count("worker.autofail.outcomes", len(participant_ids), attributes={"status": "applied"})
        if not participant_ids:
            # The job was due, yet _eligible_draw_participant_ids_query found nothing: every overdue draw is locked
            # by an in-progress API write. next_run_at still counts those draws, so without a
            # delay the job would be due again at once and spin until that transaction commits. A batch that made
            # progress needs no delay: if more draws are overdue, next_run_at makes the job due again.
            return JobDeferred(retry_after=timedelta(seconds=1))
        return JobSucceeded()
