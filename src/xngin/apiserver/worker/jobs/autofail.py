"""Autofailing bandit draws at their individual expiry deadlines."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

import sentry_sdk
from loguru import logger
from sqlalchemy import ColumnElement, and_, exists, func, select
from sqlalchemy.orm import InstrumentedAttribute, Session, selectinload
from sqlalchemy.sql import Select

from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.bandit_outcomes import Outcome, apply_outcomes
from xngin.apiserver.worker.jobs.base import Backoff, JobDeferred, JobResult, JobSucceeded

AUTOFAIL_EXPERIMENT_TYPES = frozenset({ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE})


def is_pending_autofail_draw(experiment_id: str | InstrumentedAttribute[str]) -> ColumnElement[bool]:
    """Match the experiment's unresolved autofail draws, as indexed by ix_draws_pending_autofail."""
    return and_(
        tables.Draw.experiment_id == experiment_id,
        tables.Draw.enable_autofail.is_(True),
        tables.Draw.outcome.is_(None),
    )


def select_pending_autofail_draws[T](column: InstrumentedAttribute[T], experiment_id: str) -> Select[tuple[T]]:
    """Select a column of the experiment's unresolved autofail draws, oldest first."""
    return select(column).where(is_pending_autofail_draw(experiment_id)).order_by(tables.Draw.created_at)


def _eligible_draw_participant_ids_query(
    experiment_id: str,
    autofail_window: int,
    batch_size: int,
) -> Select[tuple[str]]:
    """Query for the next `batch_size` participant IDs whose draws have outlived the autofail window.

    Draws already locked by an in-progress write, such as an API outcome report, are skipped so
    the batch never waits on the API. AutofailJob.compute_next_run_at deliberately counts those draws, so a batch
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


def _autofail_experiment_query(experiment_id: str) -> Select[tuple[tables.Experiment]]:
    """Reads the full Experiment ORM object.

    We force some relations to be loaded because we are re-using the API endpoint's implementation so we need to match
    its expectations.
    """
    return (
        select(tables.Experiment)
        .where(tables.Experiment.id == experiment_id)
        .options(
            selectinload(tables.Experiment.arms),
            selectinload(tables.Experiment.contexts),
            selectinload(tables.Experiment.experiment_fields),
        )
    )


@dataclass(slots=True)
class AutofailJob:
    name: ClassVar[str] = "autofail"
    batch_size: int
    backoff: Backoff

    def needs_refresh(self) -> ColumnElement[bool]:
        # Inserting draws does not update the experiment, so the refresh must find the first pending draw itself.
        # Later draws expire after the earliest pending one, so they never move a stored run time earlier. The
        # rest is the SQL form of compute_next_run_at's checks, so it returns a time whenever this matches. There
        # is deliberately no end date check: draws can expire long after the end date.
        return and_(
            tables.JobStatus.next_run_at.is_(None),
            tables.Experiment.state == ExperimentState.COMMITTED,
            tables.Experiment.enable_autofail.is_(True),
            tables.Experiment.experiment_type.in_(AUTOFAIL_EXPERIMENT_TYPES),
            # The refresh nests this condition in a subquery on job_statuses, and SQLAlchemy correlates only with the
            # immediately enclosing query; experiments.id here would scan every experiment's draws.
            exists().where(is_pending_autofail_draw(tables.JobStatus.experiment_id)),
        )

    def compute_next_run_at(
        self, session: Session, experiment: tables.Experiment, status: tables.JobStatus, now: datetime
    ) -> datetime | None:
        if (
            experiment.state != ExperimentState.COMMITTED
            or not experiment.enable_autofail
            or experiment.experiment_type not in AUTOFAIL_EXPERIMENT_TYPES
        ):
            return None
        # Read without locking, so draws locked by an in-progress API write still count, unlike in
        # _eligible_draw_participant_ids_query. Skipping them here could leave a draw unscheduled if that write
        # rolled back. Unresolved draws still count after the experiment ends.
        created_at = session.scalar(select_pending_autofail_draws(tables.Draw.created_at, experiment.id).limit(1))
        return created_at + timedelta(hours=experiment.autofail_window) if created_at is not None else None

    def execute(self, session: Session, experiment_id: str) -> JobResult:
        """Autofail one batch of overdue draws.

        The caller's claim holds the experiment row, so it cannot be deleted while the batch runs.
        """
        experiment = session.scalars(_autofail_experiment_query(experiment_id)).one()
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
            # by an in-progress API write. compute_next_run_at still counts those draws, so without a
            # delay the job would be due again at once and spin until that transaction commits. A batch that made
            # progress needs no delay: if more draws are overdue, compute_next_run_at makes the job due again.
            return JobDeferred(retry_after=timedelta(seconds=1))
        return JobSucceeded()
