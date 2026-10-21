"""Transaction-neutral outcome batches shared by maintenance jobs."""

from collections.abc import Iterable
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.experiments.experiments_common import update_bandit_arm_with_outcome_impl
from xngin.apiserver.sqla import tables


@dataclass(slots=True, frozen=True)
class Outcome:
    participant_id: str
    # None means no value is available yet, so the draw stays unresolved.
    value: float | None
    autofailed: bool = False


@dataclass(slots=True)
class OutcomeReport:
    # Count of successfully applied outcome updates
    ingested: int = 0
    # The value was incompatible for this experiment type, or some other failure.
    invalid: int = 0
    # Not in the warehouse yet, or the target column is still NULL. The draw keeps
    # outcome=NULL and is re-read on a later run.
    pending: int = 0


def apply_outcomes(
    session: Session,
    experiment: tables.Experiment,
    outcomes: Iterable[Outcome],
    *,
    skip_invalid: bool = False,
) -> OutcomeReport:
    """Apply a batch without committing, preserving per-draw posterior/history semantics.

    Outcomes without a value are counted as pending and left unresolved. Validation errors precede writes in the
    updater. Only warehouse ingestion may skip them; all other failures roll back through the caller's transaction
    boundary.
    """
    report = OutcomeReport()
    for outcome in outcomes:
        if outcome.value is None:
            report.pending += 1
            continue
        with logger.contextualize(participant_id=outcome.participant_id):
            try:
                update_bandit_arm_with_outcome_impl(
                    xngin_session=session,
                    experiment=experiment,
                    participant_id=outcome.participant_id,
                    outcome=outcome.value,
                    autofailed_outcome=outcome.autofailed,
                )
            except LateValidationError as exc:
                if not skip_invalid:
                    raise
                logger.info(f"Skipping invalid warehouse outcome: {exc}")
                report.invalid += 1
            else:
                report.ingested += 1
    return report
