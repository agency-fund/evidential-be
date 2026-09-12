"""Reads bandit outcomes from an organization's data warehouse.

MAB_ONLINE_DWH experiments have no outcomes pushed to the API. For those, this module finds draws
that still have no outcome, reads the experiment's target column for those participants, and applies
each value through the same per-outcome update the push API uses.

Each experiment's outcomes are applied in a single transaction, so an experiment either pulls or it
does not. A run that dies partway leaves the experiment untouched and the next run repeats it.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import sentry_sdk
from loguru import logger
from sqlalchemy import func, select, text
from sqlalchemy.orm import selectinload

from xngin.apiserver import database
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.participant_metrics_queries import get_participant_metrics
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.flags import NPROC
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType
from xngin.apiserver.routers.experiments.experiments_common import (
    MismatchedExperimentTypeError,
    update_bandit_arm_with_outcome_impl,
)
from xngin.apiserver.sqla import tables

# The amount of time one experiment's warehouse read may run before the pull is abandoned. The pull
# job can specify a different budget via command line flags.
PULL_TIMEOUT_SECS = 90


# Each concurrent pull holds an application-database connection for the whole of its warehouse read, so this should
# stay below the size of the connection pool.
MAX_PULL_WORKERS = NPROC


@dataclass(slots=True)
class PullReport:
    """Counts from one experiment's pull."""

    ingested: int = 0
    invalid: int = 0
    pending: int = 0

    def summary(self) -> str:
        return f"ingested={self.ingested} invalid={self.invalid} pending={self.pending}"


def select_experiments_to_pull() -> list[str]:
    """Return the ids of the experiments whose outcomes we read from a data warehouse.

    Covers committed MAB-DWH experiments that are running, start tomorrow, or ended yesterday. The
    day either side gives outcomes that land late a chance to arrive.

    This method establishes its own database connection.
    """
    buffer = text("interval '1 day'")
    with database.get_session() as session:
        return list(
            session.scalars(
                select(tables.Experiment.id)
                .where(
                    tables.Experiment.experiment_type == ExperimentsType.MAB_ONLINE_DWH.value,
                    tables.Experiment.state == ExperimentState.COMMITTED.value,
                    func.now().between(
                        func.date_trunc("minute", tables.Experiment.start_date - buffer),
                        func.date_trunc("minute", tables.Experiment.end_date + buffer),
                    ),
                )
                .order_by(tables.Experiment.id)
            )
        )


def pull_one_experiment(experiment_id: str, *, dwh_timeout: float = PULL_TIMEOUT_SECS) -> PullReport:
    """Read newly landed outcomes for one MAB_ONLINE_DWH experiment and apply them.

    Runs as one transaction covering all three phases:
    1. Read what the warehouse query needs, and lock the draws we intend to fill,
    2. Read the warehouse,
    3. Compute + apply the values.

    Either every valid outcome lands or none does, so a failure never leaves an experiment half
    pulled. Values the warehouse has not filled in yet, and values that fail validation, stay NULL
    and are re-read on a later run.

    The locks taken in phase 1 are held until the transaction ends. This means dwh_timeout
    is a lower limit on that lock duration, so we should be careful.
    """
    experiment_query = (
        select(tables.Experiment)
        .where(tables.Experiment.id == experiment_id)
        .options(
            selectinload(tables.Experiment.arms),
            selectinload(tables.Experiment.contexts),
            selectinload(tables.Experiment.datasource),
            selectinload(tables.Experiment.experiment_fields),
        )
    )

    with logger.contextualize(experiment_id=experiment_id), database.get_session() as session, session.begin():
        # 1. Read what the warehouse query needs: the datasource config, the target and unique-id
        # columns, and the draws that still have no outcome.
        experiment = session.execute(experiment_query).scalar_one()
        if ExperimentsType(experiment.experiment_type) != ExperimentsType.MAB_ONLINE_DWH:
            raise MismatchedExperimentTypeError(f"Cannot pull outcomes for a {experiment.experiment_type} experiment.")
        unique_id_field = experiment.unique_id_field()
        target_field = next((ef for ef in experiment.experiment_fields if ef.is_target), None)
        if experiment.datasource_table is None or unique_id_field is None or target_field is None:
            raise LateValidationError(
                "MAB-DWH experiment is missing its datasource table, unique id field, or target field."
            )
        dsconfig = experiment.datasource.get_config()
        table_name = experiment.datasource_table
        unique_id_field_name = unique_id_field.field_name
        target_field_name = target_field.field_name

        # FOR NO KEY UPDATE holds these draws for the rest of the transaction while still letting
        # foreign keys reference them. Taking the lock here, before the warehouse read, is what
        # keeps autofail off them for the whole pull.
        pending_ids = list(
            session.scalars(
                select(tables.Draw.participant_id)
                .where(
                    tables.Draw.experiment_id == experiment_id,
                    tables.Draw.outcome.is_(None),
                )
                .order_by(tables.Draw.created_at)
                .with_for_update(of=tables.Draw, key_share=True)
            )
        )
        if not pending_ids:
            return PullReport()

        # 2. Read the external DWH, under a deadline so a wedged warehouse cannot hold the locks
        # taken above indefinitely.
        with DwhSession.open(dsconfig.dwh, timeout=dwh_timeout) as dwh:
            sa_table = dwh.inspect_table(table_name)
            # model_construct: get_participant_metrics only reads field_name; the power-analysis
            # fields the validator demands (metric_pct_change/metric_target) don't apply here.
            target_metric = DesignSpecMetricRequest.model_construct(field_name=target_field_name)
            participant_outcomes = dwh.run(
                get_participant_metrics,
                sa_table,
                [target_metric],
                unique_id_field_name,
                pending_ids,
            )

        # 3. Compute the value per participant, then apply them.
        values_by_participant: dict[str, float | None] = {
            po.participant_id: next(
                (mv.metric_value for mv in po.metric_values if mv.metric_name == target_field_name),
                None,
            )
            for po in participant_outcomes
        }

        report = PullReport()
        for participant_id in pending_ids:
            value = values_by_participant.get(participant_id)
            if value is None:
                # Not in the warehouse yet, or the target column is still NULL. The draw keeps
                # outcome=NULL and is re-read on a later run.
                report.pending += 1
                continue
            try:
                update_bandit_arm_with_outcome_impl(
                    xngin_session=session,
                    experiment=experiment,
                    participant_id=participant_id,
                    outcome=value,
                )
                report.ingested += 1
            except LateValidationError as exc:
                # The reward and target guards reject the value before writing anything, so the
                # transaction stays usable and one bad warehouse value does not block the rest of
                # the experiment. The draw keeps outcome=NULL and a corrected value lands later.
                logger.info(f"{experiment_id}: skipping outcome for participant '{participant_id}': {exc}")
                report.invalid += 1
    return report


def pull_all_experiments(pull_timeout: float) -> None:
    """Pull outcomes for every experiment that needs them in parallel.

    A failure on one experiment is reported and does not stop the others.
    """
    experiment_ids = select_experiments_to_pull()
    logger.info(f"Pulling outcomes for {len(experiment_ids)} experiments.")
    sentry_sdk.metrics.count("dwh_pull.experiments", len(experiment_ids))

    failures = 0
    executor = ThreadPoolExecutor(
        max_workers=min(MAX_PULL_WORKERS, len(experiment_ids) or 1), thread_name_prefix="dwhpull"
    )
    futures = {
        experiment_id: executor.submit(pull_one_experiment, experiment_id, dwh_timeout=pull_timeout)
        for experiment_id in experiment_ids
    }
    for experiment_id, future in futures.items():
        with logger.contextualize(experiment_id=experiment_id):
            try:
                report = future.result()
            except Exception as exc:
                failures += 1
                logger.opt(exception=exc).error(f"{experiment_id}: pull failed")
                sentry_sdk.capture_exception(exc)
                sentry_sdk.metrics.count("dwh_pull.failed", 1, attributes={"experiment_id": experiment_id})
                continue
            logger.info(f"{experiment_id}: {report.summary()}")
            attributes = {"experiment_id": experiment_id}
            sentry_sdk.metrics.count("dwh_pull.ingested", report.ingested, attributes=attributes)
            sentry_sdk.metrics.count("dwh_pull.invalid", report.invalid, attributes=attributes)
            sentry_sdk.metrics.count("dwh_pull.pending", report.pending, attributes=attributes)

    executor.shutdown(wait=False, cancel_futures=True)

    logger.info(f"Pulled outcomes for {len(experiment_ids) - failures} experiments, {failures} failed.")
