import asyncio
import random
from typing import TYPE_CHECKING

import numpy as np
import sentry_sdk
from loguru import logger
from sqlalchemy import func, insert, or_, select, text
from sqlalchemy.orm import selectinload

from xngin.apiserver import database
from xngin.apiserver.routers.common_api_types import ExperimentAnalysisResponse
from xngin.apiserver.routers.common_enums import ContextType, ExperimentsType
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# The amount of time the API server will wait for a snapshot to complete when invoked in response to user request.
# The snapshotter cron job can specify a different timeout via command line flags.
SNAPSHOT_TIMEOUT_SECS = 90


async def create_pending_snapshots(snapshot_interval: int):
    """
    Identify experiments that are due for fresh snapshots, or whose most recent snapshot failed, and insert snapshots
    with status=pending for them.

    This method establishes its own database connection.
    """
    freshness_threshold = text(f"interval '{snapshot_interval} seconds'")

    async with database.async_session() as session, session.begin():
        # All active experiments will be snapshot by this method. We also include experiments
        # that start tomorrow, or ended yesterday, to collect +/- 1 day on both sides of the experiment.
        buffer = text("interval '1 day'")

        # Find all experiments that are active, and their latest snapshot.
        experiments_snapshot_status = (
            select(tables.Experiment.id, tables.Snapshot.updated_at, tables.Snapshot.status)
            .join(tables.Snapshot, isouter=True)
            .distinct(tables.Experiment.id)  # generates a PostgreSQL "DISTINCT ON"
            .where(
                func.now().between(
                    func.date_trunc("minute", tables.Experiment.start_date - buffer),
                    func.date_trunc("minute", tables.Experiment.end_date + buffer),
                ),
            )
            .order_by(tables.Experiment.id, tables.Snapshot.updated_at.desc())
            .cte()
        )
        # Filter for snapshots that are late or failed.
        candidate_experiments = select(experiments_snapshot_status.c.id).where(
            or_(
                # latest snapshot is too old
                func.date_trunc("minute", experiments_snapshot_status.c.updated_at)
                <= func.date_trunc("minute", func.now() + text("interval '1 minute'") - freshness_threshold),
                # latest snapshot failed
                experiments_snapshot_status.c.status == "failed",
                # no snapshots exist for experiment
                experiments_snapshot_status.c.status.is_(None),
            )
        )

        # Create a new snapshot with status=pending for each experiment that needs one.
        for experiment_id in (await session.execute(candidate_experiments)).scalars():
            await session.execute(insert(tables.Snapshot).values(experiment_id=experiment_id))


async def make_first_snapshot(experiment_id: str, snapshot_id: str):
    """Process a specific snapshot with status=pending.

    This method is intended to be invoked immediately after a snapshot is created in response to user request.
    """
    async with database.async_session() as session, session.begin():
        snapshot = (
            await session.execute(
                select(tables.Snapshot)
                .where(
                    tables.Snapshot.experiment_id == experiment_id,
                    tables.Snapshot.id == snapshot_id,
                    tables.Snapshot.status == "pending",
                )
                .with_for_update(skip_locked=True)
                .options(
                    selectinload(tables.Snapshot.experiment),
                    selectinload(tables.Snapshot.experiment).selectinload(tables.Experiment.arms),
                    selectinload(tables.Snapshot.experiment).selectinload(tables.Experiment.contexts),
                    selectinload(tables.Snapshot.experiment).selectinload(tables.Experiment.datasource),
                    selectinload(tables.Snapshot.experiment)
                    .selectinload(tables.Experiment.experiment_fields)
                    .selectinload(tables.ExperimentField.experiment_filters),
                )
            )
        ).scalar_one_or_none()
        if snapshot is None:
            logger.info(f"{experiment_id}.{snapshot_id} is missing or is already being processed.")
            return
        await _handle_one_snapshot_safely(session, snapshot, SNAPSHOT_TIMEOUT_SECS)


async def process_pending_snapshots(snapshot_timeout: int, *, max_jitter_secs: float = 2):
    """Processes pending snapshots, one at a time, until there are no more available.

    Interactions with the client data warehouse will be considered timed-out after snapshot_timeout seconds. These
    timeouts will raise an exception and update the snapshot with status="failed". The timeout is not guaranteed
    to be respected because some interactions with client DWH are blocking and those timeout behaviors have not yet
    been aligned.
    """
    one_pending_snapshot = (
        select(tables.Snapshot)
        .where(tables.Snapshot.status == "pending")
        .limit(1)
        .with_for_update(skip_locked=True)
        .options(
            selectinload(tables.Snapshot.experiment),
            selectinload(tables.Snapshot.experiment).selectinload(tables.Experiment.arms),
            selectinload(tables.Snapshot.experiment).selectinload(tables.Experiment.contexts),
            selectinload(tables.Snapshot.experiment).selectinload(tables.Experiment.datasource),
            selectinload(tables.Snapshot.experiment)
            .selectinload(tables.Experiment.experiment_fields)
            .selectinload(tables.ExperimentField.experiment_filters),
        )
    )

    while True:
        await asyncio.sleep(random.uniform(0, max_jitter_secs))  # jitter
        async with database.async_session() as session, session.begin():
            snapshot = (await session.execute(one_pending_snapshot)).scalar_one_or_none()
            if snapshot is None:
                logger.info("No pending snapshots available.")
                return
            _ = await _handle_one_snapshot_safely(session, snapshot, snapshot_timeout)


async def _handle_one_snapshot_safely(session: AsyncSession, snapshot: tables.Snapshot, snapshot_timeout: int):
    sentry_sdk.metrics.count("snapshots.started", 1, attributes={"experiment_id": snapshot.experiment_id})
    experiment = await snapshot.awaitable_attrs.experiment
    datasource = await experiment.awaitable_attrs.datasource
    with logger.contextualize(experiment_id=experiment.id):
        logger.info(f"{experiment.id}.{snapshot.id}: processing")
        try:
            async with asyncio.timeout(snapshot_timeout):
                result = await _query_dwh_for_snapshot_data(session, datasource, experiment)
                snapshot.data = result.model_dump(mode="json")
                snapshot.status = "success"
        except Exception as exc:
            sentry_sdk.metrics.count("snapshots.failed", 1, attributes={"experiment_id": snapshot.experiment_id})
            logger.opt(exception=exc).info(f"{experiment.id}.{snapshot.id}: exception")
            snapshot.status = "failed"
            snapshot.message = f"{type(exc).__name__}: {exc}"
        sentry_sdk.metrics.count("snapshots.finished", 1, attributes={"experiment_id": snapshot.experiment_id})
        logger.info(f"{experiment.id}.{snapshot.id}: done")


async def _query_dwh_for_snapshot_data(
    session: AsyncSession, datasource: tables.Datasource, experiment: tables.Experiment
) -> ExperimentAnalysisResponse:
    """Collect a snapshot from a customer DWH and returns the snapshot data."""
    if ExperimentsType(experiment.experiment_type).is_bayesian():
        context_vals = None

        # TODO: If the experiment is a CMAB, we need to pass in context values.
        # Ideally we should marginalize over contexts, but we'll start with just using
        # the mean context values. Captured in issue 140
        # (https://github.com/agency-fund/evidential-sprint/issues/140)
        if ExperimentsType(experiment.experiment_type).is_cmab():
            contexts = await experiment.awaitable_attrs.contexts
            sorted_context_defns = sorted(contexts, key=lambda c: c.id)
            # draw.context_vals were already sorted corresponding to the sorted_context_defns
            # ordering when the assignment was made.
            mean_context_val = await _mean_context_vals_from_draws(session, experiment.id, len(contexts))
            context_vals = [
                abs(float(np.ceil(m - 0.5))) if context.value_type == ContextType.BINARY else m
                for m, context in zip(mean_context_val, sorted_context_defns, strict=False)
            ]

        return await experiments_common.analyze_experiment_bandit_impl(
            xngin_session=session, experiment=experiment, context_vals=context_vals
        )

    if ExperimentsType(experiment.experiment_type).is_freq():
        # Look for the arm in position 1. If not found, use the first arm.
        baseline_arm = next((arm for arm in experiment.arms if arm.position == 1), experiment.arms[0])
        assert baseline_arm.id is not None
        return await experiments_common.analyze_experiment_freq_impl(
            xngin_session=session,
            dsconfig=datasource.get_config(),
            experiment=experiment,
            baseline_arm_id=baseline_arm.id,
            metrics=ExperimentStorageConverter(experiment).get_design_spec_metrics(),
        )
    raise ValueError(f"Unsupported experiment type: {experiment.experiment_type}")


async def _mean_context_vals_from_draws(session: AsyncSession, experiment_id: str, n_contexts: int) -> list[float]:
    """Per-position mean of non-null context_vals across draws.

    Returns a zero vector when no rows match. Assumes all non-null context_vals
    arrays share the same length and contain no inner NULLs — both invariants
    enforced by the assignment write path.
    """
    if n_contexts == 0:
        return []
    rows = (
        await session.execute(
            text("""
                SELECT AVG(val)
                FROM draws, unnest(context_vals) WITH ORDINALITY AS t(val, ord)
                WHERE experiment_id = :eid AND context_vals IS NOT NULL
                GROUP BY ord
                ORDER BY ord
            """),
            {"eid": experiment_id},
        )
    ).all()
    if not rows:
        return [0.0] * n_contexts
    return [r[0] for r in rows]
