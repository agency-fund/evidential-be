import asyncio
import random

from loguru import logger
from sqlalchemy import func, insert, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import database
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.sqla import tables

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
        # All active frequentist experiments will be snapshot by this method. We also include experiments
        # that start tomorrow, or ended yesterday, to collect +/- 1 day on both sides of the experiment.
        buffer = text("interval '1 day'")

        # Find all experiments that are active, and their latest snapshot.
        experiments_snapshot_status = (
            select(tables.Experiment.id, tables.Snapshot.updated_at, tables.Snapshot.status)
            .join(tables.Snapshot, isouter=True)
            .distinct(tables.Experiment.id)  # generates a PostgreSQL "DISTINCT ON"
            .where(
                tables.Experiment.experiment_type.like("freq%"),
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
            )
        ).scalar_one_or_none()
        if snapshot is None:
            logger.info(f"{experiment_id}.{snapshot} is missing or is already being processed.")
            return
        await _handle_one_snapshot_safely(session, snapshot, SNAPSHOT_TIMEOUT_SECS)


async def process_pending_snapshots(snapshot_timeout: int):
    """Processes pending snapshots, one at a time, until there are no more available.

    Interactions with the client data warehouse will be considered timed-out after snapshot_timeout seconds. These
    timeouts will raise an exception and update the snapshot with status="failed". The timeout is not guaranteed
    to be respected because some interactions with client DWH are blocking and those timeout behaviors have not yet
    been aligned.
    """
    one_pending_snapshot = (
        select(tables.Snapshot).where(tables.Snapshot.status == "pending").limit(1).with_for_update(skip_locked=True)
    )

    while True:
        await asyncio.sleep(random.uniform(0, 2))  # jitter
        async with database.async_session() as session, session.begin():
            snapshot = (await session.execute(one_pending_snapshot)).scalar_one_or_none()
            if snapshot is None:
                logger.info("No pending snapshots available.")
                return
            _ = await _handle_one_snapshot_safely(session, snapshot, snapshot_timeout)


async def _handle_one_snapshot_safely(session, snapshot, snapshot_timeout):
    experiment = await snapshot.awaitable_attrs.experiment
    datasource = await experiment.awaitable_attrs.datasource
    logger.info(f"{experiment.id}.{snapshot.id}: processing")
    try:
        async with asyncio.timeout(snapshot_timeout):
            snapshot.data = await _query_dwh_for_snapshot_data(session, datasource, experiment)
            snapshot.status = "success"
    except TimeoutError:
        logger.info(f"{experiment.id}.{snapshot.id}: timeout")
        snapshot.status = "failed"
        snapshot.message = "TimeoutError"
    except Exception as exc:
        logger.opt(exception=exc).info(f"{experiment.id}.{snapshot.id}: unhandled exception")
        snapshot.status = "failed"
        snapshot.message = f"{type(exc).__name__}: {exc}"
    logger.info(f"{experiment.id}.{snapshot.id}: done")


async def _query_dwh_for_snapshot_data(
    _session: AsyncSession, datasource: tables.Datasource, _experiment: tables.Experiment
) -> dict:
    """Collect a snapshot from a customer DWH and returns the snapshot data."""
    ds_config = datasource.get_config()
    # TODO(qixotic): replace this dummy implementation with snapshot collection behavior
    async with DwhSession(ds_config.dwh) as dwh:
        table_list = await dwh.list_tables()
    return {"todo": table_list}
