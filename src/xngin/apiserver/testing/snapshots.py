"""Capture snapshot attempts without an API background task."""

from xngin.apiserver import database
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.jobs.snapshot_analysis import capture_snapshot


def commit_snapshot(experiment_id: str, *, dwh_timeout: float = 90) -> tables.Snapshot:
    with database.get_session() as session, session.begin():
        return capture_snapshot(session, experiment_id, dwh_timeout=dwh_timeout).snapshot
