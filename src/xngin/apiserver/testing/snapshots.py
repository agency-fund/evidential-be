"""Capture snapshot attempts without an API background task."""

from xngin.apiserver import database
from xngin.apiserver.snapshots.snapshotter import capture_snapshot
from xngin.apiserver.sqla import tables


def commit_snapshot(experiment_id: str, *, dwh_timeout: float = 90) -> tables.Snapshot:
    with database.get_session() as session, session.begin():
        return capture_snapshot(session, experiment_id, dwh_timeout=dwh_timeout)
