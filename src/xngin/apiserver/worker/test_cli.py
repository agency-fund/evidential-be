import contextlib
import inspect
import json
from pathlib import Path

import pytest

from xngin.apiserver.worker import cli
from xngin.apiserver.worker.tq import task_queue_thread
from xngin.apiserver.worker.tq.task_queue import TaskQueue

RAILWAY_CONFIG = Path(__file__).parents[4] / "railway.worker.json"


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Stub the worker's process-level setup and exit, and run the task queue until stop is set.

    Returns the exit codes passed to os._exit.
    """
    exits: list[int] = []
    for setup in (cli.customlogging, cli.sentry, cli.secretservice):
        monkeypatch.setattr(setup, "setup", lambda: None)
    monkeypatch.setattr(cli.database, "setup", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(cli.database, "get_sqlalchemy_database_url", lambda: "postgresql+psycopg://example.invalid/db")
    monkeypatch.setattr(TaskQueue, "run", lambda self, stop: stop.wait())
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    return exits


def test_railway_draining_covers_default_warehouse_timeouts():
    # The task queue is joined first, then the executor drains, so shutdown can take the sum of the two.
    defaults = inspect.signature(cli.run).parameters
    draining = json.loads(RAILWAY_CONFIG.read_text())["deploy"]["drainingSeconds"]
    longest_attempt = max(defaults["dwh_pull_timeout"].default, defaults["snapshot_dwh_timeout"].default)
    assert draining >= longest_attempt + cli.TQ_JOIN_TIMEOUT_SECS


def test_escaping_exception_still_exits_immediately(exits: list[int], monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_worker", fail)
    cli.run()
    assert exits == [1]


def test_task_queue_failure_stops_the_worker(exits: list[int], monkeypatch: pytest.MonkeyPatch):
    captured: list[BaseException] = []
    stopped: list[bool] = []

    def fail(self, stop):
        raise RuntimeError("boom")

    def run_until_stopped(jobs, stop, *args):
        stopped.append(stop.wait(timeout=5))

    monkeypatch.setattr(TaskQueue, "run", fail)
    monkeypatch.setattr(task_queue_thread.sentry_sdk, "capture_exception", captured.append)
    monkeypatch.setattr(cli, "run_worker", run_until_stopped)
    cli.run()
    assert stopped == [True]
    assert exits == [0]
    assert [str(exc) for exc in captured] == ["boom"]
