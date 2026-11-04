from threading import Event

import pytest

from xngin.apiserver.worker.executor import SystemClock
from xngin.apiserver.worker.tq.task_queue import TaskQueue
from xngin.apiserver.worker.tq.task_queue_thread import running_task_queue

DATABASE_URL = "postgresql+psycopg://example.invalid/db"


def test_leaving_the_context_stops_and_joins_the_queue(monkeypatch: pytest.MonkeyPatch):
    stopped: list[bool] = []
    monkeypatch.setattr(TaskQueue, "run", lambda self, stop: stopped.append(stop.wait(timeout=5)))
    stop = Event()
    with running_task_queue(DATABASE_URL, SystemClock(), stop, max_retries=1, poll_interval=1, join_timeout=5):
        assert stopped == []
    assert stopped == [True]
