import queue
import threading
from datetime import timedelta

import psycopg
import pytest
from sqlalchemy.orm import Session

from xngin.apiserver.testing.worker_helpers import FakeClock
from xngin.apiserver.worker.executor import SystemClock
from xngin.apiserver.worker.tq import task_queue as task_queue_module
from xngin.apiserver.worker.tq.task_queue import Task, TaskQueue
from xngin.apiserver.worker.tq.tq_test_support import insert_task, tq_runner, wait_for_task_status


def test_task_queue_processes_pending_task_successfully(xngin_session: Session, tq_dsn: str):
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=1, clock=SystemClock())
    handled_task_ids: queue.SimpleQueue[str] = queue.SimpleQueue()

    def handler(task: Task) -> None:
        handled_task_ids.put(task.id)

    task_queue.register_handler("test.success", handler)
    with tq_runner(task_queue):
        task = insert_task(
            xngin_session,
            task_type="test.success",
            payload={"value": "ok"},
        )

        completed_task = wait_for_task_status(task.id, "success")

        assert handled_task_ids.get(timeout=1) == task.id
        assert handled_task_ids.empty()
        assert completed_task.retry_count == 0
        assert completed_task.message is None


def test_task_queue_marks_unhandled_task_dead_when_max_retries_zero(
    xngin_session: Session,
    tq_dsn: str,
):
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=0, poll_interval_secs=1, clock=SystemClock())
    with tq_runner(task_queue):
        task = insert_task(
            xngin_session,
            task_type="test.unhandled",
            payload={"value": "missing-handler"},
        )

        dead_task = wait_for_task_status(task.id, "dead")

        assert dead_task.retry_count == 1
        assert dead_task.message == "No handler for task type: test.unhandled"


def test_task_queue_requeues_failed_task_with_backoff(
    xngin_session: Session,
    tq_dsn: str,
):
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=1, clock=SystemClock())
    handled_task_ids: queue.SimpleQueue[str] = queue.SimpleQueue()

    def handler(task: Task) -> None:
        handled_task_ids.put(task.id)
        raise RuntimeError("handler failed")

    task_queue.register_handler("test.retry", handler)
    with tq_runner(task_queue):
        task = insert_task(
            xngin_session,
            task_type="test.retry",
            payload={"value": "retry"},
        )

        pending_task = wait_for_task_status(
            task.id,
            "pending",
            predicate=lambda row: row.retry_count == 1 and row.message == "handler failed",
        )

        assert handled_task_ids.get(timeout=1) == task.id
        assert handled_task_ids.empty()
        assert pending_task.retry_count == 1
        assert pending_task.message == "handler failed"
        assert pending_task.updated_at > task.created_at
        assert pending_task.embargo_until == pending_task.updated_at + timedelta(minutes=1)


class StoppedClock:
    """A clock that never advances, so the poll interval never elapses, and that signals when the queue reads it."""

    def __init__(self) -> None:
        self.read = threading.Event()

    def monotonic(self) -> float:
        self.read.set()
        return 0.0

    def wait(self, event: threading.Event, timeout: float) -> bool:
        return event.wait(timeout)


class SteppingClock:
    """A clock that advances by step on every read."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def wait(self, event: threading.Event, timeout: float) -> bool:
        return event.wait(timeout)


def test_task_queue_retries_after_operational_error(monkeypatch: pytest.MonkeyPatch):
    clock, stop = FakeClock(), threading.Event()
    task_queue = TaskQueue(dsn="postgresql://example.invalid/test", max_retries=1, poll_interval_secs=1, clock=clock)
    connected_at: list[float] = []

    def fake_connect(*args, **kwargs):
        connected_at.append(clock.now)
        if len(connected_at) == 2:
            stop.set()
        raise psycopg.OperationalError("boom")

    monkeypatch.setattr(task_queue_module.psycopg, "connect", fake_connect)

    task_queue.run(stop=stop)

    assert connected_at == [0.0, 5.0]


@pytest.mark.usefixtures("xngin_session")
def test_idle_task_queue_stops_promptly_when_stopped(tq_dsn: str):
    clock, stop = StoppedClock(), threading.Event()
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=60, clock=clock)
    thread = threading.Thread(target=task_queue.run, kwargs={"stop": stop})
    thread.start()
    assert clock.read.wait(timeout=5), "The queue never began waiting for notifications."
    stop.set()
    thread.join(timeout=3)
    assert not thread.is_alive()


def test_task_queue_is_woken_by_notification(xngin_session: Session, tq_dsn: str):
    # The poll interval never elapses on this clock, so only the insert's NOTIFY can wake the queue.
    clock = StoppedClock()
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=60, clock=clock)
    task_queue.register_handler("test.notify", lambda task: None)
    with tq_runner(task_queue):
        assert clock.read.wait(timeout=5), "The queue never began waiting for notifications."
        task = insert_task(xngin_session, task_type="test.notify")
        wait_for_task_status(task.id, "success")


@pytest.mark.usefixtures("xngin_session")
def test_task_queue_polls_again_when_poll_interval_elapses(tq_dsn: str, monkeypatch: pytest.MonkeyPatch):
    stop = threading.Event()
    # Each read of the clock moves past the poll interval, so the wait for a notification ends immediately.
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=60, clock=SteppingClock(step=60))
    fetch_task = task_queue._fetch_task
    polls: list[None] = []

    def counting_fetch_task(conn):
        polls.append(None)
        if len(polls) == 2:
            stop.set()
        return fetch_task(conn)

    monkeypatch.setattr(task_queue, "_fetch_task", counting_fetch_task)

    task_queue.run(stop=stop)

    assert len(polls) == 2


def test_task_queue_sets_up_notifications_once_per_connection(
    xngin_session: Session, tq_dsn: str, monkeypatch: pytest.MonkeyPatch
):
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=1, clock=SystemClock())
    task_queue.register_handler("test.success", lambda task: None)
    setup_notifications = task_queue._setup_notifications
    setups: list[None] = []

    def counting_setup_notifications(conn):
        setups.append(None)
        setup_notifications(conn)

    monkeypatch.setattr(task_queue, "_setup_notifications", counting_setup_notifications)
    tasks = [insert_task(xngin_session, task_type="test.success") for _ in range(3)]

    with tq_runner(task_queue):
        for task in tasks:
            wait_for_task_status(task.id, "success")
        # Inserted while the queue is idle, so it is found after waiting for a notification.
        wait_for_task_status(insert_task(xngin_session, task_type="test.success").id, "success")

    assert len(setups) == 1
