import asyncio
import queue
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
import pytest
from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import database
from xngin.apiserver.sqla import tables
from xngin.tq import task_queue as task_queue_module
from xngin.tq.task_queue import Task, TaskQueue

pytest_plugins = ("xngin.apiserver.conftest",)

STATUS_TIMEOUT_SECS = 5.0


@dataclass(slots=True)
class TqHandle:
    cancel: threading.Event
    error_collector: queue.SimpleQueue[BaseException]


def start_task_queue_in_thread(
    task_queue: TaskQueue, cancel: threading.Event
) -> tuple[threading.Thread, queue.SimpleQueue[BaseException]]:
    """Starts a TaskQueue in a new thread."""
    error_collector: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    def run_task_queue() -> None:
        try:
            task_queue.run(cancel=cancel)
        except BaseException as exc:
            error_collector.put(exc)

    thread = threading.Thread(target=run_task_queue, name="test-task-queue")
    thread.start()
    return thread, error_collector


async def wait_for_task_status(
    task_id: str,
    expected_status: str,
    *,
    predicate: Callable[[tables.Task], bool] | None = None,
) -> tables.Task:
    """Polls for a specific task reaching a certain status within STATUS_TIMEOUT_SECS."""
    deadline = time.monotonic() + STATUS_TIMEOUT_SECS
    latest_task: tables.Task | None = None
    while time.monotonic() < deadline:
        async with database.async_session() as session:
            latest_task = await session.get(tables.Task, task_id)
        if (
            latest_task is not None
            and latest_task.status == expected_status
            and (predicate is None or predicate(latest_task))
        ):
            return latest_task
        await asyncio.sleep(0.10)
    raise AssertionError(
        f"Task {task_id} did not reach status {expected_status!r} before timeout. Last observed task: {latest_task!r}"
    )


async def insert_task(
    xngin_session: AsyncSession,
    *,
    task_type: str,
    payload: dict | None = None,
) -> tables.Task:
    """Inserts a task using SQLAlchemy."""
    task = tables.Task(task_type=task_type, payload=payload)
    xngin_session.add(task)
    await xngin_session.commit()
    await xngin_session.refresh(task)
    return task


@pytest.fixture
def tq_dsn() -> str:
    """Converts a SQLAlchemy DSN to a Psycopg-compatible DSN."""
    url = make_url(database.get_sqlalchemy_database_url()).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


@contextmanager
def tq_runner(queue_instance: TaskQueue) -> Generator[None]:
    """Context manager for running a TaskQueue in a thread with graceful shutdown."""
    cancel = threading.Event()
    thread, error_collector = start_task_queue_in_thread(queue_instance, cancel)
    try:
        yield
    finally:
        cancel.set()
        thread.join(timeout=3)
        if thread.is_alive():
            raise AssertionError("TaskQueue thread did not stop within timeout")
        if not error_collector.empty():
            raise AssertionError(f"TaskQueue thread raised an exception: {error_collector.get()!r}")


async def test_task_queue_processes_pending_task_successfully(xngin_session: AsyncSession, tq_dsn: str):
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=1)
    handled_task_ids: queue.SimpleQueue[str] = queue.SimpleQueue()

    def handler(task: Task) -> None:
        handled_task_ids.put(task.id)

    task_queue.register_handler("test.success", handler)
    with tq_runner(task_queue):
        task = await insert_task(
            xngin_session,
            task_type="test.success",
            payload={"value": "ok"},
        )

        completed_task = await wait_for_task_status(task.id, "success")

        assert handled_task_ids.get(timeout=1) == task.id
        assert handled_task_ids.empty()
        assert completed_task.retry_count == 0
        assert completed_task.message is None


async def test_task_queue_marks_unhandled_task_dead_when_max_retries_zero(
    xngin_session: AsyncSession,
    tq_dsn: str,
):
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=0, poll_interval_secs=1)
    with tq_runner(task_queue):
        task = await insert_task(
            xngin_session,
            task_type="test.unhandled",
            payload={"value": "missing-handler"},
        )

        dead_task = await wait_for_task_status(task.id, "dead")

        assert dead_task.retry_count == 1
        assert dead_task.message == "No handler for task type: test.unhandled"


async def test_task_queue_requeues_failed_task_with_backoff(
    xngin_session: AsyncSession,
    tq_dsn: str,
):
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=1, poll_interval_secs=1)
    handled_task_ids: queue.SimpleQueue[str] = queue.SimpleQueue()

    def handler(task: Task) -> None:
        handled_task_ids.put(task.id)
        raise RuntimeError("handler failed")

    task_queue.register_handler("test.retry", handler)
    with tq_runner(task_queue):
        inserted_at = datetime.now(UTC)
        task = await insert_task(
            xngin_session,
            task_type="test.retry",
            payload={"value": "retry"},
        )

        pending_task = await wait_for_task_status(
            task.id,
            "pending",
            predicate=lambda row: row.retry_count == 1 and row.message == "handler failed",
        )

        assert handled_task_ids.get(timeout=1) == task.id
        assert handled_task_ids.empty()
        assert pending_task.retry_count == 1
        assert pending_task.message == "handler failed"
        assert pending_task.embargo_until > inserted_at


def test_task_queue_retries_after_operational_error(monkeypatch: pytest.MonkeyPatch):
    task_queue = TaskQueue(dsn="postgresql://example.invalid/test", max_retries=1, poll_interval_secs=1)
    cancel = threading.Event()
    sleep_calls: list[int] = []

    def fake_connect(*args, **kwargs):
        raise psycopg.OperationalError("boom")

    def fake_sleep(delay: int) -> None:
        sleep_calls.append(delay)
        cancel.set()

    monkeypatch.setattr(task_queue_module.psycopg, "connect", fake_connect)
    monkeypatch.setattr(task_queue_module.time, "sleep", fake_sleep)

    task_queue.run(cancel=cancel)

    assert sleep_calls == [5]
