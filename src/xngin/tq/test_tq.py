import queue
import threading
from datetime import timedelta

import psycopg
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.tq import task_queue as task_queue_module
from xngin.tq.task_queue import Task, TaskQueue
from xngin.tq.tq_test_support import insert_task, tq_runner, wait_for_task_status

pytest_plugins = ("xngin.apiserver.conftest",)


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
        assert pending_task.updated_at > task.created_at
        assert pending_task.embargo_until == pending_task.updated_at + timedelta(minutes=1)


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
