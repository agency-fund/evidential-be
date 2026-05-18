"""Shared test helpers for the tq package."""

import asyncio
import queue
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import database
from xngin.apiserver.sqla import tables
from xngin.tq.task_queue import TaskQueue

STATUS_TIMEOUT_SECS = 5.0


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
