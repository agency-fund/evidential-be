"""Running attempts on a bounded pool of threads."""

import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Protocol, Self

from loguru import logger

from xngin.apiserver.worker.attempts import FinishedAttempt, attempt_experiment
from xngin.apiserver.worker.jobs.base import Job


class AttemptExecutor:
    """Wraps a ThreadPoolExecutor with behavior that ensures we never run more than parallelism jobs at once.

    It manages each attempt from start to finish: start gives each free slot one experiment, collect_finished
    hands back finished attempts and frees their slots, and wait sleeps until one finishes or the scheduler's
    chosen time.

    Use it as a context manager. Exiting waits for running attempts to commit or roll back, then stops the
    threads, even if the scheduler loop raised.

    The methods are only called from the scheduler's threads. The jobs trigger the _wake event when they
    complete, and that job's slot stays occupied until collect_finished is called.
    """

    def __init__(self, parallelism: int, clock: Clock) -> None:
        self._parallelism = parallelism
        self._clock = clock
        self._pool = ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="worker")
        self._running: dict[Future[FinishedAttempt], str] = {}
        self._wake = Event()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        logger.info("Worker stopping; draining attempts.")
        self._pool.shutdown(wait=True)
        self._running.clear()

    def is_full(self) -> bool:
        return len(self._running) >= self._parallelism

    def in_progress(self) -> tuple[str, ...]:
        """The experiments with an attempt in flight, which the scheduler must not hand to another slot."""
        return tuple(self._running.values())

    def start(self, jobs: Sequence[Job], stop: Event, experiment_ids: Iterable[str]) -> None:
        """Start an attempt for each experiment, in order, until the slots are full or stop is set."""
        for experiment_id in experiment_ids:
            if self.is_full() or stop.is_set():
                return
            future = self._pool.submit(attempt_experiment, jobs, experiment_id)
            self._running[future] = experiment_id
            future.add_done_callback(lambda _: self._wake.set())

    def collect_finished(self) -> list[FinishedAttempt]:
        """Return the attempts that finished since the last call, freeing their slots.

        This clears the wake signal before looking, so an attempt that finishes afterward, while the caller polls,
        still ends the next wait at once.
        """
        self._wake.clear()
        finished = []
        for future in list(self._running):
            if future.done():
                del self._running[future]
                finished.append(future.result())
        return finished

    def wait(self, stop: Event, wake_at: float) -> None:
        """Sleep until the time wake_at, until an attempt finishes, or until stop is set.

        Waiting happens on the wake signal, which stop does not set, so check stop at least once a second.
        """
        while not stop.is_set():
            remaining = wake_at - self._clock.monotonic()
            if remaining <= 0 or self._clock.wait(self._wake, min(remaining, 1)):
                return


class Clock(Protocol):
    """The monotonic clock and sleeps the scheduler and executor run on, which tests replace to control time."""

    def monotonic(self) -> float: ...

    def wait(self, event: Event, timeout: float) -> bool:
        """Wait up to timeout seconds for event to be set, and return whether it was."""
        ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def wait(self, event: Event, timeout: float) -> bool:
        return event.wait(timeout)
