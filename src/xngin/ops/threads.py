"""Deadlines for blocking calls."""

import contextvars
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any


class ThreadTimeout:
    """Runs blocking calls on one helper thread, under one shared deadline.

    Obtain one from timeout_thread(). Every call runs on the same helper thread, in submission order,
    so a caller may hand the helper an object that must only ever be touched from one thread at a
    time -- a SQLAlchemy Session, say -- and keep handing it to successive calls.
    """

    def __init__(self, executor: ThreadPoolExecutor, deadline: float):
        self._executor = executor
        self._deadline = deadline
        self._expired = False

    @property
    def remaining(self) -> float:
        """Seconds left before the deadline; zero or negative once it has passed."""
        return self._deadline - time.monotonic()

    @property
    def expired(self) -> bool:
        """Whether the deadline has passed, or a run() has already given up on a call."""
        return self._expired or self.remaining <= 0

    def submit[T](self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        """Queues fn on the helper thread without waiting for it.

        Ordered behind everything already queued, including a call run() has given up on. Use this
        for work that must happen on the helper thread but that the caller need not wait for --
        releasing a resource the helper owns, typically.
        """
        # The helper thread starts with an empty contextvars context, so hand it a copy of the
        # caller's. Anything that reads contextvars -- loguru's logger.contextualize() among them --
        # then behaves on the helper as it does here. Each submit takes its own snapshot, so an
        # abandoned call still holding an older Context cannot collide with a later one.
        context = contextvars.copy_context()
        return self._executor.submit(context.run, lambda: fn(*args, **kwargs))

    def run[T](self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Calls fn on the helper thread and returns its result.

        Raises TimeoutError if the shared deadline passes first, and re-raises anything fn itself
        raised. Once the deadline has passed, raises without submitting anything: work we have
        already stopped waiting for is work not worth starting.
        """
        remaining = self.remaining
        if self.expired:
            raise TimeoutError("The deadline passed before this call could start.")
        future = self.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=remaining)
        except TimeoutError:
            if future.done():
                # fn raised TimeoutError itself; the deadline is still intact.
                raise
            self._expired = True
            raise


@contextmanager
def timeout_thread(*, seconds: float, name: str) -> Iterator[ThreadTimeout]:
    """Bounds how long a block of blocking calls may run, by running them on a helper thread.

    Use this for calls that cannot be interrupted -- a query against a customer data warehouse, say
    -- where the caller needs to stop waiting even though the call itself cannot be cancelled:

        try:
            with timeout_thread(seconds=10, name="warehouse") as timeout:
                result = timeout.run(read_the_warehouse, arg)
        except TimeoutError:
            ...

    The deadline covers the block as a whole rather than each call, because the budgets callers have
    are per-operation.

    The deadline bounds the *caller*, not the call. A call that overruns keeps running on its helper
    thread until it finishes on its own; nothing here can cancel it. It is therefore only safe to
    abandon a call whose side effects the caller is prepared to have happen late, or not at all.
    Note too that concurrent.futures registers an atexit hook that joins live worker threads, so an
    abandoned call delays interpreter exit until it finishes: this bounds a unit of work, not a
    process's wall clock.

    Args:
        seconds: How long run() waits, in total across every call, before raising TimeoutError.
        name: What the helper is waiting on, to identify it in a stack dump. Named threads are
            "timeout-{name}", so that a caller that also runs its own pool cannot end up with two
            different pools sharing one prefix.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"timeout-{name}" if name else "timeout")
    try:
        yield ThreadTimeout(executor, time.monotonic() + seconds)
    finally:
        # Return without waiting; running and queued calls continue to completion.
        executor.shutdown(wait=False)
