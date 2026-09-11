import contextvars
import operator
import threading
import time

import pytest

from xngin.ops.threads import timeout_thread


def test_run_returns_the_value():
    with timeout_thread(seconds=10, name="test") as timeout:
        assert timeout.run(operator.add, 1, 2) == 3


def test_run_reraises_what_fn_raised():
    def boom():
        raise ValueError("from the helper thread")

    with pytest.raises(ValueError, match="from the helper thread"), timeout_thread(seconds=10, name="test") as timeout:
        timeout.run(boom)


def test_run_raises_timeouterror_once_the_deadline_passes():
    release = threading.Event()
    try:
        with pytest.raises(TimeoutError), timeout_thread(seconds=0.01, name="test") as timeout:
            timeout.run(release.wait)
    finally:
        release.set()


def test_exiting_after_a_timeout_does_not_wait_for_the_abandoned_call():
    """The whole point of the deadline is that the caller stops waiting, so exiting must not rejoin."""
    release = threading.Event()
    started_at = time.monotonic()
    try:
        with pytest.raises(TimeoutError), timeout_thread(seconds=0.01, name="test") as timeout:
            timeout.run(release.wait)
        assert time.monotonic() - started_at < 1.0
    finally:
        release.set()


def test_the_helper_thread_is_named_after_the_timeout_and_the_caller():
    """The prefix carries "timeout" so it cannot collide with a pool the caller runs under the same name."""
    with timeout_thread(seconds=10, name="dwhpull") as timeout:
        assert timeout.run(lambda: threading.current_thread().name).startswith("timeout-dwhpull")


def test_run_may_be_called_repeatedly_under_one_deadline():
    with timeout_thread(seconds=10, name="test") as timeout:
        assert [timeout.run(operator.add, n, 1) for n in range(3)] == [1, 2, 3]


def test_every_call_runs_on_the_same_helper_thread():
    """DwhSession reuses one warehouse Session across calls, which is only safe if this holds."""
    with timeout_thread(seconds=10, name="test") as timeout:
        idents = {timeout.run(threading.get_ident) for _ in range(3)}
    assert len(idents) == 1
    assert idents != {threading.get_ident()}


def test_the_deadline_covers_the_calls_together_not_each_one():
    with timeout_thread(seconds=0.3, name="test") as timeout:
        timeout.run(time.sleep, 0.2)
        with pytest.raises(TimeoutError):
            timeout.run(time.sleep, 0.2)


def test_run_after_the_deadline_does_not_start_the_call():
    started = threading.Event()
    with timeout_thread(seconds=0.01, name="test") as timeout:
        time.sleep(0.05)
        with pytest.raises(TimeoutError):
            timeout.run(started.set)
    assert not started.is_set()


def test_run_after_a_timeout_fails_fast_instead_of_queueing_behind_it():
    release = threading.Event()
    try:
        with timeout_thread(seconds=0.01, name="test") as timeout:
            with pytest.raises(TimeoutError):
                timeout.run(release.wait)
            started_at = time.monotonic()
            with pytest.raises(TimeoutError):
                timeout.run(operator.add, 1, 2)
            assert time.monotonic() - started_at < 0.5
    finally:
        release.set()


def test_submit_runs_behind_a_call_that_timed_out_without_waiting_for_it():
    """DwhSession closes its warehouse connection this way: queued after the call it gave up on."""
    release = threading.Event()
    ran_late = threading.Event()
    try:
        with timeout_thread(seconds=0.01, name="test") as timeout:
            with pytest.raises(TimeoutError):
                timeout.run(release.wait)
            timeout.submit(ran_late.set)
            assert not ran_late.is_set()
            release.set()
            assert ran_late.wait(timeout=5)
    finally:
        release.set()


def test_a_timeouterror_raised_by_fn_is_not_mistaken_for_the_deadline():
    def boom():
        raise TimeoutError("fn's own timeout, not ours")

    with timeout_thread(seconds=10, name="test") as timeout:
        with pytest.raises(TimeoutError, match="fn's own timeout"):
            timeout.run(boom)
        assert not timeout.expired
        assert timeout.run(operator.add, 1, 2) == 3


def test_remaining_and_expired_track_the_deadline():
    with timeout_thread(seconds=0.05, name="test") as timeout:
        assert 0 < timeout.remaining <= 0.05
        assert not timeout.expired
        time.sleep(0.1)
        assert timeout.remaining < 0
        assert timeout.expired


def test_the_helper_thread_runs_in_a_copy_of_the_callers_context():
    """Anything reading contextvars -- logger.contextualize() included -- behaves as it does here."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar("var", default="<unset>")
    var.set("caller-value")
    with timeout_thread(seconds=10, name="test") as timeout:
        assert timeout.run(var.get) == "caller-value"

        def mutate() -> str:
            var.set("helper-value")
            return var.get()

        assert timeout.run(mutate) == "helper-value"
    assert var.get() == "caller-value"
