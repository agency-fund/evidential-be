from datetime import timedelta

from xngin.apiserver.worker.jobs.base import Backoff


def test_backoff_escalates_then_holds_at_its_last_delay():
    backoff = Backoff(delays=(timedelta(minutes=1), timedelta(minutes=5)))
    assert [backoff.delay_after(failures) for failures in (1, 2, 3, 10)] == [
        timedelta(minutes=1),
        timedelta(minutes=5),
        timedelta(minutes=5),
        timedelta(minutes=5),
    ]
    assert Backoff.constant(timedelta(hours=1)).delay_after(4) == timedelta(hours=1)
