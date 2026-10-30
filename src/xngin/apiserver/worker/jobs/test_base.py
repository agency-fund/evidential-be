from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import DateTime, literal, select

from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.worker_helpers import make_push_experiment
from xngin.apiserver.worker.jobs.base import Backoff, next_run_in_window

OPENS = datetime(2026, 1, 1, tzinfo=UTC)
CLOSES = OPENS + timedelta(days=10)


def test_backoff_escalates_then_holds_at_its_last_delay():
    backoff = Backoff(delays=(timedelta(minutes=1), timedelta(minutes=5)))
    assert [backoff.delay_after(failures) for failures in (1, 2, 3, 10)] == [
        timedelta(minutes=1),
        timedelta(minutes=5),
        timedelta(minutes=5),
        timedelta(minutes=5),
    ]
    assert Backoff.constant(timedelta(hours=1)).delay_after(4) == timedelta(hours=1)


LATE = CLOSES + timedelta(minutes=1)


@pytest.mark.parametrize(
    ("last_success_at", "last_run_at", "now", "expected"),
    [
        pytest.param(None, None, OPENS - timedelta(days=1), OPENS, id="never-ran-before-opens"),
        pytest.param(None, None, OPENS + timedelta(days=1), OPENS + timedelta(days=1), id="never-ran-in-window"),
        pytest.param(None, OPENS, LATE, None, id="never-succeeded-closed"),
        pytest.param(
            OPENS - timedelta(days=5),
            OPENS - timedelta(days=5),
            OPENS - timedelta(days=1),
            OPENS,
            id="ran-before-opens",
        ),
        pytest.param(OPENS, OPENS, OPENS, OPENS + timedelta(days=2), id="interval-fits"),
        # A 2-day interval from 1 day before the close would skip the close, so the run moves to it.
        pytest.param(
            CLOSES - timedelta(days=1), CLOSES - timedelta(days=1), CLOSES - timedelta(days=1), CLOSES, id="overshoots"
        ),
        # The attempt for the run due at the close starts a minute late: the run is still due.
        pytest.param(CLOSES - timedelta(days=1), CLOSES - timedelta(days=1), LATE, CLOSES, id="final-run-starts-late"),
        # A run due mid-window that the worker missed still gets one attempt after the close.
        pytest.param(OPENS, OPENS, LATE, OPENS + timedelta(days=2), id="missed-run-after-close"),
        # The late attempt failed, so the run already had its attempt.
        pytest.param(CLOSES - timedelta(days=1), LATE, LATE, None, id="final-run-attempted"),
        # The migration seeded the last success and stamped the migration time, after the close, as the last attempt.
        pytest.param(OPENS, LATE, LATE, None, id="seeded-after-close"),
        pytest.param(CLOSES, CLOSES, CLOSES, None, id="ran-at-close"),
    ],
)
def test_next_run_in_window(xngin_session, testing_datasource, last_success_at, last_run_at, now, expected):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    xngin_session.add(
        tables.JobStatus(
            experiment_id=experiment.id, job_name="test", last_success_at=last_success_at, last_run_at=last_run_at
        )
    )
    xngin_session.commit()

    def timestamp(value):
        return literal(value, DateTime(timezone=True))

    run_at = next_run_in_window(timestamp(now), timedelta(days=2), opens=timestamp(OPENS), closes=timestamp(CLOSES))
    assert xngin_session.scalar(select(run_at).where(tables.JobStatus.experiment_id == experiment.id)) == expected
