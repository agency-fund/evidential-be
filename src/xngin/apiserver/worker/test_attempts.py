from datetime import timedelta

import pytest

from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.worker_helpers import NOW, FakeJob, OtherFakeJob
from xngin.apiserver.worker.attempts import _select_best_next_job  # noqa: PLC2701

JOBS = (FakeJob(), OtherFakeJob())


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        # Each job maps to its (next_run_at, last_run_at).
        pytest.param({"test": (None, None), "zother": (NOW + timedelta(seconds=1), None)}, None, id="none-due-yet"),
        pytest.param({"test": (NOW, None), "zother": (None, None)}, "test", id="due-exactly-now"),
        pytest.param(
            {"test": (NOW, NOW - timedelta(days=1)), "zother": (NOW, None)}, "zother", id="never-ran-goes-first"
        ),
        pytest.param(
            {"test": (NOW, NOW - timedelta(minutes=1)), "zother": (NOW, NOW - timedelta(minutes=5))},
            "zother",
            id="least-recently-run-goes-first",
        ),
        pytest.param({"test": (NOW, None), "zother": (NOW, None)}, "test", id="ties-go-to-the-job-name"),
    ],
)
def test_select_best_next_job(runs, expected):
    statuses = {
        name: tables.JobStatus(experiment_id="e", job_name=name, next_run_at=next_run_at, last_run_at=last_run_at)
        for name, (next_run_at, last_run_at) in runs.items()
    }
    selected = _select_best_next_job(JOBS, statuses, NOW)
    assert (selected.name if selected else None) == expected
