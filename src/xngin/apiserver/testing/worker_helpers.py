"""Jobs, experiments, and status readers shared by the worker tests."""

import time
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Event
from typing import ClassVar
from unittest.mock import patch

from sqlalchemy import ColumnElement, false, select, text
from sqlalchemy.orm import Session

from xngin.apiserver import database
from xngin.apiserver.routers.common_enums import ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.apiserver.routers.experiments.test_experiments_common import insert_experiment_and_arms
from xngin.apiserver.sqla import tables
from xngin.apiserver.worker.attempts import (
    AttemptExecutor,
    AttemptResult,
    FinishedAttempt,
    attempt_experiment,
)
from xngin.apiserver.worker.jobs.base import Backoff, Job, JobSucceeded
from xngin.apiserver.worker.jobs.snapshots import SnapshotJob
from xngin.apiserver.worker.scheduler import Pacer, due_experiment_ids, run_worker

BACKOFF = Backoff(delays=(timedelta(minutes=1), timedelta(minutes=5)))
SNAPSHOT = SnapshotJob(interval=timedelta(hours=3), dwh_timeout=90, backoff=Backoff.constant(timedelta(hours=1)))
NOW = datetime(2030, 1, 1, tzinfo=UTC)


def run_nothing(session, experiment_id):
    return JobSucceeded()


def schedule_daily(status, now):
    return status.last_success_at + timedelta(days=1) if status.last_success_at else now


@dataclass(slots=True)
class FakeJob:
    name: ClassVar[str] = "test"
    backoff: Backoff = BACKOFF
    schedule: Callable = schedule_daily
    run: Callable = run_nothing

    def needs_refresh(self) -> ColumnElement[bool]:
        return false()

    def compute_next_run_at(self, session, experiment, status, now):
        return self.schedule(status, now)

    def execute(self, session, experiment_id):
        return self.run(session, experiment_id)


class OtherFakeJob(FakeJob):
    name = "zother"


def make_push_experiment(session, datasource, **kwargs):
    return insert_experiment_and_arms(
        session,
        datasource,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
        **kwargs,
    )


def attempt_due(jobs: Sequence[Job], excluded: Collection[str] = (), *, now: datetime | None = None) -> FinishedAttempt:
    """Attempt the first due experiment, as the scheduler would after a look.

    now stands in for both the scheduler's and the attempt's clock. Tests expecting nothing to be due assert on
    due_experiment_ids instead.
    """
    due = due_experiment_ids(jobs, excluded, now=now)
    if not due:
        raise LookupError("No experiment is due.")
    if now is None:
        return attempt_experiment(jobs, due[0])
    with attempt_clock(return_value=now):
        return attempt_experiment(jobs, due[0])


def attempt_clock(**kwargs):
    """Patch the database time attempts read: attempt_clock(return_value=NOW) or side_effect=[started, finished]."""
    return patch("xngin.apiserver.worker.attempts.database_now", **kwargs)


def ran_successfully(attempt: FinishedAttempt, job: Job) -> bool:
    """Whether the attempt ran the job and the job's status shows no failure.

    Jobs report exceptions, including failed assertions inside fake jobs, as job failures rather than raising.
    """
    return attempt.result is AttemptResult.JOB_RAN and status_for(attempt.experiment_id, job).consecutive_failures == 0


def status_for(experiment_id, job):
    with database.get_session() as session:
        return session.get(tables.JobStatus, (experiment_id, job.name))


def read_outcomes(session, experiment_id):
    return dict(
        session.execute(
            select(tables.Draw.participant_id, tables.Draw.outcome).where(tables.Draw.experiment_id == experiment_id)
        ).all()
    )


@dataclass(slots=True)
class FakeClock:
    """A worker clock whose time moves only while the worker waits with no attempt in flight.

    Attempts run on the executor's threads in real time, so while one is in flight a wait blocks until it finishes
    instead of advancing the clock; fake time never passes during an attempt.
    """

    now: float = 0.0
    executor: AttemptExecutor | None = None

    def monotonic(self) -> float:
        return self.now

    def wait(self, event: Event, timeout: float) -> bool:
        if self.executor is not None and self.executor.in_progress():
            if not event.wait(5):
                raise TimeoutError("The worker's attempt never finished.")
            return True
        if event.is_set():
            return True
        self.now += timeout
        return False


def run_worker_on_fake_clock(
    jobs: Sequence[Job], stop: Event, clock: FakeClock, pacer: Pacer, on_look: Callable[[], object] = lambda: None
) -> None:
    """Run the worker on the test's thread, one attempt at a time, with time moving only on the fake clock.

    on_look is called after each look has handed out its attempt, if any.
    """

    class ObservedExecutor(AttemptExecutor):
        def start(self, jobs: Sequence[Job], stop: Event, experiment_ids: Iterable[str]) -> None:
            super().start(jobs, stop, experiment_ids)
            on_look()

    executor = ObservedExecutor(1, clock)
    clock.executor = executor
    with executor:
        run_worker(jobs, stop, clock, pacer, executor)


@dataclass(slots=True)
class Pause:
    """A FakeJob run that pauses partway, so a test can act while the attempt holds its claim.

    before and after run on either side of the pause. The worker records a job's exceptions, including failed
    assertions, as job failures rather than raising them, so the pause keeps them for paused_attempt to raise.
    """

    before: Callable[[Session, str], object] | None = None
    after: Callable[[Session, str], object] | None = None
    reached: Event = field(default_factory=Event)
    resume: Event = field(default_factory=Event)
    errors: list[BaseException] = field(default_factory=list)
    # The paused attempt's database backend, so a test can terminate it and wait for it to exit.
    pid: int | None = None

    def __call__(self, session: Session, experiment_id: str) -> JobSucceeded:
        try:
            self._run(session, experiment_id)
        except BaseException as exc:
            self.errors.append(exc)
            raise
        return JobSucceeded()

    def _run(self, session: Session, experiment_id: str) -> None:
        self.pid = session.scalar(text("SELECT pg_backend_pid()"))
        if self.before:
            self.before(session, experiment_id)
        self.reached.set()
        if not self.resume.wait(5):
            raise TimeoutError("The test never resumed the paused job.")
        if self.after:
            self.after(session, experiment_id)


@contextmanager
def paused_attempt(jobs: Sequence[Job], pause: Pause, experiment_id: str) -> Iterator[Future[FinishedAttempt]]:
    """Start attempt_experiment on a thread and yield its future once the job reaches the pause.

    Leaving the block resumes the job and waits for the attempt to finish. If the job raised, that exception is
    raised here, so a test cannot pass while its job failed.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        attempt = pool.submit(attempt_experiment, jobs, experiment_id)
        try:
            if not pause.reached.wait(3):
                raise TimeoutError(f"The attempt never reached the pause: {attempt}")
            yield attempt
        finally:
            pause.resume.set()
        attempt.result(timeout=5)
    if pause.errors:
        raise pause.errors[0]


def wait_until_backend_exits(pid: int | None) -> None:
    """Wait until the database backend pid is gone, as pg_terminate_backend returns before it has released its locks."""
    deadline = time.monotonic() + 3
    with database.get_session() as observer:
        while observer.scalar(text("SELECT EXISTS (SELECT FROM pg_stat_activity WHERE pid = :pid)"), {"pid": pid}):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Backend {pid} did not exit.")
            time.sleep(0.01)
