from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event, Lock

import pytest
from sqlalchemy import case

from xngin.apiserver import database
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.worker_helpers import (
    AUTOFAIL,
    FakeClock,
    FakeJob,
    due_experiment_ids,
    make_autofail_experiment,
    make_push_experiment,
    read_outcomes,
    run_worker_on_fake_clock,
    status_for,
)
from xngin.apiserver.worker.attempts import (
    AttemptResult,
    FinishedAttempt,
    claim_experiment,
)
from xngin.apiserver.worker.executor import AttemptExecutor, SystemClock
from xngin.apiserver.worker.jobs.base import JobSucceeded
from xngin.apiserver.worker.scheduler import Pacer, Schedule, _look, run_worker  # noqa: PLC2701


def test_worker_runs_parallel_jobs_and_refills_a_slot_while_another_is_busy(xngin_session, testing_datasource):
    experiments = [make_push_experiment(xngin_session, testing_datasource.ds) for _ in range(4)]
    releases = {experiment.id: Event() for experiment in experiments}
    entered: Queue[str] = Queue()
    active: set[str] = set()
    concurrency: list[int] = []
    lock, stop = Lock(), Event()

    def pause(session, experiment):
        experiment_id = experiment.id
        with lock:
            active.add(experiment_id)
            concurrency.append(len(active))
        entered.put(experiment_id)
        try:
            assert releases[experiment_id].wait(10)
        finally:
            with lock:
                active.remove(experiment_id)
        return JobSucceeded()

    job = FakeJob(run=pause)
    clock = SystemClock()
    with ThreadPoolExecutor(max_workers=1) as pool, AttemptExecutor(2, clock) as executor:
        running = pool.submit(
            run_worker, (job,), stop, Pacer(poll_interval=60, clock=clock, exit_after_jobs=4), executor
        )
        try:
            first, second = entered.get(timeout=5), entered.get(timeout=5)
            assert first != second
            with lock:
                assert active == {first, second}

            releases[first].set()
            third = entered.get(timeout=5)
            assert third not in {first, second}
            with lock:
                assert active == {second, third}
            assert status_for(first, job).last_success_at is not None

            # Refill the same slot again while the second job stays blocked. The sixty-second
            # poll cannot account for either refill; the real completion notifications must wake it.
            releases[third].set()
            fourth = entered.get(timeout=5)
            assert {first, second, third, fourth} == set(releases)
            with lock:
                assert active == {second, fourth}
            releases[second].set()
            releases[fourth].set()
            running.result(timeout=5)
        finally:
            stop.set()
            for release in releases.values():
                release.set()

    assert max(concurrency) == 2
    assert not active
    assert entered.empty()
    for experiment in experiments:
        status = status_for(experiment.id, job)
        assert status.last_success_at is not None
        assert status.consecutive_failures == 0


def test_worker_shutdown_drains_execution(xngin_session, testing_datasource):
    make_push_experiment(xngin_session, testing_datasource.ds)
    entered, release, stop = Event(), Event(), Event()

    def pause(session, experiment):
        entered.set()
        assert release.wait(5)
        return JobSucceeded()

    clock = SystemClock()
    executor = AttemptExecutor(2, clock)

    def run_then_drain():
        with executor:
            run_worker((FakeJob(run=pause),), stop, Pacer(poll_interval=0.1, clock=clock), executor)

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(run_then_drain)
        try:
            assert entered.wait(3)
            stop.set()
            # Leaving the executor's context waits for the paused attempt, so the worker has not finished.
            assert not running.done()
        finally:
            stop.set()
            release.set()
        running.result(timeout=4)


def test_worker_without_experiments_looks_once_per_poll_interval(xngin_session):
    clock, stop, looks = FakeClock(), Event(), []

    def record_look():
        looks.append(clock.monotonic())
        if len(looks) == 3:
            stop.set()

    run_worker_on_fake_clock((FakeJob(),), stop, Pacer(poll_interval=10, clock=clock), record_look)
    # Attempts that find nothing must not trigger another look; only the poll interval does.
    assert looks == [0, 10, 20]


def test_next_look_sees_edits_and_new_experiments(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    clock, stop, calls = FakeClock(), Event(), []
    created: list[tables.Experiment] = []

    def schedule(now):
        # The first experiment is unscheduled until edited; any other is due at once.
        return case(
            (tables.Experiment.id != experiment.id, now),
            (tables.Experiment.name == "edited", now),
            else_=None,
        )

    def record(session, experiment):
        calls.append((experiment.id, clock.monotonic()))
        if len(calls) == 2:
            stop.set()
        return JobSucceeded()

    def change_after_first_look():
        if not created:
            experiment.name = "edited"
            xngin_session.commit()
            created.append(make_push_experiment(xngin_session, testing_datasource.ds))

    run_worker_on_fake_clock(
        (FakeJob(schedule=schedule, run=record),), stop, Pacer(poll_interval=10, clock=clock), change_after_first_look
    )
    # The look one poll interval in finds both the edited experiment and the new one, though neither changed any
    # worker state.
    assert sorted(calls) == sorted([(experiment.id, 10), (created[0].id, 10)])


def test_failed_look_finds_no_work_and_waits_a_poll_interval(mocker):
    mocker.patch("xngin.apiserver.worker.scheduler.read_schedule", side_effect=RuntimeError("database down"))
    capture_exception = mocker.patch("xngin.apiserver.worker.scheduler.sentry_sdk.capture_exception")
    assert _look((FakeJob(),), ()) == Schedule()
    capture_exception.assert_called_once()


def test_claims_held_elsewhere_do_not_hide_available_work(xngin_session, testing_datasource):
    for _ in range(6):
        make_push_experiment(xngin_session, testing_datasource.ds)
    job = FakeJob()
    # Hold every experiment the scheduler tries before the last one.
    *held, available = due_experiment_ids((job,))
    clock = FakeClock()
    pacer = Pacer(poll_interval=60, clock=clock, exit_after_jobs=1)
    with database.get_session() as holder, holder.begin():
        for experiment_id in held:
            assert claim_experiment(holder, experiment_id) is not None
        run_worker_on_fake_clock((job,), Event(), pacer)
    # Each attempt that could not claim its experiment skipped it and led straight to the next look, without
    # waiting for the poll interval.
    assert status_for(available, job).last_success_at is not None
    assert all(status_for(experiment_id, job) is None for experiment_id in held)
    assert set(pacer.skipped()) == set(held)
    assert clock.monotonic() == 0


def test_autofail_backlog_drains_without_waiting_for_poll(xngin_session, testing_datasource):
    # A two-draw batch keeps this test small while requiring three independent attempts.
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1", "2", "3", "4", "5"))
    clock, looks = FakeClock(), []
    run_worker_on_fake_clock(
        (AUTOFAIL,),
        Event(),
        Pacer(poll_interval=60, clock=clock, exit_after_jobs=3),
        lambda: looks.append(clock.monotonic()),
    )
    # Completion wake-ups dispatch every batch at once, without waiting for a poll interval.
    assert looks == [0, 0, 0]
    assert read_outcomes(xngin_session, experiment.id) == dict.fromkeys(("1", "2", "3", "4", "5"), 0.0)


def make_pacer(clock, **kwargs):
    return Pacer(poll_interval=10, clock=clock, **kwargs)


@pytest.mark.parametrize("result", list(AttemptResult))
def test_record_requests_a_look_after_any_attempt_finishes(result):
    clock = FakeClock()
    pacer = make_pacer(clock)
    pacer.looked(Schedule())
    clock.now = 3
    pacer.record_finished([])
    assert not pacer.should_look()
    pacer.record_finished([FinishedAttempt("e", result)])
    assert pacer.should_look()


@pytest.mark.parametrize(
    ("result", "skipped"),
    [
        (AttemptResult.JOB_RAN, False),
        (AttemptResult.NOTHING_READY, False),
        (AttemptResult.ABORTED, True),
        (AttemptResult.NOT_CLAIMED, True),
    ],
)
def test_aborted_and_unclaimable_experiments_are_skipped_for_one_poll_interval(result, skipped):
    clock = FakeClock(now=100)
    pacer = make_pacer(clock)
    pacer.record_finished([FinishedAttempt("e", result)])
    assert pacer.skipped() == (("e",) if skipped else ())
    clock.now = 109.9
    pacer.record_finished([])
    assert pacer.skipped() == (("e",) if skipped else ())
    clock.now = 110
    pacer.record_finished([])
    assert pacer.skipped() == ()


def test_should_exit_counts_executed_jobs_only():
    pacer = make_pacer(FakeClock(), exit_after_jobs=2)
    pacer.record_finished([
        FinishedAttempt("a", AttemptResult.JOB_RAN),
        FinishedAttempt("b", AttemptResult.NOTHING_READY),
        FinishedAttempt("c", AttemptResult.ABORTED),
        FinishedAttempt("d", AttemptResult.NOT_CLAIMED),
    ])
    assert not pacer.should_exit()
    pacer.record_finished([FinishedAttempt("a", AttemptResult.JOB_RAN)])
    assert pacer.should_exit()
    assert not make_pacer(FakeClock()).should_exit()


def test_look_waits_a_poll_interval_without_a_known_run_time():
    clock = FakeClock()
    pacer = make_pacer(clock)
    assert pacer.should_look()
    pacer.looked(Schedule())
    clock.now = 9.9
    assert not pacer.should_look()
    clock.now = 10
    assert pacer.should_look()


def test_known_run_time_brings_the_look_earlier_but_never_later():
    pacer = make_pacer(FakeClock(now=4))
    pacer.looked(Schedule(next_run_in=3))
    assert pacer.next_look == 7
    pacer.looked(Schedule(next_run_in=30))
    assert pacer.next_look == 14


def test_wake_ignores_the_next_look_while_every_slot_is_busy():
    pacer = make_pacer(FakeClock())
    pacer.looked(Schedule(next_run_in=2))
    assert pacer.wake_at(can_start=True) == 2
    assert pacer.wake_at(can_start=False) == float("inf")
