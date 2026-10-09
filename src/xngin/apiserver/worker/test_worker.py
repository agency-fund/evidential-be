from datetime import timedelta
from threading import Event

from sqlalchemy import select, text, update

from xngin.apiserver import database
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.worker_helpers import (
    NOW,
    FakeJob,
    OtherFakeJob,
    Pause,
    attempt_due,
    make_push_experiment,
    paused_attempt,
    ran_successfully,
    run_nothing,
    schedule_daily,
    status_for,
    wait_until_backend_exits,
)
from xngin.apiserver.worker.attempts import AttemptResult, FinishedAttempt, attempt_experiment, claim_experiment
from xngin.apiserver.worker.jobs.base import JobFailed, JobSucceeded
from xngin.apiserver.worker.scheduler import (
    _seconds_until_next_run,  # noqa: PLC2701
    _stale_experiment_ids,  # noqa: PLC2701
    due_experiment_ids,
    refresh_stale_experiments,
)


def test_cross_job_exclusion_and_parallel_experiments(xngin_session, testing_datasource):
    for _ in range(2):
        make_push_experiment(xngin_session, testing_datasource.ds)
    pause = Pause()
    jobs = (FakeJob(run=pause), OtherFakeJob())
    refresh_stale_experiments(jobs)
    first, second = due_experiment_ids(jobs)
    with paused_attempt(jobs, pause, first) as running:
        # The paused attempt holds the first experiment for every job, so a second attempt on it cannot claim it.
        assert attempt_due((jobs[1],)) == FinishedAttempt(first, AttemptResult.NOT_CLAIMED)
        other = attempt_due((jobs[1],), excluded=(first,))
        assert other.experiment_id == second
        assert ran_successfully(other, jobs[1])
    assert ran_successfully(running.result(), jobs[0])
    assert attempt_due((jobs[1],)).experiment_id == first
    assert due_experiment_ids((jobs[1],)) == ()


def test_failure_rolls_back_effects_and_keeps_backoff_across_refreshes(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    original = experiment.name

    def fail(session, experiment_id):
        session.execute(
            update(tables.Experiment).where(tables.Experiment.id == experiment_id).values(name="uncommitted")
        )
        raise ValueError("failure")

    job = FakeJob(run=fail)
    refresh_stale_experiments((job,))
    assert attempt_due((job,), now=NOW).result is AttemptResult.JOB_RAN
    xngin_session.refresh(experiment)
    assert experiment.name == original
    status = status_for(experiment.id, job)
    assert status.consecutive_failures == 1
    assert status.next_run_at == NOW + timedelta(minutes=1)
    refresh_stale_experiments((job,))
    assert due_experiment_ids((job,), now=NOW) == ()
    assert status_for(experiment.id, job).next_run_at == status.next_run_at
    assert status_for(experiment.id, job).consecutive_failures == 1
    job.run = run_nothing
    assert attempt_due((job,), now=NOW + timedelta(minutes=1)).result is AttemptResult.JOB_RAN
    assert status_for(experiment.id, job).consecutive_failures == 0


def test_long_job_errors_are_truncated_to_fit_the_status_row(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    job = FakeJob(run=lambda session, experiment_id: JobFailed(error="x" * 3000))
    refresh_stale_experiments((job,))
    assert attempt_due((job,)).result is AttemptResult.JOB_RAN
    assert status_for(experiment.id, job).last_error == "x" * 2048


def test_connection_loss_rolls_back_claim_status_and_effects(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)

    def disconnect(session, experiment_id):
        session.execute(update(tables.Experiment).where(tables.Experiment.id == experiment_id).values(name="lost"))
        session.connection().invalidate()
        raise ConnectionError("lost connection")

    job = FakeJob(run=disconnect)
    refresh_stale_experiments((job,))
    assert attempt_due((job,)).result is AttemptResult.ABORTED
    assert status_for(experiment.id, job).last_run_at is None
    xngin_session.refresh(experiment)
    assert experiment.name != "lost"
    job.run = run_nothing
    assert ran_successfully(attempt_due((job,)), job)


def stale_among(jobs, *experiments):
    with database.get_session() as session:
        return set(_stale_experiment_ids(session, jobs)) & {experiment.id for experiment in experiments}


def test_attempt_that_fails_before_claiming_aborts(xngin_session, mocker):
    mocker.patch("xngin.apiserver.worker.attempts.claim_experiment", side_effect=ConnectionError("unreachable"))
    assert attempt_experiment((FakeJob(),), "e") == FinishedAttempt("e", AttemptResult.ABORTED)


def test_parent_lock_delays_maintenance_without_claiming_work(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    job = FakeJob()
    refresh_stale_experiments((job,))
    with database.get_session() as deleter, deleter.begin():
        deleter.execute(select(tables.Experiment.id).where(tables.Experiment.id == experiment.id).with_for_update())
        assert attempt_due((job,)) == FinishedAttempt(experiment.id, AttemptResult.NOT_CLAIMED)
        assert status_for(experiment.id, job).last_run_at is None
        # A busy parent must leave the claim available.
        assert claim_experiment(deleter, experiment.id) is not None
    assert ran_successfully(attempt_due((job,)), job)


def test_due_jobs_and_experiments_get_fair_turns(xngin_session, testing_datasource):
    first = make_push_experiment(xngin_session, testing_datasource.ds)
    second = make_push_experiment(xngin_session, testing_datasource.ds)
    calls = []

    def record(session, experiment_id):
        calls.append(experiment_id)
        return JobSucceeded()

    jobs = (FakeJob(schedule=lambda status, now: now, run=record), OtherFakeJob())
    refresh_stale_experiments(jobs)
    results = [attempt_due(jobs) for _ in range(4)]
    assert all(r.result is AttemptResult.JOB_RAN for r in results)
    assert calls.count(first.id) == calls.count(second.id) == 1
    assert all(status_for(e.id, jobs[1]).last_success_at for e in (first, second))


def test_backend_termination_allows_takeover_but_old_attempt_cannot_write(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    pause = Pause(
        after=lambda session, experiment_id: session.add(tables.Snapshot(experiment_id=experiment_id, data={}))
    )
    job = FakeJob(run=pause)
    refresh_stale_experiments((job,))
    with paused_attempt((job,), pause, experiment.id) as old:
        with database.get_engine().begin() as connection:
            assert connection.scalar(text("SELECT pg_terminate_backend(:pid)"), {"pid": pause.pid})
        wait_until_backend_exits(pause.pid)
        assert ran_successfully(attempt_due((FakeJob(),)), FakeJob())
    assert old.result().result is AttemptResult.ABORTED
    with database.get_session() as session:
        assert list(session.scalars(select(tables.Snapshot))) == []
    assert status_for(experiment.id, job).last_success_at is not None


def test_refresh_failure_does_not_hide_other_experiments(xngin_session, testing_datasource):
    experiments = sorted(
        [make_push_experiment(xngin_session, testing_datasource.ds) for _ in range(2)], key=lambda e: e.id
    )
    broken, healthy = experiments

    def schedule(status, now):
        if status.experiment_id == broken.id:
            raise ValueError("Invalid configuration")
        return schedule_daily(status, now)

    job = FakeJob(schedule=schedule)
    assert broken.id in refresh_stale_experiments((job,))
    result = attempt_due((job,))
    assert ran_successfully(result, job)
    assert result.experiment_id == healthy.id


def test_refresh_ends_early_once_stop_is_set(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    stop = Event()
    stop.set()
    assert refresh_stale_experiments((FakeJob(),), stop=stop) == ()
    assert status_for(experiment.id, FakeJob()) is None


def test_refresh_covers_a_bounded_number_of_experiments_per_pass(xngin_session, testing_datasource):
    job = FakeJob()
    experiments = [make_push_experiment(xngin_session, testing_datasource.ds) for _ in range(3)]
    refresh_stale_experiments((job,), limit=2)
    assert len([experiment for experiment in experiments if status_for(experiment.id, job) is not None]) == 2
    refresh_stale_experiments((job,), limit=2)
    assert all(status_for(experiment.id, job) is not None for experiment in experiments)
    assert stale_among((job,), *experiments) == set()


def test_refresh_creates_status_rows_for_a_newly_added_job(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    refresh_stale_experiments((FakeJob(),))
    assert stale_among((FakeJob(),), experiment) == set()
    assert stale_among((FakeJob(), OtherFakeJob()), experiment) == {experiment.id}
    refresh_stale_experiments((FakeJob(), OtherFakeJob()))
    assert status_for(experiment.id, OtherFakeJob()) is not None


def test_edit_committed_during_an_attempt_is_refreshed_afterward(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    pause = Pause()
    job = FakeJob(schedule=lambda status, now: now, run=pause)
    refresh_stale_experiments((job,))
    with paused_attempt((job,), pause, experiment.id):
        # The attempt refreshed its run times from the experiment as it was before this edit.
        experiment.name = "edited"
        xngin_session.commit()
    assert stale_among((job,), experiment) == {experiment.id}


def test_seconds_until_next_run_ignores_due_and_unscheduled_jobs(xngin_session, testing_datasource):
    soon, due, unscheduled = (make_push_experiment(xngin_session, testing_datasource.ds) for _ in range(3))
    offsets = {soon.id: timedelta(hours=1), due.id: timedelta(hours=-1)}

    def schedule(status, now):
        offset = offsets.get(status.experiment_id)
        return None if offset is None else now + offset

    job = FakeJob(schedule=schedule)
    refresh_stale_experiments((job,))

    assert status_for(unscheduled.id, job).next_run_at is None
    seconds = _seconds_until_next_run((job,))
    assert seconds is not None
    assert 3500 < seconds <= 3600
    assert _seconds_until_next_run((OtherFakeJob(),)) is None
