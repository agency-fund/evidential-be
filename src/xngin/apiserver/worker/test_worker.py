from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest
from sqlalchemy import select, text, update

from xngin.apiserver import database
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.routers.admin.test_admin_api import (
    make_bandit_online_experiment,
)
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.apiserver.routers.experiments.experiments_common import (
    create_assignment_for_participant,
    update_bandit_arm_with_outcome_impl,
)
from xngin.apiserver.routers.experiments.test_experiments_common import make_create_online_bandit_experiment_request
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.worker_helpers import (
    AUTOFAIL,
    NOW,
    PULL,
    SNAPSHOT,
    FakeJob,
    OtherFakeJob,
    Pause,
    attempt_clock,
    attempt_due,
    make_autofail_experiment,
    make_dwh_experiment,
    make_push_experiment,
    paused_attempt,
    ran_successfully,
    read_outcomes,
    run_nothing,
    schedule_daily,
    status_for,
    wait_until_backend_exits,
)
from xngin.apiserver.worker.attempts import AttemptResult, FinishedAttempt, attempt_experiment, claim_experiment
from xngin.apiserver.worker.jobs import bandit_outcomes
from xngin.apiserver.worker.jobs.base import JobFailed, JobSucceeded
from xngin.apiserver.worker.scheduler import (
    _seconds_until_next_run,  # noqa: PLC2701
    _stale_experiment_ids,  # noqa: PLC2701
    due_experiment_ids,
    refresh_stale_experiments,
)
from xngin.stats.stats_errors import StatsNoParticipantsError


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


def test_future_activation_and_date_invalidation(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    experiment.start_date = NOW + timedelta(days=3)
    experiment.end_date = NOW + timedelta(days=10)
    xngin_session.commit()
    refresh_stale_experiments((SNAPSHOT,))
    assert due_experiment_ids((SNAPSHOT,), now=NOW) == ()
    assert status_for(experiment.id, SNAPSHOT).next_run_at == NOW + timedelta(days=2)
    experiment.start_date = NOW + timedelta(days=4)
    xngin_session.commit()
    refresh_stale_experiments((SNAPSHOT,))
    assert due_experiment_ids((SNAPSHOT,), now=NOW) == ()
    assert status_for(experiment.id, SNAPSHOT).next_run_at == NOW + timedelta(days=3)


def test_dwh_pull_waits_for_the_start_date(xngin_session, testing_datasource):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    experiment.start_date = NOW + timedelta(days=3)
    experiment.end_date = NOW + timedelta(days=10)
    xngin_session.commit()
    refresh_stale_experiments((PULL, SNAPSHOT))
    assert status_for(experiment.id, PULL).next_run_at == NOW + timedelta(days=3)
    assert status_for(experiment.id, SNAPSHOT).next_run_at == NOW + timedelta(days=2)


@pytest.mark.parametrize(
    ("days_since_end", "resolved", "scheduled"),
    [
        (3, False, True),
        (3, True, False),
        (9, False, False),
    ],
)
def test_dwh_pull_continues_after_the_buffer_while_draws_are_pending(
    xngin_session, testing_datasource, days_since_end, resolved, scheduled
):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    experiment.end_date = datetime.now(UTC) - timedelta(days=days_since_end)
    if resolved:
        update_bandit_arm_with_outcome_impl(xngin_session, experiment, "1", 1.0)
    xngin_session.commit()
    refresh_stale_experiments((PULL, SNAPSHOT))
    assert (status_for(experiment.id, PULL).next_run_at is not None) == scheduled
    assert status_for(experiment.id, SNAPSHOT).next_run_at is None


def test_autofail_catches_up_after_an_outage(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1", "2", "3", "4", "5"))
    refresh_stale_experiments((AUTOFAIL,))
    assert status_for(experiment.id, AUTOFAIL).next_run_at is not None

    # The worker is down for a month, long past the end of the experiment. Keeping
    # updated_at leaves the stored run time as the only thing that brings the experiment back.
    month_ago = datetime.now(UTC) - timedelta(days=30)
    xngin_session.execute(
        update(tables.Experiment)
        .where(tables.Experiment.id == experiment.id)
        .values(start_date=month_ago - timedelta(days=7), end_date=month_ago, updated_at=tables.Experiment.updated_at)
    )
    xngin_session.execute(
        update(tables.Draw)
        .where(tables.Draw.experiment_id == experiment.id)
        .values(created_at=month_ago - timedelta(days=1))
    )
    xngin_session.commit()
    assert stale_among((AUTOFAIL,), experiment) == set()

    # Five overdue draws take three batches of two.
    for remaining in (3, 1, 0):
        assert ran_successfully(attempt_due((AUTOFAIL,)), AUTOFAIL)
        assert list(read_outcomes(xngin_session, experiment.id).values()).count(None) == remaining
    assert status_for(experiment.id, AUTOFAIL).next_run_at is None


def test_autofail_one_batch_and_overdue_work_past_horizon(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds)
    experiment.end_date = datetime.now(UTC) - timedelta(days=5)
    xngin_session.commit()
    refresh_stale_experiments((AUTOFAIL,))
    assert ran_successfully(attempt_due((AUTOFAIL,)), AUTOFAIL)
    assert list(read_outcomes(xngin_session, experiment.id).values()).count(None) == 1
    assert ran_successfully(attempt_due((AUTOFAIL,)), AUTOFAIL)
    assert list(read_outcomes(xngin_session, experiment.id).values()).count(None) == 0
    assert status_for(experiment.id, AUTOFAIL).next_run_at is None


def stale_among(jobs, *experiments):
    with database.get_session() as session:
        return set(_stale_experiment_ids(session, jobs)) & {experiment.id for experiment in experiments}


def test_refresh_wakes_autofail_for_the_first_pending_draw(xngin_session, testing_datasource):
    resolved = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    pending = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    update_bandit_arm_with_outcome_impl(xngin_session, resolved, "1", 1.0)
    pending.end_date = datetime.now(UTC) - timedelta(days=5)
    xngin_session.commit()

    refresh_stale_experiments((AUTOFAIL,))

    # Ended experiments stay scheduled while they have unresolved draws.
    assert status_for(pending.id, AUTOFAIL).next_run_at is not None
    assert status_for(resolved.id, AUTOFAIL).next_run_at is None
    assert stale_among((AUTOFAIL,), resolved, pending) == set()

    # Inserting a draw does not update the experiment, but the first pending one still wakes the refresh.
    updated_at = resolved.updated_at
    create_assignment_for_participant(xngin_session, resolved, "2", None, random_state=66)
    xngin_session.commit()
    assert resolved.updated_at == updated_at
    assert stale_among((AUTOFAIL,), resolved, pending) == {resolved.id}
    refresh_stale_experiments((AUTOFAIL,))
    assert status_for(resolved.id, AUTOFAIL).next_run_at is not None
    assert stale_among((AUTOFAIL,), resolved, pending) == set()


def test_refresh_ignores_pending_draws_once_autofail_is_disabled(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    refresh_stale_experiments((AUTOFAIL,))
    experiment.enable_autofail = False
    xngin_session.commit()

    refresh_stale_experiments((AUTOFAIL,))

    # Its draws are still pending, but compute_next_run_at gives no time, so the refresh must not revisit it forever.
    assert status_for(experiment.id, AUTOFAIL).next_run_at is None
    assert stale_among((AUTOFAIL,), experiment) == set()


def test_autofail_schedules_exact_expiry_without_early_execution(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    created_at = datetime.now(UTC)
    xngin_session.execute(
        update(tables.Draw).where(tables.Draw.experiment_id == experiment.id).values(created_at=created_at)
    )
    xngin_session.commit()
    refresh_stale_experiments((AUTOFAIL,))
    assert due_experiment_ids((AUTOFAIL,)) == ()
    assert status_for(experiment.id, AUTOFAIL).next_run_at == created_at + timedelta(hours=1)
    assert read_outcomes(xngin_session, experiment.id) == {"1": None}


def test_locked_autofail_draws_keep_deadline_with_cooldown(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    refresh_stale_experiments((AUTOFAIL,))
    with database.get_session() as api, api.begin():
        api.execute(
            select(tables.Draw).where(tables.Draw.experiment_id == experiment.id).with_for_update(of=tables.Draw)
        )
        # The batch was deferred: no success, no failure, and the attempt counts as having run no job.
        assert attempt_due((AUTOFAIL,)).result is AttemptResult.NOTHING_READY
        status = status_for(experiment.id, AUTOFAIL)
        assert status.next_run_at == status.retry_not_before
        assert status.last_success_at is None
        assert status.consecutive_failures == 0
        assert due_experiment_ids((AUTOFAIL,)) == ()
    assert ran_successfully(attempt_due((AUTOFAIL,), now=status.next_run_at), AUTOFAIL)


def test_claim_corrects_a_stale_run_time_without_running_a_job(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    refresh_stale_experiments((AUTOFAIL,))
    stale = status_for(experiment.id, AUTOFAIL).next_run_at
    assert stale is not None
    # An API outcome resolves the only pending draw, changing neither the experiment nor its status row, so
    # the refresh cannot notice that the stored run time is stale.
    update_bandit_arm_with_outcome_impl(xngin_session, experiment, "1", 1.0)
    xngin_session.commit()
    assert status_for(experiment.id, AUTOFAIL).next_run_at == stale

    # The scheduler hands out the experiment because the stored time is due; the attempt finds nothing ready and
    # corrects it.
    assert attempt_due((AUTOFAIL,)) == FinishedAttempt(experiment.id, AttemptResult.NOTHING_READY)
    status = status_for(experiment.id, AUTOFAIL)
    assert status.next_run_at is None
    assert status.last_run_at is None
    assert due_experiment_ids((AUTOFAIL,)) == ()


def test_attempt_that_fails_before_claiming_aborts(xngin_session, mocker):
    mocker.patch("xngin.apiserver.worker.attempts.claim_experiment", side_effect=ConnectionError("unreachable"))
    assert attempt_experiment((FakeJob(),), "e") == FinishedAttempt("e", AttemptResult.ABORTED)


def test_api_assignment_commits_during_maintenance(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=())
    pause = Pause()
    job = FakeJob(run=pause)
    refresh_stale_experiments((job,))
    with paused_attempt((job,), pause, experiment.id) as running:
        # New assignments must commit while maintenance still holds both locks.
        xngin_session.execute(text("SET LOCAL lock_timeout = '300ms'"))
        create_assignment_for_participant(xngin_session, experiment, "new", None, random_state=66)
        xngin_session.commit()
    assert ran_successfully(running.result(), job)
    refresh_stale_experiments((AUTOFAIL,))
    assert status_for(experiment.id, AUTOFAIL).next_run_at is not None
    assert due_experiment_ids((job,)) == ()


def test_snapshot_without_participants_succeeds_without_backoff(testing_datasource, aclient: AdminAPIClient, mocker):
    experiment_id = make_bandit_online_experiment(aclient, testing_datasource.datasource_id).config.experiment_id
    mocker.patch(
        "xngin.apiserver.worker.jobs.snapshot_analysis._compute_snapshot_data",
        side_effect=StatsNoParticipantsError(),
    )
    refresh_stale_experiments((SNAPSHOT,))
    started = datetime.now(UTC)
    assert attempt_due((SNAPSHOT,), now=started).result is AttemptResult.JOB_RAN
    with database.get_session() as session:
        snapshot = session.scalars(select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id)).one()
    # The snapshot still records why there is no data, but the job counts it as a success.
    assert snapshot.status == "failed"
    assert "No participants" in (snapshot.message or "")
    status = status_for(experiment_id, SNAPSHOT)
    assert status.last_success_at == started
    assert status.consecutive_failures == 0
    assert status.last_error is None
    assert status.next_run_at == started + SNAPSHOT.interval


def test_failed_snapshots_retry_hourly_until_one_succeeds(testing_datasource, aclient: AdminAPIClient, mocker):
    experiment_id = make_bandit_online_experiment(aclient, testing_datasource.datasource_id).config.experiment_id

    def fail(session, *args):
        session.execute(update(tables.Experiment).where(tables.Experiment.id == experiment_id).values(name="partial"))
        session.execute(text("SELECT 1 / 0"))

    analysis = mocker.patch("xngin.apiserver.worker.jobs.snapshot_analysis._compute_snapshot_data", side_effect=fail)
    refresh_stale_experiments((SNAPSHOT,))
    started = datetime.now(UTC)
    assert attempt_due((SNAPSHOT,), now=started).result is AttemptResult.JOB_RAN
    with database.get_session() as session:
        # The failed analysis's writes roll back, but its failed snapshot commits.
        assert session.get(tables.Experiment, experiment_id).name != "partial"
        snapshot = session.scalars(select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id)).one()
    assert snapshot.status == "failed"
    status = status_for(experiment_id, SNAPSHOT)
    # The status row carries the snapshot's own diagnostic.
    assert status.last_error == snapshot.message
    assert "division by zero" in status.last_error
    assert status.last_success_at is None
    retry = SNAPSHOT.backoff.delay_after(1)
    assert retry < SNAPSHOT.interval
    assert status.next_run_at == started + retry
    # Refreshes and repeated failures cannot accelerate or lengthen the hourly retries.
    refresh_stale_experiments((SNAPSHOT,))
    assert due_experiment_ids((SNAPSHOT,), now=started + timedelta(minutes=1)) == ()
    second_run = started + retry
    assert attempt_due((SNAPSHOT,), now=second_run).result is AttemptResult.JOB_RAN
    assert status_for(experiment_id, SNAPSHOT).next_run_at == second_run + retry
    assert status_for(experiment_id, SNAPSHOT).consecutive_failures == 2

    mocker.stop(analysis)
    third_run = second_run + retry
    assert attempt_due((SNAPSHOT,), now=third_run).result is AttemptResult.JOB_RAN
    status = status_for(experiment_id, SNAPSHOT)
    assert status.last_success_at == third_run
    assert status.next_run_at == third_run + SNAPSHOT.interval
    assert status.consecutive_failures == 0


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


def test_warehouse_pull_and_snapshot_commit_with_their_statuses(xngin_session, testing_datasource):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds, participants=("1", "99999999999"))
    refresh_stale_experiments((PULL, SNAPSHOT))
    assert ran_successfully(attempt_due((PULL, SNAPSHOT)), PULL)
    assert read_outcomes(xngin_session, experiment.id) == {"1": 0.0, "99999999999": None}
    assert (
        status_for(experiment.id, PULL).next_run_at == status_for(experiment.id, PULL).last_success_at + PULL.interval
    )
    assert ran_successfully(attempt_due((PULL, SNAPSHOT)), SNAPSHOT)
    assert (
        xngin_session.scalar(select(tables.Snapshot.id).where(tables.Snapshot.experiment_id == experiment.id))
        is not None
    )
    assert status_for(experiment.id, SNAPSHOT).last_success_at is not None


def test_warehouse_timeout_releases_locks_and_allows_retry_while_helper_is_stalled(
    xngin_session, testing_datasource, mocker
):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds, participants=("1", "2"))
    job = replace(PULL, dwh_timeout=0.25)
    refresh_stale_experiments((job,))
    entered, release, returned = Event(), Event(), Event()

    def stall(*_args, **_kwargs):
        entered.set()
        release.wait()
        returned.set()

    stalled = mocker.patch.object(DwhSession, "_inspect_table_blocking", stall)
    started = datetime.now(UTC)
    finished = started + timedelta(seconds=17)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            mocker.patch("xngin.apiserver.worker.attempts.database_now", side_effect=[started, finished])
            attempt = pool.submit(attempt_experiment, (job,), experiment.id)
            assert entered.wait(5)
            assert attempt.result(timeout=5).result is AttemptResult.JOB_RAN
            status = status_for(experiment.id, job)
            assert status.last_run_at == started
            assert status.last_success_at is None
            assert status.consecutive_failures == 1
            assert "DwhTimeoutError:" in status.last_error
            retry_at = finished + job.backoff.delay_after(1)
            assert status.next_run_at == status.retry_not_before == retry_at
            assert read_outcomes(xngin_session, experiment.id) == {"1": None, "2": None}

            # The helper still owns its warehouse call, but neither the maintenance claim nor
            # the application database's draw locks may survive the failed attempt's commit.
            with database.get_session() as other, other.begin():
                assert claim_experiment(other, experiment.id) is not None
                participant_ids = other.scalars(
                    select(tables.Draw.participant_id)
                    .where(tables.Draw.experiment_id == experiment.id)
                    .with_for_update(nowait=True)
                ).all()
                assert set(participant_ids) == {"1", "2"}

            refresh_stale_experiments((job,))
            assert due_experiment_ids((job,), now=retry_at - timedelta(microseconds=1)) == ()
            # Restore inspection for a fresh helper without releasing the abandoned one.
            mocker.stop(stalled)
            mocker.patch("xngin.apiserver.worker.attempts.database_now", return_value=retry_at)
            retry = pool.submit(attempt_experiment, (PULL,), experiment.id)
            assert ran_successfully(retry.result(timeout=5), PULL)
            assert read_outcomes(xngin_session, experiment.id) == {"1": 0.0, "2": 1.0}
            status = status_for(experiment.id, job)
            assert status.last_success_at == retry_at
            assert status.consecutive_failures == 0
            assert status.last_error is None
            assert status.retry_not_before is None
            assert status.next_run_at == retry_at + PULL.interval
            assert not release.is_set()
            assert not returned.is_set()
        finally:
            release.set()
            assert returned.wait(5)


def test_periodic_jobs_that_never_succeeded_follow_existing_snapshots_and_outcomes(xngin_session, testing_datasource):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    now = datetime.now(UTC)
    snapshot_at, observed_at = now - timedelta(hours=1), now - timedelta(hours=2)
    xngin_session.add_all([
        tables.Snapshot(experiment_id=experiment.id, status="success", updated_at=snapshot_at),
        # A later failure doesn't count as a successful snapshot.
        tables.Snapshot(experiment_id=experiment.id, status="failed", updated_at=now),
    ])
    xngin_session.execute(
        update(tables.Draw)
        .where(tables.Draw.experiment_id == experiment.id)
        .values(outcome=1, autofailed_outcome=False, observed_at=observed_at)
    )
    xngin_session.commit()
    refresh_stale_experiments((PULL, SNAPSHOT))
    assert status_for(experiment.id, SNAPSHOT).next_run_at == snapshot_at + SNAPSHOT.interval
    assert status_for(experiment.id, PULL).next_run_at == observed_at + PULL.interval


def test_api_outcome_and_metadata_update_commit_during_worker_read(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    pause = Pause()
    job = FakeJob(run=pause)
    refresh_stale_experiments((job,))
    with paused_attempt((job,), pause, experiment.id) as worker:
        xngin_session.execute(text("SET LOCAL lock_timeout = '300ms'"))
        update_bandit_arm_with_outcome_impl(xngin_session, experiment, "1", 1.0)
        experiment.end_date += timedelta(days=1)
        xngin_session.commit()
    assert ran_successfully(worker.result(), job)
    assert read_outcomes(xngin_session, experiment.id) == {"1": 1.0}


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


def test_snapshot_connection_loss_does_not_commit_failure_or_schedule(
    testing_datasource, aclient: AdminAPIClient, mocker
):
    experiment_id = make_bandit_online_experiment(aclient, testing_datasource.datasource_id).config.experiment_id

    def disconnect(session, *args):
        session.connection().invalidate()
        raise ConnectionError("Lost attempt transaction")

    mocker.patch("xngin.apiserver.worker.jobs.snapshot_analysis._compute_snapshot_data", side_effect=disconnect)
    refresh_stale_experiments((SNAPSHOT,))
    assert attempt_due((SNAPSHOT,)).result is AttemptResult.ABORTED
    with database.get_session() as session:
        assert session.scalar(select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id)) is None
    assert status_for(experiment_id, SNAPSHOT).last_run_at is None


@pytest.mark.parametrize(
    ("job", "make_experiment"),
    [(PULL, make_dwh_experiment), (AUTOFAIL, make_autofail_experiment)],
    ids=["dwh_pull", "autofail"],
)
def test_real_job_failure_rolls_back_backs_off_and_recovers(
    xngin_session, testing_datasource, monkeypatch, job, make_experiment
):
    participants = ("1", "2", "3")
    experiment = make_experiment(xngin_session, testing_datasource.ds, participants=participants)
    real_update = bandit_outcomes.update_bandit_arm_with_outcome_impl
    updated = []
    fail = True

    def update_then_fail(**kwargs):
        # Keep the real job, warehouse reads, and outcome writes. Inject a failure only after two
        # actual writes, so a missing rollback would leave observable partial results.
        result = real_update(**kwargs)
        updated.append(kwargs["participant_id"])
        if fail and len(updated) == 2:
            raise RuntimeError("Interrupted outcome processing")
        return result

    monkeypatch.setattr(bandit_outcomes, "update_bandit_arm_with_outcome_impl", update_then_fail)
    refresh_stale_experiments((job,))
    started = datetime.now(UTC)
    finished = started + timedelta(seconds=17)
    # Distinct start and finish times show that the retry delay is measured from when the run finished.
    with attempt_clock(side_effect=[started, finished]):
        assert attempt_experiment((job,), experiment.id).result is AttemptResult.JOB_RAN
    assert len(updated) == 2
    assert read_outcomes(xngin_session, experiment.id) == dict.fromkeys(participants)
    status = status_for(experiment.id, job)
    retry_at = finished + job.backoff.delay_after(1)
    assert status.last_run_at == started
    assert status.consecutive_failures == 1
    assert status.last_error == "RuntimeError: Interrupted outcome processing"
    assert status.next_run_at == status.retry_not_before == retry_at

    # A refresh cannot bring the retry forward.
    refresh_stale_experiments((job,))
    assert due_experiment_ids((job,), now=retry_at - timedelta(microseconds=1)) == ()

    fail = False
    assert ran_successfully(attempt_due((job,), now=retry_at), job)
    status = status_for(experiment.id, job)
    assert status.last_success_at == retry_at
    assert status.last_error is None
    assert status.retry_not_before is None
    assert any(value is not None for value in read_outcomes(xngin_session, experiment.id).values())


def test_refresh_revisits_only_new_and_edited_experiments(xngin_session, testing_datasource):
    current = make_push_experiment(xngin_session, testing_datasource.ds)
    ended = make_push_experiment(xngin_session, testing_datasource.ds, end_date=datetime.now(UTC) - timedelta(days=2))
    draft = make_push_experiment(xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED)
    assert stale_among((SNAPSHOT,), current, ended, draft) == {current.id, ended.id, draft.id}

    refresh_stale_experiments((SNAPSHOT,))

    assert stale_among((SNAPSHOT,), current, ended, draft) == set()
    assert status_for(current.id, SNAPSHOT).next_run_at is not None
    assert status_for(ended.id, SNAPSHOT).next_run_at is None
    assert status_for(draft.id, SNAPSHOT).next_run_at is None

    draft.state = ExperimentState.COMMITTED
    xngin_session.commit()
    assert stale_among((SNAPSHOT,), current, ended, draft) == {draft.id}
    refresh_stale_experiments((SNAPSHOT,))
    assert status_for(draft.id, SNAPSHOT).next_run_at is not None


def test_committing_a_draft_through_the_api_schedules_its_jobs(testing_datasource, aclient: AdminAPIClient):
    jobs = (SNAPSHOT, PULL)
    datasource_id = testing_datasource.datasource_id
    request = make_create_online_bandit_experiment_request(
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        reward_type=LikelihoodTypes.BERNOULLI,
        prior_type=PriorTypes.BETA,
    )
    experiment_id = aclient.create_experiment(
        datasource_id=datasource_id, body=request, random_state=42
    ).data.experiment_id

    # The draft gets a status row for every job, none of them scheduled, and later refreshes leave it alone.
    refresh_stale_experiments(jobs)
    assert all(status_for(experiment_id, job).next_run_at is None for job in jobs)
    with database.get_session() as session:
        assert experiment_id not in _stale_experiment_ids(session, jobs)

    # Committing changes updated_at, which the refresh notices.
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
    with database.get_session() as session:
        assert experiment_id in _stale_experiment_ids(session, jobs)
    refresh_stale_experiments(jobs)
    assert all(status_for(experiment_id, job).next_run_at is not None for job in jobs)


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
