from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest
from sqlalchemy import case, select, text, update

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
    due_experiment_ids,
    make_autofail_experiment,
    make_dwh_experiment,
    make_push_experiment,
    paused_attempt,
    ran_successfully,
    read_outcomes,
    run_nothing,
    schedule_now,
    scheduled_at,
    status_for,
    wait_until_backend_exits,
)
from xngin.apiserver.worker.attempts import AttemptResult, FinishedAttempt, attempt_experiment, claim_experiment
from xngin.apiserver.worker.jobs import bandit_outcomes
from xngin.apiserver.worker.jobs.base import JobFailed, JobSucceeded
from xngin.apiserver.worker.jobs.dwh_pull import LATE_OUTCOME_GRACE
from xngin.apiserver.worker.scheduler import read_schedule
from xngin.stats.stats_errors import StatsNoParticipantsError


def test_cross_job_exclusion_and_parallel_experiments(xngin_session, testing_datasource):
    for _ in range(2):
        make_push_experiment(xngin_session, testing_datasource.ds)
    pause = Pause()
    jobs = (FakeJob(run=pause), OtherFakeJob())
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


def test_long_job_errors_are_truncated_to_fit_the_status_row(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    job = FakeJob(run=lambda session, experiment: JobFailed(error="x" * 3000))
    assert attempt_due((job,)).result is AttemptResult.JOB_RAN
    assert status_for(experiment.id, job).last_error == "x" * 2048


def test_connection_loss_rolls_back_claim_status_and_effects(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)

    def disconnect(session, experiment):
        experiment.name = "lost"
        session.flush()
        session.connection().invalidate()
        raise ConnectionError("lost connection")

    job = FakeJob(run=disconnect)
    assert attempt_due((job,)).result is AttemptResult.ABORTED
    assert status_for(experiment.id, job) is None
    xngin_session.refresh(experiment)
    assert experiment.name != "lost"
    job.run = run_nothing
    assert ran_successfully(attempt_due((job,)), job)


def test_dwh_pull_waits_for_the_start_date(xngin_session, testing_datasource):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    experiment.start_date = NOW + timedelta(days=3)
    experiment.end_date = NOW + timedelta(days=10)
    xngin_session.commit()
    assert scheduled_at(experiment.id, PULL, now=NOW) == NOW + timedelta(days=3)
    assert scheduled_at(experiment.id, SNAPSHOT, now=NOW) == NOW + timedelta(days=2)


@pytest.mark.parametrize(
    ("days_since_end", "resolved", "scheduled"),
    [
        pytest.param(-3, False, True, id="running-pending"),
        pytest.param(-3, True, False, id="running-resolved"),
        pytest.param(3, False, True, id="grace-pending"),
        pytest.param(3, True, False, id="grace-resolved"),
        pytest.param(9, False, False, id="after-grace-pending"),
    ],
)
def test_dwh_pull_runs_only_while_draws_are_pending_until_the_grace_period_ends(
    xngin_session, testing_datasource, days_since_end, resolved, scheduled
):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    experiment.end_date = datetime.now(UTC) - timedelta(days=days_since_end)
    if resolved:
        update_bandit_arm_with_outcome_impl(xngin_session, experiment, "1", 1.0)
    xngin_session.commit()
    assert (scheduled_at(experiment.id, PULL) is not None) == scheduled


def test_autofail_catches_up_after_an_outage(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1", "2", "3", "4", "5"))

    # The worker is down for a month, long past the end of the experiment.
    month_ago = datetime.now(UTC) - timedelta(days=30)
    xngin_session.execute(
        update(tables.Experiment)
        .where(tables.Experiment.id == experiment.id)
        .values(start_date=month_ago - timedelta(days=7), end_date=month_ago)
    )
    xngin_session.execute(
        update(tables.Draw)
        .where(tables.Draw.experiment_id == experiment.id)
        .values(created_at=month_ago - timedelta(days=1))
    )
    xngin_session.commit()

    # Five overdue draws take three batches of two.
    for remaining in (3, 1, 0):
        assert ran_successfully(attempt_due((AUTOFAIL,)), AUTOFAIL)
        assert list(read_outcomes(xngin_session, experiment.id).values()).count(None) == remaining
    assert scheduled_at(experiment.id, AUTOFAIL) is None


def test_autofail_schedule_follows_pending_draws(xngin_session, testing_datasource):
    resolved = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    pending = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    update_bandit_arm_with_outcome_impl(xngin_session, resolved, "1", 1.0)
    pending.end_date = datetime.now(UTC) - timedelta(days=5)
    xngin_session.commit()

    # Ended experiments stay scheduled while they have unresolved draws.
    assert scheduled_at(pending.id, AUTOFAIL) is not None
    assert scheduled_at(resolved.id, AUTOFAIL) is None

    # Inserting a draw does not update the experiment, but schedules it all the same.
    create_assignment_for_participant(xngin_session, resolved, "2", None, random_state=66)
    xngin_session.commit()
    assert scheduled_at(resolved.id, AUTOFAIL) is not None

    # Disabling autofail unschedules the experiment, though its draws are still pending.
    pending.enable_autofail = False
    xngin_session.commit()
    assert scheduled_at(pending.id, AUTOFAIL) is None


def test_autofail_schedules_exact_expiry_without_early_execution(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    created_at = datetime.now(UTC)
    xngin_session.execute(
        update(tables.Draw).where(tables.Draw.experiment_id == experiment.id).values(created_at=created_at)
    )
    xngin_session.commit()
    assert due_experiment_ids((AUTOFAIL,)) == ()
    assert scheduled_at(experiment.id, AUTOFAIL) == created_at + timedelta(hours=1)
    assert read_outcomes(xngin_session, experiment.id) == {"1": None}


def test_locked_autofail_draws_defer_the_job(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    with database.get_session() as api, api.begin():
        api.execute(
            select(tables.Draw).where(tables.Draw.experiment_id == experiment.id).with_for_update(of=tables.Draw)
        )
        # The batch was deferred: no success, no failure, and the attempt counts as having run no job.
        assert attempt_due((AUTOFAIL,)).result is AttemptResult.NOTHING_READY
        status = status_for(experiment.id, AUTOFAIL)
        assert scheduled_at(experiment.id, AUTOFAIL) == status.retry_not_before
        assert status.last_success_at is None
        assert status.consecutive_failures == 0
        assert due_experiment_ids((AUTOFAIL,)) == ()
    assert ran_successfully(attempt_due((AUTOFAIL,), now=status.retry_not_before), AUTOFAIL)


def test_attempt_rechecks_run_times_under_its_claim(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    assert due_experiment_ids((AUTOFAIL,)) == (experiment.id,)
    # An API outcome resolves the only pending draw after the look found the experiment due.
    update_bandit_arm_with_outcome_impl(xngin_session, experiment, "1", 1.0)
    xngin_session.commit()

    assert attempt_experiment((AUTOFAIL,), experiment.id) == FinishedAttempt(experiment.id, AttemptResult.NOTHING_READY)
    assert status_for(experiment.id, AUTOFAIL) is None
    assert due_experiment_ids((AUTOFAIL,)) == ()


def test_metrics_failure_after_commit_does_not_abort_the_attempt(xngin_session, testing_datasource, mocker):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    job = FakeJob()
    # Only job metrics reported after the commit use distribution().
    mocker.patch("xngin.apiserver.worker.attempts.sentry_sdk.metrics.distribution", side_effect=RuntimeError("sentry"))
    assert ran_successfully(attempt_experiment((job,), experiment.id), job)


def test_a_bug_in_the_warehouse_read_alerts(xngin_session, testing_datasource, mocker):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    mocker.patch.object(DwhSession, "get_participant_metrics", side_effect=TypeError("bug"))
    capture_exception = mocker.patch("xngin.apiserver.worker.attempts.sentry_sdk.capture_exception")
    assert attempt_due((PULL,)).result is AttemptResult.JOB_RAN
    status = status_for(experiment.id, PULL)
    assert status.consecutive_failures == 1
    assert status.last_error == "TypeError: bug"
    capture_exception.assert_called_once()


def test_api_writes_commit_while_a_worker_holds_the_experiment(xngin_session, testing_datasource):
    experiment = make_autofail_experiment(xngin_session, testing_datasource.ds, participants=("1",))
    pause = Pause()
    job = FakeJob(run=pause)
    with paused_attempt((job,), pause, experiment.id) as running:
        # Assignments, outcomes, and experiment edits must commit while maintenance still holds both locks.
        xngin_session.execute(text("SET LOCAL lock_timeout = '300ms'"))
        create_assignment_for_participant(xngin_session, experiment, "new", None, random_state=66)
        update_bandit_arm_with_outcome_impl(xngin_session, experiment, "1", 1.0)
        experiment.end_date += timedelta(days=1)
        xngin_session.commit()
    assert ran_successfully(running.result(), job)
    assert read_outcomes(xngin_session, experiment.id) == {"1": 1.0, "new": None}


def test_snapshot_without_participants_succeeds_without_backoff(testing_datasource, aclient: AdminAPIClient, mocker):
    experiment_id = make_bandit_online_experiment(aclient, testing_datasource.datasource_id).config.experiment_id
    mocker.patch(
        "xngin.apiserver.worker.jobs.snapshot_analysis._compute_snapshot_data",
        side_effect=StatsNoParticipantsError(),
    )
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
    assert scheduled_at(experiment_id, SNAPSHOT, now=started) == started + SNAPSHOT.interval


def test_failed_snapshots_retry_hourly_until_one_succeeds(testing_datasource, aclient: AdminAPIClient, mocker):
    experiment_id = make_bandit_online_experiment(aclient, testing_datasource.datasource_id).config.experiment_id

    def fail(session, *args):
        session.execute(update(tables.Experiment).where(tables.Experiment.id == experiment_id).values(name="partial"))
        session.execute(text("SELECT 1 / 0"))

    analysis = mocker.patch("xngin.apiserver.worker.jobs.snapshot_analysis._compute_snapshot_data", side_effect=fail)
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
    assert scheduled_at(experiment_id, SNAPSHOT, now=started) == started + retry
    # Repeated failures cannot accelerate or lengthen the hourly retries.
    assert due_experiment_ids((SNAPSHOT,), now=started + timedelta(minutes=1)) == ()
    second_run = started + retry
    assert attempt_due((SNAPSHOT,), now=second_run).result is AttemptResult.JOB_RAN
    assert scheduled_at(experiment_id, SNAPSHOT, now=second_run) == second_run + retry
    assert status_for(experiment_id, SNAPSHOT).consecutive_failures == 2

    mocker.stop(analysis)
    third_run = second_run + retry
    assert attempt_due((SNAPSHOT,), now=third_run).result is AttemptResult.JOB_RAN
    status = status_for(experiment_id, SNAPSHOT)
    assert status.last_success_at == third_run
    assert scheduled_at(experiment_id, SNAPSHOT, now=third_run) == third_run + SNAPSHOT.interval
    assert status.consecutive_failures == 0


def test_parent_lock_delays_maintenance_without_claiming_work(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    job = FakeJob()
    with database.get_session() as deleter, deleter.begin():
        deleter.execute(select(tables.Experiment.id).where(tables.Experiment.id == experiment.id).with_for_update())
        assert attempt_due((job,)) == FinishedAttempt(experiment.id, AttemptResult.NOT_CLAIMED)
        assert status_for(experiment.id, job) is None
        # A busy parent must leave the claim available.
        assert claim_experiment(deleter, experiment.id) is not None
    assert ran_successfully(attempt_due((job,)), job)


def test_due_jobs_and_experiments_get_fair_turns(xngin_session, testing_datasource):
    first = make_push_experiment(xngin_session, testing_datasource.ds)
    second = make_push_experiment(xngin_session, testing_datasource.ds)
    calls = []

    def record(session, experiment):
        calls.append(experiment.id)
        return JobSucceeded()

    jobs = (FakeJob(schedule=schedule_now, run=record), OtherFakeJob())
    results = [attempt_due(jobs) for _ in range(4)]
    assert all(r.result is AttemptResult.JOB_RAN for r in results)
    assert calls.count(first.id) == calls.count(second.id) == 1
    assert all(status_for(e.id, jobs[1]).last_success_at for e in (first, second))


def test_backend_termination_allows_takeover_but_old_attempt_cannot_write(xngin_session, testing_datasource):
    experiment = make_push_experiment(xngin_session, testing_datasource.ds)
    pause = Pause(after=lambda session, experiment: session.add(tables.Snapshot(experiment_id=experiment.id, data={})))
    job = FakeJob(run=pause)
    with paused_attempt((job,), pause, experiment.id) as old:
        with database.get_engine().begin() as connection:
            assert connection.scalar(text("SELECT pg_terminate_backend(:pid)"), {"pid": pause.pid})
        wait_until_backend_exits(pause.pid)
        assert ran_successfully(attempt_due((FakeJob(),)), FakeJob())
    assert old.result().result is AttemptResult.ABORTED
    with database.get_session() as session:
        assert list(session.scalars(select(tables.Snapshot))) == []
    assert status_for(experiment.id, job).last_success_at is not None


def test_warehouse_timeout_releases_locks_and_allows_retry_while_helper_is_stalled(
    xngin_session, testing_datasource, mocker
):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds, participants=("1", "2"))
    job = replace(PULL, dwh_timeout=0.25)
    entered, release, returned = Event(), Event(), Event()

    def stall(*_args, **_kwargs):
        entered.set()
        release.wait()
        returned.set()

    stalled = mocker.patch.object(DwhSession, "_inspect_table_blocking", stall)
    capture_exception = mocker.patch("xngin.apiserver.worker.attempts.sentry_sdk.capture_exception")
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
            # A customer's warehouse failing is not our bug, so it does not alert.
            capture_exception.assert_not_called()
            retry_at = finished + job.backoff.delay_after(1)
            assert scheduled_at(experiment.id, job, now=finished) == status.retry_not_before == retry_at
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
            # Every draw has its outcome, so there is nothing left to pull.
            assert scheduled_at(experiment.id, PULL, now=retry_at) is None
            assert not release.is_set()
            assert not returned.is_set()
        finally:
            release.set()
            assert returned.wait(5)


def test_final_run_due_at_the_close_still_runs_after_it(xngin_session, testing_datasource):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    closes = datetime.now(UTC) - timedelta(minutes=1)
    experiment.end_date = closes - LATE_OUTCOME_GRACE
    # The last pull was an hour before the close, so the next interval overshoots it and the final pull is due at it.
    last_pull = closes - timedelta(hours=1)
    xngin_session.add(
        tables.JobStatus(
            experiment_id=experiment.id, job_name=PULL.name, last_success_at=last_pull, last_run_at=last_pull
        )
    )
    xngin_session.commit()
    assert scheduled_at(experiment.id, PULL) == closes
    # The attempt rechecks the run time after the close, and still runs the pull.
    assert ran_successfully(attempt_due((PULL,)), PULL)
    assert read_outcomes(xngin_session, experiment.id) == {"1": 0.0}
    assert scheduled_at(experiment.id, PULL) is None


def test_misconfigured_dwh_pull_fails_without_alerting(xngin_session, testing_datasource, mocker):
    experiment = make_dwh_experiment(xngin_session, testing_datasource.ds)
    experiment.datasource_table = None
    xngin_session.commit()
    capture_exception = mocker.patch("xngin.apiserver.worker.attempts.sentry_sdk.capture_exception")
    assert attempt_due((PULL,)).result is AttemptResult.JOB_RAN
    status = status_for(experiment.id, PULL)
    assert status.consecutive_failures == 1
    assert status.last_error.startswith("LateValidationError: MAB-DWH experiment is missing")
    capture_exception.assert_not_called()


def test_snapshot_connection_loss_does_not_commit_failure_or_schedule(
    testing_datasource, aclient: AdminAPIClient, mocker
):
    experiment_id = make_bandit_online_experiment(aclient, testing_datasource.datasource_id).config.experiment_id

    def disconnect(session, *args):
        session.connection().invalidate()
        raise ConnectionError("Lost attempt transaction")

    mocker.patch("xngin.apiserver.worker.jobs.snapshot_analysis._compute_snapshot_data", side_effect=disconnect)
    assert attempt_due((SNAPSHOT,)).result is AttemptResult.ABORTED
    with database.get_session() as session:
        assert session.scalar(select(tables.Snapshot).where(tables.Snapshot.experiment_id == experiment_id)) is None
    assert status_for(experiment_id, SNAPSHOT) is None


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
    assert scheduled_at(experiment.id, job, now=finished) == status.retry_not_before == retry_at
    assert due_experiment_ids((job,), now=retry_at - timedelta(microseconds=1)) == ()

    fail = False
    assert ran_successfully(attempt_due((job,), now=retry_at), job)
    status = status_for(experiment.id, job)
    assert status.last_success_at == retry_at
    assert status.last_error is None
    assert status.retry_not_before is None
    assert any(value is not None for value in read_outcomes(xngin_session, experiment.id).values())


def test_snapshots_schedule_committed_experiments_within_their_window(xngin_session, testing_datasource):
    current = make_push_experiment(xngin_session, testing_datasource.ds)
    ended = make_push_experiment(xngin_session, testing_datasource.ds, end_date=datetime.now(UTC) - timedelta(days=2))
    draft = make_push_experiment(xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED)
    assert scheduled_at(current.id, SNAPSHOT) is not None
    assert scheduled_at(ended.id, SNAPSHOT) is None
    assert scheduled_at(draft.id, SNAPSHOT) is None


def test_committing_a_draft_through_the_api_schedules_its_jobs(testing_datasource, aclient: AdminAPIClient):
    datasource_id = testing_datasource.datasource_id
    request = make_create_online_bandit_experiment_request(
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        reward_type=LikelihoodTypes.BERNOULLI,
        prior_type=PriorTypes.BETA,
    )
    experiment_id = aclient.create_experiment(
        datasource_id=datasource_id, body=request, random_state=42
    ).data.experiment_id

    assert scheduled_at(experiment_id, SNAPSHOT) is None
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
    assert scheduled_at(experiment_id, SNAPSHOT) is not None
    # Pulls wait for a draw to pull.
    assert scheduled_at(experiment_id, PULL) is None


def test_schedule_reports_the_next_future_run_and_ignores_unscheduled_jobs(xngin_session, testing_datasource):
    soon, due, unscheduled = (make_push_experiment(xngin_session, testing_datasource.ds) for _ in range(3))

    def schedule(now):
        experiment_id = tables.Experiment.id
        return case(
            (experiment_id == soon.id, now + timedelta(hours=1)),
            (experiment_id == due.id, now - timedelta(hours=1)),
            else_=None,
        )

    job = FakeJob(schedule=schedule)
    assert scheduled_at(unscheduled.id, job) is None
    looked = read_schedule((job,))
    assert looked.due == (due.id,)
    assert looked.next_run_in is not None
    assert 3500 < looked.next_run_in <= 3600
    assert read_schedule((OtherFakeJob(),)).next_run_in is None
