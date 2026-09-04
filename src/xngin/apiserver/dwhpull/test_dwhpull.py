import asyncio
import contextlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwhpull import cli
from xngin.apiserver.dwhpull import dwhpull as dwhpull_mod
from xngin.apiserver.dwhpull.dwhpull import (
    PULL_TIMEOUT_SECS,
    pull_all_experiments,
    pull_one_experiment,
    select_experiments_to_pull,
)
from xngin.apiserver.routers.common_api_types import ExperimentsType, LikelihoodTypes, PriorTypes
from xngin.apiserver.routers.common_enums import ExperimentState
from xngin.apiserver.routers.experiments.experiments_common import (
    MismatchedExperimentTypeError,
    create_assignment_for_participant,
)
from xngin.apiserver.routers.experiments.test_experiments_common import insert_experiment_and_arms
from xngin.apiserver.sqla import tables


async def make_mab_dwh_experiment(xngin_session, datasource, **kwargs) -> tables.Experiment:
    return await insert_experiment_and_arms(
        xngin_session,
        datasource,
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
        **kwargs,
    )


async def read_outcomes(xngin_session, experiment_id: str) -> dict[str, float | None]:
    rows = await xngin_session.execute(
        select(tables.Draw.participant_id, tables.Draw.outcome).where(tables.Draw.experiment_id == experiment_id)
    )
    return dict(rows.all())


async def test_pull_one_experiment_boolean_target(xngin_session, testing_datasource):
    """A pull reads target values for outcome-less draws, applies them, and reports counts."""
    experiment = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    # Testing DWH: id=1 has is_onboarded=false, id=2 has is_onboarded=true.
    await create_assignment_for_participant(xngin_session, experiment, "1", None, random_state=66)
    await create_assignment_for_participant(xngin_session, experiment, "2", None, random_state=67)
    # Not present in the testing DWH.
    await create_assignment_for_participant(xngin_session, experiment, "99999999999", None, random_state=68)

    report = await pull_one_experiment(experiment.id)
    assert (report.ingested, report.invalid, report.pending) == (2, 0, 1)
    assert await read_outcomes(xngin_session, experiment.id) == {"1": 0.0, "2": 1.0, "99999999999": None}

    # The posteriors moved: one success adds 1 to an arm's alpha, one failure adds 1 to a beta.
    alpha_gain = 0.0
    beta_gain = 0.0
    for arm in experiment.arms:
        await xngin_session.refresh(arm)
        assert arm.alpha is not None and arm.alpha_init is not None
        assert arm.beta is not None and arm.beta_init is not None
        alpha_gain += arm.alpha - arm.alpha_init
        beta_gain += arm.beta - arm.beta_init
    assert (alpha_gain, beta_gain) == (1.0, 1.0)

    # Re-running ingests nothing new: filled draws no longer match the outcome-less query.
    report = await pull_one_experiment(experiment.id)
    assert (report.ingested, report.invalid, report.pending) == (0, 0, 1)


async def test_pull_one_experiment_with_no_outcomeless_draws_skips_the_warehouse(xngin_session, testing_datasource):
    """The common case: nothing to read, so the run does not touch the data warehouse at all."""
    experiment = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds)

    report = await pull_one_experiment(experiment.id)

    assert (report.ingested, report.invalid, report.pending) == (0, 0, 0)


async def test_pull_one_experiment_numeric_target_accepts_any_float(xngin_session, testing_datasource):
    """A numeric target under a Normal reward takes arbitrary float values, not just 0/1.

    Moved here from the outcome API, which no longer accepts pushes for this type.
    """
    experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.MAB_ONLINE_DWH,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
        target_field_name="current_income",
    )
    await create_assignment_for_participant(xngin_session, experiment, "1", None, random_state=66)

    report = await pull_one_experiment(experiment.id)

    assert (report.ingested, report.invalid, report.pending) == (1, 0, 0)
    outcomes = await read_outcomes(xngin_session, experiment.id)
    assert outcomes["1"] == 29912.0


async def test_pull_one_experiment_null_target_value_stays_pending(xngin_session, testing_datasource):
    """A row whose target column is NULL is left pending, to be re-read on a later run."""
    experiment = await make_mab_dwh_experiment(
        xngin_session,
        testing_datasource.ds,
        # NULL for low ids in the testing DWH.
        target_field_name="is_onboarded_onetime",
    )
    await create_assignment_for_participant(xngin_session, experiment, "1", None, random_state=66)

    report = await pull_one_experiment(experiment.id)
    assert (report.ingested, report.invalid, report.pending) == (0, 0, 1)


async def test_pull_one_experiment_invalid_value_skipped_and_left_null(xngin_session, testing_datasource):
    """A pulled value that fails validation is skipped and the draw stays NULL, so a corrected
    warehouse value is picked up on a later run."""
    experiment = await make_mab_dwh_experiment(
        xngin_session,
        testing_datasource.ds,
        # id=1's current_income is 29912.0, which fails the Bernoulli 0/1 guard.
        target_field_name="current_income",
    )
    await create_assignment_for_participant(xngin_session, experiment, "1", None, random_state=66)

    report = await pull_one_experiment(experiment.id)
    assert (report.ingested, report.invalid, report.pending) == (0, 1, 0)
    assert await read_outcomes(xngin_session, experiment.id) == {"1": None}
    for arm in experiment.arms:
        await xngin_session.refresh(arm)
        assert arm.alpha == arm.alpha_init
        assert arm.beta == arm.beta_init


async def test_pull_one_experiment_continues_after_a_rejected_value(xngin_session, testing_datasource):
    """A rejected value must not stop the participants after it in the same run."""
    experiment = await make_mab_dwh_experiment(
        xngin_session,
        testing_datasource.ds,
        # No current_income in the testing DWH is 0 or 1, so every value fails the Bernoulli guard.
        target_field_name="current_income",
    )
    await create_assignment_for_participant(xngin_session, experiment, "1", None, random_state=66)
    await create_assignment_for_participant(xngin_session, experiment, "3", None, random_state=67)

    report = await pull_one_experiment(experiment.id)

    # invalid=2 means the loop reached the second participant after rolling back the first.
    assert (report.ingested, report.invalid, report.pending) == (0, 2, 0)


async def test_pull_one_experiment_leaves_resolved_draws_alone(xngin_session, testing_datasource):
    """A draw another writer already resolved, autofail included, is not overwritten by a pull.

    The pull holds FOR NO KEY UPDATE on its draws for the whole run precisely so autofail cannot
    resolve one mid-pull. This covers the other side of that: whatever is already resolved stays.
    """
    experiment = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    await create_assignment_for_participant(xngin_session, experiment, "1", None, random_state=66)
    await create_assignment_for_participant(xngin_session, experiment, "2", None, random_state=67)

    # Stand in for autofail having already closed out participant 1 with its 0.0.
    await xngin_session.execute(
        update(tables.Draw)
        .where(tables.Draw.experiment_id == experiment.id, tables.Draw.participant_id == "1")
        .values(outcome=0.0, observed_at=datetime.now(UTC), autofailed_outcome=True)
    )
    await xngin_session.commit()

    report = await pull_one_experiment(experiment.id)

    # Only participant 2 was outstanding. Participant 1 keeps the autofailed 0.0, even though the
    # warehouse says is_onboarded=false for them anyway.
    assert (report.ingested, report.invalid, report.pending) == (1, 0, 0)
    assert await read_outcomes(xngin_session, experiment.id) == {"1": 0.0, "2": 1.0}


async def test_pull_all_experiments_abandons_an_experiment_that_exceeds_its_timeout(
    xngin_session, testing_datasource, mocker
):
    """A warehouse that hangs must not hold the transaction open indefinitely.

    Holding an application-database transaction across the warehouse read is only safe because
    pull_all_experiments bounds each experiment. Uses a tiny budget against a stalled read, so the
    test cancels immediately rather than waiting.
    """
    experiment = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    await create_assignment_for_participant(xngin_session, experiment, "2", None, random_state=67)

    async def never_returns(self, table_name):
        await asyncio.sleep(30)

    mocker.patch.object(DwhSession, "inspect_table", never_returns)

    # Reported as a failure. One warehouse does not stop the other experiments.
    await pull_all_experiments(pull_timeout=0.05)

    # The transaction rolled back, so the draw is untouched and the next run retries it.
    assert await read_outcomes(xngin_session, experiment.id) == {"2": None}


async def test_pull_one_experiment_applies_all_outcomes_or_none(xngin_session, testing_datasource, mocker):
    """A failure partway through leaves the experiment untouched, so a rerun repeats it cleanly."""
    experiment = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    await create_assignment_for_participant(xngin_session, experiment, "1", None, random_state=66)
    await create_assignment_for_participant(xngin_session, experiment, "2", None, random_state=67)

    real_update = dwhpull_mod.update_bandit_arm_with_outcome_impl
    seen = []

    async def fail_on_the_second(**kwargs):
        seen.append(kwargs["participant_id"])
        if len(seen) == 2:
            raise RuntimeError("warehouse pull interrupted")
        return await real_update(**kwargs)

    mocker.patch.object(dwhpull_mod, "update_bandit_arm_with_outcome_impl", side_effect=fail_on_the_second)

    with pytest.raises(RuntimeError):
        await pull_one_experiment(experiment.id)

    # The first outcome applied in memory but never committed, so neither draw is resolved.
    assert len(seen) == 2
    assert await read_outcomes(xngin_session, experiment.id) == {"1": None, "2": None}


async def test_pull_one_experiment_rejects_non_dwh_experiment(xngin_session, testing_datasource):
    experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
    )
    with pytest.raises(MismatchedExperimentTypeError):
        await pull_one_experiment(experiment.id)


async def test_select_experiments_to_pull_skips_other_types_and_states(xngin_session, testing_datasource):
    due = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds)
    await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED)
    await make_mab_dwh_experiment(
        xngin_session,
        testing_datasource.ds,
        end_date=datetime.now(UTC) - timedelta(days=30),
    )
    await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.MAB_ONLINE,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
    )

    assert await select_experiments_to_pull() == [due.id]


async def test_pull_all_experiments_isolates_a_failing_experiment(xngin_session, testing_datasource):
    """One experiment that cannot be read must not stop the others in the same run."""
    broken = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    broken.datasource_table = "no_such_table"
    healthy = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    await xngin_session.commit()
    await create_assignment_for_participant(xngin_session, broken, "1", None, random_state=66)
    await create_assignment_for_participant(xngin_session, healthy, "2", None, random_state=67)

    await pull_all_experiments(PULL_TIMEOUT_SECS)

    assert await read_outcomes(xngin_session, broken.id) == {"1": None}
    assert await read_outcomes(xngin_session, healthy.id) == {"2": 1.0}


async def test_pull_all_experiments_covers_every_due_experiment(xngin_session, testing_datasource):
    first = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    second = await make_mab_dwh_experiment(xngin_session, testing_datasource.ds, target_field_name="is_onboarded")
    await create_assignment_for_participant(xngin_session, first, "1", None, random_state=66)
    await create_assignment_for_participant(xngin_session, second, "2", None, random_state=67)

    await pull_all_experiments(PULL_TIMEOUT_SECS)

    assert await read_outcomes(xngin_session, first.id) == {"1": 0.0}
    assert await read_outcomes(xngin_session, second.id) == {"2": 1.0}


async def test_apull_runs_within_a_database_session(mocker):
    @contextlib.asynccontextmanager
    async def noop_setup():
        yield

    mocker.patch("xngin.apiserver.dwhpull.cli.database.setup", noop_setup)
    pull_mock = mocker.patch("xngin.apiserver.dwhpull.cli.dwhpull.pull_all_experiments")

    await cli.apull(pull_timeout=42)

    pull_mock.assert_awaited_once_with(42)
