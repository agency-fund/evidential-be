"""Benchmarks (vibed) for the Experiment.draws access paths.

Run explicitly with:
    task test -- -m benchmark -s src/xngin/apiserver/snapshots/test_draws_perf.py

Per-scenario setup is hybrid: a fixed N_WARMUP of participants is assigned
and their outcomes recorded via the real assignment + update-outcome APIs
(to drive the bandit posterior updates, giving the arms realistic
alpha/beta/mu/covariance state). The remaining draws are bulk-seeded via
`INSERT ... SELECT generate_series` so setup stays tractable at 250k rows.
"""

import random
import statistics
import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    CMABContextInputRequest,
    CMABExperimentSpec,
    Context,
    ContextInput,
    ContextType,
    CreateExperimentRequest,
    ExperimentsType,
    LikelihoodTypes,
    MABExperimentSpec,
    PriorTypes,
    UpdateBanditArmOutcomeRequest,
)
from xngin.apiserver.snapshots.snapshotter import make_first_snapshot
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.experiments_api_client import ExperimentsAPIClient

pytestmark = pytest.mark.benchmark

DRAW_COUNTS = [10_000, 250_000]
EXPERIMENT_TYPES = [ExperimentsType.MAB_ONLINE, ExperimentsType.CMAB_ONLINE]
N_WARMUP = 200


def _iterations(n_draws: int) -> int:
    return 5 if n_draws <= 10_000 else 3


async def _create_bandit_experiment(
    aclient: AdminAPIClient, testing_datasource, experiment_type: ExperimentsType
) -> str:
    match experiment_type:
        case ExperimentsType.MAB_ONLINE:
            design_spec: MABExperimentSpec | CMABExperimentSpec = MABExperimentSpec(
                experiment_type=experiment_type,
                experiment_name="perf mab",
                description="perf mab",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    ArmBandit(arm_name="control", arm_description="", alpha_init=1, beta_init=1),
                    ArmBandit(arm_name="treatment", arm_description="", alpha_init=1, beta_init=1),
                ],
                prior_type=PriorTypes.BETA,
                reward_type=LikelihoodTypes.BERNOULLI,
            )
        case ExperimentsType.CMAB_ONLINE:
            design_spec = CMABExperimentSpec(
                experiment_type=experiment_type,
                experiment_name="perf cmab",
                description="perf cmab",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime.now(UTC) + timedelta(days=1),
                arms=[
                    ArmBandit(arm_name="control", arm_description="", mu_init=0, sigma_init=1),
                    ArmBandit(arm_name="treatment", arm_description="", mu_init=0, sigma_init=1),
                ],
                contexts=[
                    Context(context_name=f"c{i}", context_description="", value_type=ContextType.REAL_VALUED)
                    for i in range(3)
                ],
                prior_type=PriorTypes.NORMAL,
                reward_type=LikelihoodTypes.BERNOULLI,
            )
        case _:
            raise ValueError(f"Unsupported: {experiment_type}")
    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.ds.id,
        body=CreateExperimentRequest(design_spec=design_spec),
        desired_n=2,
        random_state=42,
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.ds.id, experiment_id=experiment_id)
    return experiment_id


async def _warmup_via_api(
    eclient: ExperimentsAPIClient,
    testing_datasource,
    experiment_id: str,
    *,
    is_cmab: bool,
    n_warmup: int,
    context_inputs: list[ContextInput] | None,
) -> None:
    """Assign and update outcomes for n_warmup participants through the real API.

    This drives the bandit posterior update path so the arm parameters
    (alpha/beta/mu/covariance) end up in a realistic, non-default state. Per-call
    cost grows with draws already in the arm, so keep n_warmup modest.
    """
    rng = random.Random(42)
    for i in range(n_warmup):
        participant_id = f"api_{i}"
        if is_cmab:
            assert context_inputs is not None
            eclient.get_assignment_cmab(
                api_key=testing_datasource.key,
                body=CMABContextInputRequest(context_inputs=context_inputs),
                experiment_id=experiment_id,
                participant_id=participant_id,
            )
        else:
            eclient.get_assignment(
                api_key=testing_datasource.key,
                experiment_id=experiment_id,
                participant_id=participant_id,
            )
        eclient.update_bandit_arm_with_participant_outcome(
            api_key=testing_datasource.key,
            body=UpdateBanditArmOutcomeRequest(outcome=float(rng.randint(0, 1))),
            experiment_id=experiment_id,
            participant_id=participant_id,
        )


async def _bulk_insert_draws(xngin_session, experiment_id: str, arm_ids: list[str], n_draws: int, *, is_cmab: bool):
    """Server-side bulk insert of n_draws rows via INSERT ... SELECT generate_series.

    Outcome alternates null/non-null so the stddev path filters roughly half the rows.
    For CMAB, context_vals is a 3-float array; for MAB it is NULL. Participant ids use
    a `p` prefix, disjoint from the `api_` prefix used by _warmup_via_api.
    """
    context_expr = (
        "ARRAY[MOD(gs, 5)::float8, MOD(gs + 1, 7)::float8, MOD(gs + 2, 3)::float8]" if is_cmab else "NULL::float8[]"
    )
    arm0 = arm_ids[0]
    arm1 = arm_ids[1] if len(arm_ids) > 1 else arm0
    sql = text(f"""
        INSERT INTO draws (experiment_id, participant_id, participant_type, arm_id, outcome, context_vals)
        SELECT
            :eid,
            'p' || gs::text,
            '',
            CASE WHEN MOD(gs, 2) = 0 THEN :arm0 ELSE :arm1 END,
            CASE WHEN MOD(gs, 2) = 0 THEN MOD(gs, 3)::float8 ELSE NULL END,
            {context_expr}
        FROM generate_series(1, :n) AS gs
    """)
    await xngin_session.execute(sql, {"eid": experiment_id, "arm0": arm0, "arm1": arm1, "n": n_draws})
    await xngin_session.commit()


def _summary_line(name: str, timings: list[float]) -> str:
    return (
        f"BENCH {name}: n={len(timings)} "
        f"min={min(timings):.3f}s median={statistics.median(timings):.3f}s "
        f"mean={statistics.mean(timings):.3f}s max={max(timings):.3f}s"
    )


async def _arm_ids(xngin_session, experiment_id: str) -> list[str]:
    return list(
        (await xngin_session.execute(select(tables.Arm.id).where(tables.Arm.experiment_id == experiment_id))).scalars()
    )


async def _context_inputs(xngin_session, experiment_id: str) -> list[ContextInput]:
    contexts = list(
        (
            await xngin_session.execute(select(tables.Context).where(tables.Context.experiment_id == experiment_id))
        ).scalars()
    )
    return [ContextInput(context_id=c.id, context_value=1.0) for c in sorted(contexts, key=lambda c: c.id)]


async def _setup_experiment_with_draws(
    xngin_session,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    testing_datasource,
    experiment_type: ExperimentsType,
    n_draws: int,
    label: str,
) -> tuple[str, list[ContextInput] | None]:
    experiment_id = await _create_bandit_experiment(aclient, testing_datasource, experiment_type)
    arm_ids = await _arm_ids(xngin_session, experiment_id)
    is_cmab = experiment_type == ExperimentsType.CMAB_ONLINE
    context_inputs = await _context_inputs(xngin_session, experiment_id) if is_cmab else None

    t_setup = time.perf_counter()
    await _warmup_via_api(
        eclient,
        testing_datasource,
        experiment_id,
        is_cmab=is_cmab,
        n_warmup=N_WARMUP,
        context_inputs=context_inputs,
    )
    t_warmup_done = time.perf_counter()
    remaining = n_draws - N_WARMUP
    if remaining > 0:
        await _bulk_insert_draws(xngin_session, experiment_id, arm_ids, remaining, is_cmab=is_cmab)
    print(
        f"\nSETUP {label} {experiment_type.value} n={n_draws}: "
        f"warmup={t_warmup_done - t_setup:.2f}s "
        f"copy={time.perf_counter() - t_warmup_done:.2f}s"
    )
    return experiment_id, context_inputs


@pytest.mark.parametrize("n_draws", DRAW_COUNTS)
@pytest.mark.parametrize("experiment_type", EXPERIMENT_TYPES)
async def test_analyze_endpoint_perf(
    xngin_session,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    testing_datasource,
    experiment_type: ExperimentsType,
    n_draws: int,
):
    experiment_id, context_inputs = await _setup_experiment_with_draws(
        xngin_session, aclient, eclient, testing_datasource, experiment_type, n_draws, "analyze"
    )
    is_cmab = experiment_type == ExperimentsType.CMAB_ONLINE

    timings: list[float] = []
    for _ in range(_iterations(n_draws)):
        t0 = time.perf_counter()
        if is_cmab:
            assert context_inputs is not None
            resp = aclient.analyze_cmab_experiment(
                datasource_id=testing_datasource.ds.id,
                experiment_id=experiment_id,
                body=CMABContextInputRequest(context_inputs=context_inputs),
            )
        else:
            resp = aclient.analyze_experiment(
                datasource_id=testing_datasource.ds.id,
                experiment_id=experiment_id,
            )
        timings.append(time.perf_counter() - t0)
        assert resp.status == 200, resp.data

    print(_summary_line(f"analyze_endpoint[{experiment_type.value},{n_draws}]", timings))


@pytest.mark.parametrize("n_draws", DRAW_COUNTS)
@pytest.mark.parametrize("experiment_type", EXPERIMENT_TYPES)
async def test_snapshot_path_perf(
    xngin_session,
    aclient: AdminAPIClient,
    eclient: ExperimentsAPIClient,
    testing_datasource,
    experiment_type: ExperimentsType,
    n_draws: int,
):
    experiment_id, _ = await _setup_experiment_with_draws(
        xngin_session, aclient, eclient, testing_datasource, experiment_type, n_draws, "snapshot"
    )

    timings: list[float] = []
    for _ in range(_iterations(n_draws)):
        snap = tables.Snapshot(experiment_id=experiment_id)
        xngin_session.add(snap)
        await xngin_session.commit()
        snapshot_id = snap.id

        t0 = time.perf_counter()
        await make_first_snapshot(experiment_id, snapshot_id)
        timings.append(time.perf_counter() - t0)

        await xngin_session.refresh(snap)
        assert snap.status == "success", snap.message

    print(_summary_line(f"snapshot_path[{experiment_type.value},{n_draws}]", timings))
