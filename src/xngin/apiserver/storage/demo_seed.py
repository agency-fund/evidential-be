"""Creates the Project Wellbeing demo organization for developer and staging environments.

Project Wellbeing is a fictional school-health program used to demo Evidential. It is created by the first-time-user
setup (see bootstrap.create_entities_for_first_time_user) next to the generic developer samples, whenever a dev data
warehouse with the demo table (tools/generate_project_wellbeing_dwh.py) is configured.

Demo creation is isolated: it runs in a savepoint and never commits, so a failure rolls back only the demo and the
caller's login or user creation still succeeds.

Every demo is identical. Participants are chosen deterministically from generator-owned columns (not with the
warehouse's unseeded random sampling), assignment uses fixed random states, and outcomes written back to the
warehouse are pure functions of each participant's baseline, assigned arm, and a fixed seed. Creating the demo again
for another user therefore rewrites the same outcome values and leaves earlier demos unchanged.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import sqlalchemy
from loguru import logger
from sqlalchemy import bindparam, delete, func, select, update
from sqlalchemy.orm import Session

from xngin.apiserver import apikeys, flags
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.routers.admin import admin_common
from xngin.apiserver.routers.common_api_types import (
    Arm,
    ArmBandit,
    CMABExperimentSpec,
    Context,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    MABDwhExperimentSpec,
    MABExperimentSpec,
    OnlineFrequentistExperimentSpec,
    PowerRequest,
    PowerResponse,
    PreassignedFrequentistExperimentSpec,
    Stratum,
)
from xngin.apiserver.routers.common_enums import (
    ContextType,
    LikelihoodTypes,
    PriorTypes,
    Relation,
    StopAssignmentReason,
)
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.settings import Dsn, RemoteDatabaseConfig
from xngin.apiserver.snapshots import snapshotter
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.project_wellbeing_dwh_def import PROJECT_WELLBEING_DWH_TABLE_NAME

DEMO_ORGANIZATION_NAME = "Project Wellbeing"
DEMO_WAREHOUSE_DATASOURCE_NAME = "Project Wellbeing Warehouse"

TABLE = PROJECT_WELLBEING_DWH_TABLE_NAME
PRIMARY_KEY = "id"
SCHOOL = "school_id"

# Base seed for every random choice the demo makes. Changing it changes every demo result.
DEMO_SEED = 20261012

GLASSES_N = 900
IRON_PILOT_SCHOOLS = 20
IRON_SUPPORT_SCHOOLS = 90
# Iron Support is designed to detect a 12% gain (about 88 schools); the simulated champions effect of 14% exceeds it.
IRON_SUPPORT_DESIGN_PCT_CHANGE = 0.12
# Glasses is designed around a ~10-point (7%) minimum detectable effect at 900 students.
GLASSES_DESIGN_PCT_CHANGE = 0.07

HEALTH_REPORT_PARENTS = 600
# Number of simulated arrival batches; a real analysis snapshot is captured after each one.
SNAPSHOT_BATCHES = 8
# Experiments are designed this long before they launch.
DESIGN_LEAD_TIME = timedelta(days=7)
HANDWASHING_DRAWS = 400
DENTAL_KIT_DRAWS = 200
# The last Dental Kit batch is left pending (outcomes in the warehouse, not yet pulled) for a live dwh-pull demo.
DENTAL_KIT_PENDING = 20
NUTRITION_DRAWS = 360


@dataclass(frozen=True)
class _Window:
    """An experiment's dates as day offsets from the moment the demo is created (negative = in the past)."""

    start_offset_days: int
    end_offset_days: int

    def dates(self, now: datetime) -> tuple[datetime, datetime]:
        return now + timedelta(days=self.start_offset_days), now + timedelta(days=self.end_offset_days)


# The school year as the demo tells it: the iron pilot comes first and informs the larger iron trial.
IRON_PILOT_WINDOW = _Window(-240, -180)
GLASSES_WINDOW = _Window(-210, -120)
IRON_SUPPORT_WINDOW = _Window(-170, -60)
# In-progress experiments end well in the future so the demo stays live for months after it is created.
HEALTH_REPORT_WINDOW = _Window(-45, 180)
# Bandits come after the A/B tests: optimize the winners, then personalize.
HANDWASHING_WINDOW = _Window(-150, -90)
DENTAL_KIT_WINDOW = _Window(-60, 180)
NUTRITION_WINDOW = _Window(-40, 180)


@dataclass
class _DemoContext:
    session: Session
    organization: tables.Organization
    warehouse: tables.Datasource
    dsn: Dsn
    now: datetime


def maybe_create_project_wellbeing(session: Session, user: tables.User, dwh_dsn: str | None) -> None:
    """Creates the Project Wellbeing demo organization for the user when the demo warehouse is available.

    Skips (with a log line) when demo creation is disabled, no dev warehouse is configured, or the demo table is not
    loaded. Any error rolls back only the demo's savepoint and is logged; it never propagates to the caller.
    """
    if not flags.XNGIN_CREATE_DEMO_ORG:
        logger.info("Skipping Project Wellbeing demo: XNGIN_CREATE_DEMO_ORG is off.")
        return
    if not dwh_dsn:
        return

    dsn = Dsn.from_url(dwh_dsn)
    try:
        with DwhSession.open(dsn) as dwh:
            table_loaded = TABLE in dwh.list_tables()
    except Exception:
        logger.exception("Skipping Project Wellbeing demo: could not list tables in the dev warehouse.")
        return
    if not table_loaded:
        logger.info(
            f"Skipping Project Wellbeing demo: table {TABLE} is not loaded (run `task bootstrap-dwh-database`)."
        )
        return

    try:
        with session.begin_nested():
            _create_demo(session, user, dsn)
    except Exception:
        logger.exception("Project Wellbeing demo creation failed; continuing without it.")


def _create_demo(session: Session, user: tables.User, dsn: Dsn) -> None:
    """Creates the demo organization, its datasources, an API key, and the experiments inside the caller's savepoint."""
    organization = admin_common.create_organization_impl(session, user, DEMO_ORGANIZATION_NAME)
    warehouse = admin_common.create_datasource_impl(
        session,
        organization,
        DEMO_WAREHOUSE_DATASOURCE_NAME,
        RemoteDatabaseConfig(type="remote", dwh=dsn),
    )
    # The plaintext key is intentionally discarded; `xngin-cli add-user` prints fresh keys, and presenters can create
    # one in the UI when they need to show integration calls.
    label, key = apikeys.make_key()
    warehouse.api_keys.append(tables.ApiKey(id=label, key=apikeys.hash_key_or_raise(key)))
    session.flush()

    ctx = _DemoContext(session=session, organization=organization, warehouse=warehouse, dsn=dsn, now=datetime.now(UTC))
    _create_iron_pilot(ctx)
    _create_glasses(ctx)
    _create_iron_support(ctx)
    _create_health_report(ctx)
    _create_handwashing(ctx)
    _create_dental_kit(ctx)
    _create_nutrition(ctx)


# --- Warehouse access ----------------------------------------------------------------------------------------------


def _fetch_rows(
    dwh_session: Session, sa_table: sqlalchemy.Table, columns: Sequence[str], where: sqlalchemy.ColumnElement[bool]
) -> list[sqlalchemy.Row[Any]]:
    """Returns matching rows ordered by primary key, so the result never depends on warehouse row order."""
    query = select(*(sa_table.c[name] for name in columns)).where(where).order_by(sa_table.c[PRIMARY_KEY])
    return list(dwh_session.execute(query).fetchall())


def _write_outcomes(dwh_session: Session, sa_table: sqlalchemy.Table, column: str, values: dict[int, Any]) -> None:
    """Writes one seed-owned outcome column for the given participant IDs."""
    stmt = (
        update(sa_table)
        .where(sa_table.c[PRIMARY_KEY] == bindparam("participant_id"))
        .values({column: bindparam("outcome_value")})
    )
    dwh_session.execute(stmt, [{"participant_id": pid, "outcome_value": v} for pid, v in sorted(values.items())])
    dwh_session.commit()


def _with_warehouse[T](ctx: _DemoContext, fn: Callable[[DwhSession, sqlalchemy.Table], T]) -> T:
    with DwhSession.open(ctx.dsn) as dwh:
        return fn(dwh, dwh.inspect_table(TABLE))


# --- Experiment helpers --------------------------------------------------------------------------------------------


def _power(ctx: _DemoContext, request: PowerRequest) -> PowerResponse:
    """Runs Evidential's power check exactly as the power endpoint does."""
    from xngin.apiserver.routers.admin import admin_api  # noqa: PLC0415 - avoids an import cycle through auth.

    return admin_api.power_check(ctx.warehouse, request)


def _create_preassigned(
    ctx: _DemoContext,
    spec: PreassignedFrequentistExperimentSpec,
    sa_table: sqlalchemy.Table,
    participants: list,  # untyped like GetParticipantsResult.participants, which this mirrors
    power_analyses: PowerResponse | None,
    random_state: int,
) -> tables.Experiment:
    """Creates and commits a preassigned experiment from already-chosen participants with Evidential's real
    assignment, balance check, and storage."""
    result = experiments_common.create_preassigned_experiment_impl(
        CreateExperimentRequest(design_spec=spec, power_analyses=power_analyses),
        datasource_id=ctx.warehouse.id,
        organization_id=ctx.organization.id,
        dwh_sa_table=sa_table,
        dwh_participants=participants,
        random_state=random_state,
        xngin_session=ctx.session,
        stratify_on_metrics=False,
        validated_webhooks=[],
        field_type_map=experiments_common.fetch_fields_or_raise(ctx.warehouse, spec),
    )
    experiment = ctx.session.get_one(tables.Experiment, result.experiment_id)
    experiments_common.commit_experiment_impl(ctx.session, experiment)
    # Place the experiment in the story's school year: designed a week before launch, everyone assigned at launch.
    experiment.created_at = experiment.start_date - DESIGN_LEAD_TIME
    experiment.stopped_assignments_at = experiment.start_date
    ctx.session.execute(
        update(tables.ArmAssignment)
        .where(tables.ArmAssignment.experiment_id == experiment.id)
        .values(created_at=experiment.start_date)
    )
    return experiment


def _create_and_commit(
    ctx: _DemoContext, datasource: tables.Datasource, request: CreateExperimentRequest
) -> tables.Experiment:
    """Creates and commits an online or bandit experiment through Evidential's normal creation path."""
    result = experiments_common.create_experiment_impl(
        request, datasource, ctx.session, stratify_on_metrics=False, random_state=None, validated_webhooks=[]
    )
    experiment = ctx.session.get_one(tables.Experiment, result.experiment_id)
    if request.design_spec.experiment_type.is_bandit():
        _order_arm_ids_by_position(ctx, experiment, request)
    experiments_common.commit_experiment_impl(ctx.session, experiment)
    experiment.created_at = experiment.start_date - DESIGN_LEAD_TIME
    return experiment


def _order_arm_ids_by_position(
    ctx: _DemoContext, experiment: tables.Experiment, request: CreateExperimentRequest
) -> None:
    """Prefixes each arm's random ID with its position so IDs sort in catalog order.

    Bandit sampling orders arms by ID (stats.bandit_sampling.choose_arm), and arm IDs are random, so the same seed
    would pick different arms in different demos. Runs before commit and before any draw, so nothing references the
    old IDs yet. The random remainder keeps every ID unique.
    """
    # experiment.arms is ordered by position (the order arms were given in the design spec).
    for index, arm in enumerate(experiment.arms):
        prefix, _, suffix = arm.id.partition("_")
        arm.id = f"{prefix}_{index:02d}{suffix[2:]}"
    # Context values are likewise passed in context-ID order, so contexts get the same treatment (spec order).
    if experiment.contexts:
        assert isinstance(request.design_spec, CMABExperimentSpec)  # only contextual bandits have contexts
        spec_order = {c.context_name: i for i, c in enumerate(request.design_spec.contexts or [])}
        for context in experiment.contexts:
            prefix, _, suffix = context.id.partition("_")
            context.id = f"{prefix}_{spec_order[context.name]:02d}{suffix[2:]}"
    ctx.session.flush()


def _api_only_datasource(ctx: _DemoContext) -> tables.Datasource:
    return next(ds for ds in ctx.organization.datasources if ds.name == admin_common.DEFAULT_NO_DWH_SOURCE_NAME)


def _mark_finished(experiment: tables.Experiment) -> None:
    """Records that assignment stopped at the end date, as the assignment path does lazily for ended experiments."""
    experiment.stopped_assignments_at = experiment.end_date
    experiment.stopped_assignments_reason = StopAssignmentReason.END_DATE


def _assigned_arms(ctx: _DemoContext, experiment: tables.Experiment) -> list[tuple[int, str]]:
    """Returns (participant_id, arm_name) for every assignment, ordered by participant ID."""
    arm_names = {arm.id: arm.name for arm in experiment.arms}
    rows = ctx.session.execute(
        select(tables.ArmAssignment.participant_id, tables.ArmAssignment.arm_id).where(
            tables.ArmAssignment.experiment_id == experiment.id
        )
    ).all()
    return sorted((int(pid), arm_names[arm_id]) for pid, arm_id in rows)


def _record_decision(experiment: tables.Experiment, *, impact: str, decision: str) -> None:
    experiment.impact = impact
    experiment.decision = decision


def _store_snapshot(ctx: _DemoContext, experiment: tables.Experiment, taken_at: datetime) -> None:
    """Runs the real snapshot analysis on the experiment's current state and stores it as if taken at taken_at."""
    ctx.session.flush()
    _add_snapshot(ctx, experiment, _snapshot_analysis(ctx, experiment), taken_at)


def _snapshot_analysis(ctx: _DemoContext, experiment: tables.Experiment):
    # Reuses the snapshotter's own analysis so demo history matches what the snapshot job would have recorded.
    return snapshotter._compute_snapshot_data(ctx.session, ctx.warehouse, experiment, flags.DWH_TIMEOUT_SECS)


def _add_snapshot(ctx: _DemoContext, experiment: tables.Experiment, analysis, taken_at: datetime) -> None:
    ctx.session.add(
        tables.Snapshot(
            experiment_id=experiment.id,
            created_at=taken_at,
            updated_at=taken_at,
            status="success",
            data=analysis.model_copy(update={"created_at": taken_at}).model_dump(mode="json"),
        )
    )


def _store_results_as_they_arrive(
    ctx: _DemoContext, experiment: tables.Experiment, arrival_order: list[int], times: list[datetime]
) -> None:
    """Stores real snapshots of a preassigned experiment as if outcomes arrived in arrival_order over time.

    Every participant is assigned at launch and every outcome is already in the warehouse, so each earlier snapshot
    is computed with the not-yet-arrived participants' assignments temporarily removed inside a savepoint that is
    rolled back. The analysis itself is the snapshotter's own; only its input is restricted.
    """
    ctx.session.flush()
    for k, taken_at in enumerate(times, start=1):
        arrived = {str(pid) for pid in arrival_order[: round(len(arrival_order) * k / len(times))]}
        savepoint = ctx.session.begin_nested()
        try:
            ctx.session.execute(
                delete(tables.ArmAssignment).where(
                    tables.ArmAssignment.experiment_id == experiment.id,
                    tables.ArmAssignment.participant_id.not_in(arrived),
                )
            )
            analysis = _snapshot_analysis(ctx, experiment)
        finally:
            savepoint.rollback()
        _add_snapshot(ctx, experiment, analysis, taken_at)


def _spread_timestamps(
    ctx: _DemoContext,
    model: type[tables.Draw | tables.ArmAssignment],
    experiment: tables.Experiment,
    ordered_participant_ids: Sequence[str],
    start: datetime,
    end: datetime,
    observed_after: timedelta | None = None,
) -> None:
    """Rewrites assignment or draw times so participants arrive evenly between start and end in the given order.

    Batches of the simulation line up with the batch snapshot times because both divide the same span evenly. For
    draws, observed outcomes are stamped observed_after the draw (never later than now); pending draws stay pending.
    """
    # Core statements (not ORM bulk updates) so executemany can match rows by experiment and participant ID.
    table = cast(sqlalchemy.Table, model.__table__)
    step = (end - start) / len(ordered_participant_ids)
    rows = [
        {"row_participant_id": pid, "row_created_at": start + step * (i + 0.5)}
        for i, pid in enumerate(ordered_participant_ids)
    ]
    ctx.session.connection().execute(
        update(table)
        .where(table.c.experiment_id == experiment.id, table.c.participant_id == bindparam("row_participant_id"))
        .values(created_at=bindparam("row_created_at")),
        rows,
    )
    if observed_after is not None:
        ctx.session.connection().execute(
            update(table)
            .where(table.c.experiment_id == experiment.id, table.c.outcome.is_not(None))
            .values(observed_at=func.least(table.c.created_at + observed_after, ctx.now))
        )


def _batch_times(window: _Window, now: datetime, batches: int) -> list[datetime]:
    """Evenly spaced times from the experiment start up to now, one per batch (the last one is now)."""
    start, _ = window.dates(now)
    step = (now - start) / batches
    return [start + step * (i + 1) for i in range(batches)]


def _pick_schools(rng: np.random.Generator, schools: Sequence[int], count: int) -> list[int]:
    return sorted(int(s) for s in rng.choice(sorted(schools), size=count, replace=False))


# --- Iron Pilot ----------------------------------------------------------------------------------------------------


def _create_iron_pilot(ctx: _DemoContext) -> None:
    rng = np.random.default_rng(DEMO_SEED + 1)
    pilot_schools = _pick_schools(rng, range(1, 121), IRON_PILOT_SCHOOLS)

    def fetch(dwh: DwhSession, sa_table: sqlalchemy.Table):
        rows = dwh.run(
            _fetch_rows,
            sa_table,
            [PRIMARY_KEY, SCHOOL, "baseline_iron_adherence_pct"],
            sa_table.c[SCHOOL].in_(pilot_schools),
        )
        return sa_table, rows

    sa_table, rows = _with_warehouse(ctx, fetch)
    start, end = IRON_PILOT_WINDOW.dates(ctx.now)
    spec = PreassignedFrequentistExperimentSpec(
        experiment_name="Iron Pilot: SMS reminders",
        description="SMS reminders to parents increase students' weekly iron-tablet adherence.",
        table_name=TABLE,
        primary_key=PRIMARY_KEY,
        start_date=start,
        end_date=end,
        arms=[
            Arm(arm_name="Control", arm_description="No reminders"),
            Arm(arm_name="SMS reminders", arm_description="Weekly SMS reminder to parents"),
        ],
        filters=[],
        strata=[],
        metrics=[DesignSpecMetricRequest(field_name="iron_adherence_pct_pilot", metric_pct_change=0.12)],
        cluster_key=SCHOOL,
        desired_n_clusters=IRON_PILOT_SCHOOLS,
    )
    experiment = _create_preassigned(ctx, spec, sa_table, rows, None, random_state=DEMO_SEED + 1)

    # Outcome: SMS reminders make no real difference. Adherence stays near each student's baseline.
    baseline = {row.id: row.baseline_iron_adherence_pct for row in rows}
    noise_rng = np.random.default_rng(DEMO_SEED + 101)
    outcomes = {}
    for pid, arm in _assigned_arms(ctx, experiment):
        lift = 0.5 if arm == "SMS reminders" else 0.0
        outcomes[pid] = round(float(np.clip(baseline[pid] + lift + noise_rng.normal(0, 6), 0, 100)), 1)
    _with_warehouse(ctx, lambda dwh, t: dwh.run(_write_outcomes, t, "iron_adherence_pct_pilot", outcomes))

    _record_decision(
        experiment,
        impact="low",
        decision=(
            "SMS reminders alone did not change adherence. Add an in-person support arm and size the next trial "
            "using the school-level ICC measured here (0.15)."
        ),
    )


# --- Glasses -------------------------------------------------------------------------------------------------------


def _create_glasses(ctx: _DemoContext) -> None:
    columns = [PRIMARY_KEY, "gender", "grade", "baseline_reading_score", "reading_measured_last_term"]

    def fetch(dwh: DwhSession, sa_table: sqlalchemy.Table):
        return sa_table, dwh.run(_fetch_rows, sa_table, columns, sa_table.c.failed_vision_screen.is_(True))

    sa_table, failed = _with_warehouse(ctx, fetch)
    measured_stats = _power(
        ctx,
        PowerRequest(
            table_name=TABLE,
            metrics=[
                DesignSpecMetricRequest(field_name="reading_score_endline", metric_pct_change=GLASSES_DESIGN_PCT_CHANGE)
            ],
            filters=[
                Filter(field_name="failed_vision_screen", relation=Relation.INCLUDES, value=[True]),
                Filter(field_name="reading_measured_last_term", relation=Relation.INCLUDES, value=[True]),
            ],
            n_arms=2,
        ),
    )
    # One-time mode assigns only students without a reading score yet. Choosing them from the generator-owned
    # reading_measured_last_term flag (rather than from NULLs, which earlier demos fill in) keeps every demo identical.
    pool = [row for row in failed if not row.reading_measured_last_term]
    rng = np.random.default_rng(DEMO_SEED + 2)
    chosen_ids = set(int(i) for i in rng.choice([row.id for row in pool], size=GLASSES_N, replace=False))
    participants = [row for row in pool if row.id in chosen_ids]

    # The stored power analysis describes the design as it looked before the trial: baseline stats from last term's
    # measured students and the not-yet-measured pool size. Echoing those stats keeps it identical across demos.
    measured = measured_stats.analyses[0].metric_spec
    primary = DesignSpecMetricRequest(
        field_name="reading_score_endline",
        metric_pct_change=GLASSES_DESIGN_PCT_CHANGE,
        use_one_time_metric=True,
        metric_type=measured.metric_type,
        metric_baseline=measured.metric_baseline,
        metric_stddev=measured.metric_stddev,
        available_n=len(failed),
        available_nonnull_n=len(failed) - len(pool),
    )
    power = _power(
        ctx,
        PowerRequest(
            table_name=TABLE,
            metrics=[primary],
            n_arms=2,
            desired_n=GLASSES_N,
            desired_ns=[300, 600, 900, 1200, 1500, 1800],
        ),
    )

    start, end = GLASSES_WINDOW.dates(ctx.now)
    spec = PreassignedFrequentistExperimentSpec(
        experiment_name="Glasses for reading",
        description=(
            "Students who failed the vision screening score higher on reading when given free glasses, because they "
            "can see the text."
        ),
        table_name=TABLE,
        primary_key=PRIMARY_KEY,
        start_date=start,
        end_date=end,
        arms=[
            Arm(arm_name="Control", arm_description="Usual classroom support"),
            Arm(arm_name="Free glasses", arm_description="Free prescription glasses fitted at school"),
        ],
        filters=[Filter(field_name="failed_vision_screen", relation=Relation.INCLUDES, value=[True])],
        strata=[Stratum(field_name="gender"), Stratum(field_name="grade")],
        metrics=[
            DesignSpecMetricRequest(
                field_name="reading_score_endline",
                metric_pct_change=GLASSES_DESIGN_PCT_CHANGE,
                use_one_time_metric=True,
            ),
            DesignSpecMetricRequest(field_name="glasses_attendance_pct", metric_pct_change=0.05),
        ],
        desired_n=GLASSES_N,
    )
    experiment = _create_preassigned(ctx, spec, sa_table, participants, power, random_state=DEMO_SEED + 2)

    # Outcome: glasses lift reading about 33% over control; attendance improves a little.
    baseline = {row.id: row.baseline_reading_score for row in participants}
    noise_rng = np.random.default_rng(DEMO_SEED + 102)
    reading, attendance = {}, {}
    for pid, arm in _assigned_arms(ctx, experiment):
        treated = arm == "Free glasses"
        score = baseline[pid] * (1.33 if treated else 1.0) + noise_rng.normal(0, 15)
        reading[pid] = round(float(np.clip(score, 0, 400)), 1)
        attendance[pid] = round(float(np.clip((88 if treated else 84) + noise_rng.normal(0, 8), 0, 100)), 1)

    def write(dwh: DwhSession, sa_table: sqlalchemy.Table) -> None:
        dwh.run(_write_outcomes, sa_table, "reading_score_endline", reading)
        dwh.run(_write_outcomes, sa_table, "glasses_attendance_pct", attendance)

    _with_warehouse(ctx, write)

    # Reading scores came in school by school over the term; show the results as they arrived.
    arrival_rng = np.random.default_rng(DEMO_SEED + 202)
    arrival_order = [int(pid) for pid in arrival_rng.permutation(sorted(reading))]
    glasses_start, glasses_end = GLASSES_WINDOW.dates(ctx.now)
    step = (glasses_end - glasses_start) / SNAPSHOT_BATCHES
    times = [glasses_start + step * (i + 1) for i in range(SNAPSHOT_BATCHES)]
    _store_results_as_they_arrive(ctx, experiment, arrival_order, times)

    _record_decision(
        experiment,
        impact="high",
        decision=(
            "Free glasses raised reading scores by about a third. Scale vision screening and glasses to all schools."
        ),
    )


# --- Iron Support --------------------------------------------------------------------------------------------------


def _create_iron_support(ctx: _DemoContext) -> None:
    # The full trial runs in schools that were not in the pilot.
    pilot_schools = set(_pick_schools(np.random.default_rng(DEMO_SEED + 1), range(1, 121), IRON_PILOT_SCHOOLS))
    candidates = [s for s in range(1, 121) if s not in pilot_schools]
    schools = _pick_schools(np.random.default_rng(DEMO_SEED + 3), candidates, IRON_SUPPORT_SCHOOLS)
    arm_weights = [20, 40, 40]

    def fetch(dwh: DwhSession, sa_table: sqlalchemy.Table):
        rows = dwh.run(
            _fetch_rows, sa_table, [PRIMARY_KEY, SCHOOL, "baseline_iron_adherence_pct"], sa_table.c[SCHOOL].in_(schools)
        )
        return sa_table, rows

    sa_table, rows = _with_warehouse(ctx, fetch)
    # Size the trial from the iron baseline, which carries the school-level ICC the pilot measured.
    baseline_stats = _power(
        ctx,
        PowerRequest(
            table_name=TABLE,
            metrics=[
                DesignSpecMetricRequest(
                    field_name="baseline_iron_adherence_pct", metric_pct_change=IRON_SUPPORT_DESIGN_PCT_CHANGE
                )
            ],
            cluster_key=SCHOOL,
            n_arms=3,
            arm_weights=arm_weights,
        ),
    )
    stats = baseline_stats.analyses[0].metric_spec
    power = _power(
        ctx,
        PowerRequest(
            table_name=TABLE,
            metrics=[
                DesignSpecMetricRequest(
                    field_name="iron_adherence_pct_support",
                    metric_pct_change=IRON_SUPPORT_DESIGN_PCT_CHANGE,
                    metric_type=stats.metric_type,
                    metric_baseline=stats.metric_baseline,
                    metric_stddev=stats.metric_stddev,
                    available_n=stats.available_n,
                    available_nonnull_n=stats.available_nonnull_n,
                    icc=stats.icc,
                    avg_cluster_size=stats.avg_cluster_size,
                    cv=stats.cv,
                )
            ],
            cluster_key=SCHOOL,
            n_arms=3,
            arm_weights=arm_weights,
        ),
    )

    start, end = IRON_SUPPORT_WINDOW.dates(ctx.now)
    spec = PreassignedFrequentistExperimentSpec(
        experiment_name="Iron Support: peer champions",
        description=(
            "In-person peer health champions raise students' iron-tablet adherence more than SMS reminders or no "
            "support."
        ),
        table_name=TABLE,
        primary_key=PRIMARY_KEY,
        start_date=start,
        end_date=end,
        arms=[
            Arm(arm_name="Control", arm_description="No extra support", arm_weight=arm_weights[0]),
            Arm(arm_name="SMS reminders", arm_description="Weekly SMS reminder to parents", arm_weight=arm_weights[1]),
            Arm(
                arm_name="Peer champions",
                arm_description="Trained student health champions in each class",
                arm_weight=arm_weights[2],
            ),
        ],
        filters=[],
        strata=[],
        metrics=[
            DesignSpecMetricRequest(
                field_name="iron_adherence_pct_support", metric_pct_change=IRON_SUPPORT_DESIGN_PCT_CHANGE
            ),
            DesignSpecMetricRequest(field_name="iron_support_hb_change", metric_pct_change=0.10),
        ],
        cluster_key=SCHOOL,
        desired_n_clusters=IRON_SUPPORT_SCHOOLS,
    )
    experiment = _create_preassigned(ctx, spec, sa_table, rows, power, random_state=DEMO_SEED + 3)

    # Outcome: peer champions add about 14% over control; SMS adds a little.
    baseline = {row.id: row.baseline_iron_adherence_pct for row in rows}
    # Champions lift is set so the measured gain lands near 14%: the control schools happen to start ~1.5 points
    # lower under the fixed assignment, which the analysis attributes to the treatment arms.
    lift = {"Control": 0.0, "SMS reminders": 2.0, "Peer champions": 6.2}
    hb_mean = {"Control": 0.1, "SMS reminders": 0.2, "Peer champions": 0.5}
    noise_rng = np.random.default_rng(DEMO_SEED + 103)
    adherence, hb_change = {}, {}
    for pid, arm in _assigned_arms(ctx, experiment):
        adherence[pid] = round(float(np.clip(baseline[pid] + lift[arm] + noise_rng.normal(0, 6), 0, 100)), 1)
        hb_change[pid] = round(float(hb_mean[arm] + noise_rng.normal(0, 0.3)), 2)

    def write(dwh: DwhSession, sa_table: sqlalchemy.Table) -> None:
        dwh.run(_write_outcomes, sa_table, "iron_adherence_pct_support", adherence)
        dwh.run(_write_outcomes, sa_table, "iron_support_hb_change", hb_change)

    _with_warehouse(ctx, write)
    _record_decision(
        experiment,
        impact="medium",
        decision="Peer champions raised adherence by about 14%. Roll out champions; drop SMS-only reminders.",
    )


# --- Health Report -------------------------------------------------------------------------------------------------


def _create_health_report(ctx: _DemoContext) -> None:
    grade_range = [3, 8]

    def fetch(dwh: DwhSession, sa_table: sqlalchemy.Table):
        return dwh.run(_fetch_rows, sa_table, [PRIMARY_KEY], sa_table.c.grade.between(grade_range[0], grade_range[1]))

    eligible = [int(row.id) for row in _with_warehouse(ctx, fetch)]
    # Parents "open the app" in a fixed, shuffled order.
    rng = np.random.default_rng(DEMO_SEED + 4)
    arrivals = [int(i) for i in rng.choice(eligible, size=HEALTH_REPORT_PARENTS, replace=False)]

    start, end = HEALTH_REPORT_WINDOW.dates(ctx.now)
    spec = OnlineFrequentistExperimentSpec(
        experiment_name="Health Report redesign",
        description="A simpler, redesigned health report leads more parents to book a health check-up for their child.",
        table_name=TABLE,
        primary_key=PRIMARY_KEY,
        start_date=start,
        end_date=end,
        arms=[
            Arm(arm_name="Current report", arm_description="Existing multi-page health report"),
            Arm(arm_name="New report", arm_description="One-page report with a clear next step"),
        ],
        filters=[Filter(field_name="grade", relation=Relation.BETWEEN, value=grade_range)],
        strata=[],
        metrics=[
            DesignSpecMetricRequest(field_name="health_checkup_booked", metric_pct_change=0.10),
            DesignSpecMetricRequest(field_name="health_portal_logins", metric_pct_change=0.10),
        ],
    )
    experiment = _create_and_commit(ctx, ctx.warehouse, CreateExperimentRequest(design_spec=spec))

    # Outcome: the new report adds about 8% to check-up bookings, too little for this sample to be conclusive yet.
    booking_rate = {"Current report": 0.40, "New report": 0.432}
    login_rate = {"Current report": 3.0, "New report": 3.1}
    outcome_rng = np.random.default_rng(DEMO_SEED + 104)
    # Bookings follow each arm's rate as a running quota (with a random starting phase per arm) rather than
    # independent coin flips: with only ~300 parents per arm, coin-flip noise can erase or double an 8% lift.
    booking_phase = {arm: float(outcome_rng.random()) for arm in booking_rate}
    seen = dict.fromkeys(booking_rate, 0)
    batch_size = HEALTH_REPORT_PARENTS // SNAPSHOT_BATCHES
    for batch, taken_at in enumerate(_batch_times(HEALTH_REPORT_WINDOW, ctx.now, SNAPSHOT_BATCHES)):
        booked, logins = {}, {}
        for offset, pid in enumerate(arrivals[batch * batch_size : (batch + 1) * batch_size]):
            assignment = experiments_common.insert_assignment_no_commit(
                ctx.session, experiment, str(pid), random_state=DEMO_SEED + 400 + batch * batch_size + offset
            )
            assert assignment is not None  # online experiments always assign
            arm = assignment.arm_name
            rate, phase, n = booking_rate[arm], booking_phase[arm], seen[arm]
            booked[pid] = int(rate * (n + 1) + phase) > int(rate * n + phase)
            seen[arm] = n + 1
            logins[pid] = int(outcome_rng.poisson(login_rate[arm]))

        def write(dwh: DwhSession, sa_table: sqlalchemy.Table, booked=booked, logins=logins) -> None:
            dwh.run(_write_outcomes, sa_table, "health_checkup_booked", booked)
            dwh.run(_write_outcomes, sa_table, "health_portal_logins", logins)

        _with_warehouse(ctx, write)
        _store_snapshot(ctx, experiment, taken_at)

    arrived = [str(pid) for pid in arrivals[: batch_size * SNAPSHOT_BATCHES]]
    _spread_timestamps(ctx, tables.ArmAssignment, experiment, arrived, start, ctx.now)
    _record_decision(
        experiment,
        impact="unclear",
        decision="Too early to call: the new report looks about 8% better, but the range still includes no effect. "
        "Keep running.",
    )


# --- Bandits -------------------------------------------------------------------------------------------------------


def _draw(ctx: _DemoContext, experiment: tables.Experiment, participant_id: str, seed: int, context=None) -> str:
    """Makes one real bandit draw and returns the chosen arm's name."""
    assignment = experiments_common.insert_assignment_no_commit(
        ctx.session, experiment, participant_id, sorted_context_vals=context, random_state=seed
    )
    assert assignment is not None  # bandits always draw
    return assignment.arm_name


def _create_handwashing(ctx: _DemoContext) -> None:
    # True wash-before-lunch rates; the social-norm message is the clear winner.
    true_rates = {
        "Simple reminder": 0.12,
        "Germ facts": 0.15,
        "Your classmates do it": 0.25,
        "Teacher praise": 0.17,
        "Soap star sticker": 0.13,
    }
    start, end = HANDWASHING_WINDOW.dates(ctx.now)
    spec = MABExperimentSpec(
        experiment_name="Handwashing reminders",
        description=(
            "Among five reminder messages, a social-norm message about classmates leads the most children to wash "
            "their hands before lunch."
        ),
        start_date=start,
        end_date=end,
        prior_type=PriorTypes.BETA,
        reward_type=LikelihoodTypes.BERNOULLI,
        arms=[
            ArmBandit(arm_name=name, arm_description=f"Lunchtime message: {name}", alpha_init=1.0, beta_init=1.0)
            for name in true_rates
        ],
    )
    experiment = _create_and_commit(ctx, _api_only_datasource(ctx), CreateExperimentRequest(design_spec=spec))

    rng = np.random.default_rng(DEMO_SEED + 105)
    for i in range(HANDWASHING_DRAWS):
        pid = f"hw-{i + 1:04d}"
        arm = _draw(ctx, experiment, pid, DEMO_SEED + 5000 + i)
        washed = 1.0 if rng.random() < true_rates[arm] else 0.0
        experiments_common.update_bandit_arm_with_outcome_impl(ctx.session, experiment, pid, washed)

    # Children wash (or not) at lunch the same day they get the message.
    pids = [f"hw-{i + 1:04d}" for i in range(HANDWASHING_DRAWS)]
    _spread_timestamps(ctx, tables.Draw, experiment, pids, start, end, observed_after=timedelta(hours=3))
    _mark_finished(experiment)
    _record_decision(
        experiment,
        impact="high",
        decision=(
            'The "your classmates do it" message works best (about 1 in 4 children wash). Use it in all schools; '
            "the bandit already sent most children to it during the test."
        ),
    )


def _create_dental_kit(ctx: _DemoContext) -> None:
    true_days = {"Standard kit": 14.0, "Sticker chart": 17.0, "Story book": 20.0}

    def fetch(dwh: DwhSession, sa_table: sqlalchemy.Table):
        return dwh.run(_fetch_rows, sa_table, [PRIMARY_KEY], sa_table.c.grade <= 4)

    eligible = [int(row.id) for row in _with_warehouse(ctx, fetch)]
    rng = np.random.default_rng(DEMO_SEED + 6)
    students = [int(i) for i in rng.choice(eligible, size=DENTAL_KIT_DRAWS, replace=False)]

    start, end = DENTAL_KIT_WINDOW.dates(ctx.now)
    spec = MABDwhExperimentSpec(
        experiment_name="Dental kit design",
        description=(
            "A dental kit with an illustrated story book leads to more brushing days than a standard kit or a "
            "sticker chart."
        ),
        start_date=start,
        end_date=end,
        table_name=TABLE,
        primary_key=PRIMARY_KEY,
        target_field_name="brushing_days_dental_kit",
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.NORMAL,
        arms=[
            ArmBandit(arm_name=name, arm_description=f"Kit: {name}", mu_init=15.0, sigma_init=5.0) for name in true_days
        ],
    )
    experiment = _create_and_commit(ctx, ctx.warehouse, CreateExperimentRequest(design_spec=spec))

    outcome_rng = np.random.default_rng(DEMO_SEED + 106)
    batch_size = DENTAL_KIT_DRAWS // SNAPSHOT_BATCHES
    for batch, taken_at in enumerate(_batch_times(DENTAL_KIT_WINDOW, ctx.now, SNAPSHOT_BATCHES)):
        days: dict[int, float] = {}
        for offset, pid in enumerate(students[batch * batch_size : (batch + 1) * batch_size]):
            arm = _draw(ctx, experiment, str(pid), DEMO_SEED + 6000 + batch * batch_size + offset)
            days[pid] = round(float(np.clip(outcome_rng.normal(true_days[arm], 4.0), 0, 30)), 1)

        # Brushing days land in the warehouse first, as they would from the school's own records.
        def write(dwh: DwhSession, sa_table: sqlalchemy.Table, days=days) -> None:
            dwh.run(_write_outcomes, sa_table, "brushing_days_dental_kit", days)

        _with_warehouse(ctx, write)
        ready = sorted(days)[: len(days) - DENTAL_KIT_PENDING] if batch == SNAPSHOT_BATCHES - 1 else sorted(days)
        for pid in ready:
            # Applies the warehouse value the same way the dwh-pull job does.
            experiments_common.update_bandit_arm_with_outcome_impl(ctx.session, experiment, str(pid), days[pid])
        _store_snapshot(ctx, experiment, taken_at)

    # Brushing days are counted over the following weeks and land in the school records later.
    drawn = [str(pid) for pid in students[: batch_size * SNAPSHOT_BATCHES]]
    _spread_timestamps(ctx, tables.Draw, experiment, drawn, start, ctx.now, observed_after=timedelta(days=3))


def _create_nutrition(ctx: _DemoContext) -> None:
    # Chance a family tries a healthy recipe, by tip format and smartphone access.
    true_rates = {
        "Recipe video": {True: 0.50, False: 0.08},
        "Printed recipe card": {True: 0.20, False: 0.55},
        "SMS tip": {True: 0.25, False: 0.20},
    }

    def fetch(dwh: DwhSession, sa_table: sqlalchemy.Table):
        return dwh.run(_fetch_rows, sa_table, [PRIMARY_KEY, "has_smartphone_household", "grade"], sa_table.c.id > 0)

    rows = _with_warehouse(ctx, fetch)
    rng = np.random.default_rng(DEMO_SEED + 7)
    order = rng.choice(len(rows), size=NUTRITION_DRAWS, replace=False)
    families = [rows[int(i)] for i in order]

    start, end = NUTRITION_WINDOW.dates(ctx.now)
    spec = CMABExperimentSpec(
        experiment_name="Nutrition tips: who responds to what",
        description=(
            "The best nutrition tip format depends on the family: a recipe video for families with a smartphone, a "
            "printed recipe card for families without one."
        ),
        start_date=start,
        end_date=end,
        prior_type=PriorTypes.NORMAL,
        reward_type=LikelihoodTypes.BERNOULLI,
        arms=[
            ArmBandit(arm_name=name, arm_description=f"Weekly tip as: {name}", mu_init=0.0, sigma_init=1.0)
            for name in true_rates
        ],
        contexts=[
            Context(
                context_name="has_smartphone",
                context_description="1 if the household has a smartphone, otherwise 0",
                value_type=ContextType.BINARY,
            ),
            Context(
                context_name="grade",
                context_description="Child's school grade (1-8)",
                value_type=ContextType.REAL_VALUED,
            ),
        ],
    )
    experiment = _create_and_commit(ctx, _api_only_datasource(ctx), CreateExperimentRequest(design_spec=spec))
    # Context values are passed in context-ID order, as the assignment API requires.
    names_in_id_order = [c.name for c in sorted(experiment.contexts, key=lambda c: c.id)]

    outcome_rng = np.random.default_rng(DEMO_SEED + 107)
    batch_size = NUTRITION_DRAWS // SNAPSHOT_BATCHES
    for batch, taken_at in enumerate(_batch_times(NUTRITION_WINDOW, ctx.now, SNAPSHOT_BATCHES)):
        for offset, family in enumerate(families[batch * batch_size : (batch + 1) * batch_size]):
            values = {"has_smartphone": float(family.has_smartphone_household), "grade": float(family.grade)}
            pid = f"nu-{int(family.id)}"
            arm = _draw(
                ctx,
                experiment,
                pid,
                DEMO_SEED + 7000 + batch * batch_size + offset,
                context=[values[name] for name in names_in_id_order],
            )
            tried = 1.0 if outcome_rng.random() < true_rates[arm][bool(family.has_smartphone_household)] else 0.0
            experiments_common.update_bandit_arm_with_outcome_impl(ctx.session, experiment, pid, tried)
        _store_snapshot(ctx, experiment, taken_at)

    drawn = [f"nu-{int(family.id)}" for family in families[: batch_size * SNAPSHOT_BATCHES]]
    _spread_timestamps(ctx, tables.Draw, experiment, drawn, start, ctx.now, observed_after=timedelta(days=2))
