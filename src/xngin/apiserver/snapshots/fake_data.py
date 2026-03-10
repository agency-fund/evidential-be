"""Helpers for generating fake frequentist snapshot data."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import cycle

import numpy as np
from scipy import stats
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, joinedload

from xngin.apiserver.routers.common_api_types import (
    ArmAnalysis,
    DesignSpecMetricRequest,
    FreqExperimentAnalysisResponse,
    MetricAnalysis,
)
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter

VALID_SNAPSHOT_FIELDS = ["num_missing_values", "estimate", "std_error", "p_value", "t_stat"]


@dataclass(frozen=True)
class HistoricalSnapshotProfile:
    """Configures a deterministic fake historical trend for a developer experiment."""

    seed: int
    snapshot_count: int
    start_days_ago: int
    participant_start: int
    participant_step: int
    treatment_effect_start: float
    treatment_effect_end: float
    baseline_scale: float = 0.12
    std_error_start: float = 0.72
    std_error_end: float = 0.31


# uv run xngin-cli snapshots create-fake --dsn="$DATABASE_URL" --exp-id="$EXPERIMENT_ID" --n=10 --random-seed=101
STEADY_GAIN = HistoricalSnapshotProfile(
    seed=101,
    snapshot_count=10,
    start_days_ago=9,
    participant_start=440,
    participant_step=65,
    treatment_effect_start=0.04,
    treatment_effect_end=0.32,
)

# uv run xngin-cli snapshots create-fake --dsn="$DATABASE_URL" --exp-id="$EXPERIMENT_ID" --n=10 --random-seed=202
LATE_BREAKOUT = HistoricalSnapshotProfile(
    seed=202,
    snapshot_count=10,
    start_days_ago=9,
    participant_start=360,
    participant_step=95,
    treatment_effect_start=-0.04,
    treatment_effect_end=0.62,
    baseline_scale=0.1,
    std_error_start=0.95,
    std_error_end=0.22,
)


def recalculate_t_and_p(arm_analysis: ArmAnalysis, df: int = 100) -> None:
    """Recalculate t-stat and p-value after estimate/std-error changes."""
    if arm_analysis.std_error is not None and arm_analysis.std_error != 0:
        arm_analysis.t_stat = arm_analysis.estimate / arm_analysis.std_error
        arm_analysis.p_value = float(2 * (1 - stats.t.cdf(abs(arm_analysis.t_stat), df=df)))
    elif arm_analysis.estimate > 0:
        arm_analysis.t_stat = float("inf")
        arm_analysis.p_value = 0.0
    elif arm_analysis.estimate < 0:
        arm_analysis.t_stat = float("-inf")
        arm_analysis.p_value = 0.0
    else:
        arm_analysis.t_stat = None
        arm_analysis.p_value = None


def validate_freq_experiment(experiment: tables.Experiment) -> None:
    """Raise if the experiment cannot hold frequentist snapshot analysis data."""
    if experiment.experiment_type not in {ExperimentsType.FREQ_ONLINE, ExperimentsType.FREQ_PREASSIGNED}:
        raise ValueError(f"Experiment type must be freq_online or freq_preassigned, got {experiment.experiment_type}")

    design_spec_fields = ExperimentStorageConverter(experiment).get_design_spec_fields()
    if not design_spec_fields.metrics:
        raise ValueError("Experiment has no metrics defined")
    if not experiment.arms:
        raise ValueError("Experiment has no arms defined")


def get_metric_names(experiment: tables.Experiment) -> list[str]:
    """Return metric names defined on a frequentist experiment."""
    return [m.field_name for m in ExperimentStorageConverter(experiment).get_design_spec_fields().metrics or []]


def get_arm_ids(experiment: tables.Experiment) -> list[str]:
    """Return all arm ids defined on an experiment."""
    return [arm.id for arm in experiment.arms]


def _get_progress(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return index / (count - 1)


def _interpolate(start: float, end: float, progress: float) -> float:
    return start + ((end - start) * progress)


def _get_baseline_arm_id(experiment: tables.Experiment) -> str:
    baseline_arm = next(arm for arm in experiment.arms if arm.position == 1)
    return baseline_arm.id


def _make_arm_analysis(
    arm: tables.Arm,
    *,
    is_baseline: bool,
    estimate: float,
    std_error: float,
    num_missing_values: int,
    df: int = 100,
) -> ArmAnalysis:
    arm_analysis = ArmAnalysis(
        arm_id=arm.id,
        arm_name=arm.name,
        arm_description=arm.description,
        is_baseline=is_baseline,
        num_missing_values=num_missing_values,
        estimate=estimate,
        std_error=std_error,
        t_stat=None,
        p_value=None,
    )
    recalculate_t_and_p(arm_analysis, df=df)
    return arm_analysis


def create_freq_experiment_analysis(
    experiment: tables.Experiment,
    created_at: datetime,
    *,
    metric_name: str | None = None,
    arm_id: str | None = None,
    field: str | None = None,
    value: float | None = None,
    num_participants: int | None = None,
    num_missing_participants: int | None = None,
    treatment_effect: float | None = None,
    baseline_scale: float = 0.15,
    std_error: float | None = None,
    rng: np.random.Generator | None = None,
) -> FreqExperimentAnalysisResponse:
    """Create one fake frequentist analysis payload."""
    rng = rng or np.random.default_rng()
    validate_freq_experiment(experiment)

    metric_analyses = []
    baseline_arm_id = _get_baseline_arm_id(experiment)

    for metric_index, metric_obj in enumerate(
        ExperimentStorageConverter(experiment).get_design_spec_fields().metrics or []
    ):
        metric_field_name = metric_obj.field_name
        arm_analyses = []

        for arm_index, arm in enumerate(experiment.arms):
            is_baseline = arm.id == baseline_arm_id
            metric_offset = metric_index * 0.015
            baseline_estimate = float(rng.normal(0, baseline_scale)) + metric_offset
            treatment_estimate = (
                treatment_effect if treatment_effect is not None else float(rng.normal(0.18 + metric_offset, 0.11))
            )
            estimate = baseline_estimate if is_baseline else treatment_estimate + float(rng.normal(0, 0.035))
            derived_std_error = max(
                0.05,
                std_error if std_error is not None else abs(float(rng.normal(0.6, 0.08))),
            )
            missing_values = max(0, int(rng.normal(2 + metric_index + arm_index, 1.2)))

            arm_analysis = _make_arm_analysis(
                arm,
                is_baseline=is_baseline,
                estimate=estimate,
                std_error=derived_std_error,
                num_missing_values=missing_values,
            )

            should_apply = (
                (metric_name is None or metric_name == metric_field_name)
                and (arm_id is None or arm_id == arm.id)
                and field is not None
                and value is not None
            )
            if should_apply:
                assert field is not None
                setattr(arm_analysis, field, value)
                if field in {"estimate", "std_error"}:
                    recalculate_t_and_p(arm_analysis)

            arm_analyses.append(arm_analysis)

        metric_analyses.append(
            MetricAnalysis(
                metric_name=metric_field_name,
                metric=DesignSpecMetricRequest(
                    field_name=metric_field_name,
                    metric_target=metric_obj.metric_target,
                    metric_pct_change=metric_obj.metric_pct_change,
                ),
                arm_analyses=arm_analyses,
            )
        )

    return FreqExperimentAnalysisResponse(
        experiment_id=experiment.id,
        metric_analyses=metric_analyses,
        num_participants=num_participants if num_participants is not None else int(rng.integers(500, 2001)),
        num_missing_participants=(
            num_missing_participants if num_missing_participants is not None else int(rng.integers(0, 51))
        ),
        created_at=created_at,
    )


def build_historical_snapshot_payloads(
    experiment: tables.Experiment,
    profile: HistoricalSnapshotProfile,
    *,
    end_at: datetime | None = None,
) -> list[FreqExperimentAnalysisResponse]:
    """Build a deterministic series of historical snapshots for a dev experiment."""
    validate_freq_experiment(experiment)

    rng = np.random.default_rng(profile.seed)
    end_at = end_at or datetime.now(UTC)
    start_at = end_at - timedelta(days=profile.start_days_ago)
    payloads: list[FreqExperimentAnalysisResponse] = []

    for snapshot_index in range(profile.snapshot_count):
        progress = _get_progress(snapshot_index, profile.snapshot_count)
        created_at = start_at + timedelta(days=snapshot_index)
        num_participants = profile.participant_start + round(profile.participant_step * snapshot_index)
        missing_participants = max(0, round((profile.snapshot_count - snapshot_index - 1) * 0.9) + (snapshot_index % 2))
        treatment_effect = _interpolate(
            profile.treatment_effect_start,
            profile.treatment_effect_end,
            progress,
        ) + float(rng.normal(0, 0.018))
        std_error = max(
            0.05,
            _interpolate(profile.std_error_start, profile.std_error_end, progress) + float(rng.normal(0, 0.02)),
        )

        payloads.append(
            create_freq_experiment_analysis(
                experiment,
                created_at,
                num_participants=num_participants,
                num_missing_participants=min(missing_participants, num_participants // 8),
                treatment_effect=treatment_effect,
                baseline_scale=profile.baseline_scale,
                std_error=std_error,
                rng=rng,
            )
        )

    return payloads


async def seed_historical_snapshots(
    session: AsyncSession,
    experiment: tables.Experiment,
    profile: HistoricalSnapshotProfile,
) -> int:
    """Insert fake historical snapshots for one experiment if none exist yet."""
    existing_snapshot_id = await session.scalar(
        select(tables.Snapshot.id).where(tables.Snapshot.experiment_id == experiment.id).limit(1)
    )
    if existing_snapshot_id is not None:
        return 0

    # Refresh relationships here because bootstrap has only loaded the Experiment row, not its arms.
    await session.refresh(experiment, ["arms"])
    payloads = build_historical_snapshot_payloads(experiment, profile)
    snapshots = [
        tables.Snapshot(
            experiment_id=experiment.id,
            created_at=payload.created_at,
            updated_at=payload.created_at,
            status="success",
            data=payload.model_dump(mode="json"),
        )
        for payload in payloads
    ]
    session.add_all(snapshots)
    return len(snapshots)


def get_freq_experiment_for_cli(session: Session, exp_id: str) -> tables.Experiment:
    """Load an experiment and its arms for CLI snapshot operations."""
    stmt = select(tables.Experiment).where(tables.Experiment.id == exp_id).options(joinedload(tables.Experiment.arms))
    experiment = session.scalars(stmt).unique().one_or_none()
    if experiment is None:
        raise ValueError(f"Experiment {exp_id} not found")
    validate_freq_experiment(experiment)
    return experiment


def create_fake_snapshots(
    session: Session,
    experiment: tables.Experiment,
    *,
    start_date: datetime | None = None,
    n: int = 1,
    arm_id: str | None = None,
    metric_name: str | None = None,
    field: str | None = None,
    values: list[float] | None = None,
    random_seed: int | None = None,
) -> list[tables.Snapshot]:
    """Insert fake snapshots for manual CLI use."""
    base_date = start_date or datetime.now(UTC)
    value_cycle = cycle(values) if values else None
    rng = np.random.default_rng(random_seed)
    snapshots = []

    for index in range(n):
        snapshot_date = base_date + timedelta(days=index)
        value = next(value_cycle) if value_cycle else None
        snapshot_data = create_freq_experiment_analysis(
            experiment=experiment,
            created_at=snapshot_date,
            metric_name=metric_name,
            arm_id=arm_id,
            field=field,
            value=value,
            rng=rng,
        )
        snapshot = tables.Snapshot(
            experiment_id=experiment.id,
            created_at=snapshot_date,
            updated_at=snapshot_date,
            status="success",
            data=snapshot_data.model_dump(mode="json"),
        )
        session.add(snapshot)
        snapshots.append(snapshot)

    return snapshots
