"""
Fake snapshot generator for FreqExperimentAnalysisResponse.

This CLI tool helps inject fake Snapshot data into a PostgreSQL database for testing purposes.
It supports two main modes:
1. CREATE (-c): Insert new dummy Snapshot data with mostly default values
2. UPDATE (-u): Update existing Snapshots for a specific metric > arm > field with array of values

example usages:
    uv run python tools/snapshot_sim.py create \
        --dsn "postgresql+psycopg://postgres:postgres@localhost:5499/xngin?sslmode=disable" \
        --exp-id exp_TBM4J2KfOq5J3SMZ \
        -n 7

    # Note the use of -- below to distinguish negatives from flags in positional args.
    uv run python tools/snapshot_sim.py update \
        --dsn "postgresql+psycopg://postgres:postgres@localhost:5499/xngin?sslmode=disable" \
        --exp-id exp_TBM4J2KfOq5J3SMZ \
        --arm-id arm_am9zslDFJ940VO7p \
        --metric current_income \
        --field estimate \
        -n 7 \
        -- 0 3 5 10 4 1 -8
"""

import random
from datetime import UTC, datetime, timedelta
from itertools import cycle
from typing import Annotated

import typer
from scipy import stats
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, attributes, joinedload

from xngin.apiserver.routers.common_api_types import (
    ArmAnalysis,
    DesignSpecMetricRequest,
    FreqExperimentAnalysisResponse,
    MetricAnalysis,
)
from xngin.apiserver.routers.common_enums import ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter

app = typer.Typer()


def generate_default_arm_analysis(arm: tables.Arm, is_baseline: bool, rng: random.Random, df: int = 100) -> dict:
    """Generate default arm analysis with random noise."""
    estimate = rng.gauss(0, 1) if not is_baseline else rng.gauss(0, 1) * 10
    std_error = abs(rng.gauss(1, 0.2))
    t_stat = estimate / std_error if std_error != 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    return ArmAnalysis(
        arm_id=arm.id,
        arm_name=arm.name,
        arm_description=arm.description,
        is_baseline=is_baseline,
        num_missing_values=0,
        estimate=estimate,
        std_error=std_error,
        t_stat=t_stat,
        p_value=float(p_value),
    )


def create_freq_experiment_analysis(
    experiment: tables.Experiment,
    created_at: datetime,
    metric_name: str | None = None,
    arm_id: str | None = None,
    field: str | None = None,
    value: float | None = None,
    rng: random.Random | None = None,
) -> dict:
    """Create a FreqExperimentAnalysisResponse dict."""
    if rng is None:
        rng = random.Random()

    converter = ExperimentStorageConverter(experiment)
    design_spec_fields = converter.get_design_spec_fields()

    if not design_spec_fields.metrics:
        raise ValueError("❌ Experiment has no metrics defined")

    metric_analyses = []

    for metric_obj in design_spec_fields.metrics:
        metric_field_name = metric_obj.field_name
        arm_analyses = []

        # Make all the arm analyses for this metric
        for idx, arm in enumerate(experiment.arms):
            # We're assuming a t-distribution with 100 degrees of freedom, 2-sided test.
            arm_analysis = generate_default_arm_analysis(arm, idx == 0, rng, df=100)

            # Apply custom values if specified
            should_apply = (
                (metric_name is None or metric_name == metric_field_name)
                and (arm_id is None or arm_id == arm.id)
                and field is not None
                and value is not None
            )

            if should_apply:
                setattr(arm_analysis, field, value)
                # Recalculate dependent fields if needed
                if field in {"estimate", "std_error"} and arm_analysis.std_error != 0:
                    arm_analysis.t_stat = arm_analysis.estimate / arm_analysis.std_error
                    arm_analysis.p_value = float(2 * (1 - stats.t.cdf(abs(arm_analysis.t_stat), df=100)))

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
        num_participants=rng.randint(500, 2000),
        num_missing_participants=rng.randint(0, 50),
        created_at=created_at,
    )


@app.command()
def create(
    dsn: Annotated[str, typer.Option("--dsn", "-d", help="Database connection string")],
    exp_id: Annotated[str, typer.Option("--exp-id", "-e", help="Experiment ID")],
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", "-s", help="Start date (ISO format, default: now)"),
    ] = None,
    n: Annotated[int, typer.Option("--n", "-n", help="Number of snapshots to create, one per day from start_date")] = 1,
    arm_id: Annotated[str | None, typer.Option("--arm-id", "-a", help="Arm ID to apply values to")] = None,
    metric: Annotated[str | None, typer.Option("--metric", "-m", help="Metric name to apply values to")] = None,
    field: Annotated[
        str | None,
        typer.Option(
            "--field",
            "-f",
            help="Field name: num_missing_values, estimate, std_error, p_value, t_stat",
        ),
    ] = None,
    values: Annotated[list[float] | None, typer.Argument(help="Values to apply to the field")] = None,
    random_seed: Annotated[int | None, typer.Option("--random-seed", "-r", help="Random seed")] = None,
) -> None:
    """Create new fake snapshots for a frequentist experiment."""
    engine = create_engine(dsn)

    with Session(engine) as session:
        # Fetch experiment with arms
        stmt = (
            select(tables.Experiment).where(tables.Experiment.id == exp_id).options(joinedload(tables.Experiment.arms))
        )
        experiment = session.scalars(stmt).unique().one_or_none()

        if not experiment:
            typer.echo(f"❌ Experiment {exp_id} not found", err=True)
            raise typer.Exit(1)

        if experiment.experiment_type not in {ExperimentsType.FREQ_ONLINE, ExperimentsType.FREQ_PREASSIGNED}:
            typer.echo(
                f"❌ Experiment type must be freq_online or freq_preassigned, got {experiment.experiment_type}",
                err=True,
            )
            raise typer.Exit(1)

        # Validate metric if provided
        if metric:
            converter = ExperimentStorageConverter(experiment)
            design_spec_fields = converter.get_design_spec_fields()
            metric_names = [m.field_name for m in (design_spec_fields.metrics or [])]
            if metric not in metric_names:
                typer.echo(
                    f"❌ Metric '{metric}' not found in experiment. Available: {metric_names}",
                    err=True,
                )
                raise typer.Exit(1)

        # Validate arm_id if provided
        if arm_id:
            arm_ids = [arm.id for arm in experiment.arms]
            if arm_id not in arm_ids:
                typer.echo(
                    f"❌ Arm ID '{arm_id}' not found in experiment. Available: {arm_ids}",
                    err=True,
                )
                raise typer.Exit(1)

        # Validate field if provided
        if field:
            valid_fields = ["num_missing_values", "estimate", "std_error", "p_value", "t_stat"]
            if field not in valid_fields:
                typer.echo(
                    f"❌ Field '{field}' not valid. Must be one of: {valid_fields}",
                    err=True,
                )
                raise typer.Exit(1)

        # Parse start date
        base_date = datetime.now(UTC)
        if start_date:
            base_date = datetime.fromisoformat(start_date)
            if base_date.tzinfo is None:
                base_date = base_date.replace(tzinfo=UTC)

        # Create value cycle if provided
        value_cycle = cycle(values) if values else None

        # Create snapshots
        rng = random.Random(random_seed)
        for i in range(n):
            snapshot_date = base_date + timedelta(days=i)
            value = next(value_cycle) if value_cycle else None

            snapshot_data = create_freq_experiment_analysis(
                experiment=experiment,
                created_at=snapshot_date,
                metric_name=metric,
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
            typer.echo(f"Created snapshot at {snapshot_date.isoformat()}")

        session.commit()
        typer.echo(f"Successfully created {n} snapshots for experiment {exp_id}")


@app.command()
def update(
    dsn: Annotated[str, typer.Option("--dsn", "-d", help="Database connection string")],
    exp_id: Annotated[str, typer.Option("--exp-id", "-e", help="Experiment ID")],
    arm_id: Annotated[str, typer.Option("--arm-id", "-a", help="Arm ID (REQUIRED)")],
    metric: Annotated[str, typer.Option("--metric", "-m", help="Metric name (REQUIRED)")],
    field: Annotated[
        str,
        typer.Option(
            "--field",
            "-f",
            help="Field name: one of {num_missing_values, estimate, std_error, p_value, t_stat}."
            "Note: when the arm is not the baseline,'estimate' is an offset relative to the "
            "baseline! (REQUIRED)",
        ),
    ],
    values: Annotated[list[float], typer.Argument(help="Values to apply (REQUIRED)")],
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", "-s", help="Start date to filter snapshots (ISO format)"),
    ] = None,
    n: Annotated[int, typer.Option("--n", "-n", help="Maximum number of snapshots to update")] = 1,
    echo: Annotated[bool, typer.Option("--echo", "-e", help="Echo SQL queries")] = False,
) -> None:
    """Update existing snapshots with specific field values."""
    if not values:
        typer.echo("❌ values are REQUIRED for update mode", err=True)
        raise typer.Exit(1)

    engine = create_engine(dsn, echo=echo)

    with Session(engine) as session:
        # Fetch experiment
        stmt = select(tables.Experiment).where(tables.Experiment.id == exp_id)
        experiment = session.execute(stmt).scalar_one_or_none()

        if not experiment:
            typer.echo(f"❌ Experiment {exp_id} not found", err=True)
            raise typer.Exit(1)

        # Validate field
        valid_fields = ["num_missing_values", "estimate", "std_error", "p_value", "t_stat"]
        if field not in valid_fields:
            typer.echo(
                f"❌ Field '{field}' invalid. Must be one of: {valid_fields}",
                err=True,
            )
            raise typer.Exit(1)

        # Build query for snapshots
        snapshot_stmt = (
            select(tables.Snapshot)
            .where(tables.Snapshot.experiment_id == exp_id)
            .order_by(tables.Snapshot.created_at.asc())
            .limit(n)
        )
        if start_date:
            filter_date = datetime.fromisoformat(start_date)
            if filter_date.tzinfo is None:
                filter_date = filter_date.replace(tzinfo=UTC)
            snapshot_stmt = snapshot_stmt.where(tables.Snapshot.created_at >= filter_date)
        snapshots = list(session.execute(snapshot_stmt).scalars().all())

        if not snapshots:
            typer.echo("No snapshots found matching criteria", err=True)
            raise typer.Exit(1)

        # Update snapshots
        value_cycle = cycle(values)
        updated_count = 0

        for snapshot in snapshots:
            if snapshot.data is None:
                typer.echo(f"Warning: Snapshot {snapshot.id} has no data, skipping")
                continue

            value = next(value_cycle)

            # Find and update the specific metric and arm
            updated = False
            new_data = FreqExperimentAnalysisResponse.model_validate(snapshot.data)
            for metric_analysis in new_data.metric_analyses:
                if metric_analysis.metric_name == metric:
                    for arm_analysis in metric_analysis.arm_analyses:
                        if arm_analysis.arm_id == arm_id:
                            # Update the field
                            setattr(arm_analysis, field, value)

                            # Recalculate dependent fields if needed
                            if field in {"estimate", "std_error"}:
                                std_error = arm_analysis.std_error
                                if std_error != 0:
                                    arm_analysis.t_stat = arm_analysis.estimate / std_error
                                    arm_analysis.p_value = float(
                                        2 * (1 - stats.t.cdf(abs(arm_analysis.t_stat), df=100))
                                    )

                            updated = True
                            break
                if updated:
                    break

            if updated:
                # Tell SQLAlchemy to update the updated_at column with the same value.
                # We don't actually change updated_at since that is the timestamp used for grouping
                # and ordering the snapshots in the plots, and we assume the existing time reflects
                # what the user wants, e.g. you could have modified it manually in the db.
                attributes.flag_modified(snapshot, "updated_at")
                snapshot.data = new_data.model_dump(mode="json")
                updated_count += 1
                typer.echo(f"✅ {snapshot.id} (created at {snapshot.created_at.isoformat()}) with {field}={value}")
            else:
                typer.echo(f"⚠️ metric {metric} arm {arm_id} not found in {snapshot.id}")

        session.commit()
        typer.echo(f"➡️ Successfully updated {updated_count} snapshots")


if __name__ == "__main__":
    app()
