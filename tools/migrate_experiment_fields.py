"""Migrates experiment.design_spec_fields JSONB data to new relational tables.

Fetches all frequentist experiments with design_spec_fields set, reconstructs each experiment's
DesignSpec, then re-saves it using set_design_spec_fields() so that only the new experiment_fields &
experiment_filters rows are written (the deprecated JSONB column is left untouched).

Usage:
    uv run python tools/migrate_experiment_fields.py --dry-run \
        "postgresql+psycopg://postgres:postgres@localhost:5499/xngin?sslmode=disable"
"""

import asyncio
from typing import Annotated

import typer
from pydantic import TypeAdapter
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, joinedload

from xngin.apiserver.database import generic_url_to_sa_url
from xngin.apiserver.routers.common_enums import ExperimentState, ExperimentsType
from xngin.apiserver.settings import DatasourceConfig
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter

app = typer.Typer()


async def _migrate(dsn: str, dry_run: bool) -> None:
    engine = create_engine(generic_url_to_sa_url(dsn), echo=dry_run)

    with Session(engine) as session:
        result = session.execute(
            select(tables.Experiment)
            .where(tables.Experiment.design_spec_fields.isnot(None))
            .where(tables.Experiment.state.notin_([ExperimentState.DESIGNING, ExperimentState.ABANDONED]))
            .where(
                tables.Experiment.experiment_type.in_([ExperimentsType.FREQ_ONLINE, ExperimentsType.FREQ_PREASSIGNED])
            )
            .options(
                joinedload(tables.Experiment.datasource).joinedload(tables.Datasource.organization),
            )
        )
        experiments = result.unique().scalars().all()

        typer.echo(f"Found {len(experiments)} eligible experiments.")

        migrated = 0
        errors = 0

        for exp in experiments:
            try:
                # Directly load the config to bypass get_config()'s decryption of the dsn which we don't want.
                ds_config = TypeAdapter(DatasourceConfig).validate_python(exp.datasource.config)
                participants_schema = ds_config.find_participants_or_none(exp.participant_type)
                if participants_schema is None:  # we'll have UNKNOWN data_types
                    typer.echo(
                        f"  [WARN] No participant_type={exp.participant_type!r} found for experiment {exp.id!r} "
                    )

                # Core conversion logic: Given the old pt and design spec, write out the new fields.
                exp.datasource_table = participants_schema.table_name if participants_schema else None
                converter = ExperimentStorageConverter(exp)
                design_spec = await converter.get_design_spec()
                converter.set_design_spec_fields(
                    design_spec,
                    participants_schema=participants_schema,
                    include_deprecated_design_spec_fields=False,
                )

                typer.echo(
                    f"  [OK] {exp.id} ({exp.datasource.organization.name!r} / {exp.datasource.name!r} / {exp.name!r})"
                )
                migrated += 1

            except Exception as e:
                session.expire(exp)
                typer.echo(f"  [ERROR] {exp.id}: {e}")
                errors += 1

        typer.echo(f"\nSummary: {migrated} migrated, {errors} error(s).")
        if dry_run:
            typer.echo("🔄 Dry run — rolling back.")
            session.rollback()
        else:
            session.commit()
            typer.echo("✅ Changes committed.")

    engine.dispose()


@app.command()
def migrate(
    dsn: Annotated[str, typer.Argument(help="PostgreSQL DSN for the application database")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without committing to the database")
    ] = False,
) -> None:
    """Migrate experiment design_spec_fields JSONB to the experiment_fields & experiment_filters tables."""
    asyncio.run(_migrate(dsn, dry_run))


if __name__ == "__main__":
    app()
