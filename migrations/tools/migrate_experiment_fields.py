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


async def _migrate(dsn: str, dry_run: bool, raise_on_error: bool) -> None:
    engine = create_engine(generic_url_to_sa_url(dsn), echo=dry_run)

    try:
        with Session(engine) as session:
            result = session.execute(
                select(tables.Experiment)
                .where(tables.Experiment.design_spec_fields.isnot(None))
                .where(tables.Experiment.state.notin_([ExperimentState.ABANDONED]))
                .where(
                    tables.Experiment.experiment_type.in_([
                        ExperimentsType.FREQ_ONLINE,
                        ExperimentsType.FREQ_PREASSIGNED,
                    ])
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
                    # If a user deleted a participant type that used to back an experiment,
                    # we'll just store UNKNOWN as the data_types for non-filter fields.
                    if participants_schema is None:
                        typer.echo(f"  [WARN] No participant_type={exp.participant_type!r} for experiment {exp.id!r}")

                    # Core conversion logic: Given the old pt and design spec, write out the new fields.
                    converter = ExperimentStorageConverter(exp)
                    design_spec = await converter.get_design_spec()
                    field_type_map = (
                        {field.field_name: field.data_type for field in participants_schema.fields}
                        if participants_schema
                        else {}
                    )
                    unique_id_name = participants_schema.get_unique_id_field() if participants_schema else None
                    converter.set_design_spec_fields(
                        design_spec,
                        field_type_map=field_type_map,
                        unique_id_name=unique_id_name,
                        set_deprecated_design_spec_fields=False,
                    )
                    # Finally set the datasource table name.
                    exp.datasource_table = participants_schema.table_name if participants_schema else None

                    typer.echo(
                        f"  [OK] {exp.id} "
                        f"({exp.datasource.organization.name!r} / {exp.datasource.name!r} / {exp.name!r})"
                    )
                    migrated += 1

                except Exception as e:
                    # An exception can occur if a filter exists with an UNKNOWN data_type due to the
                    # field missing in the participant type.
                    session.expunge(exp)
                    typer.echo(f"  [ERROR] {exp.id}: {e}")
                    errors += 1
                    if raise_on_error:
                        raise

            typer.echo(f"\nSummary: {migrated} migrated, {errors} error(s).")
            if dry_run:
                typer.echo("🔄 Dry run — rolling back.")
                session.rollback()
            else:
                session.commit()
                typer.echo("✅ Changes committed.")

    finally:
        engine.dispose()


@app.command()
def migrate(
    dsn: Annotated[str, typer.Argument(help="PostgreSQL DSN for the application database")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without committing to the database")
    ] = False,
    raise_error: Annotated[
        bool, typer.Option("--raise-error/--no-raise-error", help="Raise an error if any experiment fails to migrate")
    ] = True,
) -> None:
    """Migrate experiment design_spec_fields JSONB to the experiment_fields & experiment_filters tables."""
    asyncio.run(_migrate(dsn, dry_run, raise_error))


if __name__ == "__main__":
    app()
