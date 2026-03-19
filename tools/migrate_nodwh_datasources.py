"""Adds a NoDwh datasource for every organization that doesn't already have one.

Usage:
    uv run python tools/migrate_nodwh_datasources.py --dry-run \
        "postgresql+psycopg://postgres:postgres@localhost:5499/xngin?sslmode=disable"
"""

from typing import Annotated

import typer
from sqlalchemy import create_engine, exists, select
from sqlalchemy.orm import Session

from xngin.apiserver.database import generic_url_to_sa_url
from xngin.apiserver.settings import NoDwh, RemoteDatabaseConfig
from xngin.apiserver.sqla import tables

DEFAULT_NO_DWH_SOURCE_NAME = "API Only"

app = typer.Typer()


def _migrate(dsn: str, dry_run: bool) -> None:
    engine = create_engine(generic_url_to_sa_url(dsn), echo=dry_run)

    try:
        with Session(engine) as session:
            # Find orgs that have no NoDwh datasource
            stmt = select(tables.Organization).where(
                ~exists(
                    select(tables.Datasource.id).where(
                        tables.Datasource.organization_id == tables.Organization.id,
                        tables.Datasource.config["dwh"]["driver"].as_string() == "none",
                    )
                )
            )
            orgs = session.scalars(stmt).all()
            typer.echo(f"Found {len(orgs)} organization(s) missing a NoDwh datasource.")

            nodwh_config = RemoteDatabaseConfig(participants=[], type="remote", dwh=NoDwh())

            for org in orgs:
                ds = tables.Datasource(
                    id=tables.datasource_id_factory(),
                    name=DEFAULT_NO_DWH_SOURCE_NAME,
                    organization=org,
                )
                ds.set_config(nodwh_config)
                session.add(ds)
                typer.echo(f"  [ADD] org={org.id!r} ({org.name!r}) → datasource id={ds.id!r}")

            if dry_run:
                typer.echo("Dry run — rolling back.")
                session.rollback()
            else:
                session.commit()
                typer.echo(f"Done. {len(orgs)} datasource(s) added.")
    finally:
        engine.dispose()


@app.command()
def migrate(
    dsn: Annotated[str, typer.Argument(help="PostgreSQL DSN for the application database")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Add a NoDwh datasource for every organization that doesn't already have one."""
    _migrate(dsn, dry_run)


if __name__ == "__main__":
    app()
