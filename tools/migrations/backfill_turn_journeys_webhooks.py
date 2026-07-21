"""Creates a "turn.journeys_changed" webhook for every organization that has a
Turn connection but is missing the webhook.

The webhook (with its id and auth_token) is now created automatically whenever a
Turn connection is set, but organizations that connected Turn before that logic
existed have a turn_connections row and no webhook. This backfills them.

Creating the webhook does NOT require the Turn API token — only the
organization id — so no secrets keyset needs to be configured to run this.

Usage:
    uv run python tools/migrations/backfill_turn_journeys_webhooks.py --dry-run \
        "postgresql+psycopg://postgres:postgres@localhost:5499/xngin?sslmode=disable"
"""

from typing import Annotated

import typer
from sqlalchemy import create_engine, exists, select
from sqlalchemy.orm import Session

from xngin.apiserver.database import generic_url_to_sa_url
from xngin.apiserver.routers.admin import admin_common
from xngin.apiserver.routers.admin.admin_api_types import AddTurnJourneysChangedWebhookRequest
from xngin.apiserver.sqla import tables

TURN_WEBHOOK_NAME = "Turn Journeys Changed Webhook"

app = typer.Typer()


def _migrate(dsn: str, dry_run: bool) -> None:
    engine = create_engine(generic_url_to_sa_url(dsn), echo=dry_run)

    try:
        with Session(engine) as session:
            stmt = select(tables.TurnConnection).where(
                ~exists(
                    select(tables.Webhook.id).where(
                        tables.Webhook.organization_id == tables.TurnConnection.organization_id,
                        tables.Webhook.type == "turn.journeys_changed",
                    )
                )
            )
            turn_connections = session.scalars(stmt).all()
            typer.echo(f"Found {len(turn_connections)} Turn connection(s) missing a journeys-changed webhook.")

            for tc in turn_connections:
                _, wh = admin_common.create_webhook_impl(
                    session,
                    tc.organization_id,
                    AddTurnJourneysChangedWebhookRequest(name=TURN_WEBHOOK_NAME),
                )
                typer.echo(f"  [ADD] org={tc.organization_id!r} → webhook id={wh.id!r}")

            if dry_run:
                typer.echo("Dry run — rolling back.")
                session.rollback()
            else:
                session.commit()
                typer.echo(f"Done. {len(turn_connections)} webhook(s) added.")
    finally:
        engine.dispose()


@app.command()
def migrate(
    dsn: Annotated[str, typer.Argument(help="PostgreSQL DSN for the application database")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Backfill the turn.journeys_changed webhook for orgs with a Turn connection but no webhook."""
    _migrate(dsn, dry_run)


if __name__ == "__main__":
    app()
