"""FastAPI dependencies that resolve integration API path parameters to the resources they name.

The counterpart to admin_dependencies for the public integration routes, and read the same way at the call
site: `ideps.turn_webhook` resolves the resource its route names and rejects a caller who may not have it.
What differs is the principal. These routes are called by Turn.io rather than by a signed-in user, so the
caller proves itself with the webhook's own token, or with an API key when the route names an experiment.

Requiring these dependencies means a route cannot reach a resource without proving access to it first, and
the checks stay in one place rather than in each handler's opening lines.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Path, status
from sqlalchemy.orm import Session

from xngin.apiserver import constants
from xngin.apiserver.dependencies import xngin_sync_db_session
from xngin.apiserver.routers.admin_integrations.admin_integrations_api import get_turn_webhook_or_raise
from xngin.apiserver.routers.experiments import experiments_dependencies as edeps
from xngin.apiserver.sqla import tables


def _authenticated(auth_token: str | None, wh: tables.Webhook | None) -> tables.Webhook:
    """Returns the webhook if the caller presented its token, and raises 401 otherwise."""
    if not auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing webhook auth token.")
    if not wh or auth_token != wh.auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook auth token.")
    return wh


def turn_webhook(
    webhook_id: Annotated[str, Path()],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    auth_token: Annotated[str | None, Header(alias=constants.HEADER_WEBHOOK_TOKEN)] = None,
) -> tables.Webhook:
    """Resolves the Turn.io webhook a route names, for a caller presenting its token.

    Requires {webhook_id} in the route path and the webhook's token in the Webhook-Token header.

    A webhook that does not exist is reported before the token is examined, which tells an unauthenticated
    caller whether a webhook id is in use. That is how these routes have always answered, and Turn.io relies
    on the 404 to distinguish a retired webhook from a rejected token.
    """
    wh = get_turn_webhook_or_raise(session, webhook_id=webhook_id, allow_missing=False)
    return _authenticated(auth_token, wh)


def turn_connection(
    wh: Annotated[tables.Webhook, Depends(turn_webhook)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> tables.TurnConnection:
    """Resolves the Turn.io connection owning the webhook a route names.

    Requires {webhook_id} in the route path and the webhook's token in the Webhook-Token header.
    """
    conn = session.get(tables.TurnConnection, wh.organization_id)
    if conn is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Turn.io connection configured for this organization."
        )
    return conn


def turn_config(
    exp: Annotated[tables.Experiment, Depends(edeps.experiment)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> tables.ExperimentTurnConfig:
    """Resolves the Turn.io journey mapping of the experiment a route names.

    Requires {experiment_id} in the route path and an API key authorized on its datasource.
    """
    cfg = session.get(tables.ExperimentTurnConfig, exp.id)
    if cfg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Turn.io mapping configured for this experiment."
        )
    return cfg
