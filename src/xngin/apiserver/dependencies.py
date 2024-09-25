from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from xngin.apiserver.database import SessionLocal
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.settings import get_settings_for_server, XnginSettings


def settings_dependency():
    """Provides the settings for the server.

    This may be overridden by tests using the FastAPI dependency override features.
    """
    return get_settings_for_server()


def config_dependency(
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
    config_id: Annotated[str | None, Header(example="testing")] = None,
):
    """Returns the configuration for the current request, as determined by the Config-ID HTTP request header."""
    if not config_id:
        return None
    return settings.get_client_config(config_id)


def xngin_db_session() -> Session:
    """Returns a database connection to the xngin sqlite database (not customer data warehouse)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def gsheet_cache(xngin_db: Annotated[Session, Depends(xngin_db_session)]):
    return GSheetCache(xngin_db)
