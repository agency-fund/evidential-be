from typing import Annotated

import psycopg2
from fastapi import Depends, Header

from xngin.apiserver.settings import get_settings_for_server, XnginSettings

type Dwh = psycopg2.extensions.connection


def settings_dependency():
    """Provides the settings for the server.

    This may be overridden by tests using the FastAPI dependency override features.
    """
    return get_settings_for_server()


def config_dependency(
    settings: Annotated[XnginSettings, Depends(settings_dependency)],
    config_id: Annotated[str | None, Header()] = None,
):
    """Returns the configuration for the current request, as determined by the Config-ID HTTP request header."""
    if not config_id:
        return None
    return settings.get_client_config(config_id)
