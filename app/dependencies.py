from typing import Annotated

import psycopg2

from app.settings import get_settings_for_server, XnginSettings

type Dwh = psycopg2.extensions.connection


def settings_dependency():
    """Provides the settings for the server.

    This may be overridden by tests using the FastAPI dependency override features.
    """
    return get_settings_for_server()


def dwh_dependency(settings: Annotated[XnginSettings, settings_dependency]):
    """Placeholder for the dependency on the data warehouse connection."""
    with psycopg2.connect(**settings.customer.dwh.model_dump()) as conn:
        yield conn
