"""Tests of date and time columns through the admin API, against the stress test table in the local testing warehouse.

The warehouse session's time zone must not change which instants a filter matches, so the tests also run as a role
whose sessions are in Asia/Kathmandu (UTC+05:45).
"""

from collections.abc import Iterator

import pytest
import sqlalchemy
from sqlalchemy import text

from xngin.apiserver import flags
from xngin.apiserver.settings import Dsn
from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.dwh_time_types_helpers import check_time_fields
from xngin.apiserver.testing.pg_helpers import with_name_suffix
from xngin.apiserver.testing.stress_loaders import pg_quote, postgres_stress_schema

SESSION_TIME_ZONE = "Asia/Kathmandu"


@pytest.fixture(name="stress_schema", scope="module")
def fixture_stress_schema() -> Iterator[str]:
    with postgres_stress_schema("stress_time") as schema:
        yield schema


@pytest.fixture(name="kathmandu_dsn", scope="module")
def fixture_kathmandu_dsn(stress_schema: str) -> Iterator[Dsn]:
    """A Dsn for a role that may read stress_schema, and whose sessions are in SESSION_TIME_ZONE."""
    role, password = with_name_suffix("stress_time_kathmandu"), "kathmandu"
    admin = sqlalchemy.create_engine(Dsn.from_url(flags.XNGIN_DEVDWH_DSN).to_sqlalchemy_url())
    statements = [
        f"CREATE ROLE {pg_quote(role)} LOGIN PASSWORD '{password}'",
        f"ALTER ROLE {pg_quote(role)} SET timezone = '{SESSION_TIME_ZONE}'",
        f"GRANT USAGE ON SCHEMA {pg_quote(stress_schema)} TO {pg_quote(role)}",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA {pg_quote(stress_schema)} TO {pg_quote(role)}",
    ]
    try:
        with admin.begin() as conn:
            conn.execute(text(f"DROP ROLE IF EXISTS {pg_quote(role)}"))
            for statement in statements:
                conn.execute(text(statement))
        dsn: Dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN)
        yield dsn.model_copy(update={"user": role, "password": password, "search_path": pg_quote(stress_schema)})
    finally:
        with admin.begin() as conn:
            conn.execute(text(f"DROP OWNED BY {pg_quote(role)}"))
            conn.execute(text(f"DROP ROLE IF EXISTS {pg_quote(role)}"))
        admin.dispose()


def test_postgres_date_and_time_fields(aclient: AdminAPIClient, stress_schema: str):
    dsn: Dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN)
    check_time_fields(aclient, dsn.model_copy(update={"search_path": pg_quote(stress_schema)}), stress_dwh_def.POSTGRES)


def test_postgres_date_and_time_fields_in_a_session_time_zone_other_than_utc(
    aclient: AdminAPIClient, kathmandu_dsn: Dsn
):
    check_time_fields(aclient, kathmandu_dsn, stress_dwh_def.POSTGRES)
