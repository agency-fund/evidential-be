"""Tests of the table, schema and column names a data source can use, through the admin API, against the local
Postgres testing warehouse.
"""

from collections.abc import Iterator

import pytest

from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.assertions import expect_status_code
from xngin.apiserver.testing.dwh_identifiers_helpers import (
    PG_OTHER_SCHEMA,
    PG_SCHEMAS,
    PG_TABLE_NAMES,
    create_datasource,
    exercise_through_api,
    expect_unusable_table_name,
    pg_dsn,
    pg_identifier_schemas,
    pg_quote,
    with_name_suffix,
)

pytestmark = pytest.mark.usefixtures("pg_schemas")


@pytest.fixture(name="pg_schemas", scope="module")
def fixture_pg_schemas() -> Iterator[None]:
    with pg_identifier_schemas():
        yield


@pytest.mark.parametrize("schema", PG_SCHEMAS)
@pytest.mark.parametrize("table_name", PG_TABLE_NAMES)
def test_postgres_reads_tables_whose_names_need_quoting(aclient: AdminAPIClient, schema: str, table_name: str):
    # The search_path is the customer's own value, so names that need quoting arrive quoted.
    exercise_through_api(aclient, pg_dsn("postgresql+psycopg", pg_quote(with_name_suffix(schema))), table_name)


def test_postgres_search_path_entries_after_the_first_are_searched(aclient: AdminAPIClient):
    search_path = f"public, {pg_quote(with_name_suffix('Mixed Schema-Name'))}"
    exercise_through_api(aclient, pg_dsn("postgresql+psycopg", search_path), "has space")


def test_postgres_search_path_is_the_boundary_for_unqualified_names(aclient: AdminAPIClient):
    """A table outside the search_path is neither listed nor found by its name in another schema."""
    datasource_id = create_datasource(aclient, pg_dsn("postgresql+psycopg", with_name_suffix("dwh_tn_plain")))
    assert "elsewhere" not in aclient.inspect_datasource(datasource_id=datasource_id).data.tables
    with expect_status_code(404, message_contains="The table 'elsewhere' does not exist"):
        aclient.inspect_table_in_datasource(datasource_id=datasource_id, table_name="elsewhere")


@pytest.mark.parametrize(
    "table_name",
    [
        f"{PG_OTHER_SCHEMA}.elsewhere",
        f"{pg_quote(PG_OTHER_SCHEMA)}.elsewhere",
        "dwh_tn_plain.plain_t",
        "public.anything",
    ],
)
def test_postgres_dotted_names_do_not_reach_other_schemas(aclient: AdminAPIClient, table_name: str):
    """A dot is part of the quoted table name, so schema.table names a table in the search_path, not in schema."""
    for schema in (PG_OTHER_SCHEMA, "dwh_tn_plain"):
        table_name = table_name.replace(schema, with_name_suffix(schema))
    dsn = pg_dsn("postgresql+psycopg", pg_quote(with_name_suffix("dwh_tn_plain")))
    expect_unusable_table_name(aclient, dsn, table_name, "does not exist")
