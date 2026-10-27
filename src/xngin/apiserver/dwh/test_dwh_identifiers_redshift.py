"""Tests of the table, schema and column names a data source can use, through the admin API, against a real
Redshift.

They are marked integration and need XNGIN_DEVDWH_AWS_REDSHIFT_DSN: a Redshift user that may create schemas, with
sslmode=verify-full.
"""

import os
from collections.abc import Iterator

import pytest

from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.settings import Dsn
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.dwh_identifiers_helpers import (
    PG_SCHEMAS,
    PG_TABLE_NAMES,
    exercise_through_api,
    expect_unusable_table_name,
    pg_identifier_schemas,
    pg_quote,
    with_name_suffix,
)

# These tests run with case-sensitive identifiers off, Redshift's default: it stores and lists every name in lowercase,
# so they read the lowercase names, and check that a mixed-case spelling still resolves.

REDSHIFT_DSN_VAR = "XNGIN_DEVDWH_AWS_REDSHIFT_DSN"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get(REDSHIFT_DSN_VAR), reason=f"{REDSHIFT_DSN_VAR} is required for Redshift integration tests"
    ),
    pytest.mark.usefixtures("redshift_schemas", "real_dns"),
]


def _redshift_dsn(schema: str) -> Dsn:
    """The test Redshift, with a search_path naming schema as Redshift stores it: in lowercase."""
    dsn: Dsn = Dsn.from_url(os.environ[REDSHIFT_DSN_VAR])
    return dsn.model_copy(update={"search_path": pg_quote(with_name_suffix(schema).lower())})


@pytest.fixture(name="redshift_schemas", scope="module")
def fixture_redshift_schemas() -> Iterator[None]:
    with pg_identifier_schemas(Dsn.from_url(os.environ[REDSHIFT_DSN_VAR])):
        yield


@pytest.fixture(name="real_dns")
def fixture_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves the Redshift host for real; the test suite otherwise sends every connection to 127.0.0.1."""
    monkeypatch.setattr(safe_resolve, "INTERCEPT_DNS_FOR_TESTING", False)


@pytest.mark.parametrize("schema", PG_SCHEMAS)
@pytest.mark.parametrize("table_name", PG_TABLE_NAMES)
def test_redshift_reads_tables_whose_names_need_quoting(aclient: AdminAPIClient, schema: str, table_name: str):
    exercise_through_api(aclient, _redshift_dsn(schema), table_name.lower())


def test_redshift_resolves_a_mixed_case_name_to_its_lowercase_table(aclient: AdminAPIClient):
    exercise_through_api(aclient, _redshift_dsn("dwh_tn_plain"), "MixedCase", listed_as="mixedcase")


@pytest.mark.parametrize("table_name", ["public.plain_t", "dwh_tn_other.elsewhere"])
def test_redshift_dotted_names_do_not_reach_other_schemas(aclient: AdminAPIClient, table_name: str):
    table_name = table_name.replace("dwh_tn_other", with_name_suffix("dwh_tn_other").lower())
    expect_unusable_table_name(aclient, _redshift_dsn("dwh_tn_plain"), table_name, "does not exist")
