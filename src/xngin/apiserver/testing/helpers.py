"""Helpers shared by apiserver tests: test database URIs, datasource metadata, and API response assertions.

Fixtures live in src/xngin/conftest.py. Keep this an ordinary module (not a conftest) so that importing it never
registers fixtures or hooks a second time.
"""

import base64
import dataclasses
import enum
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never, cast

import pytest
import sqlalchemy
from sqlalchemy import delete, make_url
from sqlalchemy.dialects.postgresql import psycopg
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import Session
from sqlalchemy_bigquery import dialect as bigquery_dialect

from xngin.apiserver import settings
from xngin.apiserver.dwh import dwh_utils
from xngin.apiserver.routers.admin import admin_api_types as aapi
from xngin.apiserver.settings import Dsn
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.pg_helpers import with_database_suffix


class RowProtocolMixin:
    @property
    def _mapping(self) -> Mapping[str, Any]:
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)  # type: ignore[unreachable]
        raise RuntimeError("RowProtocolMixin is only defined for use with dataclasses.")


class DbType(enum.StrEnum):
    RS = "redshift"
    PG = "postgres"
    BQ = "bigquery"

    def dialect(self) -> Dialect:
        """Returns the SQLAlchemy dialect most appropriate for this DbType."""
        match self:
            case DbType.RS:
                return sqlalchemy.dialects.postgresql.psycopg2.dialect()
            case DbType.PG:
                return psycopg.dialect()
            case DbType.BQ:
                # No type hints so cast
                # https://github.com/googleapis/python-bigquery-sqlalchemy/issues/355
                return cast(Dialect, bigquery_dialect())
        assert_never(self)


@dataclass
class TestUriInfo:
    """Holds information about a test database URI."""

    connect_url: URL
    db_type: DbType

    def __str__(self):
        """Display-safe string representation of the URL."""

        safe_url = self.connect_url.set(password=None)
        if "password" in safe_url.query:
            safe_url = safe_url.update_query_dict({"password": "***"})
        if "credentials_base64" in safe_url.query:
            safe_url = safe_url.update_query_dict({"credentials_base64": "REDACTED"})
        return f"{safe_url} (detected type: {self.db_type})"


def get_queries_test_uri() -> TestUriInfo:
    """Gets the DSN of the testing data warehouse to use in tests.

    A Postgres database name is suffixed to be unique to this test run and worker; see pg_helpers.

    See xngin.apiserver.dwh.test_queries.fixture_queries_dwh_session.
    """
    connection_uri = os.environ.get("XNGIN_QUERIES_TEST_URI", "")
    if not connection_uri:
        raise ValueError("XNGIN_QUERIES_TEST_URI must be set.")
    info = get_test_uri_info(connection_uri)
    if info.db_type == DbType.PG:
        info = get_test_uri_info(with_database_suffix(connection_uri))
    return info


def get_test_uri_info(connection_uri: str) -> TestUriInfo:
    """Returns a TestUriInfo dataclass about a test database given its connection_uri."""
    if not connection_uri:
        raise ValueError("connection_uri is empty")

    parsed_url = make_url(connection_uri)
    if dwh_utils.is_bigquery(parsed_url):
        dbtype = DbType.BQ
    elif dwh_utils.is_redshift(parsed_url):
        dbtype = DbType.RS
    elif dwh_utils.is_postgres(parsed_url):
        dbtype = DbType.PG
    else:
        raise ValueError(
            f"connection_uri is not recognized as a BigQuery, Redshift, or Postgres database: {connection_uri}"
        )

    return TestUriInfo(connect_url=parsed_url, db_type=dbtype)


def delete_seeded_users(xngin_session: Session):
    """Deletes users created by the xngin_session fixture."""
    xngin_session.execute(delete(tables.User))
    xngin_session.commit()
    xngin_session.reset()


@dataclass(slots=True)
class DatasourceMetadata:
    """Describes an ephemeral datasource, organization, and API key.

    Tests should prefer to use the API responses and avoid using the ORM entities. We return the ORM entities only for
    tests that are not yet migrated to use the API endpoints, or that cannot use them because they are testing a lower
    level intentionally.
    """

    org: tables.Organization
    ds: tables.Datasource
    api_org: aapi.GetOrganizationResponse
    api_ds: aapi.GetDatasourceResponse
    organization_id: str
    datasource_id: str

    # An API key suitable for use in the Authorization: header.
    key: str
    key_id: str


def convert_dwh_to_create_api_dsn(dwh: settings.Dwh) -> aapi.Dsn:
    """Converts a trusted settings DWH config into a create_datasource request payload with revealed credentials."""
    match dwh:
        case Dsn():
            if dwh.is_redshift():
                return aapi.RedshiftDsn(
                    dbname=dwh.dbname,
                    host=dwh.host,
                    password=aapi.RevealedStr(value=dwh.password),
                    port=dwh.port,
                    search_path=dwh.search_path,
                    user=dwh.user,
                )
            return aapi.PostgresDsn(
                dbname=dwh.dbname,
                host=dwh.host,
                password=aapi.RevealedStr(value=dwh.password),
                port=dwh.port,
                search_path=dwh.search_path,
                sslmode=dwh.sslmode,
                user=dwh.user,
            )
        case settings.BqDsn():
            match dwh.credentials:
                case settings.GcpServiceAccountInfo():
                    credentials_content = base64.standard_b64decode(dwh.credentials.content_base64).decode()
                case settings.GcpServiceAccountFile():
                    credentials_content = Path(dwh.credentials.path).read_text(encoding="utf-8")
                case _:
                    raise TypeError(f"Unsupported BigQuery credentials type: {type(dwh.credentials).__name__}")
            return aapi.BqDsn(
                project_id=dwh.project_id,
                dataset_id=dwh.dataset_id,
                credentials=aapi.GcpServiceAccount(content=credentials_content),
            )
        case _:
            raise TypeError(f"Unsupported DWH type for test datasource creation: {type(dwh).__name__}")


# The service account key file that BigQuery integration tests authenticate with.
GCP_CREDENTIALS_VAR = "XNGIN_DEVDWH_GCP_CREDENTIALS"

skip_unless_gcp_credentials = pytest.mark.skipif(
    not os.environ.get(GCP_CREDENTIALS_VAR), reason=f"{GCP_CREDENTIALS_VAR} is required for BigQuery integration tests"
)
