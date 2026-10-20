"""Helpers shared by apiserver tests: test database URIs, datasource metadata, and API response assertions.

Fixtures live in src/xngin/conftest.py. Keep this an ordinary module (not a conftest) so that importing it never
registers fixtures or hooks a second time.
"""

import base64
import dataclasses
import enum
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
from xngin.apiserver.exceptionhandlers import XHTTPValidationError
from xngin.apiserver.routers.admin import admin_api_types as aapi
from xngin.apiserver.settings import Dsn
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClientNotDefaultStatusError
from xngin.apiserver.testing.admin_integrations_api_client import (
    AdminIntegrationsAPIClientNotDefaultStatusError,
)
from xngin.apiserver.testing.integrations_api_client import IntegrationsAPIClientNotDefaultStatusError
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


@dataclass
class StatusCodeMatcher:
    exc: (
        AdminAPIClientNotDefaultStatusError
        | AdminIntegrationsAPIClientNotDefaultStatusError
        | IntegrationsAPIClientNotDefaultStatusError
        | None
    ) = None

    def http_response(self):
        """Returns the httpx2 Response."""
        if self.exc is None:
            raise AssertionError("StatusCodeMatcher has no captured response yet.")
        return self.exc.result.response

    def _detail_messages(self) -> list[str]:
        response = self.http_response()
        parsed_error = XHTTPValidationError.model_validate(response.json())
        return [detail.msg for detail in parsed_error.detail]

    def _has_text(self, text: str) -> bool:
        response = self.http_response()
        body_text = response.text if response.text is not None else str(response.content)
        return text in body_text

    def _has_message(self, msg: str, *, contains: bool = False) -> bool:
        response = self.http_response()
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError(f"Expected JSON object response body, got {type(body).__name__}.")
        actual_msg = body.get("message")
        if not isinstance(actual_msg, str):
            raise TypeError(f'Expected JSON object with string "message", got: {body!r}')
        actual_msg = actual_msg.lower()
        expected_msg = msg.lower()
        return expected_msg in actual_msg if contains else actual_msg == expected_msg


@contextmanager
def expect_status_code(
    status_code: int,
    *,
    message_contains: str | None = None,
    message_eq: str | None = None,
    detail_contains: str | None = None,
    detail_eq: str | None = None,
    text: str | None = None,
) -> Iterator[StatusCodeMatcher]:
    """Like pytest.raises(), but for checking the non-default response codes of an AdminAPIClient request."""
    match = StatusCodeMatcher()
    with pytest.raises((
        AdminAPIClientNotDefaultStatusError,
        AdminIntegrationsAPIClientNotDefaultStatusError,
        IntegrationsAPIClientNotDefaultStatusError,
    )) as exc:
        yield match
    match.exc = cast(
        "AdminAPIClientNotDefaultStatusError"
        " | AdminIntegrationsAPIClientNotDefaultStatusError"
        " | IntegrationsAPIClientNotDefaultStatusError",
        exc.value,
    )
    http_response = match.http_response()
    assert http_response.status_code == status_code, (
        f"Expected '{status_code}' response code but got {http_response.status_code}: {http_response.content}"
    )
    if message_eq is not None:
        assert match._has_message(message_eq), (
            f"Expected '{message_eq}' to be equal to the .message field: {http_response.content}"
        )
    if message_contains is not None:
        assert match._has_message(message_contains, contains=True), (
            f"Expected '{message_contains}' to be a substring of the .message field: {http_response.content}"
        )
    if detail_eq is not None:
        assert any(msg == detail_eq for msg in match._detail_messages()), (
            f"Expected '{detail_eq}' to be one of the .msg fields in the response: {http_response.content}"
        )
    if detail_contains is not None:
        messages = match._detail_messages()
        assert any(detail_contains in msg for msg in messages), (
            f"Expected '{detail_contains}' to be in one of the .msg fields in the response: {http_response.content}"
            f"\nmsg list (length {len(messages)}): \n\t{'\n\t'.join(messages)}"
        )
    if text is not None:
        assert match._has_text(text), f"Expected '{text}' to be in the response: {http_response.content}"
