"""conftest configures FastAPI dependency injection for testing and also does some setup before tests in
this module are run."""

import base64
import contextlib
import dataclasses
import enum
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, assert_never, cast

import pytest
import sqlalchemy
from sqlalchemy import delete, make_url
from sqlalchemy.dialects.postgresql import psycopg
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)
from sqlalchemy.orm import selectinload
from sqlalchemy_bigquery import dialect as bigquery_dialect
from starlette.testclient import TestClient

from xngin.apiserver import database, flags, settings
from xngin.apiserver.dependencies import (
    random_seed_dependency,
)
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.exceptionhandlers import XHTTPValidationError
from xngin.apiserver.main import app
from xngin.apiserver.routers.admin import admin_api_types as aapi
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.apiserver.routers.auth.auth_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.settings import (
    Dsn,
    ParticipantsDef,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing import (
    admin_api_client,
    experiments_api_client,
)
from xngin.apiserver.testing.admin_api_client import AdminAPIClientNotDefaultStatusError
from xngin.apiserver.testing.pg_helpers import create_database_if_not_exists_pg
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF
from xngin.db_extensions import custom_functions

# SQLAlchemy's logger will append this to the name of its loggers used for the application database; e.g.
# sqlalchemy.engine.Engine.xngin_app.
SA_LOGGER_NAME_FOR_APP = "xngin_app"


class RowProtocolMixin:
    @property
    def _mapping(self) -> Mapping[str, Any]:
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
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

    See xngin.apiserver.dwh.test_queries.fixture_queries_dwh_session.
    """
    connection_uri = os.environ.get("XNGIN_QUERIES_TEST_URI", "")
    if not connection_uri:
        raise ValueError("XNGIN_QUERIES_TEST_URI must be set.")
    return get_test_uri_info(connection_uri)


@pytest.fixture(scope="session", autouse=True)
def print_database_env_vars():
    """Prints debugging information sometimes useful for working with tests to stdout."""

    database_url = None
    with contextlib.suppress(ValueError):
        database_url = get_test_uri_info(database.get_server_database_url())

    queries_url = None
    with contextlib.suppress(ValueError):
        queries_url = get_queries_test_uri()

    dwh_url = None
    with contextlib.suppress(ValueError):
        dwh_url = get_test_uri_info(flags.XNGIN_DEVDWH_DSN)

    print(
        "Running tests with "
        f"\n\tDATABASE_URL: {database_url}"
        f"\n\tXNGIN_DEVDWH_DSN: {dwh_url}"
        f"\n\tXNGIN_QUERIES_TEST_URI: {queries_url}"
    )


def get_random_seed_for_test():
    """Returns a seed for testing."""
    return 42


@pytest.fixture(scope="session", autouse=True)
def allow_connecting_to_private_ips():
    safe_resolve.ALLOW_CONNECTING_TO_PRIVATE_IPS = True


@pytest.fixture
def disable_safe_resolve_check():
    prev = flags.DISABLE_SAFEDNS_CHECK
    flags.DISABLE_SAFEDNS_CHECK = True
    try:
        yield
    finally:
        flags.DISABLE_SAFEDNS_CHECK = prev


def get_test_uri_info(connection_uri: str) -> TestUriInfo:
    """Returns a TestUriInfo dataclass about a test database given its connection_uri."""
    if connection_uri.startswith("bigquery"):
        dbtype = DbType.BQ
    elif "redshift.amazonaws.com" in connection_uri:
        dbtype = DbType.RS
    elif connection_uri.startswith("postgres"):
        dbtype = DbType.PG
    else:
        raise ValueError(
            f"connection_uri is not recognized as a BigQuery, Redshift, or Postgres database: {connection_uri}"
        )
    return TestUriInfo(connect_url=make_url(connection_uri), db_type=dbtype)


@pytest.fixture(scope="session", autouse=True)
def fixture_override_app_dependencies():
    """Configures FastAPI dependencies for testing.

    This uses FastAPI's dependency override mechanism: https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
    """

    app.dependency_overrides[random_seed_dependency] = get_random_seed_for_test

    auth_dependencies.disable(app)
    auth_dependencies.enable_testing_tokens()


@pytest.fixture(name="client")
def fixture_client(xngin_session):
    """Returns a FastAPI TestClient.

    TestClient manages the lifecycle of the app and will invoke the FastAPI app and router @lifespan methods.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(name="eclient")
def fixture_experiments_api_client(xngin_session):
    """Returns a generated API client for the Integration API.

    The generated client uses TestClient under the hood. TestClient manages the lifecycle of the app and will invoke
    the FastAPI app and router @lifespan methods.
    """
    with experiments_api_client.ExperimentsAPIClient.from_app(app) as eapi_client:
        yield eapi_client


@pytest.fixture(name="aclient")
def fixture_admin_api_client(xngin_session):
    """Returns a generated API client for privileged Admin API requests."""
    with TestClient(app, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}) as client:
        yield admin_api_client.AdminAPIClient(client)


@pytest.fixture(name="aclient_unpriv")
def fixture_admin_api_client_unpriv(xngin_session):
    """Returns a generated API client for unprivileged Admin API requests."""
    with TestClient(app, headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING}"}) as client:
        yield admin_api_client.AdminAPIClient(client)


@pytest.fixture(scope="session")
async def fixture_initialize_xngin_db_schema():
    """Create the application schema once for the test session."""
    async with database.setup():
        create_database_if_not_exists_pg(database.get_sqlalchemy_database_url())
        async with database.get_async_engine().begin() as conn:
            await conn.run_sync(tables.Base.metadata.create_all)


@pytest.fixture(name="xngin_session")
async def fixture_xngin_db_session(fixture_initialize_xngin_db_schema):
    """Yields a SQLAlchemy session suitable for direct interaction with the database.

    This will delete all rows from the application tables at the beginning of every test. The users table will be seeded
    with
    a privileged user (pget, ppost, ...) and an unprivileged user (uget, upost, ...). These users can be removed by
    individual tests by calling delete_seeded_users().

    Where possible, prefer using the API methods to test functionality rather than touching the database
    directly.
    """
    async with database.setup():
        async with database.get_async_engine().begin() as conn:
            for table in reversed(tables.Base.metadata.sorted_tables):
                await conn.execute(sqlalchemy.delete(table))
        async with database.async_session() as session:
            session.add_all([
                tables.User(email=PRIVILEGED_EMAIL, is_privileged=True),
                tables.User(email=UNPRIVILEGED_EMAIL, is_privileged=False),
            ])
            await session.commit()
        async with database.async_session() as sess:
            try:
                yield sess
            finally:
                await sess.close()


async def delete_seeded_users(xngin_session: AsyncSession):
    """Deletes users created by the xngin_session fixture."""
    await xngin_session.execute(delete(tables.User))
    await xngin_session.commit()
    await xngin_session.reset()


@pytest.fixture(name="use_deterministic_random")
def fixture_use_deterministic_random():
    """Tests that want deterministic SQL random() behavior can request this fixture. This will only affect
    SQLAlchemy expressions that use custom_functions.our_random().
    """
    original = custom_functions.USE_DETERMINISTIC_RANDOM
    try:
        custom_functions.USE_DETERMINISTIC_RANDOM = True
        yield
    finally:
        custom_functions.USE_DETERMINISTIC_RANDOM = original


@dataclass(slots=True)
class DatasourceMetadata:
    """Describes an ephemeral datasource, organization, and API key."""

    org: tables.Organization
    ds: tables.Datasource
    api_org: aapi.GetOrganizationResponse
    api_ds: aapi.GetDatasourceResponse
    organization_id: str
    datasource_id: str
    pt: ParticipantsDef

    # The SQLAlchemy DSN
    dsn: str

    # An API key suitable for use in the Authorization: header.
    key: str
    key_id: str


@pytest.fixture(name="testing_datasource")
async def fixture_testing_datasource(
    xngin_session: AsyncSession, aclient: admin_api_client.AdminAPIClient
) -> DatasourceMetadata:
    """Creates a datasource fixture using the Admin API."""
    return await _make_datasource_metadata(
        xngin_session,
        aclient=aclient,
        new_name="testing datasource",
    )


@pytest.fixture(name="testing_datasource_other")
async def fixture_testing_datasource_other(
    xngin_session: AsyncSession,
    aclient: admin_api_client.AdminAPIClient,
) -> DatasourceMetadata:
    """Creates a second datasource fixture using the Admin API."""
    return await _make_datasource_metadata(
        xngin_session,
        aclient=aclient,
        new_name="testing datasource other",
    )


async def _make_datasource_metadata(
    xngin_session: AsyncSession,
    *,
    aclient: admin_api_client.AdminAPIClient,
    new_name: str,
    participants_def_list: list[ParticipantsDef] | None = None,
) -> DatasourceMetadata:
    """Generates a new Organization, Datasource, and API key for testing.

    Args:
    aclient - API client used to create the test entities.
    new_name - the friendly name of the datasource.
    participants_def_list - Allows overriding the new ds's `participants` list of participant types.
    """
    pt_list = participants_def_list or [TESTING_DWH_PARTICIPANT_DEF]
    dwh = Dsn.from_url(flags.XNGIN_DEVDWH_DSN)
    org_id = aclient.create_organizations(aapi.CreateOrganizationRequest(name="test organization")).data.id

    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=org_id,
            name=new_name,
            dsn=_convert_dwh_to_create_api_dsn(dwh),
        )
    ).data.id

    for participants_def in pt_list:
        aclient.create_participant_type(
            datasource_id=datasource_id,
            body=aapi.CreateParticipantsTypeRequest(
                participant_type=participants_def.participant_type,
                schema_def=participants_def,
            ),
        )

    key_response = aclient.create_api_key(datasource_id=datasource_id).data
    api_org = aclient.get_organization(organization_id=org_id).data
    api_ds = aclient.get_datasource(datasource_id=datasource_id).data

    org = await xngin_session.get_one(tables.Organization, org_id)
    datasource = await xngin_session.get_one(
        tables.Datasource,
        datasource_id,
        options=[selectinload(tables.Datasource.organization)],
    )
    datasource_config = datasource.get_config()

    return DatasourceMetadata(
        ds=datasource,
        api_org=api_org,
        api_ds=api_ds,
        organization_id=api_org.id,
        datasource_id=api_ds.id,
        pt=datasource_config.participants[0],
        dsn=datasource_config.to_sqlalchemy_url().render_as_string(False),
        key=key_response.key,
        key_id=key_response.id,
        org=org,
    )


def _convert_dwh_to_create_api_dsn(dwh: settings.Dwh) -> aapi.Dsn:
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
                    with open(dwh.credentials.path, encoding="utf-8") as credentials_file:
                        credentials_content = credentials_file.read()
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
    exc: AdminAPIClientNotDefaultStatusError | None = None

    def http_response(self):
        """Returns the httpx Response."""
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
    with pytest.raises(AdminAPIClientNotDefaultStatusError) as exc:
        yield match
    match.exc = exc.value
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
        assert any(detail_contains in msg for msg in match._detail_messages()), (
            f"Expected '{detail_eq}' to be in one of the .msg fields in the response: {http_response.content}"
        )
    if text is not None:
        assert match._has_text(text), f"Expected '{text}' to be in the response: {http_response.content}"
