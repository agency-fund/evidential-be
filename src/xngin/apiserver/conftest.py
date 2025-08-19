"""conftest configures FastAPI dependency injection for testing and also does some setup before tests in
this module are run."""

import contextlib
import enum
import os
import secrets
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import assert_never, cast

import pytest
import sqlalchemy
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import delete, make_url, select
from sqlalchemy.dialects.postgresql import psycopg
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)
from sqlalchemy.orm import selectinload
from sqlalchemy_bigquery import dialect as bigquery_dialect
from starlette.testclient import TestClient

from xngin.apiserver import constants, database, flags
from xngin.apiserver.apikeys import hash_key_or_raise, make_key
from xngin.apiserver.dependencies import (
    random_seed_dependency,
)
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.main import app
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.apiserver.routers.auth.auth_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.settings import (
    Dsn,
    ParticipantsConfig,
    RemoteDatabaseConfig,
    SettingsForTesting,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.pg_helpers import create_database_if_not_exists_pg
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF
from xngin.db_extensions import custom_functions

# SQLAlchemy's logger will append this to the name of its loggers used for the application database; e.g.
# sqlalchemy.engine.Engine.xngin_app.
SA_LOGGER_NAME_FOR_APP = "xngin_app"


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
        return f"{self.connect_url} (detected type: {self.db_type})"


@pytest.fixture(name="static_settings")
def fixture_static_settings() -> SettingsForTesting:
    """Reads the xngin.testing.settings.json file."""
    filename = Path(__file__).resolve().parent / "testdata/xngin.testing.settings.json"
    with open(filename) as f:
        try:
            contents = f.read()
            return TypeAdapter(SettingsForTesting).validate_json(contents)
        except ValidationError as pyve:
            print(f"Failed to parse {filename}. Contents:\n{contents}\n\nError:{pyve}")
            raise


def get_queries_test_uri() -> TestUriInfo:
    """Gets the DSN of the testing data warehouse to use in tests.

    See xngin.apiserver.dwh.test_queries.fixture_queries_session.
    """
    connection_uri = os.environ.get("XNGIN_QUERIES_TEST_URI", "")
    if not connection_uri:
        raise ValueError("XNGIN_QUERIES_TEST_URI must be set.")
    return get_test_uri_info(connection_uri)


@pytest.fixture(scope="session", autouse=True)
def print_database_env_vars():
    """Prints debugging information sometimes useful for working with tests to stdout."""

    database_url = "(unset)"
    with contextlib.suppress(ValueError):
        database_url = database.get_server_database_url()

    queries_url = "(unset)"
    with contextlib.suppress(ValueError):
        queries_url = str(get_queries_test_uri())

    dwh_url = flags.XNGIN_DEVDWH_DSN or "(unset)"

    print(
        "Running tests with "
        f"\n\tDATABASE_URL: {database_url} "
        f"\n\tXNGIN_DEVDWH_DSN: {dwh_url}"
        f"\n\tXNGIN_QUERIES_TEST_URI: {queries_url}"
    )


def get_random_seed_for_test():
    """Returns a seed for testing."""
    return 42


@pytest.fixture(scope="session", autouse=True)
def allow_connecting_to_private_ips():
    safe_resolve.ALLOW_CONNECTING_TO_PRIVATE_IPS = True


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="session", name="client")
def fixture_client():
    """Returns a FastAPI TestClient.

    TestClient manages the lifecycle of the app and will invoke the FastAPI app and router @lifespan methods.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session", name="client_v1")
def fixture_client_v1():
    """Returns a FastAPI TestClient with the {constants.API_PREFIX_V1} as a prefix on the request path.

    TestClient manages the lifecycle of the app and will invoke the FastAPI app and router @lifespan methods.
    """
    with TestClient(app, base_url=f"http://testserver{constants.API_PREFIX_V1}") as client:
        yield client


@pytest.fixture(scope="session", name="pget")
def fixture_pget(client):
    return partial(client.get, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"})


@pytest.fixture(scope="session", name="ppost")
def fixture_ppost(client):
    return partial(client.post, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"})


@pytest.fixture(scope="session", name="ppatch")
def fixture_ppatch(client):
    return partial(
        client.patch,
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )


@pytest.fixture(scope="session", name="pdelete")
def fixture_pdelete(client):
    return partial(
        client.delete,
        headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"},
    )


@pytest.fixture(scope="session", name="udelete")
def fixture_udelete(client):
    return partial(
        client.delete,
        headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING}"},
    )


@pytest.fixture(scope="session", name="uget")
def fixture_uget(client):
    return partial(
        client.get,
        headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING}"},
    )


@pytest.fixture(scope="function", name="xngin_session", autouse=True)
async def fixture_xngin_db_session():
    """Yields a SQLAlchemy session suitable for direct interaction with the database.

    This will drop and recreate all tables at the beginning of every test. The users table will be seeded with
    a privileged user (pget, ppost, ...) and an unprivileged user (uget, upost, ...). These users can be removed by
    individual tests by calling delete_seeded_users().

    Where possible, prefer using the API methods to test functionality rather than touching the database
    directly.

    This fixture uses autouse=True to ensure all tests begin with a clean database state. Since some tests only
    interact with the API server through API methods (without explicitly using the xngin_session fixture),
    autouse=True guarantees this cleanup runs for every test, preventing any shared state between test runs.
    """
    async with database.setup():
        create_database_if_not_exists_pg(database.get_sqlalchemy_database_url())
        async with database.get_async_engine().begin() as conn:
            await conn.run_sync(tables.Base.metadata.drop_all)
            await conn.run_sync(tables.Base.metadata.create_all)
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


@dataclass
class DatasourceMetadata:
    """Describes an ephemeral datasource, organization, and API key."""

    org: tables.Organization
    ds: tables.Datasource

    # The SQLAlchemy DSN
    dsn: str

    # An API key suitable for use in the Authorization: header.
    key: str
    key_id: str


@pytest.fixture(name="testing_datasource", scope="function")
async def fixture_testing_datasource(xngin_session: AsyncSession) -> DatasourceMetadata:
    """Adds to db a new Organization, Datasource, and API key."""
    return await _make_datasource_metadata(
        xngin_session,
        new_name="testing datasource",
    )


@pytest.fixture(name="testing_datasource_with_user", scope="function")
async def fixture_testing_datasource_with_user(
    xngin_session: AsyncSession,
) -> DatasourceMetadata:
    """Adds to db a new Org w/ PRIVILEGED_EMAIL user, Datasource, and API key."""
    metadata = await _make_datasource_metadata(
        xngin_session,
        new_name="testing datasource with user",
    )
    user = (
        await xngin_session.execute(
            select(tables.User)
            .options(selectinload(tables.User.organizations))
            .where(tables.User.email == PRIVILEGED_EMAIL)
        )
    ).scalar_one()
    user.organizations.append(metadata.org)
    await xngin_session.commit()
    return metadata


async def _make_datasource_metadata(
    xngin_session: AsyncSession,
    *,
    new_name: str,
    new_datasource_id: str | None = None,
    participants_def_list: list[ParticipantsConfig] | None = None,
) -> DatasourceMetadata:
    """Generates a new Organization, Datasource, and API key in the database for testing.

    Args:
    db_session - the database session to use for adding our objects to the database.
    new_name - the friendly name of the datasource.
    new_datasource_id - unique ID of the datasource. If not provided, it will be randomly generated.
    participants_def_list - Allows overriding the new ds's `participants` list of participant types.
    """
    run_id = secrets.token_hex(8)

    org = tables.Organization(id="org" + run_id, name="test organization")

    # Now make a new test datasource attached to the org. Use a random ID if none is specified.
    new_datasource_id = new_datasource_id or "ds" + run_id
    datasource = tables.Datasource(id=new_datasource_id, name=new_name)
    datasource.organization = org
    # Initialize our ds config to point to our testing dwh.
    test_dwh_dsn = Dsn.from_url(flags.XNGIN_DEVDWH_DSN)
    # If no override is provided, use the default testing participant type.
    pt_list = participants_def_list or [TESTING_DWH_PARTICIPANT_DEF]
    datasource.set_config(RemoteDatabaseConfig(type="remote", participants=pt_list, dwh=test_dwh_dsn))

    # Make this ds also accessible via an API key.
    key_id, key = make_key()
    kt = tables.ApiKey(id=key_id, key=hash_key_or_raise(key))
    kt.datasource = datasource

    xngin_session.add_all([org, datasource, kt])
    await xngin_session.commit()

    return DatasourceMetadata(
        ds=datasource,
        dsn=datasource.get_config().to_sqlalchemy_url().render_as_string(False),
        key=key,
        key_id=key_id,
        org=org,
    )
