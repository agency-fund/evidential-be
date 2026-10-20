"""conftest configures FastAPI dependency injection for testing and also does some setup before tests in
this package are run.

It lives above xngin.apiserver so that its fixtures also serve the tests of other packages, such as xngin.tq, without
pytest_plugins. Helpers that tests import belong in xngin.apiserver.apiserver_test_support instead: importing a conftest
as a module loads a second copy of it.
"""

import contextlib
import os
import sys

import pytest
import sqlalchemy
from loguru import logger
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from xngin.apiserver import database, flags, request_context_middleware
from xngin.apiserver.apiserver_test_support import (
    DatasourceMetadata,
    convert_dwh_to_create_api_dsn,
    get_queries_test_uri,
    get_test_uri_info,
)
from xngin.apiserver.dependencies import (
    random_seed_dependency,
)
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.main import app
from xngin.apiserver.routers.admin import admin_api_types as aapi
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.apiserver.routers.auth.auth_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_EMAIL_2,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
    UNPRIVILEGED_TOKEN_FOR_TESTING_2,
)
from xngin.apiserver.settings import (
    Dsn,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing import (
    admin_api_client,
    admin_integrations_api_client,
    experiments_api_client,
    integrations_api_client,
)
from xngin.apiserver.testing.pg_helpers import (
    create_test_database_pg,
    finish_test_databases_pg,
    with_database_suffix,
)
from xngin.db_extensions import custom_functions

# SQLAlchemy's logger will append this to the name of its loggers used for the application database; e.g.
# sqlalchemy.engine.Engine.xngin_app.
SA_LOGGER_NAME_FOR_APP = "xngin_app"


def pytest_configure():
    """Points the application at a database unique to this test run and worker; see pg_helpers."""
    if database_url := os.environ.get("DATABASE_URL"):
        flags.DATABASE_URL = with_database_suffix(database_url)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    """Drops this process's test databases, unless a test failed or the run was cut short."""
    if not flags.DATABASE_URL:
        return
    keep = bool(session.testsfailed) or exitstatus not in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    kept = finish_test_databases_pg(flags.DATABASE_URL, keep=keep)
    if keep and kept:
        print(f"\nKept test databases for inspection (the next run drops them): {', '.join(kept)}", file=sys.stderr)


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
def safe_resolve_testing_mode():
    """Configure safe_resolve for unit tests: never perform real DNS resolution and allow private IPs."""
    safe_resolve.ALLOW_CONNECTING_TO_PRIVATE_IPS = True
    safe_resolve.INTERCEPT_DNS_FOR_TESTING = True


@pytest.fixture(scope="session", autouse=True)
def fixture_override_app_dependencies():
    """Configures FastAPI dependencies for testing.

    This uses FastAPI's dependency override mechanism: https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
    """

    app.dependency_overrides[random_seed_dependency] = get_random_seed_for_test

    auth_dependencies.disable(app)
    auth_dependencies.enable_testing_tokens()


@pytest.fixture
def records():
    """Captures the Loguru records emitted during the test, with request context applied as in production."""
    logger.configure(patcher=request_context_middleware.add_request_context)
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record), level="INFO")
    yield captured
    logger.remove(handler_id)


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


@pytest.fixture(name="aclient_unpriv_2")
def fixture_admin_api_client_unpriv_2(xngin_session):
    """Returns a second generated API client for unprivileged Admin API requests, for cross-tenant tests."""
    with TestClient(app, headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING_2}"}) as client:
        yield admin_api_client.AdminAPIClient(client)


@pytest.fixture(name="iaclient")
def fixture_integrations_admin_api_client(xngin_session):
    """Returns a generated API client for privileged Admin API requests."""
    with TestClient(app, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}) as client:
        yield admin_integrations_api_client.AdminIntegrationsAPIClient(client)


@pytest.fixture(name="iclient")
def fixture_integrations_api_client(xngin_session):
    """Returns a generated API client for privileged Integrations API requests."""
    with TestClient(app, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}) as client:
        yield integrations_api_client.IntegrationsAPIClient(client)


@pytest.fixture(scope="session")
def fixture_initialize_xngin_db_schema():
    """Create a fresh application database for the test session from a template having the current schema.

    The application database engine is held open for the whole session so that its connection pool is reused across
    tests; the database.setup() call made by the FastAPI lifespan reuses it.
    """
    with database.setup():
        create_test_database_pg(database.get_sqlalchemy_database_url(), tables.Base.metadata)
        yield


@pytest.fixture(name="xngin_session")
def fixture_xngin_db_session(fixture_initialize_xngin_db_schema):
    """Yields a SQLAlchemy session suitable for direct interaction with the database.

    This will delete all rows from the application tables at the beginning of every test. The users table will be seeded
    with
    a privileged user (pget, ppost, ...) and two unprivileged users (uget/upost and the second unprivileged user for
    cross-tenant tests). These users can be removed by individual tests by calling delete_seeded_users().

    Where possible, prefer using the API methods to test functionality rather than touching the database
    directly.
    """
    with database.get_engine().begin() as conn:
        for table in reversed(tables.Base.metadata.sorted_tables):
            conn.execute(sqlalchemy.delete(table))
    with database.get_session() as session:
        session.add_all([
            tables.User(email=PRIVILEGED_EMAIL, is_privileged=True),
            tables.User(email=UNPRIVILEGED_EMAIL, is_privileged=False),
            tables.User(email=UNPRIVILEGED_EMAIL_2, is_privileged=False),
        ])
        session.commit()
    with database.get_session() as sess:
        yield sess


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


@pytest.fixture(name="testing_datasource")
def fixture_testing_datasource(xngin_session: Session, aclient: admin_api_client.AdminAPIClient) -> DatasourceMetadata:
    """Creates a datasource fixture using the Admin API."""
    return _make_datasource_metadata(
        xngin_session,
        aclient=aclient,
        org_name="testing datasource",
    )


@pytest.fixture(name="testing_datasource_other")
def fixture_testing_datasource_other(
    xngin_session: Session,
    aclient: admin_api_client.AdminAPIClient,
) -> DatasourceMetadata:
    """Creates a second datasource fixture using the Admin API."""
    return _make_datasource_metadata(
        xngin_session,
        aclient=aclient,
        org_name="testing datasource other",
    )


def _make_datasource_metadata(
    xngin_session: Session, *, aclient: admin_api_client.AdminAPIClient, org_name: str
) -> DatasourceMetadata:
    """Generates a new Organization, Datasource, and API key for testing.

    Args:
    aclient - API client used to create the test entities.
    org_name - the friendly name of the datasource.
    """
    dwh = Dsn.from_url(flags.XNGIN_DEVDWH_DSN)
    org_id = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="test organization")).data.id

    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=org_id,
            name=org_name,
            dsn=convert_dwh_to_create_api_dsn(dwh),
        )
    ).data.id

    key_response = aclient.create_api_key(datasource_id=datasource_id).data
    api_org = aclient.get_organization(organization_id=org_id).data
    api_ds = aclient.get_datasource(datasource_id=datasource_id).data

    org = xngin_session.get_one(tables.Organization, org_id)
    datasource = xngin_session.get_one(tables.Datasource, datasource_id)

    return DatasourceMetadata(
        ds=datasource,
        api_org=api_org,
        api_ds=api_ds,
        organization_id=api_org.id,
        datasource_id=api_ds.id,
        key=key_response.key,
        key_id=key_response.id,
        org=org,
    )
