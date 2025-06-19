"""conftest configures FastAPI dependency injection for testing and also does some setup before tests in this module are run."""

import enum
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import assert_never, cast

import pytest
import sqlalchemy
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import StaticPool, make_url
from sqlalchemy.dialects.postgresql import psycopg
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy_bigquery import dialect as bigquery_dialect
from starlette.testclient import TestClient

from xngin.apiserver import constants, database, flags
from xngin.apiserver.apikeys import hash_key_or_raise, make_key
from xngin.apiserver.dependencies import (
    random_seed_dependency,
    settings_dependency,
    xngin_db_session,
)
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.main import app
from xngin.apiserver.models import tables
from xngin.apiserver.routers.oidc_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.settings import ParticipantsDef, SettingsForTesting, XnginSettings
from xngin.apiserver.testing.pg_helpers import create_database_if_not_exists_pg
from xngin.db_extensions import custom_functions

# SQLAlchemy's logger will append this to the name of its loggers used for the application database; e.g.
# sqlalchemy.engine.Engine.xngin_app.
SA_LOGGER_NAME_FOR_APP = "xngin_app"


class DeveloperErrorRunFromRootOfRepositoryPleaseError(Exception):
    def __init__(self):
        super().__init__("Tests must be run from the root of the repository.")


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


def get_settings_for_test() -> XnginSettings:
    filename = Path(__file__).parent / "testdata/xngin.testing.settings.json"
    with open(filename) as f:
        try:
            contents = f.read()
            return TypeAdapter(SettingsForTesting).validate_json(contents)
        except ValidationError as pyve:
            print(f"Failed to parse {filename}. Contents:\n{contents}\n\nError:{pyve}")
            raise


def get_test_appdb_info() -> TestUriInfo:
    """Gets the DSN of the application database to use in tests."""
    connection_uri = os.environ.get("XNGIN_TEST_APPDB_URI", "")
    if not connection_uri:
        raise ValueError("XNGIN_TEST_APPDB_URI must be set")
    info = get_test_uri_info(connection_uri)
    if info.db_type != DbType.PG:
        raise ValueError(
            f"{info.db_type} is not a supported app database: {connection_uri}"
        )
    return info


def get_test_dwh_info() -> TestUriInfo:
    """Gets the DSN of the testing data warehouse to use in tests.

    See xngin.apiserver.dwh.test_queries.fixture_dwh_session.
    """
    connection_uri = os.environ.get("XNGIN_TEST_DWH_URI", "")
    if not connection_uri:
        raise ValueError("XNGIN_TEST_DWH_URI must be set.")
    return get_test_uri_info(connection_uri)


@pytest.fixture(scope="session", autouse=True)
def setup_debug_logging():
    print(
        "Running tests with "
        f"\n\tXNGIN_TEST_APPDB_URI: {get_test_appdb_info()} "
        f"\n\tXNGIN_TEST_DWH_URI  : {get_test_dwh_info()}"
    )


def get_random_seed_for_test():
    """Returns a seed for testing."""
    return 42


@pytest.fixture(scope="session", autouse=True)
def allow_connecting_to_private_ips():
    safe_resolve.ALLOW_CONNECTING_TO_PRIVATE_IPS = True


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


def make_engine():
    """Returns a SQLA engine for XNGIN_TEST_APPDB_URI; db will be created if it does not exist."""
    appdb_info = get_test_appdb_info()
    create_database_if_not_exists_pg(appdb_info.connect_url)
    db_engine = sqlalchemy.create_engine(
        appdb_info.connect_url,
        logging_name=SA_LOGGER_NAME_FOR_APP,
        execution_options={"logging_token": "app"},
        poolclass=StaticPool,
        echo=flags.ECHO_SQL_APP_DB,
    )
    # Create all the ORM tables.
    tables.Base.metadata.create_all(bind=db_engine)

    return db_engine


def get_test_sessionmaker(db_engine: sqlalchemy.engine.Engine):
    """Returns a session bound to the db_engine; the returned database is not guaranteed unique."""
    # Hack: Cause any direct access to production code from code to fail during tests.
    database.SessionLocal = None

    testing_session_local = sessionmaker(bind=db_engine, expire_on_commit=False)

    def get_db_for_test():
        sess = testing_session_local()
        try:
            yield sess
        finally:
            sess.close()

    return get_db_for_test


@pytest.fixture(scope="session", autouse=True)
def fixture_override_app_dependencies():
    """Configures FastAPI dependencies for testing."""
    # https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
    app.dependency_overrides[xngin_db_session] = get_test_sessionmaker(make_engine())
    app.dependency_overrides[settings_dependency] = get_settings_for_test
    app.dependency_overrides[random_seed_dependency] = get_random_seed_for_test


@pytest.fixture(scope="session", name="client")
def fixture_client():
    return TestClient(app)


@pytest.fixture(scope="session", name="client_v1")
def fixture_client_v1():
    client = TestClient(app)
    client.base_url = client.base_url.join(constants.API_PREFIX_V1)
    return client


@pytest.fixture(scope="session", name="pget")
def fixture_pget(client):
    return partial(
        client.get, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}
    )


@pytest.fixture(scope="session", name="ppost")
def fixture_ppost(client):
    return partial(
        client.post, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}
    )


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


@pytest.fixture(scope="session", name="test_engine")
def fixture_test_engine():
    """Create a SQLA engine for testing that is safely disposed of after all tests are run."""
    db_engine = make_engine()
    try:
        yield db_engine
    finally:
        db_engine.dispose()


@pytest.fixture(name="xngin_session")
def fixture_xngin_db_session(test_engine):
    """Use this session directly if you don't need to test the FastAPI app itself."""
    test_sessionmaker = get_test_sessionmaker(test_engine)
    return next(test_sessionmaker())


@pytest.fixture(scope="session", autouse=True)
def ensure_correct_working_directory():
    """Ensures the tests are being run from the root of the repo.

    This is important because the tests generate some temporary data on disk and we want the paths to be right.
    """
    cwd_or_raise_unless_running_from_top_directory()


def cwd_or_raise_unless_running_from_top_directory():
    """Helper to manage the current working directory of unit tests.

    When the code is located under the home directory, this will automatically change the working directory to the root
    of the repository. This is helpful for developers because they can now run the tests from any directory without
    worrying about their working directory.

    When the code is not under the home directory, this will raise an exception unless the current working directory
    is the root of the repository. This is to avoid problems on CI or other automated runs that where it might not be
    safe to traverse parents.
    """
    current_dir = Path.cwd()
    operating_outside_homedir = Path.home() not in current_dir.parents
    if operating_outside_homedir:
        if not Path.exists(current_dir / "pyproject.toml"):
            raise DeveloperErrorRunFromRootOfRepositoryPleaseError()
    else:
        while current_dir != Path.home():
            if (current_dir / "pyproject.toml").exists():
                os.chdir(current_dir)
                return
            current_dir = current_dir.parent

    raise DeveloperErrorRunFromRootOfRepositoryPleaseError()


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


def get_settings_datasource(datasource_id: str):
    """Pull a datasource from the xngin.testing.settings.json file."""
    return get_settings_for_test().get_datasource(datasource_id)


@dataclass
class DatasourceMetadata:
    """Describes an ephemeral datasource, organization, and API key."""

    org: tables.Organization
    ds: tables.Datasource

    # The SQLAlchemy DSN
    dsn: str

    # An API key suitable for use in the Authorization: header.
    key: str


@pytest.fixture(name="testing_datasource", scope="function")
def fixture_testing_datasource(xngin_session: Session) -> DatasourceMetadata:
    """
    Generates a new Organization, Datasource, and API key for a test.

    This is NOT the same as the default Org+Datasource auto-generated for privileged users (i.e. cases where we use PRIVILEGED_TOKEN_FOR_TESTING+PRIVILEGED_EMAIL).

    Note: The datasource configured in this fixture represents a customer database. This is
    *different* than the xngin server-side database configured by conftest.setup().
    """
    return make_datasource_metadata(xngin_session)


@pytest.fixture(name="testing_datasource_with_user", scope="function")
def fixture_testing_datasource_with_user(xngin_session: Session) -> DatasourceMetadata:
    """
    Generates a new Organization, Datasource, and API key. Adds our privileged user to the org.
    """
    metadata = make_datasource_metadata(xngin_session)
    user = tables.User(
        id="user_" + secrets.token_hex(8), email=PRIVILEGED_EMAIL, is_privileged=True
    )
    user.organizations.append(metadata.org)
    xngin_session.add(user)
    xngin_session.commit()
    return metadata


def make_datasource_metadata(
    xngin_session: Session,
    *,
    datasource_id: str | None = None,
    name="test ds",
    datasource_id_for_config="testing-remote",
    participants_def_list: list[ParticipantsDef] | None = None,
) -> DatasourceMetadata:
    """Generates a new Organization, Datasource, and API key in the database for testing.

    Args:
    db_session - the database session to use for adding our objects to the database.
    datasource_id - the unique ID of the datasource. If not provided, it will be randomly generated.
    name - the friendly name of the datasource.
    datasource_id_for_config - the unique ID of the datasource **from our test settings json**
        to use for the DatasourceConfig.
    participants_def_list - set to override the top-level datasource_id_for_config's `participants`
        list of participant types, to allow for testing with simulated inline schemas.
    """
    run_id = secrets.token_hex(8)
    datasource_id = datasource_id or "ds" + run_id

    # We derive a new test datasource from the standard static "testing-remote" datasource by
    # randomizing its unique ID and marking it as requiring an API key.
    test_ds = get_settings_datasource(datasource_id_for_config).config
    if participants_def_list:
        test_ds.participants = participants_def_list

    org = tables.Organization(id="org" + run_id, name="test organization")
    datasource = tables.Datasource(id=datasource_id, name=name)
    datasource.set_config(test_ds)
    datasource.organization = org

    key_id, key = make_key()
    kt = tables.ApiKey(id=key_id, key=hash_key_or_raise(key))
    kt.datasource = datasource

    xngin_session.add_all([org, datasource, kt])
    xngin_session.commit()
    return DatasourceMetadata(
        ds=datasource,
        dsn=datasource.get_config().to_sqlalchemy_url().render_as_string(False),
        key=key,
        org=org,
    )


def dates_equal(db_date: datetime, request_date: datetime):
    """Compare dates with or without timezone info, honoring the db_date's timezone."""
    if db_date.tzinfo is None:
        return db_date == request_date.replace(tzinfo=None)
    return db_date == request_date
