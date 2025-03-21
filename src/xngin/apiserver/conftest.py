"""conftest configures FastAPI dependency injection for testing and also does some setup before tests in this module are run."""

import enum
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import pytest
import sqlalchemy
import sqlalchemy_bigquery
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import StaticPool, make_url
from sqlalchemy.dialects.postgresql import psycopg
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import sessionmaker

from xngin.apiserver import database, flags
from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.dependencies import settings_dependency, xngin_db_session
from xngin.apiserver.models import tables
from xngin.apiserver.models.tables import Organization, Datasource, ApiKey
from xngin.apiserver.settings import XnginSettings, SettingsForTesting
from xngin.apiserver.testing import testing_dwh
from xngin.db_extensions import custom_functions

logger = logging.getLogger(__name__)


class DeveloperErrorRunFromRootOfRepositoryPleaseError(Exception):
    def __init__(self):
        super().__init__("Tests must be run from the root of the repository.")


class DbType(enum.StrEnum):
    SL = "sqlite"
    RS = "redshift"
    PG = "postgres"
    BQ = "bigquery"

    def dialect(self) -> Dialect:
        """Returns the SQLAlchemy dialect most appropriate for this DbType."""
        match self:
            case DbType.SL:
                return sqlalchemy.dialects.sqlite.dialect()
            case DbType.RS:
                return sqlalchemy.dialects.postgresql.psycopg2.dialect()
            case DbType.PG:
                return psycopg.dialect()
            case DbType.BQ:
                return sqlalchemy_bigquery.dialect()
        assert_never(self)


def get_settings_for_test() -> XnginSettings:
    filename = Path(__file__).parent / "testdata/xngin.testing.settings.json"
    with open(filename) as f:
        try:
            contents = f.read()
            return TypeAdapter(SettingsForTesting).validate_json(contents)
        except ValidationError as pyve:
            print(f"Failed to parse {filename}. Contents:\n{contents}\n\nError:{pyve}")
            raise


def get_test_appdb_info():
    """Use this for tests of our application db, e.g. for caching user table confgs."""
    connection_uri = os.environ.get("XNGIN_TEST_APPDB_URI", "sqlite:///:memory:")
    return get_test_uri_info(connection_uri)


def get_test_dwh_info():
    """Use this for tests that skip settings.json and directly connect to a simulated DWH."""
    connection_uri = os.environ.get("XNGIN_TEST_DWH_URI", "sqlite:///:memory:")
    return get_test_uri_info(connection_uri)


@pytest.fixture(scope="session", autouse=True)
def setup_debug_logging():
    print(
        f"Running tests with XNGIN_TEST_APPDB_URI: {get_test_appdb_info()} "
        f"and XNGIN_TEST_DWH_URI: {get_test_dwh_info()}"
    )


def get_test_uri_info(connection_uri: str):
    """Returns a tuple of info about a test database given its connection_uri.

    Returns:
    - connection string uri
    - type of dwh backend derived from the uri
    - map of connection args for use with SQLAlchemy's create_engine()
    """
    connect_args = {}
    if connection_uri.startswith("sqlite"):
        dbtype = DbType.SL
        connect_args = {"check_same_thread": False}
    elif connection_uri.startswith("bigquery"):
        dbtype = DbType.BQ
    elif "redshift.amazonaws.com" in connection_uri:
        dbtype = DbType.RS
    elif connection_uri.startswith("postgres"):
        dbtype = DbType.PG
    else:
        raise ValueError(
            f"connection_uri is not recognized as a SQLite, BigQuery, Redshift, or Postgres database: {connection_uri}"
        )
    return make_url(connection_uri), dbtype, connect_args


def get_test_sessionmaker():
    """
    Returns a function to create a Session generator for use in tests as our app db.

    The backing db is configured by XNGIN_TEST_APPDB_URI, defaulting to an in-memory ephemeral db.
    """
    connect_url, _, connect_args = get_test_appdb_info()
    db_engine = sqlalchemy.create_engine(
        connect_url,
        connect_args=connect_args,
        poolclass=StaticPool,
        echo=flags.ECHO_SQL,
    )

    testing_session_local = sessionmaker(bind=db_engine)
    # Hack: Cause any direct access to production code from code to fail during tests.
    database.SessionLocal = None
    # Create all the ORM tables.
    tables.Base.metadata.create_all(bind=db_engine)

    def get_db_for_test():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    return get_db_for_test


def setup(app):
    """Configures FastAPI dependencies for testing."""
    # https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
    app.dependency_overrides[xngin_db_session] = get_test_sessionmaker()
    app.dependency_overrides[settings_dependency] = get_settings_for_test


@pytest.fixture(scope="session", autouse=True)
def ensure_correct_working_directory():
    """Ensures the tests are being run from the root of the repo.

    This is important because the tests generate some temporary data on disk and we want the paths to be right.
    """
    cwd_or_raise_unless_running_from_top_directory()


@pytest.fixture(scope="session", autouse=True)
def ensure_dwh_sqlite_database_exists(ensure_correct_working_directory):
    """Create testing_dwh.db, if it doesn't already exist."""
    testing_dwh.create_dwh_sqlite_database()


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

    org: Organization
    ds: Datasource

    # The SQLAlchemy DSN
    dsn: str

    # An API key suitable for use in the Authorization: header.
    key: str


@pytest.fixture(name="testing_datasource", scope="function")
def fixture_testing_datasource(db_session) -> DatasourceMetadata:
    """
    Generates a new Organization, Datasource, and API key for a test.

    This is NOT the same as the default Org+Datasource auto-generated for privileged users (i.e. cases where we use PRIVILEGED_TOKEN_FOR_TESTING+PRIVILEGED_EMAIL).
    """
    run_id = secrets.token_hex(8)
    datasource_id = "ds" + run_id

    # We derive a new test datasource from the standard static "testing-remote" datasource by
    # randomizing its unique ID and marking it as requiring an API key.
    #
    # Note: The datasource configured in this fixture represents a customer database. This is *different* than the
    # xngin server-side database configured by conftest.setup().
    test_ds = get_settings_datasource("testing-remote").config

    org = Organization(id="org" + run_id, name="test organization")
    datasource = Datasource(id=datasource_id, name="test ds")
    datasource.set_config(test_ds)
    datasource.organization = org

    key_id, key = make_key()
    kt = ApiKey(id=key_id, key=hash_key(key))
    kt.datasource = datasource

    db_session.add_all([org, datasource, kt])
    db_session.commit()
    return DatasourceMetadata(
        ds=datasource,
        dsn=datasource.get_config().dwh.to_sqlalchemy_url().render_as_string(False),
        key=key,
        org=org,
    )


@dataclass
class DatasourceMetadata:
    """Describes an ephemeral datasource, organization, and API key."""

    org: Organization
    ds: Datasource

    # The SQLAlchemy DSN
    dsn: str

    # An API key suitable for use in the Authorization: header.
    key: str
