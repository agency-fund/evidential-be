"""conftest configures FastAPI dependency injection for testing and also does some setup before tests in this module are run."""

import logging
import os
from pathlib import Path

import pytest
import sqlalchemy
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import StaticPool
from sqlalchemy.orm import sessionmaker

from xngin.apiserver import database
from xngin.apiserver.dependencies import settings_dependency, xngin_db_session
from xngin.apiserver.models import tables
from xngin.apiserver.settings import XnginSettings, SettingsForTesting
from xngin.apiserver.testing import testing_dwh
from xngin.sqlite_extensions import custom_functions

logger = logging.getLogger(__name__)


class DeveloperErrorRunFromRootOfRepositoryPleaseError(Exception):
    def __init__(self):
        super().__init__("Tests must be run from the root of the repository.")


def get_settings_for_test() -> XnginSettings:
    filename = Path(__file__).parent / "testdata/xngin.testing.settings.json"
    with open(filename) as f:
        try:
            contents = f.read()
            return TypeAdapter(SettingsForTesting).validate_json(contents)
        except ValidationError as pyve:
            print(f"Failed to parse {filename}. Contents:\n{contents}\n\nError:{pyve}")
            raise


def get_test_sessionmaker():
    """Returns a Session generator backed by an ephemeral db for use in tests as our app db."""
    # We use an in-memory ephemeral database for the xngindb during tests.
    db_engine = sqlalchemy.create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=os.environ.get("ECHO_SQL", "").lower() in ("true", "1"),
    )

    testing_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )
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
    raise_unless_running_from_top_directory()


@pytest.fixture(scope="session", autouse=True)
def ensure_dwh_sqlite_database_exists(ensure_correct_working_directory):
    """Create testing_dwh.db, if it doesn't already exist."""
    testing_dwh.create_dwh_sqlite_database()


def raise_unless_running_from_top_directory():
    """Raises an exception unless the current working directory is the root of the project."""
    pypt = Path(os.getcwd()) / "pyproject.toml"
    if not pypt.exists():
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
