"""conftest configures FastAPI dependency injection for testing and also does some setup before tests in this module are run."""

import os
from pathlib import Path
import logging
import pytest
import sqlalchemy
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import StaticPool
from sqlalchemy.orm import sessionmaker

from xngin.apiserver import database
from xngin.apiserver.dependencies import settings_dependency, db_session
from xngin.apiserver.settings import XnginSettings, SettingsForTesting
from xngin.apiserver.testing import testing_dwh

logger = logging.getLogger(__name__)


def get_settings_for_test() -> XnginSettings:
    filename = Path(__file__).parent / "testdata/xngin.testing.settings.json"
    with open(filename) as f:
        try:
            contents = f.read()
            return TypeAdapter(SettingsForTesting).validate_json(contents)
        except ValidationError as pyve:
            print(f"Failed to parse {filename}. Contents:\n{contents}")
            raise pyve


def setup(app):
    """Configures FastAPI dependencies for testing."""

    db_engine = sqlalchemy.create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )
    # Hack: Cause any direct access to production code from code to fail during tests.
    database.SessionLocal = None
    # Create all the ORM tables.
    database.Base.metadata.create_all(bind=db_engine)

    def get_db_for_test():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    # https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
    app.dependency_overrides[db_session] = get_db_for_test
    app.dependency_overrides[settings_dependency] = get_settings_for_test


@pytest.fixture(scope="module", autouse=True)
def ensure_correct_working_directory():
    """Ensures the tests are being run from the root of the repo.

    This is important because the tests generate some temporary data on disk and we want the paths to be right.
    """
    pypt = Path(os.getcwd()) / "pyproject.toml"
    if not pypt.exists():
        raise Exception("Tests must be run from the root of the repository.")


@pytest.fixture(scope="module", autouse=True)
def ensure_dwh_sqlite_database_exists(ensure_correct_working_directory):
    """Create testing_dwh.db, if it doesn't already exist."""
    testing_dwh.create_dwh_sqlite_database()
