"""conftest defines some testing fixtures that will be run automatically before tests in this module are run."""

import os
import sqlite3
from pathlib import Path
import logging
import pandas as pd
import pytest

logger = logging.getLogger(__name__)


def import_csv_to_sqlite(source_csv, destination, table_name="sample_group"):
    """Imports a CSV file to a SQLite database."""
    print(f"Importing {source_csv} to {destination}")
    df = pd.read_csv(source_csv)
    with sqlite3.connect(destination) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    conn.close()


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
    """Create testing_dwh.sqlite, if it doesn't already exist."""
    db = Path(__file__).parent / "testdata/testing_dwh.sqlite"
    if db.exists():
        return
    src = Path(__file__).parent / "testdata/testing_dwh.csv.zst"
    logger.warning(f"Creating temporary SQLite database {db} from {src}.")
    import_csv_to_sqlite(src, db)
