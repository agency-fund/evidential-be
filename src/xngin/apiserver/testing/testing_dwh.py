"""Utilities for the creation of the local testing database.

This corresponds to the "testing" config specified in xngin.testing.settings.json.
"""

import hashlib
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

TESTING_DWH_SQLITE_PATH = Path(__file__).parent.parent / "testdata/testing_dwh.db"
TESTING_DWH_RAW_DATA = Path(__file__).parent.parent / "testdata/testing_dwh.csv.zst"


class FailedSettingVersionError(Exception):
    """Raised when setting schema version on the test database fails."""

    pass


def import_csv_to_sqlite(
    source_csv: Path,
    src_version: int,
    db_path: Path,
    table_name="test_participant_type",
):
    """Imports a CSV file to a SQLite database.

    We store src_version in the database's `user_version` field so that we can automatically recreate it if the version
    changes.
    """
    logger.info(f"Importing {source_csv} to {db_path}")
    df = pd.read_csv(source_csv)
    with closing(sqlite3.connect(db_path)) as conn:
        # TODO: Replace to_sql with something that lets us mark a column as a primary key.
        row_count = df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(
            f"{row_count:,} rows inserted into {table_name} (data version {src_version})."
        )
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA user_version = {src_version}")
        if cursor.execute("PRAGMA user_version").fetchone()[0] != src_version:
            raise FailedSettingVersionError()


def read_user_version_from_sqlite(db_path: Path):
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        return cursor.execute("PRAGMA user_version").fetchone()[0]


def compact_hash(path: Path):
    """Computes a hash of the input CSV so that we can determine whether to re-create the test warehouse."""
    with open(path, "rb") as source:
        h = hashlib.blake2b(digest_size=2)
        h.update(source.read())
    return int.from_bytes(h.digest())


def create_dwh_sqlite_database(
    src: Path = TESTING_DWH_RAW_DATA,
    dest: Path = TESTING_DWH_SQLITE_PATH,
    *,
    force: bool = False,
):
    """Imports src into a sqlite database dest.

    Unless force is set, import will be skipped if the destination exists already.
    """
    src_version = compact_hash(src)
    if (
        not force
        and dest.exists()
        and read_user_version_from_sqlite(dest) == src_version
    ):
        logger.info(f"{dest} exists (data version {src_version}), skipping import.")
        return
    if not src.exists():
        raise FileNotFoundError(src)
    import_csv_to_sqlite(src, src_version, dest)
