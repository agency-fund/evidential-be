"""Utilities for the creation of the local testing database.

This corresponds to the "testing" config specified in xngin.testing.settings.json.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TESTING_DWH_SQLITE_PATH = Path(__file__).parent.parent / "testdata/testing_dwh.db"
TESTING_DWH_RAW_DATA = Path(__file__).parent.parent / "testdata/testing_dwh.csv.zst"


def import_csv_to_sqlite(source_csv, destination, table_name="test_unit_type"):
    """Imports a CSV file to a SQLite database."""
    logger.info(f"Importing {source_csv} to {destination}")
    df = pd.read_csv(source_csv)
    with sqlite3.connect(destination) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        cursor = conn.cursor()
        row_count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info(f"{row_count:,} rows inserted into {table_name}.")
        conn.commit()
    conn.close()


def create_dwh_sqlite_database(
    src: Path = TESTING_DWH_RAW_DATA,
    dest: Path = TESTING_DWH_SQLITE_PATH,
    *,
    force: bool = False,
):
    """Imports src into a sqlite database dest.

    Unless force is set, import will be skipped if the destination exists already.
    """
    if not force and dest.exists():
        logger.info(f"{dest} exists, skipping import.")
        return
    if not src.exists():
        raise FileNotFoundError(src)
    import_csv_to_sqlite(src, dest)
