import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import Connection, Select, MetaData, Table, select, func


def import_csv_to_sqlite(source_csv, destination, table_name="imported"):
    """Imports a CSV file to a SQLite database.

    If destination is present, we assume the import succeeded previously.
    """
    if Path(destination).exists():
        return
    print(f"Importing {source_csv} to {destination}")
    df = pd.read_csv(source_csv)
    with sqlite3.connect(destination) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()


def print_and_execute(conn: Connection, stmt: Select[Any]):
    result = conn.execute(stmt.limit(3))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    print(f"\nResults of query: {stmt}\n")
    print(df)


def test_sqlite_example():
    """
    Reads a CSV file into a SQLite table and then queries that table using SQLAlchemy's support for "reflecting" the
    schema of an existing table.

    (This is not a test.)

    The import of test_data.csv generates the following SQLite schema (as determined by pandas read_csv):

    CREATE TABLE IF NOT EXISTS "imported" (
      "gender" TEXT,
      "ethnicity" TEXT,
      "first_name" TEXT,
      "last_name" TEXT,
      "id" INTEGER,
      "income" REAL,
      "potential_0" REAL,
      "potential_1" INTEGER,
      "is_recruited" INTEGER,
      "is_registered" INTEGER,
      "is_onboarded" INTEGER,
      "is_engaged" INTEGER,
      "is_retained" INTEGER,
      "baseline_income" REAL,
      "current_income" REAL
    );
    """
    temp_database = Path(tempfile.gettempdir()) / "dwh.sqlite"
    import_csv_to_sqlite(Path(__file__).parent / "testdata/dwh.csv.zst", temp_database)

    engine = sqlalchemy.create_engine(f"sqlite:///{temp_database}?mode=ro")
    metadata = MetaData()

    dwh = Table("imported", metadata, autoload_with=engine)

    # Example queries
    with engine.connect() as conn:
        # Refer to columns by name
        stmt = select(dwh).where(dwh.c.gender == "Male")
        print_and_execute(conn, stmt)

        # Refer to columns by string literal
        stmt = select(dwh).where(dwh.c["gender"] == "Female")
        print_and_execute(conn, stmt)

        # Select a subset of columns
        stmt = (
            select(dwh.c.id, dwh.c.last_name, dwh.c.current_income)
            .where(dwh.c.current_income <= 30000)
            .where(dwh.c.current_income >= 25000)
        )
        print_and_execute(conn, stmt)

        # Aggregations
        stmt = select(
            dwh.c.gender,
            func.min(dwh.c.income),
            func.avg(dwh.c.income),
            func.max(dwh.c.income),
        ).group_by(dwh.c.gender)
        print_and_execute(conn, stmt)
