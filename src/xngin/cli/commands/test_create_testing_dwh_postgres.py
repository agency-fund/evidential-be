"""Tests of create-testing-dwh against the local Postgres testing warehouse."""

import csv
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from sqlalchemy import make_url

from xngin.apiserver import flags
from xngin.cli.commands.create_testing_dwh import (
    create_testing_dwh,
)

# Names that break unless quoted, including "%", which SQLAlchemy's Postgres quoting would write as "%%".
AWKWARD_COLUMNS = ["id", "Score (%)", 'say "hi"', "with, comma", "MixedCase"]


AWKWARD_ROWS = [["1", "12.5", "yes", "a,b", "x"], ["2", "", 'a "quote"', "", "y"]]


def _write_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(AWKWARD_COLUMNS)
        writer.writerows(AWKWARD_ROWS)


def _psycopg_dsn() -> str:
    """The testing DWH's DSN, in the form psycopg.connect accepts."""
    return make_url(flags.XNGIN_DEVDWH_DSN).set(drivername="postgresql").render_as_string(hide_password=False)


@pytest.fixture(name="schema_name")
def fixture_schema_name() -> Iterator[str]:
    """A new schema name that needs quoting, for a test to load into the testing DWH. It is dropped afterwards."""
    schema_name = f'Test %"Schema" {uuid.uuid4().hex[:8]}'
    yield schema_name
    with psycopg.connect(_psycopg_dsn(), autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name.replace('"', '""')}" CASCADE')


def _columns_and_rows(schema_name: str, table_name: str) -> tuple[list[str], list[tuple]]:
    with psycopg.connect(_psycopg_dsn()) as conn:
        columns = [
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s "
                "ORDER BY ordinal_position",
                (schema_name, table_name),
            )
        ]
        quoted = ", ".join('"' + c.replace('"', '""') + '"' for c in columns)
        schema = schema_name.replace('"', '""')
        table = table_name.replace('"', '""')
        rows = conn.execute(f'SELECT {quoted} FROM "{schema}"."{table}" ORDER BY 1').fetchall()
    return columns, rows


@pytest.mark.parametrize("driver", ["postgresql+psycopg", "postgresql+psycopg2"])
@pytest.mark.parametrize("with_ddl", [True, False])
def test_names_that_need_quoting_load_into_postgres(tmp_path: Path, schema_name: str, driver: str, with_ddl: bool):
    src = tmp_path / "awkward.csv"
    _write_csv(src)
    if with_ddl:
        (tmp_path / "awkward.postgres.ddl").write_text(
            'CREATE TABLE {{table_name}} (id INTEGER, "Score (%)" NUMERIC, "say ""hi""" TEXT, '
            '"with, comma" TEXT, "MixedCase" TEXT);'
        )
    table_name = 'Table %"1"-x'
    dsn = make_url(flags.XNGIN_DEVDWH_DSN).set(drivername=driver).render_as_string(hide_password=False)

    create_testing_dwh(dsn=dsn, src=src, schema_name=schema_name, table_name=table_name, views='a %"view"')

    columns, rows = _columns_and_rows(schema_name, table_name)
    assert columns == AWKWARD_COLUMNS
    assert [[str(v) if v is not None else "" for v in row] for row in rows] == [
        ["1", "12.5", "yes", "a,b", "x"],
        ["2", "", 'a "quote"', "", "y"],
    ]
    view_columns, view_rows = _columns_and_rows(schema_name, 'a %"view"')
    assert view_columns == AWKWARD_COLUMNS
    assert view_rows == rows

    # The table is found, and left alone, when it exists.
    create_testing_dwh(dsn=dsn, src=src, schema_name=schema_name, table_name=table_name, allow_existing=True)
    assert _columns_and_rows(schema_name, table_name) == (columns, rows)


def test_a_template_table_has_its_own_schema_and_bigquery_source(tmp_path: Path, schema_name: str):
    _write_csv(tmp_path / "awkward.csv")
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps({
            "tables": [
                {
                    "src": "awkward.csv",
                    # Postgres loads src; a .jsonl could only be loaded into BigQuery.
                    "bigquery_src": "awkward.jsonl",
                    "schema_name": schema_name,
                    "table_name": "From Template",
                }
            ]
        })
    )

    create_testing_dwh(dsn=flags.XNGIN_DEVDWH_DSN, template=template, schema_name="ignored because the table has one")

    columns, rows = _columns_and_rows(schema_name, "From Template")
    assert columns == AWKWARD_COLUMNS
    assert len(rows) == len(AWKWARD_ROWS)
