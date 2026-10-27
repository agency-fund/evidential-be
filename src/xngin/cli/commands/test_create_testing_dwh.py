"""Tests of create-testing-dwh that need no warehouse.

See test_create_testing_dwh_postgres.py for the tests that load one.
"""

from compression import zstd
from pathlib import Path

import pytest
import typer
from sqlalchemy import make_url

from xngin.apiserver import flags
from xngin.cli.commands.create_testing_dwh import (
    _build_source,  # noqa: PLC2701
    _provided_ddl,  # noqa: PLC2701
    _template_src,  # noqa: PLC2701
    _TemplateTable,  # noqa: PLC2701
    create_testing_dwh,
)

TESTDATA = Path(__file__).resolve().parent.parent.parent / "apiserver/testdata"


def test_jsonl_is_refused_for_postgres(tmp_path: Path):
    src = tmp_path / "rows.jsonl"
    src.write_text('{"id": 1}\n')
    with pytest.raises(typer.Exit) as exc:
        create_testing_dwh(dsn=flags.XNGIN_DEVDWH_DSN, src=src, table_name="never_created")
    assert exc.value.exit_code == 2


def test_jsonl_requires_bigquery_ddl(tmp_path: Path):
    src = tmp_path / "rows.jsonl.zst"
    src.write_bytes(zstd.compress(b'{"id": 1}\n'))
    with pytest.raises(typer.Exit) as exc:
        create_testing_dwh(dsn="bigquery://some-project/some_dataset", src=src, table_name="never_created")
    assert exc.value.exit_code == 2


def test_a_template_table_loads_its_own_source_into_each_warehouse():
    table = _TemplateTable(src="x.csv", bigquery_src="x.jsonl", redshift_src="x.redshift.csv", table_name="x")
    redshift = make_url("postgresql+psycopg2://u:p@wg.123.us-west-2.redshift-serverless.amazonaws.com:5439/dev")
    assert _template_src(table, redshift) == "x.redshift.csv"
    assert _template_src(table, make_url("bigquery://project/dataset")) == "x.jsonl"
    assert _template_src(table, make_url(flags.XNGIN_DEVDWH_DSN)) == "x.csv"
    assert _template_src(_TemplateTable(src="x.csv", table_name="x"), redshift) == "x.csv"


def test_a_flavored_source_shares_the_flavor_ddl(tmp_path: Path):
    (tmp_path / "x.redshift.ddl").write_text("CREATE TABLE {{table_name}} (id INTEGER);")
    for name in ("x.csv.zst", "x.redshift.csv.zst"):
        source = _build_source(src=tmp_path / name, nrows=None)
        assert _provided_ddl(source, "redshift", '"t"') == 'CREATE TABLE "t" (id INTEGER);'
        assert _provided_ddl(source, "postgres", '"t"') is None
