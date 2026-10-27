"""Generates the stress testing warehouse files that `xngin-cli create-testing-dwh` loads.

The rows, names and types come from xngin.apiserver.testing.stress_dwh_def. Run from the repository root:

    uv run python tools/generate_stress_dwh.py

Outputs, in src/xngin/apiserver/testdata/ (regenerating them produces identical files):
  - stress_dwh.csv.zst: the rows for Postgres, with a header of the Postgres column names.
  - stress_dwh.redshift.csv.zst: the same rows for Redshift, whose SUPER columns parse JSON arrays and objects.
  - stress_dwh.postgres.ddl, stress_dwh.redshift.ddl
  - stress_dwh.jsonl.zst: the rows for BigQuery, as newline-delimited JSON (BigQuery's CSV loads cannot populate
    RECORD or REPEATED columns).
  - stress_dwh.bigquery.ddl
"""

import csv
import datetime as dt
import io
import json
from compression import zstd
from decimal import Decimal
from pathlib import Path

from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.stress_dwh_def import StressTable

TESTDATA_DIR = Path(__file__).resolve().parent.parent / "src/xngin/apiserver/testdata"
ZSTD_LEVEL = 19


def _csv_value(value: object, *, json_arrays: bool) -> object:
    """A value as Postgres's or Redshift's CSV parser expects it. None becomes an empty field, which both read as NULL.

    Arrays are written as JSON for Redshift's SUPER (json_arrays), else as Postgres array literals for TEXT[]. Objects
    are written as JSON, which suits both JSONB and SUPER.
    """
    match value:
        case bool():
            return "true" if value else "false"
        case Decimal():
            return format(value, "f")
        case dt.datetime():
            return value.isoformat(sep=" ")
        case dt.date():
            return value.isoformat()
        case dt.time():
            return value.isoformat()
        case list() if json_arrays:
            return json.dumps(value, ensure_ascii=False)
        case list():
            return "{" + ",".join('"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"' for v in value) + "}"
        case dict():
            return json.dumps(value, ensure_ascii=False)
        case _:
            return value


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _ddl(table: StressTable, quote) -> str:
    columns = ",\n".join(f"  {quote(c.name)} {c.ddl_type}" for c in table.columns)
    return "CREATE TABLE {{table_name}} (\n" + columns + "\n);\n"


def _write_zstd(path: Path, text: str) -> None:
    path.write_bytes(zstd.compress(text.encode(), level=ZSTD_LEVEL))
    print(f"Wrote {path}.")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text)
    print(f"Wrote {path}.")


def main() -> None:
    rows = stress_dwh_def.generate_rows()

    for table, name, json_arrays in (
        (stress_dwh_def.POSTGRES, "stress_dwh.csv.zst", False),
        (stress_dwh_def.REDSHIFT, "stress_dwh.redshift.csv.zst", True),
    ):
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([c.name for c in table.columns])
        writer.writerows([_csv_value(row[c.key], json_arrays=json_arrays) for c in table.columns] for row in rows)
        _write_zstd(TESTDATA_DIR / name, buf.getvalue())
    _write_text(TESTDATA_DIR / "stress_dwh.postgres.ddl", _ddl(stress_dwh_def.POSTGRES, _quote))
    _write_text(TESTDATA_DIR / "stress_dwh.redshift.ddl", _ddl(stress_dwh_def.REDSHIFT, _quote))

    jsonl = "".join(json.dumps(stress_dwh_def.bigquery_row(row), ensure_ascii=False) + "\n" for row in rows)
    _write_zstd(TESTDATA_DIR / "stress_dwh.jsonl.zst", jsonl)
    _write_text(TESTDATA_DIR / "stress_dwh.bigquery.ddl", _ddl(stress_dwh_def.BIGQUERY, lambda name: f"`{name}`"))


if __name__ == "__main__":
    main()
