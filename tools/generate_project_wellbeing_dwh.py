#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy==2.2.3",
#     "typer==0.15.1",
# ]
# ///
"""Generates the Project Wellbeing demo data warehouse: a fictional school-health program.

About 6,000 synthetic students in 120 schools. The file holds attribute and baseline columns only; outcome columns
are left empty and are filled by the demo seeding code after experiments assign participants (see
src/xngin/apiserver/storage/demo_seed.py). All values are synthetic; there are no names or contact fields.

Outputs:
  - src/xngin/apiserver/testdata/project_wellbeing_dwh.csv.zst  (compressed CSV data)
  - src/xngin/apiserver/testdata/project_wellbeing_dwh.postgres.ddl
  - src/xngin/apiserver/testdata/project_wellbeing_dwh.redshift.ddl
"""

from __future__ import annotations

import csv
import io
from compression import zstd
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA_DIR = REPO_ROOT / "src" / "xngin" / "apiserver" / "testdata"
BASENAME = "project_wellbeing_dwh"

# Iron adherence: school effect plus student noise, sized so that Evidential's one-way ANOVA ICC estimator
# (xngin.stats.cluster_icc) reads 0.15. With 50 students per school that gives a design effect of 1 + 49 * 0.15 = 8.35.
IRON_MEAN = 55.0
IRON_STUDENT_SD = 18.4
IRON_ICC = 0.15

# Reading: an SD of ~53 makes the minimum detectable effect ~10 points for 900 students split across two arms
# (MDE ~= 2.8 * SD * sqrt(2 / 450)).
READING_MEAN = 150.0
READING_SD = 53.0


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str  # Postgres type; Redshift uses the same names for the types used here
    nullable: bool
    is_unique_id: bool = False
    comment: str = ""


# Column order here is the CSV and DDL order.
COLUMNS = [
    Column("id", "BIGINT", nullable=False, is_unique_id=True),
    Column("school_id", "INTEGER", nullable=False, comment="120 schools of 50 students"),
    Column("grade", "INTEGER", nullable=False),
    Column("gender", "VARCHAR(16)", nullable=False),
    Column("age", "INTEGER", nullable=False),
    Column("has_smartphone_household", "BOOLEAN", nullable=False),
    Column("failed_vision_screen", "BOOLEAN", nullable=False),
    Column("reading_measured_last_term", "BOOLEAN", nullable=False),
    Column("baseline_reading_score", "DOUBLE PRECISION", nullable=False),
    Column(
        "reading_score_endline", "DOUBLE PRECISION", nullable=True, comment="NULL unless reading_measured_last_term"
    ),
    Column("baseline_iron_adherence_pct", "DOUBLE PRECISION", nullable=False, comment="school ICC ~0.15"),
    # Seed-owned outcome columns: empty here, written by the demo seeding code.
    Column("glasses_attendance_pct", "DOUBLE PRECISION", nullable=True, comment="seed-owned"),
    Column("iron_adherence_pct_pilot", "DOUBLE PRECISION", nullable=True, comment="seed-owned"),
    Column("iron_adherence_pct_support", "DOUBLE PRECISION", nullable=True, comment="seed-owned"),
    Column("iron_support_hb_change", "DOUBLE PRECISION", nullable=True, comment="seed-owned"),
    Column("health_checkup_booked", "BOOLEAN", nullable=True, comment="seed-owned"),
    Column("health_portal_logins", "INTEGER", nullable=True, comment="seed-owned"),
    Column("brushing_days_dental_kit", "DOUBLE PRECISION", nullable=True, comment="seed-owned"),
]

app = typer.Typer()


def _standardize(values: np.ndarray) -> np.ndarray:
    """Shifts and scales values to exactly mean 0 and sample standard deviation 1."""
    return (values - values.mean()) / values.std(ddof=1)


def generate(n_schools: int, students_per_school: int, seed: int) -> dict[str, list]:
    """Returns column name -> list of values (None for NULL), rows ordered by id.

    Every random draw comes from one generator in a fixed order so that the output is identical for the same inputs.
    """
    rng = np.random.default_rng(seed)
    n = n_schools * students_per_school

    ids = np.arange(1, n + 1)
    school_id = np.repeat(np.arange(1, n_schools + 1), students_per_school)
    grade = rng.integers(1, 9, size=n)
    gender = rng.choice(["female", "male"], size=n)
    age = grade + 5 + rng.integers(0, 2, size=n)
    has_smartphone = rng.random(n) < 0.55
    failed_vision = rng.random(n) < 0.45
    # Only some students who failed the screening already have an endline reading score from last term; the rest
    # are the not-yet-measured pool that one-time metric mode assigns.
    measured_last_term = failed_vision & (rng.random(n) < 0.25)

    baseline_reading = np.clip(rng.normal(READING_MEAN, READING_SD, size=n), 0, 400)
    endline_noise = rng.normal(0, READING_SD * 0.2, size=n)
    reading_endline = np.clip(baseline_reading + endline_noise, 0, 400)

    # Student noise is centered exactly within each school (below), so school means equal the school effects and the
    # ANOVA estimator gives ICC = (m*v - s2) / (m*v + (m - 1)*s2) for school-effect variance v and student variance s2.
    # Solving for v: v = s2 * (1 + (m - 1) * ICC) / (m * (1 - ICC)).
    student_sd = IRON_STUDENT_SD
    m = students_per_school
    school_sd = np.sqrt(student_sd**2 * (1 + (m - 1) * IRON_ICC) / (m * (1 - IRON_ICC)))
    # Rescale both draws to their exact target spreads. With only 120 schools, raw sampling noise moves the measured
    # ICC by several hundredths; the demo story depends on it reading as 0.15.
    school_effect = _standardize(rng.normal(size=n_schools)) * school_sd
    student_noise = np.empty(n)
    for school in range(n_schools):
        rows = slice(school * students_per_school, (school + 1) * students_per_school)
        student_noise[rows] = _standardize(rng.normal(size=students_per_school)) * student_sd
    iron = IRON_MEAN + school_effect[school_id - 1] + student_noise
    iron = np.clip(iron, 0, 100)

    data: dict[str, list] = {
        "id": ids.tolist(),
        "school_id": school_id.tolist(),
        "grade": grade.tolist(),
        "gender": gender.tolist(),
        "age": age.tolist(),
        "has_smartphone_household": has_smartphone.tolist(),
        "failed_vision_screen": failed_vision.tolist(),
        "reading_measured_last_term": measured_last_term.tolist(),
        "baseline_reading_score": np.round(baseline_reading, 1).tolist(),
        "reading_score_endline": [
            round(float(v), 1) if measured else None
            for v, measured in zip(reading_endline, measured_last_term, strict=True)
        ],
        "baseline_iron_adherence_pct": np.round(iron, 1).tolist(),
    }
    for col in COLUMNS:
        data.setdefault(col.name, [None] * n)
    return data


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(data: dict[str, list], path: Path) -> None:
    names = [c.name for c in COLUMNS]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(names)
    for row in zip(*(data[name] for name in names), strict=True):
        writer.writerow([format_value(v) for v in row])
    # One-shot compress() always writes a complete frame (a bare ZstdCompressor.compress() call does not flush).
    path.write_bytes(zstd.compress(buf.getvalue().encode("utf-8"), level=19))


def ddl(include_primary_key: bool) -> str:
    lines = []
    for col in COLUMNS:
        line = f"  {col.name} {col.sql_type}"
        if col.is_unique_id and include_primary_key:
            line += " PRIMARY KEY"
        if not col.nullable:
            line += " NOT NULL"
        lines.append(line)
    header = "-- Project Wellbeing demo data warehouse. Generated by tools/generate_project_wellbeing_dwh.py\n"
    return header + "CREATE TABLE {{table_name}} (\n" + ",\n".join(lines) + "\n);\n"


@app.command()
def main(
    n_schools: int = typer.Option(120, help="Number of schools."),
    students_per_school: int = typer.Option(50, help="Students per school."),
    seed: int = typer.Option(20261012, help="Random seed for reproducibility."),
):
    data = generate(n_schools, students_per_school, seed)
    n = n_schools * students_per_school
    pool = sum(
        1 for f, m in zip(data["failed_vision_screen"], data["reading_measured_last_term"], strict=True) if f and not m
    )
    print(f"Generated {n} students in {n_schools} schools; not-yet-measured reading pool: {pool}")

    write_csv(data, TESTDATA_DIR / f"{BASENAME}.csv.zst")
    # Redshift does not enforce primary keys, so only the Postgres DDL declares one (same as the other test DWHs).
    (TESTDATA_DIR / f"{BASENAME}.postgres.ddl").write_text(ddl(include_primary_key=True))
    (TESTDATA_DIR / f"{BASENAME}.redshift.ddl").write_text(ddl(include_primary_key=False))
    print(f"Wrote {BASENAME}.csv.zst and DDLs to {TESTDATA_DIR}")


if __name__ == "__main__":
    app()
