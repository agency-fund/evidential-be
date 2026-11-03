"""The stress testing warehouses: one 100-row stress test table each for Postgres, Redshift and BigQuery.

Their table, schema and column names need quoting (spaces, dashes, dots, mixed case, non-ASCII letters, punctuation,
reserved words), and their columns use types beyond the plain ones of the other testing warehouses, including
BigQuery's RECORD and REPEATED fields. They exist for exploring that support interactively.
tools/generate_stress_dwh.py writes them to the testdata/stress_dwh.* files that `xngin-cli create-testing-dwh
--template src/xngin/apiserver/testdata/warehouses.json` loads, and bootstrap.py adds them as data sources for
the dev environment's first user.

Every warehouse gets the same rows, generated from a fixed seed, under its own names and types.
"""

import datetime as dt
import json
import random
import uuid
from dataclasses import dataclass
from decimal import Decimal

ROW_COUNT = 100
_SEED = 20261026

FIRST_NAMES = ["Ada", "Bao", "Chiara", "Dmitri", "Émilie", "Farah", "Gustavo", "Hana", "Ibrahim", "Jun"]
REGIONS = ["Île-de-France", "Bayern", "São Paulo", "Ōsaka", "Ontario"]
CLINICS = [f"Clinic {letter}" for letter in "ABCDEFGH"]
CITIES = ["Paris", "München", "São Paulo", "大阪", "Toronto"]
TAGS = ["vip", "new", "returning"]
TIERS = ["bronze", "silver", "gold"]
EVENTS = ["signup", "visit", "purchase"]
# The offsets the source files write time zone-aware values in, including ones that are not whole hours. Warehouses
# store the instant, not the offset.
OFFSETS = [
    dt.UTC,
    dt.timezone(dt.timedelta(hours=5, minutes=45)),
    dt.timezone(dt.timedelta(hours=-3, minutes=-30)),
    dt.timezone(dt.timedelta(hours=-8)),
]
# A leap day, and a time that does not exist in US time zones that switch to daylight saving time that night.
LEAP_DAY = dt.date(2000, 2, 29)
DST_GAP = dt.datetime(2026, 3, 8, 2, 30)
# One login at a known instant, so that a filter around it shows which time zone a value without one is taken in.
KNOWN_LOGIN = dt.datetime(2026, 7, 1, 0, 30, tzinfo=dt.UTC)


def generate_rows() -> list[dict[str, object]]:
    """The rows every stress testing warehouse holds, keyed by the logical column names that Column.key refers to."""
    rng = random.Random(_SEED)  # noqa: S311 - reproducible sample data
    # The dates and times added later draw from their own generator, so the earlier columns keep their values.
    time_rng = random.Random(_SEED + 1)  # noqa: S311 - reproducible sample data
    rows = []
    for i in range(1, ROW_COUNT + 1):
        region = rng.randrange(len(REGIONS))
        signup = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(365))
        minutes, micros = rng.randrange(60 * 24 * 300), time_rng.randrange(10**6)
        # Midnight and the last microsecond of the day, then random times of day.
        check_in = {1: dt.time(0, 0), 2: dt.time(23, 59, 59, 999_999)}.get(i) or dt.time(
            time_rng.randrange(24), time_rng.randrange(60), time_rng.randrange(60), time_rng.randrange(10**6)
        )
        rows.append({
            "id": i,
            "first_name": rng.choice(FIRST_NAMES),
            "email": f"participant{i}@example.org",
            "signup_date": signup,
            "last_login": (
                KNOWN_LOGIN
                if i == 5
                else dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(minutes=minutes, microseconds=micros)
            ).astimezone(OFFSETS[i % len(OFFSETS)]),
            "score": Decimal(rng.randrange(10_000)) / 100,
            "is_active": rng.random() < 0.6,
            # About one income in ten is unknown.
            "income": None if rng.random() < 0.1 else round(rng.gauss(50_000, 15_000), 2),
            "region": REGIONS[region],
            "clinic": CLINICS[rng.randrange(len(CLINICS))],
            "visits": rng.randrange(20),
            "reserved": rng.randrange(100),
            "quoted": rng.choice(["yes", "no", "maybe"]),
            "uuid": str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            "tags": sorted(rng.sample(TAGS, rng.randrange(len(TAGS) + 1))),
            "profile": {"tier": rng.choice(TIERS), "referrals": rng.randrange(5)},
            # Beyond NUMERIC's 29 integer and 9 fractional digits, within BIGNUMERIC's 38 and 38.
            "big_number": Decimal(f"{rng.randrange(10**12) * 10**20}.{i:030d}"),
            "city": CITIES[region],
            "postal_code": f"{rng.randrange(10_000, 99_999)}",
            "latitude": round(rng.uniform(-60, 60), 6),
            "longitude": round(rng.uniform(-180, 180), 6),
            # About one in ten of each is unknown. Births reach back before 1970; appointments are local times, with no
            # time zone, before 1970 and after 2038.
            "birth_date": (
                LEAP_DAY
                if i == 3
                else None
                if time_rng.random() < 0.1
                else dt.date(1930, 1, 1) + dt.timedelta(days=time_rng.randrange(80 * 365))
            ),
            "appointment": (
                DST_GAP
                if i == 4
                else None
                if time_rng.random() < 0.1
                else dt.datetime(1965, 1, 1)
                + dt.timedelta(seconds=time_rng.randrange(80 * 365 * 86_400), microseconds=time_rng.randrange(10**6))
            ),
            "check_in": check_in,
            "check_in_tz": check_in.replace(tzinfo=OFFSETS[i % len(OFFSETS)]),
            "events": [
                {"name": rng.choice(EVENTS), "at": signup + dt.timedelta(days=rng.randrange(30))}
                for _ in range(rng.randrange(4))
            ],
        })
    return rows


@dataclass(frozen=True)
class Column:
    """A column of a stress test table: the logical key of its value in generate_rows(), its name, and its type."""

    key: str
    name: str
    ddl_type: str


@dataclass(frozen=True)
class StressTable:
    schema: str | None
    table: str
    columns: tuple[Column, ...]

    def column(self, key: str) -> Column:
        return next(c for c in self.columns if c.key == key)


# Postgres keeps every name exactly as quoted.
POSTGRES = StressTable(
    schema="Stress Schema",
    table="Participant Data-2026",
    columns=(
        Column("id", "participant.id", "BIGINT PRIMARY KEY"),
        Column("first_name", "First Name", "TEXT"),
        Column("email", "e-mail", "VARCHAR(255)"),
        Column("signup_date", "Signup Date", "DATE"),
        Column("last_login", "last.login", "TIMESTAMPTZ"),
        Column("birth_date", "Date of Birth", "DATE"),
        Column("appointment", "appointment at (local)", "TIMESTAMP"),
        Column("check_in", "check-in time", "TIME"),
        Column("check_in_tz", "check-in time (tz)", "TIMETZ"),
        Column("score", "Score (%)", "NUMERIC(5, 2)"),
        Column("is_active", "isActive", "BOOLEAN"),
        Column("income", "Income $", "DOUBLE PRECISION"),
        Column("region", "région", "TEXT"),
        Column("clinic", "Clinic Name", "TEXT"),
        Column("visits", "visits-count", "INTEGER"),
        Column("reserved", "select", "INTEGER"),
        Column("quoted", 'say "hi"', "TEXT"),
        Column("uuid", "uuid col", "UUID"),
        Column("tags", "tags", "TEXT[]"),
        Column("profile", "profile", "JSONB"),
    ),
)

# Redshift has no UUID, array or JSONB types; SUPER holds semi-structured values, which Redshift parses from a CSV as
# JSON. With enable_case_sensitive_identifier off (the default), Redshift stores and lists every name in lowercase, so
# the data source's search_path must name the schema in lowercase too.
REDSHIFT = StressTable(
    schema="Stress Schema",
    table="Participant Data-2026",
    columns=tuple(
        Column(
            c.key,
            c.name,
            {"UUID": "CHAR(36)", "TEXT[]": "SUPER", "JSONB": "SUPER", "TEXT": "VARCHAR(256)"}.get(
                c.ddl_type, c.ddl_type
            ),
        )
        for c in POSTGRES.columns
    ),
)
REDSHIFT_SEARCH_PATH = '"stress schema"'

# BigQuery column names may not contain dots or "$"; a dot reaches into a RECORD instead, as in address.geo.lat. The
# table lives in the data source's dataset, so it has no schema. "events" is a REPEATED RECORD and "tags" a REPEATED
# field, which are reflected as ARRAY columns and are not usable as fields.
BIGQUERY = StressTable(
    schema=None,
    table="Participant Data-2026",
    columns=(
        Column("id", "Participant ID", "INT64"),
        Column("first_name", "First Name", "STRING"),
        Column("email", "e-mail", "STRING"),
        Column("signup_date", "Signup Date", "DATE"),
        Column("last_login", "Last Login", "TIMESTAMP"),
        Column("birth_date", "Date of Birth", "DATE"),
        Column("appointment", "Appointment At", "DATETIME"),
        Column("check_in", "Check-in Time", "TIME"),
        Column("score", "Score %", "NUMERIC(5, 2)"),
        Column("is_active", "isActive", "BOOL"),
        Column("income", "Income", "FLOAT64"),
        Column("region", "région", "STRING"),
        Column("clinic", "Clinic Name", "STRING"),
        Column("visits", "visits-count", "INT64"),
        Column("reserved", "select", "INT64"),
        Column("quoted", "it's", "STRING"),
        Column("uuid", "uuid col", "STRING"),
        Column("tags", "tags", "ARRAY<STRING>"),
        Column("profile", "profile", "JSON"),
        Column("big_number", "Big Number", "BIGNUMERIC"),
        Column("location", "location", "GEOGRAPHY"),
        Column(
            "address",
            "address",
            "STRUCT<city STRING, `postal code` STRING, geo STRUCT<lat FLOAT64, lng FLOAT64>>",
        ),
        Column("events", "events", "ARRAY<STRUCT<name STRING, `at` DATE>>"),
    ),
)


@dataclass(frozen=True)
class ExperimentFields:
    """The columns a bootstrapped experiment on a stress test table uses. RECORD sub-fields are named by their path."""

    primary_key: str
    metric: str
    filter_field: str
    filter_values: tuple[object, ...]
    # A time zone-aware timestamp; the experiment includes the participants who logged in since LOGGED_IN_SINCE.
    last_login: str
    strata: str


# A UTC instant, as filter values for timestamps must be.
LOGGED_IN_SINCE = "2026-02-01T00:00:00Z"


POSTGRES_EXPERIMENT = ExperimentFields(
    primary_key="participant.id",
    metric="visits-count",
    filter_field="isActive",
    filter_values=(True,),
    last_login="last.login",
    strata="région",
)
# Redshift lists isActive as isactive; the other names are already lowercase.
REDSHIFT_EXPERIMENT = ExperimentFields(
    primary_key="participant.id",
    metric="visits-count",
    filter_field="isactive",
    filter_values=(True,),
    last_login="last.login",
    strata="région",
)
BIGQUERY_EXPERIMENT = ExperimentFields(
    primary_key="Participant ID",
    metric="visits-count",
    filter_field="address.city",
    filter_values=("Paris", "München", "Toronto"),
    last_login="Last Login",
    strata="région",
)


def bigquery_row(row: dict[str, object]) -> dict[str, object]:
    """A row as BigQuery's JSON load job expects it, under BIGQUERY's column names."""

    def scalar(key: str) -> object:
        value = row[key]
        if isinstance(value, Decimal):
            # Fixed-point, since BigQuery does not parse exponents in a NUMERIC or BIGNUMERIC value.
            return format(value, "f")
        if isinstance(value, dt.date | dt.time):
            return value.isoformat()
        return value

    values: dict[str, object] = {}
    for column in BIGQUERY.columns:
        match column.key:
            case "profile":
                values[column.name] = json.dumps(row["profile"])
            case "location":
                values[column.name] = f"POINT({row['longitude']} {row['latitude']})"
            case "address":
                values[column.name] = {
                    "city": row["city"],
                    "postal code": row["postal_code"],
                    "geo": {"lat": row["latitude"], "lng": row["longitude"]},
                }
            case "events":
                events = row["events"]
                if not isinstance(events, list):
                    raise TypeError(f"events must be a list, not {type(events).__name__}")
                values[column.name] = [{"name": e["name"], "at": e["at"].isoformat()} for e in events]
            case _:
                values[column.name] = scalar(column.key)
    return values
