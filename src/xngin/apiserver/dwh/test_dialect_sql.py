"""Tests that exercise the query generation stack (our types, our query composition code, SQLAlchemy's handling of
the generated queries, and variants between dialects).

Datetimes and timestamp type fields are interesting because of the variety of implementation details between data
storage systems. Some don't support DATETIME types, some format datetime string literals differently than others, etc.

Common functions can have different names, too; e.g. one database's RANDOM can be another's RAND.

None of the tests in this file actually execute queries -- it tests the query generation, but not the query execution.
"""

from datetime import UTC, datetime
import re
from dataclasses import dataclass

import pytest
import sqlalchemy
import sqlalchemy_bigquery
from sqlalchemy import TIMESTAMP, Boolean, DateTime, Float, Integer, String, Table
from sqlalchemy.dialects.postgresql import psycopg, psycopg2
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.sql.ddl import CreateTable
from xngin.apiserver.routers.stateless_api_types import (
    Arm,
    AudienceSpecFilter,
    BaseDesignSpec,
    DesignSpecMetricRequest,
    Relation,
)
from xngin.apiserver.conftest import DbType
from xngin.apiserver.dwh.queries import compose_query, create_query_filters_from_spec


class HelpfulBase(DeclarativeBase):
    @classmethod
    def get_table(cls) -> Table:
        """Helper to return a sqlalchemy.schema.Table"""
        # Also gets around mypy typing issues, e.g. get() can return none, and SampleTable.__table__
        # is of type FromClause, but we know it's a Table and must exist.
        table = HelpfulBase.metadata.tables.get(cls.__tablename__)
        assert table is not None
        return table


class Datetimes(HelpfulBase):
    __tablename__ = "dtt"

    id = mapped_column(Integer, primary_key=True, autoincrement=False)
    dt_col = mapped_column(DateTime(timezone=False), nullable=False)
    ts_col = mapped_column(TIMESTAMP(timezone=False), nullable=False)


@dataclass
class DateTimeTestCase:
    dialect: sqlalchemy.engine.Dialect
    ddl: str
    sql: str


DATETIME_SCENARIOS = [
    DateTimeTestCase(
        sqlalchemy_bigquery.dialect(),
        "CREATE TABLE `dtt` ( `id` INT64 NOT NULL, `dt_col` DATETIME NOT NULL, `ts_col` TIMESTAMP NOT NULL)",
        "SELECT `dtt`.`id`, `dtt`.`dt_col`, `dtt`.`ts_col` "
        "FROM `dtt` "
        "WHERE `dtt`.`ts_col` >= TIMESTAMP '2020-01-01 00:00:00' "
        "AND `dtt`.`dt_col` BETWEEN DATETIME '2023-06-01 12:34:56' AND DATETIME '2024-01-01 00:00:00' "
        "ORDER BY rand() "
        "LIMIT 2",
    ),
    DateTimeTestCase(
        psycopg2.dialect(),
        "CREATE TABLE dtt ( id INTEGER NOT NULL, dt_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, ts_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, PRIMARY KEY (id))",
        "SELECT dtt.id, dtt.dt_col, dtt.ts_col "
        "FROM dtt "
        "WHERE dtt.ts_col >= '2020-01-01 00:00:00' "
        "AND dtt.dt_col BETWEEN '2023-06-01 12:34:56' AND '2024-01-01 00:00:00' "
        "ORDER BY random()  LIMIT 2",
    ),
    DateTimeTestCase(
        psycopg.dialect(),
        "CREATE TABLE dtt ( id INTEGER NOT NULL, dt_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, ts_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, PRIMARY KEY (id))",
        "SELECT dtt.id, dtt.dt_col, dtt.ts_col "
        "FROM dtt "
        "WHERE dtt.ts_col >= '2020-01-01 00:00:00' "
        "AND dtt.dt_col BETWEEN '2023-06-01 12:34:56' AND '2024-01-01 00:00:00' "
        "ORDER BY random()  LIMIT 2",
    ),
]


def make_design_spec(filters: list[AudienceSpecFilter]) -> BaseDesignSpec:
    """Makes a test BaseDesignSpec just for filter testing, with everything else defaults."""
    return BaseDesignSpec(
        filters=filters,
        # Other fields below should be ignored in tests.
        participant_type="ignored",
        experiment_type="preassigned",
        experiment_name="ignored",
        description="ignored",
        start_date=datetime(2025, 1, 1, tzinfo=UTC),
        end_date=datetime(2025, 2, 1, tzinfo=UTC),
        arms=[Arm(arm_name="A"), Arm(arm_name="B")],
        metrics=[DesignSpecMetricRequest(field_name="ignored", metric_pct_change=1)],
        strata_field_names=[],
    )


@pytest.mark.parametrize("testcase", DATETIME_SCENARIOS)
def test_datetimes(testcase: DateTimeTestCase):
    """Exercises various SQLAlchemy dialects handling of datetime and timestamp types."""
    sa_table = Datetimes.get_table()
    q = compose_query(
        sa_table,
        2,
        create_query_filters_from_spec(
            sa_table,
            make_design_spec([
                AudienceSpecFilter(
                    field_name="ts_col",
                    relation=Relation.BETWEEN,
                    value=["2020-01-01 00:00:00", None],
                ),
                AudienceSpecFilter(
                    field_name="dt_col",
                    relation=Relation.BETWEEN,
                    value=["2023-06-01T12:34:56", "2024-01-01 00:00:00Z"],
                ),
            ]),
        ),
    )
    ddl = str(CreateTable(sa_table).compile(dialect=testcase.dialect))
    normalized_ddl = re.sub(r"\s+", " ", ddl.replace("\n", "").strip())
    assert normalized_ddl == testcase.ddl, f"DDL: {normalized_ddl}"
    sql = str(
        q.compile(
            dialect=testcase.dialect,
            compile_kwargs={"literal_binds": True},
        )
    )
    assert sql.replace("\n", "") == testcase.sql, (
        f"DIALECT {type(testcase.dialect)}\nSQL = {sql}"
    )


@dataclass
class WhereTestCase:
    filters: list[AudienceSpecFilter]
    where: dict[DbType, str]

    def __str__(self):
        return " and ".join([
            f"{f.field_name} {f.relation.name} {f.value}" for f in self.filters
        ])


class WhereTable(HelpfulBase):
    __tablename__ = "tt"

    id = mapped_column(Integer, primary_key=True, autoincrement=False)
    int_col = mapped_column(Integer, nullable=False)
    float_col = mapped_column(Float, nullable=False)
    bool_col = mapped_column(Boolean, nullable=False)
    string_col = mapped_column(String, nullable=False)
    experiment_ids = mapped_column(String, nullable=False)
    dt_col = mapped_column(DateTime, nullable=False)
    ts_col = mapped_column(TIMESTAMP, nullable=False)


WHERE_TESTCASES = [
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="float_col", relation=Relation.EXCLUDES, value=[2, 3]
            )
        ],
        where={
            DbType.RS: "tt.float_col IS NULL OR (tt.float_col NOT IN (2, 3)) ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.float_col IS NULL OR (tt.float_col NOT IN (2, 3)) ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`float_col` IS NULL OR (`tt`.`float_col` NOT IN (2, 3)) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.EXCLUDES,
                value=[None],
            )
        ],
        where={
            DbType.RS: "tt.int_col IS NOT NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col IS NOT NULL ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` IS NOT NULL ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.EXCLUDES,
                value=[1],
            )
        ],
        where={
            DbType.RS: "tt.int_col IS NULL OR (tt.int_col NOT IN (1)) ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col IS NULL OR (tt.int_col NOT IN (1)) ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` IS NULL OR (`tt`.`int_col` NOT IN (1)) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.EXCLUDES,
                value=[None, 1],
            )
        ],
        where={
            DbType.RS: "tt.int_col IS NOT NULL AND (tt.int_col NOT IN (1)) ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col IS NOT NULL AND (tt.int_col NOT IN (1)) ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` IS NOT NULL AND (`tt`.`int_col` NOT IN (1)) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="dt_col",
                relation=Relation.EXCLUDES,
                value=["2024-01-01"],
            )
        ],
        where={
            DbType.RS: "tt.dt_col IS NULL OR (tt.dt_col NOT IN ('2024-01-01 00:00:00')) ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.dt_col IS NULL OR (tt.dt_col NOT IN ('2024-01-01 00:00:00')) ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`dt_col` IS NULL OR (`tt`.`dt_col` NOT IN (DATETIME '2024-01-01 00:00:00')) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="dt_col",
                relation=Relation.EXCLUDES,
                value=["2024-01-01", None],
            )
        ],
        where={
            DbType.RS: "tt.dt_col IS NOT NULL AND (tt.dt_col NOT IN ('2024-01-01 00:00:00')) ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.dt_col IS NOT NULL AND (tt.dt_col NOT IN ('2024-01-01 00:00:00')) ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`dt_col` IS NOT NULL AND (`tt`.`dt_col` NOT IN (DATETIME '2024-01-01 00:00:00')) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[None],
            )
        ],
        where={
            DbType.RS: "tt.int_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` IS NULL ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[42],
            )
        ],
        where={
            DbType.RS: "NOT (tt.int_col IS NULL OR (tt.int_col NOT IN (42))) ORDER BY random()  LIMIT 3",
            DbType.PG: "NOT (tt.int_col IS NULL OR (tt.int_col NOT IN (42))) ORDER BY random()  LIMIT 3",
            DbType.BQ: "NOT (`tt`.`int_col` IS NULL OR (`tt`.`int_col` NOT IN (42))) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[None, 1],
            )
        ],
        where={
            DbType.RS: "NOT (tt.int_col IS NOT NULL AND (tt.int_col NOT IN (1))) ORDER BY random()  LIMIT 3",
            DbType.PG: "NOT (tt.int_col IS NOT NULL AND (tt.int_col NOT IN (1))) ORDER BY random()  LIMIT 3",
            DbType.BQ: "NOT (`tt`.`int_col` IS NOT NULL AND (`tt`.`int_col` NOT IN (1))) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="dt_col",
                relation=Relation.INCLUDES,
                value=["2024-01-01"],
            )
        ],
        where={
            DbType.RS: "NOT (tt.dt_col IS NULL OR (tt.dt_col NOT IN ('2024-01-01 00:00:00'))) ORDER BY random()  LIMIT 3",
            DbType.PG: "NOT (tt.dt_col IS NULL OR (tt.dt_col NOT IN ('2024-01-01 00:00:00'))) ORDER BY random()  LIMIT 3",
            DbType.BQ: "NOT (`tt`.`dt_col` IS NULL OR (`tt`.`dt_col` NOT IN (DATETIME '2024-01-01 00:00:00'))) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="dt_col",
                relation=Relation.INCLUDES,
                value=["2024-01-01", None],
            )
        ],
        where={
            DbType.RS: "NOT (tt.dt_col IS NOT NULL AND (tt.dt_col NOT IN ('2024-01-01 00:00:00'))) ORDER BY random()  LIMIT 3",
            DbType.PG: "NOT (tt.dt_col IS NOT NULL AND (tt.dt_col NOT IN ('2024-01-01 00:00:00'))) ORDER BY random()  LIMIT 3",
            DbType.BQ: "NOT (`tt`.`dt_col` IS NOT NULL AND (`tt`.`dt_col` NOT IN (DATETIME '2024-01-01 00:00:00'))) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="dt_col",
                relation=Relation.BETWEEN,
                value=["2024-01-01", "2024-01-02"],
            )
        ],
        where={
            DbType.RS: "tt.dt_col BETWEEN '2024-01-01 00:00:00' AND '2024-01-02 00:00:00' ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.dt_col BETWEEN '2024-01-01 00:00:00' AND '2024-01-02 00:00:00' ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`dt_col` BETWEEN DATETIME '2024-01-01 00:00:00' AND DATETIME '2024-01-02 00:00:00' ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="ts_col",
                relation=Relation.BETWEEN,
                value=["2024-01-01", "2024-01-02"],
            )
        ],
        where={
            DbType.RS: "tt.ts_col BETWEEN '2024-01-01 00:00:00' AND '2024-01-02 00:00:00' ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.ts_col BETWEEN '2024-01-01 00:00:00' AND '2024-01-02 00:00:00' ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`ts_col` BETWEEN TIMESTAMP '2024-01-01 00:00:00' AND TIMESTAMP '2024-01-02 00:00:00' ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="ts_col",
                relation=Relation.BETWEEN,
                value=["2024-01-01 01:02:03.100000", "2024-01-02"],
            )
        ],
        where={
            DbType.RS: "tt.ts_col BETWEEN '2024-01-01 01:02:03' AND '2024-01-02 00:00:00' ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.ts_col BETWEEN '2024-01-01 01:02:03' AND '2024-01-02 00:00:00' ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`ts_col` BETWEEN TIMESTAMP '2024-01-01 01:02:03' AND TIMESTAMP '2024-01-02 00:00:00' ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[42, -17],
            ),
            AudienceSpecFilter(
                field_name="experiment_ids",
                relation=Relation.INCLUDES,
                value=["b", "C"],
            ),
        ],
        where={
            DbType.BQ: "NOT (`tt`.`int_col` IS NULL OR (`tt`.`int_col` NOT IN (42, -17))) AND REGEXP_CONTAINS(lower(`tt`.`experiment_ids`), '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY rand() LIMIT 3",
            DbType.PG: "NOT (tt.int_col IS NULL OR (tt.int_col NOT IN (42, -17))) AND lower(tt.experiment_ids) ~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)' ORDER BY random()  LIMIT 3",
            DbType.RS: "NOT (tt.int_col IS NULL OR (tt.int_col NOT IN (42, -17))) AND lower(tt.experiment_ids) ~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)' ORDER BY random()  LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[42, -17],
            ),
            AudienceSpecFilter(
                field_name="experiment_ids",
                relation=Relation.EXCLUDES,
                value=["b", "c"],
            ),
        ],
        where={
            DbType.BQ: "NOT (`tt`.`int_col` IS NULL OR (`tt`.`int_col` NOT IN (42, -17))) AND (`tt`.`experiment_ids` IS NULL OR char_length(`tt`.`experiment_ids`) = 0 OR NOT REGEXP_CONTAINS(lower(`tt`.`experiment_ids`), '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)')) ORDER BY rand() LIMIT 3",
            DbType.PG: "NOT (tt.int_col IS NULL OR (tt.int_col NOT IN (42, -17))) AND (tt.experiment_ids IS NULL OR char_length(tt.experiment_ids) = 0 OR lower(tt.experiment_ids) !~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY random()  LIMIT 3",
            DbType.RS: "NOT (tt.int_col IS NULL OR (tt.int_col NOT IN (42, -17))) AND (tt.experiment_ids IS NULL OR char_length(tt.experiment_ids) = 0 OR lower(tt.experiment_ids) !~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY random()  LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col", relation=Relation.BETWEEN, value=[-17, 42]
            )
        ],
        where={
            DbType.RS: "tt.int_col BETWEEN -17 AND 42 ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col BETWEEN -17 AND 42 ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` BETWEEN -17 AND 42 ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.INCLUDES,
                value=[True],
            )
        ],
        where={
            DbType.RS: "tt.bool_col IS true ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS true ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`bool_col` IS true ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.EXCLUDES,
                value=[True],
            )
        ],
        where={
            DbType.RS: "tt.bool_col IS NOT true ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NOT true ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`bool_col` IS NOT true ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.EXCLUDES,
                value=[None],
            )
        ],
        where={
            DbType.RS: "tt.bool_col IS NOT NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NOT NULL ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`bool_col` IS NOT NULL ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.INCLUDES,
                value=[None],
            )
        ],
        where={
            DbType.RS: "tt.bool_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`bool_col` IS NULL ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col", relation=Relation.INCLUDES, value=[False]
            )
        ],
        where={
            DbType.BQ: "`tt`.`bool_col` IS false ORDER BY rand() LIMIT 3",
            DbType.RS: "tt.bool_col IS false ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS false ORDER BY random()  LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.EXCLUDES,
                value=[None, True],
            )
        ],
        where={
            DbType.RS: "tt.bool_col IS NOT NULL AND tt.bool_col IS NOT true ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NOT NULL AND tt.bool_col IS NOT true ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`bool_col` IS NOT NULL AND `tt`.`bool_col` IS NOT true ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.INCLUDES,
                value=[None, False],
            )
        ],
        where={
            DbType.RS: "tt.bool_col IS NULL OR tt.bool_col IS false ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NULL OR tt.bool_col IS false ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`bool_col` IS NULL OR `tt`.`bool_col` IS false ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.INCLUDES,
                value=[True, None],
            )
        ],
        where={
            DbType.BQ: "`tt`.`bool_col` IS true OR `tt`.`bool_col` IS NULL ORDER BY rand() LIMIT 3",
            DbType.RS: "tt.bool_col IS true OR tt.bool_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS true OR tt.bool_col IS NULL ORDER BY random()  LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.EXCLUDES,
                value=[False, True],
            )
        ],
        where={
            DbType.RS: "tt.bool_col IS NOT false AND tt.bool_col IS NOT true ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NOT false AND tt.bool_col IS NOT true ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`bool_col` IS NOT false AND `tt`.`bool_col` IS NOT true ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.EXCLUDES,
                value=[False, None],
            )
        ],
        where={
            DbType.BQ: "`tt`.`bool_col` IS NOT false AND `tt`.`bool_col` IS NOT NULL ORDER BY rand() LIMIT 3",
            DbType.RS: "tt.bool_col IS NOT false AND tt.bool_col IS NOT NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NOT false AND tt.bool_col IS NOT NULL ORDER BY random()  LIMIT 3",
        },
    ),
]


@pytest.mark.parametrize("testcase", WHERE_TESTCASES, ids=lambda d: str(d))
def test_where(testcase: WhereTestCase):
    # When adding new cases, we want to cover all of our target databases, so provide some default
    # values to compare against. This makes it easier to add new supported backends (and port
    # existing tests).
    for variant in [dbtype for dbtype in DbType if dbtype.is_supported_dwh()]:
        if variant not in testcase.where:
            testcase.where[variant] = ""

    sa_table = WhereTable.get_table()
    design_spec = make_design_spec(testcase.filters)
    filters = create_query_filters_from_spec(sa_table, design_spec)
    q = compose_query(sa_table, 3, filters)

    failures = {}
    for dbtype in testcase.where:
        sql = str(
            q.compile(dialect=dbtype.dialect(), compile_kwargs={"literal_binds": True})
        )
        normalized = sql.replace("\n", "")
        normalized = normalized[normalized.find("WHERE ") + len("WHERE ") :]
        if normalized != testcase.where[dbtype]:
            failures[str(dbtype)] = normalized

    def for_copypaste():
        formatted = repr(failures)
        for variant in DbType:
            formatted = formatted.replace(
                f"'{variant.value}'", f"DbType.{variant.name}"
            )
        return formatted

    assert not failures, for_copypaste()
