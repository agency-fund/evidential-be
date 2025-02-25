"""Tests that exercise the query generation stack (our types, our query composition code, SQLAlchemy's handling of
the generated queries, and variants between dialects).

Datetimes and timestamp type fields are interesting because of the variety of implementation details between data
storage systems. Some don't support DATETIME types, some format datetime string literals differently than others, etc.

Common functions can have different names, too; e.g. one database's RANDOM can be another's RAND.

None of the tests in this file actually execute queries -- it tests the query generation, but not the query execution.
"""

import re
from dataclasses import dataclass

import pytest
import sqlalchemy
import sqlalchemy_bigquery
from sqlalchemy import Integer, DateTime, TIMESTAMP, Table, Float, Boolean, String
from sqlalchemy.dialects.postgresql import psycopg2, psycopg
from sqlalchemy.orm import mapped_column, DeclarativeBase
from sqlalchemy.sql.ddl import CreateTable

from xngin.apiserver.api_types import AudienceSpec, AudienceSpecFilter, Relation
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
        sqlalchemy.dialects.sqlite.dialect(),
        "CREATE TABLE dtt ( id INTEGER NOT NULL, dt_col DATETIME NOT NULL, ts_col TIMESTAMP NOT NULL, PRIMARY KEY (id))",
        "SELECT dtt.id, dtt.dt_col, dtt.ts_col "
        "FROM dtt WHERE dtt.ts_col >= '2020-01-01 00:00:00.000000' "
        "AND dtt.dt_col BETWEEN '2023-06-01 12:34:56.000000' AND '2024-01-01 00:00:00.000000' "
        "ORDER BY random() LIMIT 2 OFFSET 0",
    ),
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


@pytest.mark.parametrize("testcase", DATETIME_SCENARIOS)
def test_datetimes(testcase: DateTimeTestCase):
    """Exercises various SQLAlchemy dialects handling of datetime and timestamp types."""
    sa_table = Datetimes.get_table()
    q = compose_query(
        sa_table,
        2,
        create_query_filters_from_spec(
            sa_table,
            AudienceSpec(
                participant_type="ignored",
                filters=[
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
                ],
            ),
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
            AudienceSpecFilter(field_name="int_col", relation=Relation.IS, value=1),
            AudienceSpecFilter(
                field_name="float_col",
                relation=Relation.IS,
                value=None,
            ),
        ],
        where={
            DbType.SL: "tt.int_col = 1 AND tt.float_col IS NULL ORDER BY random() LIMIT 3 OFFSET 0",
            DbType.RS: "tt.int_col = 1 AND tt.float_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col = 1 AND tt.float_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` = 1 AND `tt`.`float_col` IS NULL ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="dt_col",
                relation=Relation.IS,
                value="2024-01-01",
            ),
            AudienceSpecFilter(
                field_name="ts_col",
                relation=Relation.IS,
                value=None,
            ),
        ],
        where={
            DbType.SL: "tt.dt_col = '2024-01-01' AND tt.ts_col IS NULL ORDER BY random() LIMIT 3 OFFSET 0",
            DbType.RS: "tt.dt_col = '2024-01-01' AND tt.ts_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.dt_col = '2024-01-01' AND tt.ts_col IS NULL ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`dt_col` = '2024-01-01' AND `tt`.`ts_col` IS NULL ORDER BY rand() LIMIT 3",
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
            DbType.SL: "tt.dt_col BETWEEN '2024-01-01 00:00:00.000000' AND '2024-01-02 00:00:00.000000' ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.ts_col BETWEEN '2024-01-01 00:00:00.000000' AND '2024-01-02 00:00:00.000000' ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.ts_col BETWEEN '2024-01-01 01:02:03.000000' AND '2024-01-02 00:00:00.000000' ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.BQ: "`tt`.`int_col` IN (42, -17) AND REGEXP_CONTAINS(lower(`tt`.`experiment_ids`), '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY rand() LIMIT 3",
            DbType.SL: "tt.int_col IN (42, -17) AND lower(tt.experiment_ids) REGEXP '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)' ORDER BY random() LIMIT 3 OFFSET 0",
            DbType.PG: "tt.int_col IN (42, -17) AND lower(tt.experiment_ids) ~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)' ORDER BY random()  LIMIT 3",
            DbType.RS: "tt.int_col IN (42, -17) AND lower(tt.experiment_ids) ~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)' ORDER BY random()  LIMIT 3",
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
            DbType.BQ: "`tt`.`int_col` IN (42, -17) AND (`tt`.`experiment_ids` IS NULL OR char_length(`tt`.`experiment_ids`) = 0 OR NOT REGEXP_CONTAINS(lower(`tt`.`experiment_ids`), '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)')) ORDER BY rand() LIMIT 3",
            DbType.PG: "tt.int_col IN (42, -17) AND (tt.experiment_ids IS NULL OR char_length(tt.experiment_ids) = 0 OR lower(tt.experiment_ids) !~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY random()  LIMIT 3",
            DbType.RS: "tt.int_col IN (42, -17) AND (tt.experiment_ids IS NULL OR char_length(tt.experiment_ids) = 0 OR lower(tt.experiment_ids) !~ '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY random()  LIMIT 3",
            DbType.SL: "tt.int_col IN (42, -17) AND (tt.experiment_ids IS NULL OR length(tt.experiment_ids) = 0 OR lower(tt.experiment_ids) NOT REGEXP '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.int_col IN (42) ORDER BY random() LIMIT 3 OFFSET 0",
            DbType.RS: "tt.int_col IN (42) ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col IN (42) ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` IN (42) ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="int_col", relation=Relation.BETWEEN, value=[-17, 42]
            )
        ],
        where={
            DbType.SL: "tt.int_col BETWEEN -17 AND 42 ORDER BY random() LIMIT 3 OFFSET 0",
            DbType.RS: "tt.int_col BETWEEN -17 AND 42 ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.int_col BETWEEN -17 AND 42 ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`int_col` BETWEEN -17 AND 42 ORDER BY rand() LIMIT 3",
        },
    ),
    WhereTestCase(
        filters=[
            AudienceSpecFilter(
                field_name="float_col", relation=Relation.EXCLUDES, value=[2, 3]
            )
        ],
        where={
            DbType.SL: "tt.float_col IS NULL OR (tt.float_col NOT IN (2, 3)) ORDER BY random() LIMIT 3 OFFSET 0",
            DbType.RS: "tt.float_col IS NULL OR (tt.float_col NOT IN (2, 3)) ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.float_col IS NULL OR (tt.float_col NOT IN (2, 3)) ORDER BY random()  LIMIT 3",
            DbType.BQ: "`tt`.`float_col` IS NULL OR (`tt`.`float_col` NOT IN (2, 3)) ORDER BY rand() LIMIT 3",
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
            DbType.SL: "tt.bool_col IS 1 ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS NOT 1 ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS NOT NULL ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS NULL ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS 0 ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS NOT NULL AND tt.bool_col IS NOT 1 ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS NULL OR tt.bool_col IS 0 ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS 1 OR tt.bool_col IS NULL ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS NOT 0 AND tt.bool_col IS NOT 1 ORDER BY random() LIMIT 3 OFFSET 0",
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
            DbType.SL: "tt.bool_col IS NOT 0 AND tt.bool_col IS NOT NULL ORDER BY random() LIMIT 3 OFFSET 0",
            DbType.BQ: "`tt`.`bool_col` IS NOT false AND `tt`.`bool_col` IS NOT NULL ORDER BY rand() LIMIT 3",
            DbType.RS: "tt.bool_col IS NOT false AND tt.bool_col IS NOT NULL ORDER BY random()  LIMIT 3",
            DbType.PG: "tt.bool_col IS NOT false AND tt.bool_col IS NOT NULL ORDER BY random()  LIMIT 3",
        },
    ),
]


@pytest.mark.parametrize("testcase", WHERE_TESTCASES, ids=lambda d: str(d))
def test_where(testcase: WhereTestCase):
    # When adding new cases, we want to cover all of our target databases, so provide some default values to compare
    # against. This makes it easier to add new supported backends (and port existing tests).
    for variant in DbType:
        if variant not in testcase.where:
            testcase.where[variant] = ""

    sa_table = WhereTable.get_table()
    audience_spec = AudienceSpec(participant_type="na", filters=testcase.filters)
    filters = create_query_filters_from_spec(sa_table, audience_spec)
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
