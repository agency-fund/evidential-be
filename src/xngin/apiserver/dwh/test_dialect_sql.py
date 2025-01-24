"""Datetimes and timestamp type fields are interesting because of the variety of implementation details between data
storage systems.

The tests in this file are not as much tests as they are a testbed for understanding the differences between database
engines and how SQLAlchemy translates intention into SQL and DDL.
"""

import re

import pytest
import sqlalchemy
import sqlalchemy_bigquery
from sqlalchemy import Integer, DateTime, TIMESTAMP, Table
from sqlalchemy.dialects.postgresql import psycopg2, psycopg
from sqlalchemy.orm import mapped_column, DeclarativeBase
from sqlalchemy.sql.ddl import CreateTable

from xngin.apiserver.api_types import AudienceSpec, AudienceSpecFilter, Relation
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


DATETIME_SCENARIOS = [
    (
        sqlalchemy.dialects.sqlite.dialect(),
        "CREATE TABLE dtt ( id INTEGER NOT NULL, dt_col DATETIME NOT NULL, ts_col TIMESTAMP NOT NULL, PRIMARY KEY (id))",
        "SELECT dtt.id, dtt.dt_col, dtt.ts_col "
        "FROM dtt WHERE dtt.ts_col >= '2024-01-01 00:00:00.000000' "
        "AND dtt.dt_col BETWEEN '2023-06-01 12:34:56.000000' AND '2024-01-01 00:00:00.000000' "
        "ORDER BY random() LIMIT 2 OFFSET 0",
    ),
    (
        sqlalchemy_bigquery.dialect(),
        "CREATE TABLE `dtt` ( `id` INT64 NOT NULL, `dt_col` DATETIME NOT NULL, `ts_col` TIMESTAMP NOT NULL)",
        "SELECT `dtt`.`id`, `dtt`.`dt_col`, `dtt`.`ts_col` "
        "FROM `dtt` "
        "WHERE `dtt`.`ts_col` >= TIMESTAMP '2024-01-01 00:00:00' "
        "AND `dtt`.`dt_col` BETWEEN DATETIME '2023-06-01 12:34:56' AND DATETIME '2024-01-01 00:00:00' "
        "ORDER BY rand() "
        "LIMIT 2",
    ),
    (
        psycopg2.dialect(),
        "CREATE TABLE dtt ( id INTEGER NOT NULL, dt_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, ts_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, PRIMARY KEY (id))",
        "SELECT dtt.id, dtt.dt_col, dtt.ts_col "
        "FROM dtt "
        "WHERE dtt.ts_col >= '2024-01-01 00:00:00' "
        "AND dtt.dt_col BETWEEN '2023-06-01 12:34:56' AND '2024-01-01 00:00:00' "
        "ORDER BY random()  LIMIT 2",
    ),
    (
        psycopg.dialect(),
        "CREATE TABLE dtt ( id INTEGER NOT NULL, dt_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, ts_col TIMESTAMP WITHOUT TIME ZONE NOT NULL, PRIMARY KEY (id))",
        "SELECT dtt.id, dtt.dt_col, dtt.ts_col "
        "FROM dtt "
        "WHERE dtt.ts_col >= '2024-01-01 00:00:00' "
        "AND dtt.dt_col BETWEEN '2023-06-01 12:34:56' AND '2024-01-01 00:00:00' "
        "ORDER BY random()  LIMIT 2",
    ),
]


@pytest.mark.parametrize(
    "dialect,expected_ddl,expected_sql", DATETIME_SCENARIOS, ids=lambda s: type(s)
)
def test_datetimes(dialect, expected_ddl, expected_sql):
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
                        value=["2024-01-01 00:00:00", None],
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
    ddl = str(CreateTable(sa_table).compile(dialect=dialect))
    normalized_ddl = re.sub(r"\s+", " ", ddl.replace("\n", "").strip())
    assert normalized_ddl == expected_ddl, f"DDL: {normalized_ddl}"
    sql = str(
        q.compile(
            dialect=dialect,
            compile_kwargs={"literal_binds": True},
        )
    )
    assert sql.replace("\n", "") == expected_sql, (
        f"DIALECT {type(dialect)}\nSQL = {sql}"
    )
