from dataclasses import dataclass
from datetime import date, datetime
from typing import NamedTuple

import pytest
import sqlalchemy
from sqlalchemy import Table, create_engine, make_url, text
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column
from sqlalchemy.types import Boolean, Date, DateTime, Float, Integer, String

from xngin.apiserver.conftest import DbType, get_queries_test_uri
from xngin.apiserver.settings import SA_LOGGER_NAME_FOR_DWH


class Base(DeclarativeBase):
    @classmethod
    def get_table(cls) -> Table:
        """Helper to return a sqlalchemy.schema.Table."""
        table = Base.metadata.tables.get(cls.__tablename__)
        assert table is not None
        return table


class SampleNullableTable(Base):
    __tablename__ = "test_nullable_table"

    id = mapped_column(Integer, primary_key=True, autoincrement=False)
    bool_col = mapped_column(Boolean, nullable=True)
    int_col = mapped_column(Integer, nullable=True)
    float_col = mapped_column(Float, nullable=True)
    string_col = mapped_column(String, nullable=True)
    datetime_col = mapped_column(DateTime, nullable=True)
    date_col = mapped_column(Date, nullable=True)


@dataclass
class NullableRow:
    id: int
    bool_col: bool | None
    int_col: int | None
    float_col: float | None
    string_col: str | None
    datetime_col: datetime | None
    date_col: date | None


ROW_10 = NullableRow(
    id=10,
    bool_col=None,
    int_col=None,
    float_col=1.01,
    string_col="10",
    datetime_col=datetime(2025, 1, 1, 0, 0),
    date_col=date(2025, 1, 1),
)
ROW_20 = NullableRow(
    id=20,
    bool_col=True,
    int_col=1,
    float_col=2.02,
    string_col=None,
    datetime_col=datetime.fromisoformat("2025-01-02"),
    date_col=date(2025, 1, 2),
)
ROW_30 = NullableRow(
    id=30,
    bool_col=False,
    int_col=3,
    float_col=None,
    string_col="30",
    datetime_col=None,
    date_col=None,
)
SAMPLE_NULLABLE_TABLE_ROWS = [
    ROW_10,
    ROW_20,
    ROW_30,
]


class SampleTable(Base):
    __tablename__ = "test_table"

    id = mapped_column(Integer, primary_key=True, autoincrement=False)
    int_col = mapped_column(Integer, nullable=False)
    float_col = mapped_column(Float, nullable=False)
    bool_col = mapped_column(Boolean, nullable=False)
    string_col = mapped_column(String, nullable=False)
    experiment_ids = mapped_column(String, nullable=False)


class SampleTables(NamedTuple):
    sample_table: Table
    sample_nullable_table: Table


@dataclass
class Row:
    id: int
    int_col: int
    float_col: float
    bool_col: bool
    string_col: str
    experiment_ids: str


ROW_100 = Row(
    id=100,
    int_col=42,
    float_col=3.14,
    bool_col=True,
    string_col="hello",
    experiment_ids="a",
)
ROW_200 = Row(
    id=200,
    int_col=-17,
    float_col=2.718,
    bool_col=False,
    string_col="world",
    experiment_ids="a,B",
)
ROW_300 = Row(
    id=300,
    int_col=100,
    float_col=1.618,
    bool_col=True,
    string_col="goodbye",
    experiment_ids="A,b,c",
)
SAMPLE_TABLE_ROWS = [
    ROW_100,
    ROW_200,
    ROW_300,
]


@dataclass
class Case:
    filters: list
    matches: list[Row | NullableRow]
    desired_n: int = 3

    def __str__(self):
        return " and ".join([f"{f.field_name} {f.relation.name} {f.value}" for f in self.filters])


@pytest.fixture(name="queries_dwh_engine", scope="module")
def fixture_queries_dwh_engine():
    """Yield a SQLAlchemy engine for DWH query tests."""
    test_db = get_queries_test_uri()
    if test_db.db_type == DbType.PG:
        management_db = make_url(test_db.connect_url)._replace(database=None)
        default_engine = create_engine(
            management_db,
            logging_name=SA_LOGGER_NAME_FOR_DWH,
            poolclass=sqlalchemy.pool.NullPool,
            execution_options={"logging_token": "dwh"},
        )
        with default_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for stmt in (
                f"DROP DATABASE IF EXISTS {test_db.connect_url.database}",
                f"CREATE DATABASE {test_db.connect_url.database}",
            ):
                conn.execute(text(stmt))
        default_engine.dispose()

    engine = create_engine(
        test_db.connect_url,
        logging_name=SA_LOGGER_NAME_FOR_DWH,
        execution_options={"logging_token": "dwh"},
    )
    try:
        if test_db.db_type is DbType.RS and hasattr(engine.dialect, "_set_backslash_escapes"):
            engine.dialect._set_backslash_escapes = lambda _: None

        yield engine
    finally:
        engine.dispose()


@pytest.fixture(name="shared_sample_tables", scope="module")
def fixture_shared_sample_tables(queries_dwh_engine):
    """Create and populate SampleTable and SampleNullableTable."""
    Base.metadata.drop_all(queries_dwh_engine)
    Base.metadata.create_all(queries_dwh_engine)
    with Session(queries_dwh_engine) as session:
        for row in SAMPLE_TABLE_ROWS:
            session.add(SampleTable(**row.__dict__))
        for nullable_row in SAMPLE_NULLABLE_TABLE_ROWS:
            session.add(SampleNullableTable(**nullable_row.__dict__))
        session.commit()

    try:
        yield SampleTables(SampleTable.get_table(), SampleNullableTable.get_table())
    finally:
        Base.metadata.drop_all(queries_dwh_engine)


@pytest.fixture(name="queries_dwh_session")
def fixture_queries_dwh_session(queries_dwh_engine):
    with Session(queries_dwh_engine) as session:
        yield session
