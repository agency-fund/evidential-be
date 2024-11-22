"""Stand-alone test cases for basic dynamic query generation."""

import math
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, event
from sqlalchemy.orm import declarative_base, Session

from xngin.apiserver.api_types import AudienceSpec, Relation, AudienceSpecFilter
from xngin.sqlite_extensions.custom_functions import NumpyStddev
from xngin.apiserver.dwh.queries import (
    compose_query,
    create_filters,
    query_baseline_for_metrics,
)

Base = declarative_base()


class SampleTable(Base):
    __tablename__ = "test_table"

    id = Column(Integer, primary_key=True)
    int_col = Column(Integer, nullable=False)
    float_col = Column(Float, nullable=False)
    bool_col = Column(Boolean, nullable=False)
    string_col = Column(String, nullable=False)


@dataclass
class Data:
    id: int
    int_col: int
    float_col: float
    bool_col: bool
    string_col: str


ROW_100 = Data(id=100, int_col=42, float_col=3.14, bool_col=True, string_col="hello")
ROW_200 = Data(id=200, int_col=-17, float_col=2.718, bool_col=False, string_col="world")
ROW_300 = Data(
    id=300, int_col=100, float_col=1.618, bool_col=True, string_col="goodbye"
)
SAMPLE_TABLE_ROWS = [
    ROW_100,
    ROW_200,
    ROW_300,
]


@pytest.fixture
def db_session():
    """Creates an in-memory SQLite database with test data."""
    engine = create_engine("sqlite:///:memory:", echo=False)

    @event.listens_for(engine, "connect")
    def register_sqlite_functions(dbapi_connection, _):
        NumpyStddev.register(dbapi_connection)

    Base.metadata.create_all(engine)
    session = Session(engine)
    for data in SAMPLE_TABLE_ROWS:
        session.add(SampleTable(**data.__dict__))
    session.commit()

    yield session

    session.close()
    Base.metadata.drop_all(engine)


def test_compose_query_with_no_filters(db_session):
    q = compose_query(db_session, SampleTable, 2, [])
    sql = str(q.statement.compile(compile_kwargs={"literal_binds": True}))
    assert sql.replace("\n", "") == (
        """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col """
        """FROM test_table ORDER BY random() LIMIT 2"""
    )


@dataclass
class Case:
    filters: list[AudienceSpecFilter]
    expected_query: str
    expected_ids: list[int]
    chosen_n: int = 3


FILTER_GENERATION_SUBCASES = [
    # int_col
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col],
            )
        ],
        expected_query=(
            """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col """
            """FROM test_table """
            """WHERE test_table.int_col IN (42) """
            """ORDER BY random() LIMIT 3"""
        ),
        expected_ids=[ROW_100.id],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col", relation=Relation.BETWEEN, value=[-17, 42]
            )
        ],
        expected_query=(
            """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col """
            """FROM test_table """
            """WHERE test_table.int_col BETWEEN -17 AND 42 """
            """ORDER BY random() LIMIT 3"""
        ),
        expected_ids=[ROW_100.id, ROW_200.id],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col",
                relation=Relation.EXCLUDES,
                value=[ROW_100.int_col],
            )
        ],
        expected_query=(
            """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col """
            """FROM test_table """
            """WHERE test_table.int_col IS NULL OR (test_table.int_col NOT IN (42)) """
            """ORDER BY random() LIMIT 3"""
        ),
        expected_ids=[ROW_200.id, ROW_300.id],
    ),
    # float_col
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="float_col", relation=Relation.BETWEEN, value=[2, 3]
            )
        ],
        expected_query=(
            """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col """
            """FROM test_table """
            """WHERE test_table.float_col BETWEEN 2 AND 3 """
            """ORDER BY random() LIMIT 3"""
        ),
        expected_ids=[ROW_200.id],
    ),
]


@pytest.mark.parametrize("testcase", FILTER_GENERATION_SUBCASES)
def test_compose_query(testcase, db_session):
    testcase.filters = [
        AudienceSpecFilter.model_validate(filt.model_dump())
        for filt in testcase.filters
    ]
    table = Base.metadata.tables.get(SampleTable.__tablename__)
    filters = create_filters(
        table,
        AudienceSpec(
            participant_type=SampleTable.__tablename__, filters=testcase.filters
        ),
    )
    q = compose_query(db_session, SampleTable, testcase.chosen_n, filters)
    sql = str(q.statement.compile(compile_kwargs={"literal_binds": True}))
    assert sql.replace("\n", "") == testcase.expected_query
    assert list(sorted([r.id for r in q.all()])) == list(sorted(testcase.expected_ids))


def test_query_baseline_metrics(db_session):
    table = Base.metadata.tables.get(SampleTable.__tablename__)
    row = query_baseline_for_metrics(
        db_session,
        table,
        ["int_col", "float_col"],
        AudienceSpec(
            participant_type="ignored",
            filters=[],
        ),
    )[0]._mapping
    expected = {
        "float_col__metric_count": 3,
        "float_col__metric_mean": 2.492,
        "float_col__metric_sd": 0.6415751449882287,
        "int_col__metric_count": 3,
        "int_col__metric_mean": 41.666666666666664,
        "int_col__metric_sd": 47.76563153100307,
    }
    assert set(row.keys()) == expected.keys()
    for k, v in expected.items():
        assert math.isclose(row[k], v), (k, row[k], v)
