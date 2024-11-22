"""Stand-alone test cases for basic dynamic query generation."""

import re

import pytest
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String
from sqlalchemy.orm import declarative_base, Session
from dataclasses import dataclass

from xngin.apiserver.api_types import AudienceSpec, Relation, AudienceSpecFilter
from xngin.apiserver.dwh.queries import (
    compose_query,
    create_filters,
    make_csv_regex,
)

Base = declarative_base()


class SampleTable(Base):
    __tablename__ = "test_table"

    id = Column(Integer, primary_key=True)
    int_col = Column(Integer, nullable=False)
    float_col = Column(Float, nullable=False)
    bool_col = Column(Boolean, nullable=False)
    string_col = Column(String, nullable=False)
    experiment_ids = Column(String, nullable=False)


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


@pytest.fixture
def db_session():
    """Creates an in-memory SQLite database with test data."""
    engine = create_engine("sqlite:///:memory:", echo=False)
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
        """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col, test_table.experiment_ids """
        """FROM test_table ORDER BY random() LIMIT 2"""
    )


@dataclass
class Case:
    filters: list[AudienceSpecFilter]
    where: str
    matches: list[Row]
    chosen_n: int = 3


EXPECTED_PREAMBLE = (
    """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col, test_table.experiment_ids """
    """FROM test_table """
    """WHERE """
)

FILTER_GENERATION_SUBCASES = [
    # compound filters
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col, ROW_200.int_col],
            ),
            AudienceSpecFilter(
                filter_name="experiment_ids",
                relation=Relation.INCLUDES,
                value=["b", "C"],
            ),
        ],
        where="""test_table.int_col IN (42, -17) AND lower(test_table.experiment_ids) <regexp> '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)' ORDER BY random() LIMIT 3""",
        matches=[ROW_200],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col, ROW_200.int_col],
            ),
            AudienceSpecFilter(
                filter_name="experiment_ids",
                relation=Relation.EXCLUDES,
                value=["b", "c"],
            ),
        ],
        where="""test_table.int_col IN (42, -17) AND (test_table.experiment_ids IS NULL OR char_length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) <not regexp> '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') ORDER BY random() LIMIT 3""",
        matches=[ROW_100],
    ),
    # int_col
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col],
            )
        ],
        where=(
            """test_table.int_col IN (42) """
            """ORDER BY random() LIMIT 3"""
        ),
        matches=[ROW_100],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col", relation=Relation.BETWEEN, value=[-17, 42]
            )
        ],
        where=(
            """test_table.int_col BETWEEN -17 AND 42 """
            """ORDER BY random() LIMIT 3"""
        ),
        matches=[ROW_100, ROW_200],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="int_col",
                relation=Relation.EXCLUDES,
                value=[ROW_100.int_col],
            )
        ],
        where=(
            """test_table.int_col IS NULL OR (test_table.int_col NOT IN (42)) """
            """ORDER BY random() LIMIT 3"""
        ),
        matches=[ROW_200, ROW_300],
    ),
    # float_col
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="float_col", relation=Relation.BETWEEN, value=[2, 3]
            )
        ],
        where=(
            """test_table.float_col BETWEEN 2 AND 3 """
            """ORDER BY random() LIMIT 3"""
        ),
        matches=[ROW_200],
    ),
    # regexp hacks
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.INCLUDES, value=["a"]
            )
        ],
        where="""lower(test_table.experiment_ids) <regexp> '(^(a)$)|(^(a),)|(,(a)$)|(,(a),)' ORDER BY random() LIMIT 3""",
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.INCLUDES, value=["B"]
            )
        ],
        where="""lower(test_table.experiment_ids) <regexp> '(^(b)$)|(^(b),)|(,(b)$)|(,(b),)' ORDER BY random() LIMIT 3""",
        matches=[ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.INCLUDES, value=["c"]
            )
        ],
        where="""lower(test_table.experiment_ids) <regexp> '(^(c)$)|(^(c),)|(,(c)$)|(,(c),)' ORDER BY random() LIMIT 3""",
        matches=[ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.EXCLUDES, value=["a"]
            )
        ],
        where="""test_table.experiment_ids IS NULL OR char_length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) <not regexp> '(^(a)$)|(^(a),)|(,(a)$)|(,(a),)' ORDER BY random() LIMIT 3""",
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.EXCLUDES, value=["D"]
            )
        ],
        where="""test_table.experiment_ids IS NULL OR char_length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) <not regexp> '(^(d)$)|(^(d),)|(,(d)$)|(,(d),)' ORDER BY random() LIMIT 3""",
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids",
                relation=Relation.INCLUDES,
                value=["a", "d"],
            )
        ],
        where="""lower(test_table.experiment_ids) <regexp> '(^(a|d)$)|(^(a|d),)|(,(a|d)$)|(,(a|d),)' ORDER BY random() LIMIT 3""",
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids",
                relation=Relation.EXCLUDES,
                value=["a", "d"],
            )
        ],
        where="""test_table.experiment_ids IS NULL OR char_length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) <not regexp> '(^(a|d)$)|(^(a|d),)|(,(a|d)$)|(,(a|d),)' ORDER BY random() LIMIT 3""",
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.INCLUDES, value=["d"]
            )
        ],
        where="""lower(test_table.experiment_ids) <regexp> '(^(d)$)|(^(d),)|(,(d)$)|(,(d),)' ORDER BY random() LIMIT 3""",
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.EXCLUDES, value=["d"]
            )
        ],
        where="""test_table.experiment_ids IS NULL OR char_length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) <not regexp> '(^(d)$)|(^(d),)|(,(d)$)|(,(d),)' ORDER BY random() LIMIT 3""",
        matches=[ROW_100, ROW_200, ROW_300],
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
    sql = str(q.statement.compile(compile_kwargs={"literal_binds": True})).replace(
        "\n", ""
    )
    assert sql.startswith(EXPECTED_PREAMBLE)
    sql = sql[len(EXPECTED_PREAMBLE) :]
    assert sql == testcase.where
    assert list(sorted([r.id for r in q.all()])) == list(
        sorted(r.id for r in testcase.matches)
    )


REGEX_TESTS = [
    ("", ["a"], False),
    ("a", [""], False),
    ("a", ["a"], True),
    ("a,b", ["a"], True),
    ("b,a", ["a"], True),
    ("b,a", ["a", "b"], True),
    ("b,a", ["b", "a"], True),
    ("b,a", ["b", ""], True),
    ("c,a,b,d", ["a"], True),
]


@pytest.mark.parametrize("csv,values,expected", REGEX_TESTS)
def test_make_csv_regex(csv, values, expected):
    """Tests for the regular expression, generated in isolation of the database stack.

    Null-, empty string, and negative cases are special and handled in SQL elsewhere.
    """
    r = make_csv_regex(values)
    # confirmed that sqlalchemy.dialects.sqlite.pysqlite also uses re.search
    matches = re.search(r, csv)
    actual = matches is not None
    assert actual == expected, (
        f'Expression {r} is expected to {"match" if expected else "not match"} in "{csv}". Values = {values}. Matches = {matches}.'
    )
