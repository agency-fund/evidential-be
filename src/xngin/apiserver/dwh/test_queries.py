"""Stand-alone test cases for basic dynamic query generation."""

import re
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, event
from sqlalchemy.orm import declarative_base, Session

from xngin.apiserver.api_types import (
    AudienceSpec,
    DesignSpecMetric,
    Relation,
    AudienceSpecFilter,
    MetricType,
)
from xngin.apiserver.dwh.queries import (
    compose_query,
    create_filters,
    get_stats_on_metrics,
    make_csv_regex,
)
from xngin.sqlite_extensions.custom_functions import NumpyStddev

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


@pytest.fixture(name="db_session")
def fixture_db_session():
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


@pytest.fixture(name="engine")
def fixture_engine(db_session):
    """Injects an engine into a test."""
    return db_session.get_bind()


@pytest.fixture(name="compiler")
def fixture_compiler(engine):
    """Returns a helper method to compile a SQLAlchemy Select into a SQL string."""
    return lambda query: str(
        query.compile(engine, compile_kwargs={"literal_binds": True})
    ).replace("\n", "")


def test_compose_query_with_no_filters(compiler):
    sql = compiler(compose_query(SampleTable, 2, []))
    assert sql == (
        """SELECT test_table.id, test_table.int_col, test_table.float_col, test_table.bool_col, test_table.string_col, test_table.experiment_ids """
        """FROM test_table ORDER BY random() LIMIT 2 OFFSET 0"""
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
        where="""test_table.int_col IN (42, -17) AND lower(test_table.experiment_ids) REGEXP '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)' {randomize} LIMIT 3 OFFSET 0""",
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
        where="""test_table.int_col IN (42, -17) AND (test_table.experiment_ids IS NULL OR length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) NOT REGEXP '(^(b|c)$)|(^(b|c),)|(,(b|c)$)|(,(b|c),)') {randomize} LIMIT 3 OFFSET 0""",
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
            """{randomize} LIMIT 3 OFFSET 0"""
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
            """{randomize} LIMIT 3 OFFSET 0"""
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
            """{randomize} LIMIT 3 OFFSET 0"""
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
            """{randomize} LIMIT 3 OFFSET 0"""
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
        where="""lower(test_table.experiment_ids) REGEXP '(^(a)$)|(^(a),)|(,(a)$)|(,(a),)' {randomize} LIMIT 3 OFFSET 0""",
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.INCLUDES, value=["B"]
            )
        ],
        where="""lower(test_table.experiment_ids) REGEXP '(^(b)$)|(^(b),)|(,(b)$)|(,(b),)' {randomize} LIMIT 3 OFFSET 0""",
        matches=[ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.INCLUDES, value=["c"]
            )
        ],
        where="""lower(test_table.experiment_ids) REGEXP '(^(c)$)|(^(c),)|(,(c)$)|(,(c),)' {randomize} LIMIT 3 OFFSET 0""",
        matches=[ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.EXCLUDES, value=["a"]
            )
        ],
        where="""test_table.experiment_ids IS NULL OR length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) NOT REGEXP '(^(a)$)|(^(a),)|(,(a)$)|(,(a),)' {randomize} LIMIT 3 OFFSET 0""",
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.EXCLUDES, value=["D"]
            )
        ],
        where="""test_table.experiment_ids IS NULL OR length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) NOT REGEXP '(^(d)$)|(^(d),)|(,(d)$)|(,(d),)' {randomize} LIMIT 3 OFFSET 0""",
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
        where="""lower(test_table.experiment_ids) REGEXP '(^(a|d)$)|(^(a|d),)|(,(a|d)$)|(,(a|d),)' {randomize} LIMIT 3 OFFSET 0""",
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
        where="""test_table.experiment_ids IS NULL OR length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) NOT REGEXP '(^(a|d)$)|(^(a|d),)|(,(a|d)$)|(,(a|d),)' {randomize} LIMIT 3 OFFSET 0""",
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.INCLUDES, value=["d"]
            )
        ],
        where="""lower(test_table.experiment_ids) REGEXP '(^(d)$)|(^(d),)|(,(d)$)|(,(d),)' {randomize} LIMIT 3 OFFSET 0""",
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                filter_name="experiment_ids", relation=Relation.EXCLUDES, value=["d"]
            )
        ],
        where="""test_table.experiment_ids IS NULL OR length(test_table.experiment_ids) = 0 OR lower(test_table.experiment_ids) NOT REGEXP '(^(d)$)|(^(d),)|(,(d)$)|(,(d),)' {randomize} LIMIT 3 OFFSET 0""",
        matches=[ROW_100, ROW_200, ROW_300],
    ),
]


@pytest.mark.parametrize("testcase", FILTER_GENERATION_SUBCASES)
def test_compose_query(testcase, db_session, compiler, use_deterministic_random):
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
    q = compose_query(SampleTable, testcase.chosen_n, filters)
    sql = compiler(q)
    assert sql.startswith(EXPECTED_PREAMBLE)
    sql = sql[len(EXPECTED_PREAMBLE) :]
    assert sql == str.format(testcase.where, randomize="ORDER BY test_table.id")
    query_results = db_session.scalars(q).all()
    assert list(sorted([r.id for r in query_results])) == list(
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


def test_query_baseline_metrics(db_session):
    table = Base.metadata.tables.get(SampleTable.__tablename__)
    row = get_stats_on_metrics(
        db_session,
        table,
        [
            DesignSpecMetric(metric_name="bool_col", metric_type=MetricType.BINARY),
            DesignSpecMetric(metric_name="float_col", metric_type=MetricType.NUMERIC),
            DesignSpecMetric(metric_name="int_col", metric_type=MetricType.NUMERIC),
        ],
        AudienceSpec(
            participant_type="ignored",
            filters=[],
        ),
    )
    expected = [
        DesignSpecMetric(
            metric_name="bool_col",
            metric_type=MetricType.BINARY,
            metric_baseline=0.6666666666666666,
            metric_stddev=0.4714045207910317,
            available_n=3,
        ),
        DesignSpecMetric(
            metric_name="float_col",
            metric_type=MetricType.NUMERIC,
            metric_baseline=2.492,
            metric_stddev=0.6415751449882287,
            available_n=3,
        ),
        DesignSpecMetric(
            metric_name="int_col",
            metric_type=MetricType.NUMERIC,
            metric_baseline=41.666666666666664,
            metric_stddev=47.76563153100307,
            available_n=3,
        ),
    ]
    numeric_fields = {"metric_baseline", "metric_stddev", "available_n"}
    for actual, result in zip(row, expected, strict=True):
        assert actual.metric_name == result.metric_name
        assert actual.metric_type == result.metric_type
        # pytest.approx does a reasonable fuzzy comparison of floats for non-nested dictionaries.
        assert actual.model_dump(include=numeric_fields) == pytest.approx(
            result.model_dump(include=numeric_fields)
        )
