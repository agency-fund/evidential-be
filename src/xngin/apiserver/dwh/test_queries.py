"""Stand-alone test cases for basic dynamic query generation."""

import re
from dataclasses import dataclass
from datetime import datetime

import pytest
from sqlalchemy import (
    Table,
    create_engine,
    Integer,
    Float,
    Boolean,
    String,
    event,
    Column,
    DateTime,
)
from sqlalchemy.orm import Session, DeclarativeBase, mapped_column

from xngin.apiserver.api_types import (
    AudienceSpec,
    DesignSpecMetric,
    DesignSpecMetricRequest,
    Relation,
    AudienceSpecFilter,
    MetricType,
)
from xngin.apiserver.conftest import DbType, get_test_dwh_info
from xngin.apiserver.dwh.queries import (
    compose_query,
    create_query_filters_from_spec,
    get_stats_on_metrics,
    make_csv_regex,
    create_datetime_filter,
    LateValidationError,
)
from xngin.db_extensions.custom_functions import NumpyStddev


class Base(DeclarativeBase):
    @classmethod
    def get_table(cls) -> Table:
        """Helper to return a sqlalchemy.schema.Table"""
        # Also gets around mypy typing issues, e.g. get() can return none, and SampleTable.__table__
        # is of type FromClause, but we know it's a Table and must exist.
        table = Base.metadata.tables.get(cls.__tablename__)
        assert table is not None
        return table


class SampleNullableTable(Base):
    __tablename__ = "test_nullable_table"

    id = mapped_column(Integer, primary_key=True, autoincrement=False)
    bool_col = mapped_column(Boolean, nullable=True)


@dataclass
class NullableRow:
    id: int
    bool_col: bool | None


ROW_10 = NullableRow(
    id=10,
    bool_col=None,
)
ROW_20 = NullableRow(
    id=20,
    bool_col=True,
)
ROW_30 = NullableRow(
    id=30,
    bool_col=False,
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
    filters: list[AudienceSpecFilter]
    matches: list[Row | NullableRow]
    chosen_n: int = 3


@pytest.fixture(name="db_session")
def fixture_db_session():
    """Creates an in-memory SQLite database with test data."""
    connect_url, db_type, connect_args = get_test_dwh_info()
    engine = create_engine(connect_url, connect_args=connect_args, echo=False)

    # TODO: consider trying to consolidate dwh-conditional config with that in settings.py
    if db_type is DbType.RS and hasattr(engine.dialect, "_set_backslash_escapes"):
        engine.dialect._set_backslash_escapes = lambda _: None
    elif db_type is DbType.SL:

        @event.listens_for(engine, "connect")
        def register_sqlite_functions(dbapi_connection, _):
            NumpyStddev.register(dbapi_connection)

    Base.metadata.create_all(engine)
    session = Session(engine)
    for data in SAMPLE_TABLE_ROWS:
        session.add(SampleTable(**data.__dict__))
    for data in SAMPLE_NULLABLE_TABLE_ROWS:
        session.add(SampleNullableTable(**data.__dict__))
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
    sql = compiler(compose_query(SampleTable.get_table(), 2, []))
    # regex to accommodate pg and sqlite compilers
    match = re.match(
        re.escape(
            "SELECT test_table.id, test_table.int_col, test_table.float_col,"
            " test_table.bool_col, test_table.string_col, test_table.experiment_ids "
            "FROM test_table ORDER BY random()"
        )
        + r" +LIMIT 2(?: OFFSET 0){0,1}",
        sql,
    )
    assert match is not None, sql


RELATION_CASES = [
    # compound filters
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col, ROW_200.int_col],
            ),
            AudienceSpecFilter(
                field_name="experiment_ids",
                relation=Relation.INCLUDES,
                value=["b", "C"],
            ),
        ],
        matches=[ROW_200],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col, ROW_200.int_col],
            ),
            AudienceSpecFilter(
                field_name="experiment_ids",
                relation=Relation.EXCLUDES,
                value=["b", "c"],
            ),
        ],
        matches=[ROW_100],
    ),
    # int_col
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col],
            )
        ],
        matches=[ROW_100],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="int_col", relation=Relation.BETWEEN, value=[-17, 42]
            )
        ],
        matches=[ROW_100, ROW_200],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="int_col",
                relation=Relation.EXCLUDES,
                value=[ROW_100.int_col],
            )
        ],
        matches=[ROW_200, ROW_300],
    ),
    # float_col
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="float_col", relation=Relation.BETWEEN, value=[2, 3]
            )
        ],
        matches=[ROW_200],
    ),
    # bool_col
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="bool_col",
                relation=Relation.INCLUDES,
                value=[True],
            )
        ],
        matches=[ROW_100, ROW_300],
    ),
    # regexp hacks
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids", relation=Relation.INCLUDES, value=["a"]
            )
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids", relation=Relation.INCLUDES, value=["B"]
            )
        ],
        matches=[ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids", relation=Relation.INCLUDES, value=["c"]
            )
        ],
        matches=[ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids", relation=Relation.EXCLUDES, value=["a"]
            )
        ],
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids", relation=Relation.EXCLUDES, value=["D"]
            )
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids",
                relation=Relation.INCLUDES,
                value=["a", "d"],
            )
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids",
                relation=Relation.EXCLUDES,
                value=["a", "d"],
            )
        ],
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids", relation=Relation.INCLUDES, value=["d"]
            )
        ],
        matches=[],
    ),
    Case(
        filters=[
            AudienceSpecFilter(
                field_name="experiment_ids", relation=Relation.EXCLUDES, value=["d"]
            )
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
]


@pytest.mark.parametrize("testcase", RELATION_CASES)
def test_relations(testcase, db_session, compiler, use_deterministic_random):
    testcase.filters = [
        AudienceSpecFilter.model_validate(filt.model_dump())
        for filt in testcase.filters
    ]
    filters = create_query_filters_from_spec(
        SampleTable.get_table(),
        AudienceSpec(
            participant_type=SampleTable.__tablename__, filters=testcase.filters
        ),
    )
    q = compose_query(SampleTable.get_table(), testcase.chosen_n, filters)
    query_results = db_session.execute(q).all()
    assert list(sorted([r.id for r in query_results])) == list(
        sorted(r.id for r in testcase.matches)
    ), testcase


@pytest.mark.parametrize(
    "testcase",
    [
        # Test exclusions
        Case(
            filters=[
                AudienceSpecFilter(
                    field_name="bool_col", relation=Relation.EXCLUDES, value=[True]
                )
            ],
            matches=[ROW_10, ROW_30],
        ),
        Case(
            filters=[
                AudienceSpecFilter(
                    field_name="bool_col",
                    relation=Relation.EXCLUDES,
                    value=[False, None],
                )
            ],
            matches=[ROW_20],
        ),
        # Test inclusions
        Case(
            filters=[
                AudienceSpecFilter(
                    field_name="bool_col", relation=Relation.INCLUDES, value=[False]
                )
            ],
            matches=[ROW_30],
        ),
        Case(
            filters=[
                AudienceSpecFilter(
                    field_name="bool_col",
                    relation=Relation.INCLUDES,
                    value=[True, None],
                )
            ],
            matches=[ROW_10, ROW_20],
        ),
    ],
)
def test_booleans(testcase, db_session, compiler):
    table = SampleNullableTable.get_table()
    filters = create_query_filters_from_spec(
        table,
        AudienceSpec(participant_type=table.name, filters=testcase.filters),
    )
    q = compose_query(table, testcase.chosen_n, filters)
    query_results = db_session.execute(q).all()
    assert list(sorted([r.id for r in query_results])) == list(
        sorted(r.id for r in testcase.matches)
    ), testcase


def test_datetime_filter_validation():
    col = Column("x", DateTime)
    with pytest.raises(LateValidationError) as exc:
        create_datetime_filter(
            col,
            AudienceSpecFilter(
                field_name="x", relation=Relation.EXCLUDES, value=[123, 456]
            ),
        )
    assert "only valid Relation on a datetime field is BETWEEN" in str(exc)

    with pytest.raises(LateValidationError) as exc:
        create_datetime_filter(
            col,
            AudienceSpecFilter(
                field_name="x", relation=Relation.INCLUDES, value=[123, 456]
            ),
        )
    assert "only valid Relation on a datetime field is BETWEEN" in str(exc)

    with pytest.raises(LateValidationError) as exc:
        create_datetime_filter(
            col,
            AudienceSpecFilter(
                field_name="x",
                relation=Relation.BETWEEN,
                value=["2024-01-01 00:00:00", "bark"],
            ),
        )
    assert "ISO8601 formatted date" in str(exc)

    with pytest.raises(LateValidationError) as exc:
        create_datetime_filter(
            col,
            AudienceSpecFilter(
                field_name="x",
                relation=Relation.BETWEEN,
                value=["2024-01-01 00:00:00", "2024-01-01 00:00:00+08:00"],
            ),
        )
    assert "timezone" in str(exc)


def test_allowed_datetime_filter_validation():
    col = Column("x", DateTime)

    # now without microseconds
    now = datetime.now().replace(microsecond=0)
    create_datetime_filter(
        col,
        AudienceSpecFilter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now.isoformat(), now.isoformat()],
        ),
    )

    # zero offset is allowed
    create_datetime_filter(
        col,
        AudienceSpecFilter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now.isoformat() + "Z", now.isoformat() + "-00:00"],
        ),
    )

    # now with microseconds
    now_with_microsecond = now.replace(microsecond=1)
    create_datetime_filter(
        col,
        AudienceSpecFilter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now_with_microsecond.isoformat(), None],
        ),
    )

    midnight = "2024-01-01 00:00:00"
    create_datetime_filter(
        col,
        AudienceSpecFilter(
            field_name="x", relation=Relation.BETWEEN, value=[None, midnight]
        ),
    )

    midnight_with_delim = "2024-01-01T00:00:00"
    create_datetime_filter(
        col,
        AudienceSpecFilter(
            field_name="x", relation=Relation.BETWEEN, value=[None, midnight_with_delim]
        ),
    )

    # bare dates are allowed
    bare_date = "2024-01-01"
    create_datetime_filter(
        col,
        AudienceSpecFilter(
            field_name="x", relation=Relation.BETWEEN, value=[None, bare_date]
        ),
    )


# TODO: move to api_types
def test_boolean_filter_validation():
    with pytest.raises(ValueError) as excinfo:
        AudienceSpecFilter(
            field_name="bool", relation=Relation.BETWEEN, value=[True, False]
        )
    assert "Values do not support BETWEEN." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        AudienceSpecFilter(
            field_name="bool", relation=Relation.INCLUDES, value=[True, True, True]
        )
    assert "Duplicate values" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        AudienceSpecFilter(
            field_name="bool", relation=Relation.INCLUDES, value=[True, False, None]
        )
    assert "allows all possible values" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        AudienceSpecFilter(
            field_name="bool", relation=Relation.EXCLUDES, value=[True, False, None]
        )
    assert "rejects all possible values" in str(excinfo.value)


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


def test_get_stats_on_integer_metric(db_session):
    """Test would fail on postgres and redshift without a cast to float for different reasons."""
    rows = get_stats_on_metrics(
        db_session,
        SampleTable.get_table(),
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        AudienceSpec(
            participant_type="ignored",
            filters=[],
        ),
    )

    expected = DesignSpecMetric(
        field_name="int_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=41.666666666666664,
        metric_stddev=47.76563153100307,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {"metric_baseline", "metric_stddev", "available_n"}
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # PG: assertion would fail due to a float vs decimal.Decimal comparison.
    # RS: assertion would fail due to avg() on int types keeps them as integers.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(
        expected.model_dump(include=numeric_fields)
    )


def test_get_stats_on_boolean_metric(db_session):
    """Test would fail on postgres and redshift without casting to int to float."""
    rows = get_stats_on_metrics(
        db_session,
        SampleTable.get_table(),
        [DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1)],
        AudienceSpec(
            participant_type="ignored",
            filters=[],
        ),
    )

    expected = DesignSpecMetric(
        field_name="bool_col",
        metric_type=MetricType.BINARY,
        metric_baseline=0.6666666666666666,
        metric_stddev=None,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {"metric_baseline", "metric_stddev", "available_n"}
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    assert actual.model_dump(include=numeric_fields) == pytest.approx(
        expected.model_dump(include=numeric_fields)
    )


def test_get_stats_on_numeric_metric(db_session):
    rows = get_stats_on_metrics(
        db_session,
        SampleTable.get_table(),
        [DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1)],
        AudienceSpec(
            participant_type="ignored",
            filters=[],
        ),
    )

    expected = DesignSpecMetric(
        field_name="float_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=2.492,
        metric_stddev=0.6415751449882287,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {"metric_baseline", "metric_stddev", "available_n"}
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # pytest.approx does a reasonable fuzzy comparison of floats for non-nested dictionaries.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(
        expected.model_dump(include=numeric_fields)
    )
