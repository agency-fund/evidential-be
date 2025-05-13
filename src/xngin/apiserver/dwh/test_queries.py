"""Stand-alone test cases for basic dynamic query generation."""

import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
import sqlalchemy
import sqlalchemy_bigquery
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Table,
    create_engine,
    make_url,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column
from xngin.apiserver import flags
from xngin.apiserver.routers.stateless_api_types import (
    Filter,
    DesignSpecMetric,
    DesignSpecMetricRequest,
    MetricType,
    MetricValue,
    ParticipantOutcome,
    Relation,
)
from xngin.apiserver.conftest import DbType, get_test_dwh_info
from xngin.apiserver.dwh.queries import (
    compose_query,
    create_datetime_filter,
    create_query_filters,
    get_participant_metrics,
    get_stats_on_metrics,
    make_csv_regex,
)
from xngin.apiserver.exceptions_common import LateValidationError

SA_LOGGER_NAME_FOR_DWH = "xngin_dwh"
SA_LOGGING_PREFIX_FOR_DWH = "dwh"


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
    int_col = mapped_column(Integer, nullable=True)
    float_col = mapped_column(Float, nullable=True)
    string_col = mapped_column(String, nullable=True)
    date_col = mapped_column(DateTime, nullable=True)


@dataclass
class NullableRow:
    id: int
    bool_col: bool | None
    int_col: int | None
    float_col: float | None
    string_col: str | None
    date_col: datetime | None


ROW_10 = NullableRow(
    id=10,
    bool_col=None,
    int_col=None,
    float_col=1.01,
    string_col="10",
    date_col=datetime(2025, 1, 1, 0, 0),
)
ROW_20 = NullableRow(
    id=20,
    bool_col=True,
    int_col=1,
    float_col=2.02,
    string_col=None,
    date_col=datetime.fromisoformat("2025-01-02"),
)
ROW_30 = NullableRow(
    id=30,
    bool_col=False,
    int_col=3,
    float_col=None,
    string_col="30",
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
    filters: list[Filter]
    matches: list[Row | NullableRow]
    chosen_n: int = 3

    def __str__(self):
        return " and ".join([
            f"{f.field_name} {f.relation.name} {f.value}" for f in self.filters
        ])


@pytest.fixture(name="dwh_session")
def fixture_dwh_session():
    """Yields a session that tests can use to operate on a testing database.

    On Postgres databases: a new database will be created and deleted for each use of this fixture.

    SampleTable and SampleNullableTable are created and populated on each invocation, and destroyed after the yield
    completes.
    """
    dwh_info = get_test_dwh_info()
    connect_url = dwh_info.connect_url
    db_type = dwh_info.db_type
    connect_args = dwh_info.connect_args

    default_url = make_url(connect_url)._replace(database=None)
    temporary_database_name = None
    use_temporary_database = db_type == DbType.PG

    if use_temporary_database:
        temporary_database_name = f"fixture_dwh_session_{secrets.token_hex(16)}"
        default_engine = create_engine(
            default_url,
            connect_args=connect_args,
            echo=flags.ECHO_SQL,
            logging_name=SA_LOGGER_NAME_FOR_DWH,
            poolclass=sqlalchemy.pool.NullPool,
            execution_options={"logging_token": SA_LOGGING_PREFIX_FOR_DWH},
        )
        # re: DROP and CREATE DATABASE cannot be executed inside a transaction block
        with default_engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as conn:
            for stmt in (
                f"DROP DATABASE IF EXISTS {temporary_database_name}",
                f"CREATE DATABASE {temporary_database_name}",
            ):
                conn.execute(text(stmt))
        default_engine.dispose()
        # Override the connect_url with our new database name.
        connect_url = connect_url.set(database=temporary_database_name)

    # Now we can connect to the target database
    engine = create_engine(
        connect_url,
        logging_name=SA_LOGGER_NAME_FOR_DWH,
        connect_args=connect_args,
        echo=flags.ECHO_SQL,
        execution_options={"logging_token": SA_LOGGING_PREFIX_FOR_DWH},
    )

    # TODO: consider trying to consolidate dwh-conditional config with that in settings.py
    if db_type is DbType.RS and hasattr(engine.dialect, "_set_backslash_escapes"):
        engine.dialect._set_backslash_escapes = lambda _: None

    Base.metadata.create_all(engine)
    session = Session(engine)
    for row in SAMPLE_TABLE_ROWS:
        session.add(SampleTable(**row.__dict__))
    for nullable_row in SAMPLE_NULLABLE_TABLE_ROWS:
        session.add(SampleNullableTable(**nullable_row.__dict__))

    session.commit()

    yield session

    session.close()
    engine.dispose()

    if not use_temporary_database:
        Base.metadata.drop_all(engine)
    else:
        default_engine = create_engine(
            default_url,
            logging_name=SA_LOGGER_NAME_FOR_DWH,
            execution_options={"logging_token": SA_LOGGING_PREFIX_FOR_DWH},
            connect_args=connect_args,
            echo=flags.ECHO_SQL,
            poolclass=sqlalchemy.pool.NullPool,
        )
        with default_engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as conn:
            conn.execute(text(f"DROP DATABASE {temporary_database_name}"))


def test_compile_query_without_filters_pg():
    # SQLAlchemy shares a base class for both psycopg2's and psycopg's dialect so they are very similar.
    dialects = (
        sqlalchemy.dialects.postgresql.psycopg2.dialect(),
        sqlalchemy.dialects.postgresql.psycopg.dialect(),
    )
    for dialect in dialects:
        query = compose_query(SampleTable.get_table(), 2, [])
        actual = str(
            query.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
        ).replace("\n", "")
        expectation = (
            "SELECT test_table.id, test_table.int_col, test_table.float_col,"
            " test_table.bool_col, test_table.string_col, test_table.experiment_ids "
            "FROM test_table ORDER BY random()  LIMIT 2"
        )  # two spaces!
        assert actual == expectation


def test_compile_query_without_filters_bq():
    query = compose_query(SampleTable.get_table(), 2, [])
    dialect = sqlalchemy_bigquery.dialect()
    actual = str(
        query.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
    ).replace("\n", "")
    expectation = (
        "SELECT `test_table`.`id`, `test_table`.`int_col`, `test_table`.`float_col`, "
        "`test_table`.`bool_col`, `test_table`.`string_col`, `test_table`.`experiment_ids` "
        "FROM `test_table` ORDER BY rand() LIMIT 2"
    )
    assert actual == expectation, actual


IS_NULLABLE_CASES = [
    # Verify EXCLUDES
    Case(
        filters=[
            Filter(
                field_name="bool_col",
                relation=Relation.EXCLUDES,
                value=[True],
            )
        ],
        matches=[ROW_10, ROW_30],
    ),
    Case(
        filters=[
            Filter(
                field_name="bool_col",
                relation=Relation.EXCLUDES,
                value=[False, None],
            )
        ],
        matches=[ROW_20],
    ),
    Case(
        filters=[
            Filter(
                field_name="int_col",
                relation=Relation.EXCLUDES,
                value=[None],
            ),
        ],
        matches=[ROW_20, ROW_30],
    ),
    Case(
        filters=[
            Filter(
                field_name="float_col",
                relation=Relation.EXCLUDES,
                value=[None],
            ),
        ],
        matches=[ROW_10, ROW_20],
    ),
    Case(
        filters=[
            Filter(
                field_name="string_col",
                relation=Relation.EXCLUDES,
                value=[None, ROW_10.string_col],
            ),
        ],
        matches=[ROW_30],
    ),
    Case(
        filters=[
            Filter(
                field_name="date_col",
                relation=Relation.EXCLUDES,
                value=[None, ROW_10.date_col.isoformat()],
            ),
        ],
        matches=[ROW_20],
    ),
    # Excluding a single non-null value means NULL is also included.
    Case(
        filters=[
            Filter(
                field_name="date_col",
                relation=Relation.EXCLUDES,
                value=["2025-01-01"],
            ),
        ],
        matches=[ROW_20, ROW_30],
    ),
    Case(
        filters=[
            Filter(
                field_name="float_col",
                relation=Relation.EXCLUDES,
                value=[ROW_10.float_col],
            ),
        ],
        matches=[ROW_20, ROW_30],
    ),
    # verify INCLUDES
    Case(
        filters=[
            Filter(field_name="bool_col", relation=Relation.INCLUDES, value=[False])
        ],
        matches=[ROW_30],
    ),
    Case(
        filters=[
            Filter(
                field_name="bool_col",
                relation=Relation.INCLUDES,
                value=[True, None],
            )
        ],
        matches=[ROW_10, ROW_20],
    ),
    Case(
        filters=[
            Filter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[None],
            ),
        ],
        matches=[ROW_10],
    ),
    Case(
        filters=[
            Filter(
                field_name="float_col",
                relation=Relation.INCLUDES,
                value=[None],
            ),
        ],
        matches=[ROW_30],
    ),
    Case(
        filters=[
            Filter(
                field_name="string_col",
                relation=Relation.INCLUDES,
                value=[None, ROW_10.string_col],
            ),
        ],
        matches=[ROW_10, ROW_20],
    ),
    Case(
        filters=[
            Filter(
                field_name="date_col",
                relation=Relation.INCLUDES,
                value=[None, ROW_10.date_col.isoformat()],
            ),
        ],
        matches=[ROW_10, ROW_30],
    ),
]


@pytest.mark.parametrize("testcase", IS_NULLABLE_CASES, ids=lambda d: str(d))
def test_is_nullable(testcase, dwh_session, use_deterministic_random):
    testcase.filters = [
        Filter.model_validate(filt.model_dump()) for filt in testcase.filters
    ]
    table = SampleNullableTable.get_table()
    filters = create_query_filters(table, testcase.filters)
    q = compose_query(table, testcase.chosen_n, filters)
    query_results = dwh_session.execute(q).all()
    assert list(sorted([r.id for r in query_results])) == list(
        sorted(r.id for r in testcase.matches)
    ), testcase


RELATION_CASES = [
    # compound filters
    Case(
        filters=[
            Filter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col, ROW_200.int_col],
            ),
            Filter(
                field_name="experiment_ids",
                relation=Relation.INCLUDES,
                value=["b", "C"],
            ),
        ],
        matches=[ROW_200],
    ),
    Case(
        filters=[
            Filter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col, ROW_200.int_col],
            ),
            Filter(
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
            Filter(
                field_name="int_col",
                relation=Relation.INCLUDES,
                value=[ROW_100.int_col],
            )
        ],
        matches=[ROW_100],
    ),
    Case(
        filters=[
            Filter(field_name="int_col", relation=Relation.BETWEEN, value=[-17, 42])
        ],
        matches=[ROW_100, ROW_200],
    ),
    Case(
        filters=[
            Filter(
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
            Filter(field_name="float_col", relation=Relation.BETWEEN, value=[2, 3])
        ],
        matches=[ROW_200],
    ),
    # bool_col
    Case(
        filters=[
            Filter(
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
            Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["a"])
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["B"])
        ],
        matches=[ROW_200, ROW_300],
    ),
    Case(
        filters=[
            Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["c"])
        ],
        matches=[ROW_300],
    ),
    Case(
        filters=[
            Filter(field_name="experiment_ids", relation=Relation.EXCLUDES, value=["a"])
        ],
        matches=[],
    ),
    Case(
        filters=[
            Filter(field_name="experiment_ids", relation=Relation.EXCLUDES, value=["D"])
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            Filter(
                field_name="experiment_ids",
                relation=Relation.INCLUDES,
                value=["a", "d"],
            )
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[
            Filter(
                field_name="experiment_ids",
                relation=Relation.EXCLUDES,
                value=["a", "d"],
            )
        ],
        matches=[],
    ),
    Case(
        filters=[
            Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["d"])
        ],
        matches=[],
    ),
    Case(
        filters=[
            Filter(field_name="experiment_ids", relation=Relation.EXCLUDES, value=["d"])
        ],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
]


@pytest.mark.parametrize("testcase", RELATION_CASES)
def test_relations(testcase, dwh_session, use_deterministic_random):
    testcase.filters = [
        Filter.model_validate(filt.model_dump()) for filt in testcase.filters
    ]
    filters = create_query_filters(SampleTable.get_table(), testcase.filters)
    q = compose_query(SampleTable.get_table(), testcase.chosen_n, filters)
    query_results = dwh_session.execute(q).all()
    assert list(sorted([r.id for r in query_results])) == list(
        sorted(r.id for r in testcase.matches)
    ), testcase


def test_datetime_filter_validation():
    col = Column("x", DateTime)

    with pytest.raises(LateValidationError) as exc:
        create_datetime_filter(
            col,
            Filter(field_name="x", relation=Relation.INCLUDES, value=[123, 456]),
        )
    assert "ISO8601 formatted date" in str(exc)

    with pytest.raises(LateValidationError) as exc:
        create_datetime_filter(
            col,
            Filter(
                field_name="x",
                relation=Relation.BETWEEN,
                value=["2024-01-01 00:00:00", "bark"],
            ),
        )
    assert "ISO8601 formatted date" in str(exc)

    with pytest.raises(LateValidationError) as exc:
        create_datetime_filter(
            col,
            Filter(
                field_name="x",
                relation=Relation.BETWEEN,
                value=["2024-01-01 00:00:00", "2024-01-01 00:00:00+08:00"],
            ),
        )
    assert "timezone" in str(exc)


def test_allowed_datetime_filter_validation():
    col = Column("x", DateTime)

    create_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.EXCLUDES,
            value=[None],
        ),
    )

    create_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.INCLUDES,
            value=[None],
        ),
    )

    # now without microseconds
    now = datetime.now(UTC).replace(microsecond=0)
    create_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now.isoformat(), now.isoformat()],
        ),
    )

    # zero offset is allowed
    # We strip the tz info because in the test below we want to control the tz format;
    # `now.isoformat()` by default will render with +00:00.
    now_no_tz = now.replace(tzinfo=None)
    create_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now_no_tz.isoformat() + "Z", now_no_tz.isoformat() + "-00:00"],
        ),
    )

    # now with microseconds
    now_with_microsecond = now.replace(microsecond=1)
    create_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now_with_microsecond.isoformat(), None],
        ),
    )

    midnight = "2024-01-01 00:00:00"
    create_datetime_filter(
        col,
        Filter(field_name="x", relation=Relation.BETWEEN, value=[None, midnight]),
    )

    midnight_with_delim = "2024-01-01T00:00:00"
    create_datetime_filter(
        col,
        Filter(
            field_name="x", relation=Relation.BETWEEN, value=[None, midnight_with_delim]
        ),
    )

    # bare dates are allowed
    bare_date = "2024-01-01"
    create_datetime_filter(
        col,
        Filter(field_name="x", relation=Relation.BETWEEN, value=[None, bare_date]),
    )


# TODO: move to api_types
def test_boolean_filter_validation():
    with pytest.raises(ValueError) as excinfo:
        Filter(field_name="bool", relation=Relation.BETWEEN, value=[True, False])
    assert "Values do not support BETWEEN." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Filter(field_name="bool", relation=Relation.INCLUDES, value=[True, True, True])
    assert "Duplicate values" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Filter(field_name="bool", relation=Relation.INCLUDES, value=[True, False, None])
    assert "allows all possible values" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Filter(field_name="bool", relation=Relation.EXCLUDES, value=[True, False, None])
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
    matches = re.search(r, csv)
    actual = matches is not None
    assert actual == expected, (
        f'Expression {r} is expected to {"match" if expected else "not match"} in "{csv}". Values = {values}. Matches = {matches}.'
    )


def test_get_stats_on_missing_metric_raises_error(dwh_session):
    with pytest.raises(LateValidationError) as exc:
        get_stats_on_metrics(
            dwh_session,
            SampleTable.get_table(),
            [DesignSpecMetricRequest(field_name="missing_col", metric_pct_change=0.1)],
            audience_filters=[],
        )
    assert (
        "Missing metrics (check your Datasource configuration): {'missing_col'}"
        in str(exc)
    )


def test_get_stats_on_integer_metric(dwh_session):
    """Test would fail on postgres and redshift without a cast to float for different reasons."""
    rows = get_stats_on_metrics(
        dwh_session,
        SampleTable.get_table(),
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        audience_filters=[],
    )

    expected = DesignSpecMetric(
        field_name="int_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=41.666666666666664,
        metric_stddev=47.76563153100307,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # PG: assertion would fail due to a float vs decimal.Decimal comparison.
    # RS: assertion would fail due to avg() on int types keeps them as integers.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(
        expected.model_dump(include=numeric_fields)
    )


def test_get_stats_on_nullable_integer_metric(dwh_session):
    rows = get_stats_on_metrics(
        dwh_session,
        SampleNullableTable.get_table(),
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        audience_filters=[],
    )

    expected = DesignSpecMetric(
        field_name="int_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=2.0,
        metric_stddev=1.0,
        available_nonnull_n=2,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    assert actual.model_dump(include=numeric_fields) == pytest.approx(
        expected.model_dump(include=numeric_fields)
    )


def test_get_stats_on_boolean_metric(dwh_session):
    """Test would fail on postgres and redshift without casting to int to float."""
    rows = get_stats_on_metrics(
        dwh_session,
        SampleTable.get_table(),
        [DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1)],
        audience_filters=[],
    )

    expected = DesignSpecMetric(
        field_name="bool_col",
        metric_type=MetricType.BINARY,
        metric_baseline=0.6666666666666666,
        metric_stddev=None,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    assert actual.model_dump(include=numeric_fields) == pytest.approx(
        expected.model_dump(include=numeric_fields)
    )


def test_get_stats_on_numeric_metric(dwh_session):
    rows = get_stats_on_metrics(
        dwh_session,
        SampleTable.get_table(),
        [DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1)],
        audience_filters=[],
    )

    expected = DesignSpecMetric(
        field_name="float_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=2.492,
        metric_stddev=0.6415751449882287,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # pytest.approx does a reasonable fuzzy comparison of floats for non-nested dictionaries.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(
        expected.model_dump(include=numeric_fields)
    )


def test_get_participant_metrics(dwh_session):
    participant_ids = ["100", "200"]
    rows = get_participant_metrics(
        dwh_session,
        SampleTable.get_table(),
        [
            DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1),
            DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1),
        ],
        unique_id_field="id",
        participant_ids=participant_ids,
    )

    expected = [
        ParticipantOutcome(
            participant_id="100",
            metric_values=[
                MetricValue(
                    metric_name="float_col",
                    metric_value=3.14,  # Example expected value
                ),
                MetricValue(
                    metric_name="bool_col",
                    metric_value=True,  # Example expected value
                ),
            ],
        ),
        ParticipantOutcome(
            participant_id="200",
            metric_values=[
                MetricValue(
                    metric_name="float_col",
                    metric_value=2.718,  # Example expected value
                ),
                MetricValue(
                    metric_name="bool_col",
                    metric_value=False,  # Example expected value
                ),
            ],
        ),
    ]

    assert len(rows) == len(expected)
    # Sort the rows by participant_id to make the test deterministic.
    rows = sorted(rows, key=lambda r: r.participant_id)
    for actual, exp in zip(rows, expected, strict=False):
        assert actual.participant_id == exp.participant_id
        assert actual.metric_values[0].metric_name == exp.metric_values[0].metric_name
        assert actual.metric_values[0].metric_value == exp.metric_values[0].metric_value
