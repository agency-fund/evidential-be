"""Stand-alone test cases for basic dynamic query generation."""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, NamedTuple

import pytest
import sqlalchemy
import sqlalchemy_bigquery
from sqlalchemy import (
    Column,
    Table,
    create_engine,
    make_url,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column
from sqlalchemy.types import BigInteger, Boolean, Date, DateTime, Double, Float, Integer, Numeric, String, Uuid

from xngin.apiserver import flags
from xngin.apiserver.conftest import DbType, get_queries_test_uri
from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.apiserver.dwh.queries import (
    compose_query,
    create_date_or_datetime_filter,
    create_query_filters,
    get_participant_metrics,
    get_stats_on_metrics,
    make_csv_regex,
)
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    DesignSpecMetricRequest,
    Filter,
    Relation,
)
from xngin.apiserver.routers.common_enums import DataType, MetricType
from xngin.apiserver.routers.experiments.test_property_filters import ALL_FILTER_CASES

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
    filters: list[Filter]
    matches: list[Row | NullableRow]
    chosen_n: int = 3

    def __str__(self):
        return " and ".join([f"{f.field_name} {f.relation.name} {f.value}" for f in self.filters])


@pytest.fixture(scope="module")
def queries_dwh_engine():
    """Yields a SQLAlchemy Engine for tests to ooperate on a test data warehouse.

    This dwh is specified by the XNGIN_QUERIES_TEST_URI environment variable. Usually this is a
    local Postgres, but some integration tests may point at a BigQuery dataset (see CI for example).

    SampleTable and SampleNullableTable are recreated and populated on each invocation.
    """
    test_db = get_queries_test_uri()
    if test_db.db_type == DbType.PG:
        management_db = make_url(test_db.connect_url)._replace(database=None)
        default_engine = create_engine(
            management_db,
            echo=flags.ECHO_SQL,
            logging_name=SA_LOGGER_NAME_FOR_DWH,
            poolclass=sqlalchemy.pool.NullPool,
            execution_options={"logging_token": SA_LOGGING_PREFIX_FOR_DWH},
        )
        # re: DROP and CREATE DATABASE cannot be executed inside a transaction block
        with default_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            for stmt in (
                f"DROP DATABASE IF EXISTS {test_db.connect_url.database}",
                f"CREATE DATABASE {test_db.connect_url.database}",
            ):
                conn.execute(text(stmt))
        default_engine.dispose()

    engine = create_engine(
        test_db.connect_url,
        echo=flags.ECHO_SQL,
        logging_name=SA_LOGGER_NAME_FOR_DWH,
        execution_options={"logging_token": SA_LOGGING_PREFIX_FOR_DWH},
    )
    try:
        if test_db.db_type is DbType.RS and hasattr(engine.dialect, "_set_backslash_escapes"):
            engine.dialect._set_backslash_escapes = lambda _: None

        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def shared_sample_tables(queries_dwh_engine):
    """Creates and populates SampleTable and SampleNullableTable."""
    # Create using the DeclarativeBase approach
    Base.metadata.create_all(queries_dwh_engine)
    # and then populate with our sample data.
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


@pytest.fixture
def queries_dwh_session(queries_dwh_engine):
    with Session(queries_dwh_engine) as session:
        yield session  # context manager will close it on exit


@pytest.mark.parametrize(
    "select_columns, error_message",
    [
        (set(), "select_columns must have at least one item."),
        ({"missing_column"}, "Column missing_column not found in schema."),
    ],
)
def test_compile_query_with_invalid_select_column(select_columns, error_message):
    with pytest.raises(ValueError, match=error_message):
        compose_query(SampleTable.get_table(), select_columns, [], 2)


SELECT_COLUMNS_CASES_PG = [
    pytest.param(
        set({"id", "int_col", "float_col", "bool_col", "string_col", "experiment_ids"}),
        "test_table.bool_col, test_table.experiment_ids, test_table.float_col, test_table.id, "
        "test_table.int_col, test_table.string_col",
        id="all",
    ),
    pytest.param({"id", "int_col"}, "test_table.id, test_table.int_col", id="id_and_int_col"),
]


@pytest.mark.parametrize("select_columns, expected_columns", SELECT_COLUMNS_CASES_PG)
def test_compile_query_without_filters_pg(select_columns, expected_columns):
    # SQLAlchemy shares a base class for both psycopg2's and psycopg's dialect so they are very similar.
    dialects = (
        sqlalchemy.dialects.postgresql.psycopg2.dialect(),
        sqlalchemy.dialects.postgresql.psycopg.dialect(),
    )
    for dialect in dialects:
        query = compose_query(SampleTable.get_table(), select_columns, [], 2)
        actual = str(query.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).replace("\n", "")
        expectation = f"SELECT {expected_columns} FROM test_table ORDER BY random()  LIMIT 2"  # two spaces!
        assert actual == expectation


SELECT_COLUMNS_CASES_BQ = [
    pytest.param(
        set({"id", "int_col", "float_col", "bool_col", "string_col", "experiment_ids"}),
        "`test_table`.`bool_col`, `test_table`.`experiment_ids`, `test_table`.`float_col`, `test_table`.`id`, "
        "`test_table`.`int_col`, `test_table`.`string_col`",
        id="all",
    ),
    pytest.param({"id", "int_col"}, "`test_table`.`id`, `test_table`.`int_col`", id="id_and_int_col"),
]


@pytest.mark.parametrize("select_columns, expected_columns", SELECT_COLUMNS_CASES_BQ)
def test_compile_query_without_filters_bq(select_columns, expected_columns):
    query = compose_query(SampleTable.get_table(), select_columns, [], 2)
    dialect = sqlalchemy_bigquery.dialect()
    actual = str(query.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).replace("\n", "")
    expectation = f"SELECT {expected_columns} FROM `test_table` ORDER BY rand() LIMIT 2"
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
                value=[None, ROW_10.datetime_col and ROW_10.datetime_col.isoformat()],
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
        filters=[Filter(field_name="bool_col", relation=Relation.INCLUDES, value=[False])],
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
                value=[None, ROW_10.datetime_col and ROW_10.datetime_col.isoformat()],
            ),
        ],
        matches=[ROW_10, ROW_30],
    ),
    # verify BETWEEN
    Case(
        filters=[
            Filter(field_name="float_col", relation=Relation.BETWEEN, value=[1, 3]),
        ],
        matches=[ROW_10, ROW_20],
    ),
    Case(
        filters=[
            Filter(field_name="float_col", relation=Relation.BETWEEN, value=[1, 3, None]),
        ],
        matches=[ROW_10, ROW_20, ROW_30],
    ),
    # >=
    Case(
        filters=[
            Filter(field_name="float_col", relation=Relation.BETWEEN, value=[2, None, None]),
        ],
        matches=[ROW_20, ROW_30],
    ),
    # <=
    Case(
        filters=[
            Filter(field_name="float_col", relation=Relation.BETWEEN, value=[None, 2, None]),
        ],
        matches=[ROW_10, ROW_30],
    ),
    # between datetimes
    Case(
        filters=[
            Filter(
                field_name="date_col",
                relation=Relation.BETWEEN,
                value=[
                    ROW_10.datetime_col and ROW_10.datetime_col.isoformat(),
                    ROW_20.datetime_col and ROW_20.datetime_col.isoformat(),
                    None,
                ],
            ),
        ],
        matches=[ROW_10, ROW_20, ROW_30],
    ),
    # Between dates
    Case(
        filters=[
            Filter(
                field_name="date_col",
                relation=Relation.BETWEEN,
                value=[
                    ROW_10.date_col and ROW_10.date_col.isoformat(),
                    ROW_20.date_col and ROW_20.date_col.isoformat(),
                    None,
                ],
            ),
        ],
        matches=[ROW_10, ROW_20, ROW_30],
    ),
]


@pytest.mark.parametrize("testcase", IS_NULLABLE_CASES, ids=lambda d: str(d))
def test_is_nullable(testcase, queries_dwh_session, shared_sample_tables, use_deterministic_random):
    testcase.filters = [Filter.model_validate(filt.model_dump()) for filt in testcase.filters]
    table: Table = shared_sample_tables.sample_nullable_table
    select_columns = set(table.c.keys())
    filters = create_query_filters(table, testcase.filters)
    q = compose_query(table, select_columns, filters, testcase.chosen_n)
    query_results = queries_dwh_session.execute(q)
    assert list(sorted([r.id for r in query_results])) == list(sorted(r.id for r in testcase.matches)), testcase


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
        filters=[Filter(field_name="int_col", relation=Relation.BETWEEN, value=[-17, 42])],
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
        filters=[Filter(field_name="float_col", relation=Relation.BETWEEN, value=[2, 3])],
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
        filters=[Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["a"])],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
    Case(
        filters=[Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["B"])],
        matches=[ROW_200, ROW_300],
    ),
    Case(
        filters=[Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["c"])],
        matches=[ROW_300],
    ),
    Case(
        filters=[Filter(field_name="experiment_ids", relation=Relation.EXCLUDES, value=["a"])],
        matches=[],
    ),
    Case(
        filters=[Filter(field_name="experiment_ids", relation=Relation.EXCLUDES, value=["D"])],
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
        filters=[Filter(field_name="experiment_ids", relation=Relation.INCLUDES, value=["d"])],
        matches=[],
    ),
    Case(
        filters=[Filter(field_name="experiment_ids", relation=Relation.EXCLUDES, value=["d"])],
        matches=[ROW_100, ROW_200, ROW_300],
    ),
]


@pytest.mark.parametrize("testcase", RELATION_CASES)
def test_relations(testcase, queries_dwh_session, shared_sample_tables, use_deterministic_random):
    testcase.filters = [Filter.model_validate(filt.model_dump()) for filt in testcase.filters]
    table: Table = shared_sample_tables.sample_table
    select_columns = set(table.c.keys())
    filters = create_query_filters(table, testcase.filters)
    q = compose_query(table, select_columns, filters, testcase.chosen_n)
    query_results = queries_dwh_session.execute(q)
    assert list(sorted([r.id for r in query_results])) == list(sorted(r.id for r in testcase.matches)), testcase


def _datatype_to_sqlalchemy_type(data_type: DataType):
    """Maps DataType enum to generic camel-case SQLAlchemy column type. Helper to create tables for filter tests."""
    # DDL for sqlalchemy.types.Uuid is not supported by sqlalchemy-bigquery (falls back to invalid CHAR(32)).
    my_uuid_type: Uuid = Uuid().with_variant(String(), "bigquery")
    # DDL for bigquery mapped to invalid DOUBLE, so force it to FLOAT64.
    my_double_type: Double = Double().with_variant(Float(), "bigquery")
    mapping = {
        DataType.BOOLEAN: Boolean,
        DataType.CHARACTER_VARYING: String,
        DataType.UUID: my_uuid_type,
        DataType.DATE: Date,
        DataType.INTEGER: Integer,
        DataType.DOUBLE_PRECISION: my_double_type,
        DataType.NUMERIC: Numeric,
        DataType.TIMESTAMP_WITHOUT_TIMEZONE: DateTime,
        DataType.TIMESTAMP_WITH_TIMEZONE: DateTime(timezone=True),
        DataType.BIGINT: BigInteger,
    }
    if data_type not in mapping:
        raise ValueError(f"Unsupported DataType: {data_type}")
    return mapping[data_type]


@pytest.fixture(scope="module")
def shared_filter_table(queries_dwh_engine):
    """Creates a single shared table with all columns needed for property filter tests.

    We use this to avoid repeated CREATE/DROP TABLE operations.
    Instead, each test should use a unique ID to isolate its test data.
    """
    # Collect all unique field names and types from ALL_FILTER_CASES.
    all_fields: dict[str, DataType] = {}
    for case in ALL_FILTER_CASES:
        for field_name, data_type in case.fields.items():
            # Ensure test writer didn't accidentally use multiple types for the same field.
            if field_name in all_fields and all_fields[field_name] != data_type:
                raise ValueError(f"Conflicting types for {field_name}: {all_fields[field_name]} vs {data_type}")
            all_fields[field_name] = data_type

    # Create table with all columns needed for filter tests.
    columns = [Column("id", String, primary_key=True)]
    for field_name, data_type in sorted(all_fields.items()):
        col_type = _datatype_to_sqlalchemy_type(data_type)
        columns.append(Column(field_name, col_type, nullable=True))

    metadata = sqlalchemy.MetaData()
    table = Table("shared_filter_table", metadata, *columns)
    metadata.create_all(queries_dwh_engine)

    try:
        yield table
    finally:
        metadata.drop_all(queries_dwh_engine)


@pytest.mark.parametrize("testcase", ALL_FILTER_CASES, ids=lambda d: str(d))
def test_property_filters_in_sql(testcase, shared_filter_table, queries_dwh_session):
    """Test that SQL query generation matches the in-memory filtering logic from property_filters.py."""
    test_id = str(testcase.description)

    # Insert a row with the unique test ID and properties from the test case.
    insert_values: dict[str, Any] = {"id": test_id}
    for field_name, value in testcase.props.items():
        insert_values[field_name] = value

    queries_dwh_session.execute(shared_filter_table.insert().values(insert_values))

    # Test the query with filters.
    filters = create_query_filters(shared_filter_table, testcase.filters)
    q = compose_query(shared_filter_table, select_columns={"id"}, filters=filters, chosen_n=1).where(
        shared_filter_table.c.id == test_id
    )
    result = queries_dwh_session.execute(q).scalar_one_or_none()

    if testcase.expected:
        assert result == test_id, f"Expected row to pass filters for case: {testcase}"
    else:
        assert result is None, f"Expected row to NOT pass filters for case: {testcase}"

    # Cleanup our inserted test row.
    queries_dwh_session.rollback()


@pytest.mark.parametrize("column_type", [DateTime, Date])
def test_date_or_datetime_filter_validation(column_type):
    """Test validation for DateTime and Date-typed columns."""
    col = Column("x", column_type)

    with pytest.raises(LateValidationError) as exc:
        create_date_or_datetime_filter(
            col,
            Filter(field_name="x", relation=Relation.INCLUDES, value=[123, 456]),
        )
    assert "must be strings containing an ISO8601 formatted date" in str(exc)

    with pytest.raises(LateValidationError) as exc:
        create_date_or_datetime_filter(
            col,
            Filter(
                field_name="x",
                relation=Relation.BETWEEN,
                value=["2024-01-01 00:00:00", "bark"],
            ),
        )
    assert "must be strings containing an ISO8601 formatted date" in str(exc)

    # Test timezone validation for both DateTime and Date columns
    with pytest.raises(LateValidationError) as exc:
        create_date_or_datetime_filter(
            col,
            Filter(
                field_name="x",
                relation=Relation.BETWEEN,
                value=["2024-01-01 00:00:00", "2024-01-01 00:00:00+08:00"],
            ),
        )
    assert "must be in UTC" in str(exc)


def test_date_or_datetime_filter_wrong_column_type():
    """Test that we reject non-DateTime/Date columns for datetime/date filters."""
    col = Column("x", String)
    with pytest.raises(LateValidationError) as exc:
        create_date_or_datetime_filter(col, Filter(field_name="x", relation=Relation.INCLUDES, value=["2024-01-01"]))
    assert "not a DateTime or Date type" in str(exc)


@pytest.mark.parametrize("column_type", [DateTime, Date])
def test_allowed_date_or_datetime_filter_validation(column_type):
    """Test valid Date and DateTime filter scenarios."""
    col = Column("x", column_type)

    # Singular None is allowed for both column types
    create_date_or_datetime_filter(col, Filter(field_name="x", relation=Relation.EXCLUDES, value=[None]))
    create_date_or_datetime_filter(col, Filter(field_name="x", relation=Relation.INCLUDES, value=[None]))
    # as are mixed None and date values
    create_date_or_datetime_filter(col, Filter(field_name="x", relation=Relation.BETWEEN, value=[None, "2024-12-31"]))
    create_date_or_datetime_filter(col, Filter(field_name="x", relation=Relation.BETWEEN, value=["2024-01-01", None]))

    # now without microseconds
    now = datetime.now(UTC).replace(microsecond=0)
    create_date_or_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now.isoformat(), now.isoformat()],
        ),
    )

    # zero offset is allowed (i.e. UTC timezone)
    # We strip the tz info because in the test below we want to control the tz format;
    # `now.isoformat()` by default will render with +00:00.
    now_no_tz = now.replace(tzinfo=None)
    create_date_or_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now_no_tz.isoformat() + "Z", now_no_tz.isoformat() + "-00:00"],
        ),
    )
    create_date_or_datetime_filter(
        col,
        Filter(field_name="x", relation=Relation.INCLUDES, value=["2024-01-01T12:30:00Z"]),
    )
    create_date_or_datetime_filter(
        col,
        Filter(field_name="x", relation=Relation.INCLUDES, value=["2024-01-01T12:30:00+00:00"]),
    )

    # now with microseconds
    now_with_microsecond = now.replace(microsecond=1)
    create_date_or_datetime_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now_with_microsecond.isoformat(), None],
        ),
    )

    # Check strings with and without the time delimiter.
    midnight = "2024-01-01 00:00:00"
    create_date_or_datetime_filter(
        col,
        Filter(field_name="x", relation=Relation.BETWEEN, value=[None, midnight]),
    )

    midnight_with_delim = "2024-01-01T00:00:00"
    create_date_or_datetime_filter(
        col,
        Filter(field_name="x", relation=Relation.BETWEEN, value=[None, midnight_with_delim]),
    )

    # Bare dates are allowed
    create_date_or_datetime_filter(
        col,
        Filter(field_name="x", relation=Relation.BETWEEN, value=["2024-01-01", "2024-12-31"]),
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
    matches = re.search(r, csv)
    actual = matches is not None
    assert actual == expected, (
        f'Expression {r} is expected to {"match" if expected else "not match"} in "{csv}". '
        f"Values = {values}. Matches = {matches}."
    )


def test_get_stats_on_missing_metric_raises_error(queries_dwh_session, shared_sample_tables):
    with pytest.raises(LateValidationError) as exc:
        get_stats_on_metrics(
            queries_dwh_session,
            shared_sample_tables.sample_table,
            [DesignSpecMetricRequest(field_name="missing_col", metric_pct_change=0.1)],
            filters=[],
        )
    assert "Missing metrics (check your Datasource configuration): {'missing_col'}" in str(exc)


def test_get_stats_on_integer_metric(queries_dwh_session, shared_sample_tables):
    """Test would fail on postgres and redshift without a cast to float for different reasons."""
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        filters=[],
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
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_nullable_integer_metric(queries_dwh_session, shared_sample_tables):
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_nullable_table,
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        filters=[],
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
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_boolean_metric(queries_dwh_session, shared_sample_tables):
    """Test would fail on postgres and redshift without casting to int to float."""
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1)],
        filters=[],
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
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_numeric_metric(queries_dwh_session, shared_sample_tables):
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1)],
        filters=[],
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
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_participant_metrics(queries_dwh_session, shared_sample_tables):
    participant_ids = ["100", "200"]
    rows = get_participant_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
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
