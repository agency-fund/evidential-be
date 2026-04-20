"""Stand-alone test cases for basic dynamic query generation."""

import re
from datetime import UTC, datetime
from typing import Any

import pytest
import sqlalchemy
import sqlalchemy_bigquery
from sqlalchemy import Column, Table
from sqlalchemy.types import BigInteger, Boolean, Date, DateTime, Double, Float, Integer, Numeric, String, Uuid

from xngin.apiserver.dwh.dwh_test_support import (
    ROW_10,
    ROW_20,
    ROW_30,
    ROW_100,
    ROW_200,
    ROW_300,
    Case,
    SampleTable,
)
from xngin.apiserver.dwh.query_constructors import (
    compose_query,
    create_filter,
    create_inspect_table_from_cursor_query,
    create_query_filters,
    make_csv_regex,
)
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import Filter, Relation
from xngin.apiserver.routers.common_enums import DataType
from xngin.apiserver.routers.experiments.test_property_filters import ALL_FILTER_CASES

pytest_plugins = ("xngin.apiserver.dwh.dwh_test_support",)


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
def test_is_nullable(testcase, queries_dwh_session, shared_sample_tables):
    testcase.filters = [Filter.model_validate(filt.model_dump()) for filt in testcase.filters]
    table: Table = shared_sample_tables.sample_nullable_table
    select_columns = set(table.c.keys())
    filters = create_query_filters(table, testcase.filters)
    q = compose_query(table, select_columns, filters, testcase.desired_n)
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
def test_relations(testcase, queries_dwh_session, shared_sample_tables):
    testcase.filters = [Filter.model_validate(filt.model_dump()) for filt in testcase.filters]
    table: Table = shared_sample_tables.sample_table
    select_columns = set(table.c.keys())
    filters = create_query_filters(table, testcase.filters)
    q = compose_query(table, select_columns, filters, testcase.desired_n)
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


@pytest.fixture(name="shared_filter_table", scope="module")
def fixture_shared_filter_table(queries_dwh_engine):
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
    metadata.drop_all(queries_dwh_engine)
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
    q = compose_query(shared_filter_table, select_columns={"id"}, filters=filters, desired_n=1).where(
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
        create_filter(
            col,
            Filter(field_name="x", relation=Relation.INCLUDES, value=[123, 456]),
        )
    assert "must be strings containing an ISO8601 formatted date" in str(exc)

    with pytest.raises(LateValidationError) as exc:
        create_filter(
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
        create_filter(
            col,
            Filter(
                field_name="x",
                relation=Relation.BETWEEN,
                value=["2024-01-01 00:00:00", "2024-01-01 00:00:00+08:00"],
            ),
        )
    assert "must be in UTC" in str(exc)


@pytest.mark.parametrize("column_type", [DateTime, Date])
def test_allowed_date_or_datetime_filter_validation(column_type):
    """Test valid Date and DateTime filter scenarios."""
    col = Column("x", column_type)

    # Singular None is allowed for both column types
    create_filter(col, Filter(field_name="x", relation=Relation.EXCLUDES, value=[None]))
    create_filter(col, Filter(field_name="x", relation=Relation.INCLUDES, value=[None]))
    # as are mixed None and date values
    create_filter(col, Filter(field_name="x", relation=Relation.BETWEEN, value=[None, "2024-12-31"]))
    create_filter(col, Filter(field_name="x", relation=Relation.BETWEEN, value=["2024-01-01", None]))

    # now without microseconds
    now = datetime.now(UTC).replace(microsecond=0)
    create_filter(
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
    create_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now_no_tz.isoformat() + "Z", now_no_tz.isoformat() + "-00:00"],
        ),
    )
    create_filter(
        col,
        Filter(field_name="x", relation=Relation.INCLUDES, value=["2024-01-01T12:30:00Z"]),
    )
    create_filter(
        col,
        Filter(field_name="x", relation=Relation.INCLUDES, value=["2024-01-01T12:30:00+00:00"]),
    )

    # now with microseconds
    now_with_microsecond = now.replace(microsecond=1)
    create_filter(
        col,
        Filter(
            field_name="x",
            relation=Relation.BETWEEN,
            value=[now_with_microsecond.isoformat(), None],
        ),
    )

    # Check strings with and without the time delimiter.
    midnight = "2024-01-01 00:00:00"
    create_filter(
        col,
        Filter(field_name="x", relation=Relation.BETWEEN, value=[None, midnight]),
    )

    midnight_with_delim = "2024-01-01T00:00:00"
    create_filter(
        col,
        Filter(field_name="x", relation=Relation.BETWEEN, value=[None, midnight_with_delim]),
    )

    # Bare dates are allowed
    create_filter(
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


@pytest.mark.parametrize(
    "table_name,schema_name",
    [
        pytest.param("my_table; DROP TABLE users", None, id="table_semicolon"),
        pytest.param("my_table' OR '1'='1", None, id="table_quote"),
        pytest.param("my_table\x00attack", None, id="table_null_byte"),
        pytest.param("my_table", "my_schema; DROP TABLE users", id="schema_semicolon"),
        pytest.param("my_table", "my_schema' OR '1'='1", id="schema_quote"),
        pytest.param("my_table", "my_schema\x00attack", id="schema_null_byte"),
        pytest.param("tbl;inject", "sch;inject", id="table_and_schema_injection"),
        # Embedded double-quotes must be escaped as "" so they cannot close the identifier early.
        pytest.param('tbl"name', None, id="table_embedded_dquote"),
        pytest.param("innocent_table", 'sch"name', id="schema_embedded_dquote"),
        pytest.param('tbl"x', 'sch"y', id="table_and_schema_embedded_dquote"),
    ],
)
def test_create_inspect_table_from_cursor_query_quotes_table_name(table_name, schema_name):
    """Verifies that injection payloads cannot escape the identifier context for table and/or schema."""
    dialect = sqlalchemy.dialects.postgresql.psycopg2.dialect()
    query = create_inspect_table_from_cursor_query(table_name, schema_name)
    sql = str(query.compile(dialect=dialect))

    def as_quoted_identifier(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    # Both identifiers must appear in the output. Names with embedded double-quotes are
    # always quoted (escaped as ""), so check the escaped form in that case.
    assert table_name in sql or as_quoted_identifier(table_name) in sql, f"Table name missing from: {sql}"
    if schema_name is not None:
        assert schema_name in sql or as_quoted_identifier(schema_name) in sql, f"Schema name missing from: {sql}"

    # Embedded double-quotes must be escaped as "" inside the identifier, not left bare.
    if '"' in table_name:
        assert as_quoted_identifier(table_name) in sql, f"Expected escaped table identifier in: {sql}"
    if schema_name and '"' in schema_name:
        assert as_quoted_identifier(schema_name) in sql, f"Expected escaped schema identifier in: {sql}"

    # After erasing every double-quoted identifier span (including embedded "" escapes),
    # no statement-terminating character should remain.
    sql_outside_quotes = re.sub(r'"(?:[^"]|"")*"', "", sql)
    assert ";" not in sql_outside_quotes, f"Unquoted ';' in: {sql}"


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
