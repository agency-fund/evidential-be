"""Test cases for property_filters.py functions."""

from dataclasses import dataclass

import pytest

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DataType, Filter, PropertyValueTypes
from xngin.apiserver.routers.common_enums import Relation
from xngin.apiserver.routers.experiments.property_filters import passes_filters


@dataclass
class Case:
    props: dict[str, PropertyValueTypes]  # properties to filter on
    fields: dict[str, DataType]  # simulated db fields
    filters: list[Filter]  # filters to apply
    expected: bool | None = None  # expected result: true if passes (None for error cases)
    description: str = ""  # test case name

    def __str__(self):
        if self.description:
            return self.description
        return " and ".join([f"{f.field_name} {f.relation.name} {f.value}" for f in self.filters])


@dataclass
class ErrorCase(Case):
    exception_type: type[Exception] = Exception  # expected exception type
    match_pattern: str | None = None  # optional regex pattern for error message

    def __str__(self):
        return f"{super().__str__()}_raises_{self.exception_type.__name__}"


STRING_CASES = [
    Case(
        props={"name": "Alice"},
        fields={"name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice"])],
        expected=True,
        description="string_includes_match",
    ),
    Case(
        props={"name": "Bob"},
        fields={"name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice", "bob"])],
        expected=False,
        description="string_includes_no_match",
    ),
    Case(
        props={"name": "Carl"},
        fields={"name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.EXCLUDES, value=["alice", "bob"])],
        expected=True,
        description="string_excludes_match",
    ),
    Case(
        props={"name": "Carl"},
        fields={"name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.EXCLUDES, value=["Carl", "Dave"])],
        expected=False,
        description="string_excludes_no_match",
    ),
]


INTEGER_CASES = [
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.INCLUDES, value=[25, 30])],
        expected=True,
        description="int_includes_match",
    ),
    Case(
        props={"age": 35},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.INCLUDES, value=[25, 30])],
        expected=False,
        description="int_includes_no_match",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.EXCLUDES, value=[30, 35])],
        expected=True,
        description="int_excludes_match",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.EXCLUDES, value=[25, 30])],
        expected=False,
        description="int_excludes_no_match",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.BETWEEN, value=[20, 30])],
        expected=True,
        description="int_between_in_range",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.BETWEEN, value=[30, 40])],
        expected=False,
        description="int_between_out_of_range",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.BETWEEN, value=[25, None])],
        expected=True,
        description="int_between_left_bound_only",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.BETWEEN, value=[None, 30])],
        expected=True,
        description="int_between_right_bound_only",
    ),
    Case(
        props={"age": 31},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.BETWEEN, value=[None, 30])],
        expected=False,
        description="int_between_right_bound_only_no_match",
    ),
]


# Float/Numeric tests
FLOAT_CASES = [
    Case(
        props={"score_dp": 95.5},
        fields={"score_dp": DataType.DOUBLE_PRECISION},
        filters=[Filter(field_name="score_dp", relation=Relation.INCLUDES, value=[95.5, 100.0])],
        expected=True,
        description="float_includes_match",
    ),
    Case(
        props={"score": 85.5},
        fields={"score": DataType.NUMERIC},
        filters=[Filter(field_name="score", relation=Relation.BETWEEN, value=[80.0, 90])],
        expected=True,
        description="numeric_between_in_range",
    ),
    Case(
        props={"score_dp": 95.5},
        fields={"score_dp": DataType.DOUBLE_PRECISION},
        filters=[Filter(field_name="score_dp", relation=Relation.BETWEEN, value=[80.0, 90.0])],
        expected=False,
        description="float_between_out_of_range",
    ),
]


BOOLEAN_CASES = [
    Case(
        props={"is_active": True},
        fields={"is_active": DataType.BOOLEAN},
        filters=[Filter(field_name="is_active", relation=Relation.INCLUDES, value=[True])],
        expected=True,
        description="bool_includes_true",
    ),
    Case(
        props={"is_active": False},
        fields={"is_active": DataType.BOOLEAN},
        filters=[Filter(field_name="is_active", relation=Relation.INCLUDES, value=[True])],
        expected=False,
        description="bool_includes_false",
    ),
    Case(
        props={"is_active": False},
        fields={"is_active": DataType.BOOLEAN},
        filters=[Filter(field_name="is_active", relation=Relation.EXCLUDES, value=[True])],
        expected=True,
        description="bool_excludes_match",
    ),
]


UUID_CASES = [
    Case(
        props={"user_id": "550e8400-e29b-41d4-a716-446655440000"},
        fields={"user_id": DataType.UUID},
        filters=[
            Filter(
                field_name="user_id",
                relation=Relation.INCLUDES,
                value=["550e8400-e29b-41d4-a716-446655440000"],
            )
        ],
        expected=True,
        description="uuid_includes_match",
    ),
    Case(
        props={"user_id": "550e8400-e29b-41d4-a716-446655440000"},
        fields={"user_id": DataType.UUID},
        filters=[
            Filter(
                field_name="user_id",
                relation=Relation.EXCLUDES,
                value=["550e8400-e29b-41d4-a716-446655440001"],
            )
        ],
        expected=True,
        description="uuid_excludes_match",
    ),
]


BIGINT_CASES = [
    Case(
        props={"big_number": str(2**53 + 1)},  # MAX_SAFE_INTEGER + 1
        fields={"big_number": DataType.BIGINT},
        filters=[Filter(field_name="big_number", relation=Relation.INCLUDES, value=["9007199254740993"])],
        expected=True,
        description="bigint_includes_match",
    ),
    Case(
        props={"big_number": 12345},  # int for backwards compatibility
        fields={"big_number": DataType.BIGINT},
        filters=[Filter(field_name="big_number", relation=Relation.INCLUDES, value=[12345])],
        expected=True,
        description="bigint_int_includes_match",
    ),
]


DATETIME_CASES = [
    Case(
        props={"created_at_tz": "2025-01-01T00:00:00"},
        fields={"created_at_tz": DataType.TIMESTAMP_WITH_TIMEZONE},
        filters=[Filter(field_name="created_at_tz", relation=Relation.INCLUDES, value=["2025-01-01T00:00:00"])],
        expected=True,
        description="datetime_tz_includes_match",
    ),
    Case(
        props={"created_at_tz": "2025-01-15T12:00:00"},
        fields={"created_at_tz": DataType.TIMESTAMP_WITH_TIMEZONE},
        filters=[
            Filter(
                field_name="created_at_tz",
                relation=Relation.BETWEEN,
                value=["2025-01-01T00:00:00", "2025-01-31T23:59:59"],
            )
        ],
        expected=True,
        description="datetime_tz_between_in_range",
    ),
    Case(
        props={"created_at_tz": "2025-01-15T12:00:00"},
        fields={"created_at_tz": DataType.TIMESTAMP_WITH_TIMEZONE},
        filters=[
            Filter(
                field_name="created_at_tz",
                relation=Relation.BETWEEN,
                value=["2025-01-15T12:00:00+00:00", None],
            )
        ],
        expected=True,
        description="datetime_tz_allows_zero_offset",
    ),
    Case(
        props={"created_at": "2025-02-15T12:00:00"},
        fields={"created_at": DataType.TIMESTAMP_WITHOUT_TIMEZONE},
        filters=[
            Filter(
                field_name="created_at",
                relation=Relation.BETWEEN,
                value=["2025-01-01T00:00:00", "2025-01-31T23:59:59"],
            )
        ],
        expected=False,
        description="datetime_without_tz_between_out_of_range",
    ),
    Case(
        props={"created_at": "2025-01-15T12:00:00"},
        fields={"created_at": DataType.TIMESTAMP_WITHOUT_TIMEZONE},
        filters=[
            Filter(
                field_name="created_at",
                relation=Relation.BETWEEN,
                value=[None, "2025-01-15T12:00:00+00:00"],
            )
        ],
        expected=True,
        description="datetime_without_tz_truncates_zero_offset",
    ),
    # If we didn't truncate microseconds, the stored ts would be earlier than the filtered range.
    Case(
        props={"created_at": "2025-01-15T00:00:00.001000"},
        fields={"created_at": DataType.TIMESTAMP_WITHOUT_TIMEZONE},
        filters=[
            Filter(
                field_name="created_at",
                relation=Relation.BETWEEN,
                value=["2025-01-15T00:00:00.100000", None],
            )
        ],
        expected=True,
        description="datetime_without_tz_truncates_microseconds",
    ),
]


DATE_CASES = [
    Case(
        props={"birth_date": "2000-01-01"},
        fields={"birth_date": DataType.DATE},
        filters=[Filter(field_name="birth_date", relation=Relation.INCLUDES, value=["2000-01-01"])],
        expected=True,
        description="date_includes_match",
    ),
    Case(
        props={"birth_date": "2000-06-15"},
        fields={"birth_date": DataType.DATE},
        filters=[Filter(field_name="birth_date", relation=Relation.BETWEEN, value=["2000-01-01", "2000-12-31"])],
        expected=True,
        description="date_between_in_range",
    ),
    Case(
        props={"birth_date": "2025-01-01"},
        fields={"birth_date": DataType.DATE},
        filters=[Filter(field_name="birth_date", relation=Relation.INCLUDES, value=["2025-01-01T12:00:00"])],
        expected=True,
        description="date_with_hms_truncated",
    ),
]


NULLABLE_CASES = [
    Case(
        props={"age": None},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.INCLUDES, value=[None])],
        expected=True,
        description="null_includes_match",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.INCLUDES, value=[None])],
        expected=False,
        description="null_includes_no_match",
    ),
    Case(
        props={"age": None},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.EXCLUDES, value=[None])],
        expected=False,
        description="null_excludes_no_match",
    ),
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.EXCLUDES, value=[None])],
        expected=True,
        description="null_excludes_match",
    ),
    Case(
        props={"age": None},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.BETWEEN, value=[20, 30, None])],
        expected=True,
        description="null_between_with_null_allowed",
    ),
    Case(
        props={"age": None},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.BETWEEN, value=[None, 30, None])],
        expected=True,
        description="null_lte_with_null_allowed",
    ),
    Case(
        props={"name": None},
        fields={"name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.INCLUDES, value=[None, "Alice"])],
        expected=True,
        description="null_string_includes_match",
    ),
    Case(
        props={"is_active": None},
        fields={"is_active": DataType.BOOLEAN},
        filters=[Filter(field_name="is_active", relation=Relation.INCLUDES, value=[True, None])],
        expected=True,
        description="null_bool_includes_match",
    ),
    Case(
        props={"birth_date": None},
        fields={"birth_date": DataType.DATE},
        filters=[Filter(field_name="birth_date", relation=Relation.BETWEEN, value=["2000-12-31", None, None])],
        expected=True,
        description="null_date_between_with_null_allowed",
    ),
]


COMPOUND_CASES = [
    Case(
        props={"age": 25, "name": "Alice"},
        fields={"age": DataType.INTEGER, "name": DataType.CHARACTER_VARYING},
        filters=[
            Filter(field_name="age", relation=Relation.BETWEEN, value=[20, 30]),
            Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice", "Bob"]),
        ],
        expected=True,
        description="compound_both_filters_pass",
    ),
    Case(
        props={"age": 35, "name": "Alice"},
        fields={"age": DataType.INTEGER, "name": DataType.CHARACTER_VARYING},
        filters=[
            Filter(field_name="age", relation=Relation.BETWEEN, value=[20, 30]),
            Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice", "Bob"]),
        ],
        expected=False,
        description="compound_first_filter_fails",
    ),
    Case(
        props={"age": 25, "name": "Charlie"},
        fields={"age": DataType.INTEGER, "name": DataType.CHARACTER_VARYING},
        filters=[
            Filter(field_name="age", relation=Relation.BETWEEN, value=[20, 30]),
            Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice", "Bob"]),
        ],
        expected=False,
        description="compound_second_filter_fails",
    ),
    Case(
        props={"age": 25, "is_active": True, "score_dp": 95.5},
        fields={
            "age": DataType.INTEGER,
            "is_active": DataType.BOOLEAN,
            "score_dp": DataType.DOUBLE_PRECISION,
        },
        filters=[
            Filter(field_name="age", relation=Relation.BETWEEN, value=[20, 30]),
            Filter(field_name="is_active", relation=Relation.INCLUDES, value=[True]),
            Filter(field_name="score_dp", relation=Relation.BETWEEN, value=[90.0, 100.0]),
        ],
        expected=True,
        description="compound_three_filters_all_pass",
    ),
]


OTHER_EDGE_CASES = [
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[],
        expected=True,
        description="empty_filters_returns_true",
    ),
    Case(
        props={"age": 25},  # 'name' is missing
        fields={"age": DataType.INTEGER, "name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice"])],
        expected=False,
        description="missing_property_key_treated_as_none",
    ),
    Case(
        props={},  # 'age' is missing (treated as NULL)
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.INCLUDES, value=[None, 25])],
        expected=True,
        description="missing_property_treated_as_null",
    ),
    Case(
        props={"age": None},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.INCLUDES, value=[None, 25])],
        expected=True,
        description="explicit_none_value_includes_match",
    ),
]


ALL_FILTER_CASES = (
    STRING_CASES
    + INTEGER_CASES
    + FLOAT_CASES
    + BIGINT_CASES
    + BOOLEAN_CASES
    + UUID_CASES
    + DATETIME_CASES
    + DATE_CASES
    + NULLABLE_CASES
    + COMPOUND_CASES
    + OTHER_EDGE_CASES
)


@pytest.mark.parametrize("testcase", ALL_FILTER_CASES, ids=lambda d: str(d))
def test_passes_filters_tf(testcase: Case):
    """Test that passes_filters correctly evaluates filter criteria."""
    actual = passes_filters(testcase.props, testcase.fields, testcase.filters)
    assert actual == testcase.expected, f"Failed for case: {testcase}"


ERROR_CASES = [
    ErrorCase(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="missing_field", relation=Relation.INCLUDES, value=[25])],
        exception_type=ValueError,
        match_pattern=r"Field missing_field data type is missing \(field not found\?\)",
        description="field_not_found",
    ),
    ErrorCase(
        props={"is_active": "not_a_bool"},
        fields={"is_active": DataType.BOOLEAN},
        filters=[Filter(field_name="is_active", relation=Relation.INCLUDES, value=[True])],
        exception_type=TypeError,
        match_pattern="Boolean input is not a boolean",
        description="invalid_boolean_type",
    ),
    ErrorCase(
        props={"name": 1.0},
        fields={"name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice"])],
        exception_type=TypeError,
        match_pattern="varchar input is not a string",
        description="invalid_varchar_type",
    ),
    ErrorCase(
        props={"user_id": "not-a-valid-uuid"},
        fields={"user_id": DataType.UUID},
        filters=[
            Filter(field_name="user_id", relation=Relation.INCLUDES, value=["550e8400-e29b-41d4-a716-446655440000"])
        ],
        exception_type=ValueError,
        match_pattern=None,
        description="invalid_uuid",
    ),
    ErrorCase(
        props={"age": "not_an_int"},
        fields={"age": DataType.INTEGER},
        filters=[Filter(field_name="age", relation=Relation.INCLUDES, value=[25])],
        exception_type=TypeError,
        match_pattern="Integer input must be an int",
        description="invalid_integer_type",
    ),
    ErrorCase(
        props={"name": "Alice"},
        fields={"name": DataType.CHARACTER_VARYING},
        filters=[Filter(field_name="name", relation=Relation.BETWEEN, value=["Alice", "Bob"])],
        exception_type=TypeError,
        match_pattern="BETWEEN relation is only supported for int/float/datetime/date fields",
        description="numeric_between_with_wrong_type",
    ),
    ErrorCase(
        props={"created_at": "not-a-valid-datetime"},
        fields={"created_at": DataType.TIMESTAMP_WITHOUT_TIMEZONE},
        filters=[Filter(field_name="created_at", relation=Relation.INCLUDES, value=["2025-01-01T00:00:00"])],
        exception_type=LateValidationError,
        match_pattern="created_at: datetime-type filter values must be strings containing an ISO8601 formatted date",
        description="invalid_datetime",
    ),
    ErrorCase(
        props={"created_at": "2025-01-15T12:00:00+08:00"},
        fields={"created_at": DataType.TIMESTAMP_WITH_TIMEZONE},
        filters=[Filter(field_name="created_at", relation=Relation.BETWEEN, value=[None, "2025-01-15T12:00:00+08:00"])],
        exception_type=LateValidationError,
        match_pattern="created_at: datetime-type filter values must be in UTC, and not include timezone offsets",
        description="invalid_datetime_with_nonzero_offset",
    ),
]


@pytest.mark.parametrize("testcase", ERROR_CASES, ids=lambda d: str(d))
def test_passes_filters_errors(testcase: ErrorCase):
    """Test that passes_filters raises expected exceptions for invalid inputs."""
    match_pattern = testcase.match_pattern or ".*"
    with pytest.raises(testcase.exception_type, match=match_pattern):
        passes_filters(testcase.props, testcase.fields, testcase.filters)
