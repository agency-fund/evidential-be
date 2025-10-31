"""Test cases for property_filters.py functions."""

from dataclasses import dataclass

import pytest

from xngin.apiserver.routers.common_api_types import DataType, Filter, PropertyValueTypes
from xngin.apiserver.routers.common_enums import Relation
from xngin.apiserver.routers.experiments.property_filters import passes_filters


@dataclass
class Case:
    props: dict[str, PropertyValueTypes]  # properties to filter on
    fields: dict[str, DataType]  # simulated db fields
    filters: list[Filter]  # filters to apply
    expected: bool  # expected result: true if passes
    description: str = ""  # test case name

    def __str__(self):
        if self.description:
            return self.description
        return " and ".join([f"{f.field_name} {f.relation.name} {f.value}" for f in self.filters])


BASIC_CASES = [
    Case(
        props={"age": 25},
        fields={"age": DataType.INTEGER},
        filters=[],
        expected=True,
        description="empty_filters_returns_true",
    ),
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
        props={"score": 95.5},
        fields={"score": DataType.DOUBLE_PRECISION},
        filters=[Filter(field_name="score", relation=Relation.INCLUDES, value=[95.5, 100.0])],
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
        props={"score": 95.5},
        fields={"score": DataType.DOUBLE_PRECISION},
        filters=[Filter(field_name="score", relation=Relation.BETWEEN, value=[80.0, 90.0])],
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
        props={"big_number": "18446744073709551616"},  # 2^64
        fields={"big_number": DataType.BIGINT},
        filters=[Filter(field_name="big_number", relation=Relation.INCLUDES, value=["18446744073709551616"])],
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
        props={"created_at": "2025-01-01T00:00:00"},
        fields={"created_at": DataType.TIMESTAMP_WITH_TIMEZONE},
        filters=[Filter(field_name="created_at", relation=Relation.INCLUDES, value=["2025-01-01T00:00:00"])],
        expected=True,
        description="datetime_tz_includes_match",
    ),
    Case(
        props={"created_at": "2025-01-15T12:00:00"},
        fields={"created_at": DataType.TIMESTAMP_WITH_TIMEZONE},
        filters=[
            Filter(
                field_name="created_at",
                relation=Relation.BETWEEN,
                value=["2025-01-01T00:00:00", "2025-01-31T23:59:59"],
            )
        ],
        expected=True,
        description="datetime_tz_between_in_range",
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
        description="datetime_between_out_of_range",
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
        props={"age": 25, "is_active": True, "score": 95.5},
        fields={
            "age": DataType.INTEGER,
            "is_active": DataType.BOOLEAN,
            "score": DataType.DOUBLE_PRECISION,
        },
        filters=[
            Filter(field_name="age", relation=Relation.BETWEEN, value=[20, 30]),
            Filter(field_name="is_active", relation=Relation.INCLUDES, value=[True]),
            Filter(field_name="score", relation=Relation.BETWEEN, value=[90.0, 100.0]),
        ],
        expected=True,
        description="compound_three_filters_all_pass",
    ),
]


ALL_CASES = (
    BASIC_CASES
    + INTEGER_CASES
    + FLOAT_CASES
    + BIGINT_CASES
    + BOOLEAN_CASES
    + UUID_CASES
    + DATETIME_CASES
    + DATE_CASES
    + NULLABLE_CASES
    + COMPOUND_CASES
)


@pytest.mark.parametrize("testcase", ALL_CASES, ids=lambda d: str(d))
def test_passes_filters(testcase: Case):
    """Test that passes_filters correctly evaluates filter criteria."""
    actual = passes_filters(testcase.props, testcase.fields, testcase.filters)
    assert actual == testcase.expected, f"Failed for case: {testcase}"


def test_passes_filters_field_not_found():
    """Test that ValueError is raised when field is not in fields dict."""
    props: dict[str, PropertyValueTypes] = {"age": 25}
    fields = {"age": DataType.INTEGER}
    filters = [Filter(field_name="missing_field", relation=Relation.INCLUDES, value=[25])]

    with pytest.raises(ValueError, match="Field missing_field not found in participant type"):
        passes_filters(props, fields, filters)


def test_passes_filters_invalid_boolean_type():
    """Test that TypeError is raised for invalid boolean value."""
    # Intentionally passing wrong type to test error handling
    props: dict[str, PropertyValueTypes] = {"is_active": "not_a_bool"}
    fields = {"is_active": DataType.BOOLEAN}
    filters = [Filter(field_name="is_active", relation=Relation.INCLUDES, value=[True])]

    with pytest.raises(TypeError, match="Boolean input is not a boolean"):
        passes_filters(props, fields, filters)


def test_passes_filters_invalid_varchar_type():
    """Test that TypeError is raised for invalid varchar value."""
    # Intentionally passing wrong type to test error handling
    props: dict[str, PropertyValueTypes] = {"name": 1.0}
    fields = {"name": DataType.CHARACTER_VARYING}
    filters = [Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice"])]

    with pytest.raises(TypeError, match="varchar input is not a string"):
        passes_filters(props, fields, filters)


def test_passes_filters_invalid_uuid():
    """Test that ValueError is raised for invalid UUID string."""
    props: dict[str, PropertyValueTypes] = {"user_id": "not-a-valid-uuid"}
    fields = {"user_id": DataType.UUID}
    filters = [Filter(field_name="user_id", relation=Relation.INCLUDES, value=["550e8400-e29b-41d4-a716-446655440000"])]

    with pytest.raises(ValueError):
        passes_filters(props, fields, filters)


def test_passes_filters_invalid_integer_type():
    """Test that TypeError is raised for invalid integer value."""
    # Intentionally passing wrong type to test error handling
    props: dict[str, PropertyValueTypes] = {"age": "not_an_int"}
    fields = {"age": DataType.INTEGER}
    filters = [Filter(field_name="age", relation=Relation.INCLUDES, value=[25])]

    with pytest.raises(TypeError, match="Integer input must be an int"):
        passes_filters(props, fields, filters)


def test_passes_filters_numeric_between_with_wrong_type():
    """Test that TypeError is raised when BETWEEN is used on non-numeric/datetime fields."""
    props: dict[str, PropertyValueTypes] = {"name": "Alice"}
    fields = {"name": DataType.CHARACTER_VARYING}
    filters = [Filter(field_name="name", relation=Relation.BETWEEN, value=["Alice", "Bob"])]

    with pytest.raises(TypeError, match="BETWEEN relation is only supported for int/float/datetime fields"):
        passes_filters(props, fields, filters)


def test_passes_filters_invalid_datetime():
    """Test that ValueError is raised for invalid datetime string."""
    props: dict[str, PropertyValueTypes] = {"created_at": "not-a-valid-datetime"}
    fields = {"created_at": DataType.TIMESTAMP_WITH_TIMEZONE}
    filters = [Filter(field_name="created_at", relation=Relation.INCLUDES, value=["2025-01-01T00:00:00"])]

    with pytest.raises(ValueError):
        passes_filters(props, fields, filters)


# Edge cases
def test_passes_filters_missing_property_key():
    """Test behavior when a property key is missing from props dict."""
    props: dict[str, PropertyValueTypes] = {"age": 25}  # 'name' is missing
    fields = {"age": DataType.INTEGER, "name": DataType.CHARACTER_VARYING}
    filters = [Filter(field_name="name", relation=Relation.INCLUDES, value=["Alice"])]

    # Missing property should be treated as None
    result = passes_filters(props, fields, filters)
    assert result is False


def test_passes_filters_property_with_none_value():
    """Test that properties with explicit None values are handled correctly."""
    props: dict[str, PropertyValueTypes] = {"age": None}
    fields = {"age": DataType.INTEGER}
    filters = [Filter(field_name="age", relation=Relation.INCLUDES, value=[None, 25])]

    result = passes_filters(props, fields, filters)
    assert result is True

    # Another teset of missing properties treated as NULL
    result = passes_filters({}, fields, filters)
    assert result is True
