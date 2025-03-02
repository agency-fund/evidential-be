import pytest
from pydantic import ValidationError

from xngin.apiserver.api_types import AudienceSpecFilter, Relation

VALID_COLUMN_NAMES = [
    "column_name",
    "Column_Name",
    "_hidden",
    "a123",
    "very_long_column_name_123",
]


@pytest.mark.parametrize("name", VALID_COLUMN_NAMES)
def test_valid_field_names(name):
    AudienceSpecFilter(field_name=name, relation=Relation.INCLUDES, value=[1])


INVALID_COLUMN_NAMES = [
    "123column",  # Can't start with number
    "column-name",  # No hyphens allowed
    "column.name",  # No periods allowed
    "",  # Empty string
    "column$name",  # No special characters
    "column name",  # No spaces
]


@pytest.mark.parametrize("name", INVALID_COLUMN_NAMES)
def test_invalid_field_names(name):
    with pytest.raises(
        ValidationError,
        match="field_name must start with letter/underscore and contain only letters, numbers, underscores",
    ):
        AudienceSpecFilter(field_name=name, relation=Relation.INCLUDES, value=[1])


VALID_BETWEEN = [
    ([1, 2], "integers"),
    ([1.0, 2.0], "floats"),
    (["a", "b"], "strings"),
    ([1, None], "with right None"),
    ([None, 1], "with left None"),
    ([1.0, 2], "float and int"),  # pydantic coerces to [1.0, 2.0]
    ([1.5, 2.5], "floats again"),
]


@pytest.mark.parametrize("value,descr", VALID_BETWEEN)
def test_between_relation(value, descr):
    filter_spec = AudienceSpecFilter(
        field_name="col", relation=Relation.BETWEEN, value=value
    )
    assert filter_spec.value == value, f"Failed for case: {descr}"


INVALID_BETWEEN = [
    ([1], "single value"),
    ([1, 2, 3], "three values"),
    ([None, None], "both None"),
    ([1, "2"], "int and string int"),
    (["1", 2.0], "string int and float"),
    (["1.0", 2], "string float and int"),
    (["1.0", 2.0], "string float and float"),
    ([], "empty list"),
]


@pytest.mark.parametrize("value,descr", INVALID_BETWEEN)
def test_between_relation_invalid(value, descr):
    # The third case occurs when Pydantic backtracks internally to solve the constraints on `value`.
    with pytest.raises(
        ValidationError, match=r"(BETWEEN relation|same type| validation errors )"
    ):
        v = AudienceSpecFilter(field_name="col", relation=Relation.BETWEEN, value=value)
        print(v)


VALID_OTHER = [
    (Relation.INCLUDES, [1, 2, 3]),
    (Relation.INCLUDES, ["a"]),
    (Relation.INCLUDES, [1.0, 2.0, 3.0]),
    (Relation.INCLUDES, [None, 1]),
    (Relation.INCLUDES, ["2020-01-01", None]),
    (Relation.EXCLUDES, [1.0, 2.0, 3.0]),
    (Relation.EXCLUDES, ["b"]),
    (Relation.EXCLUDES, [None, 1]),
    (Relation.EXCLUDES, [None, 1.0]),
    (Relation.EXCLUDES, ["2020-01-01", None]),
]


@pytest.mark.parametrize("relation,value", VALID_OTHER)
def test_other_relations(relation, value):
    AudienceSpecFilter(field_name="col", relation=relation, value=value)


def test_empty_value_list():
    for relation in (Relation.INCLUDES, Relation.EXCLUDES):
        with pytest.raises(ValidationError, match="value must be a non-empty list"):
            AudienceSpecFilter(field_name="col", relation=relation, value=[])


EXPERIMENT_IDS_FILTER_BAD = [
    (Relation.BETWEEN, ["a", "b"], "must have relations of type excludes, includes"),
    (Relation.INCLUDES, ["a,b", "b"], "commas"),
    (Relation.INCLUDES, [" a", "b"], "whitespace"),
]


@pytest.mark.parametrize("relation,value,descr", EXPERIMENT_IDS_FILTER_BAD)
def test_experiment_ids_hack_validators_invalid(relation, value, descr):
    with pytest.raises(ValidationError, match=descr):
        print(
            AudienceSpecFilter(
                field_name="_experiment_ids", relation=relation, value=value
            )
        )


EXPERIMENT_IDS_FILTER_GOOD = [
    (Relation.INCLUDES, ["ab", "b"], "strings"),
    (Relation.INCLUDES, ["ab", None], "None"),
    (Relation.EXCLUDES, ["a"], "mixed"),
]


@pytest.mark.parametrize("relation,value,descr", EXPERIMENT_IDS_FILTER_GOOD)
def test_experiment_ids_hack_validators_valid(relation, value, descr):
    AudienceSpecFilter(field_name="_experiment_ids", relation=relation, value=value)
