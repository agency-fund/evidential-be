import pytest
from pydantic import TypeAdapter, ValidationError

from xngin.apiserver.routers.common_api_types import Filter, PreassignedFrequentistExperimentSpec
from xngin.apiserver.routers.common_enums import Relation

VALID_COLUMN_NAMES = [
    "column_name",
    "Column_Name",
    "_hidden",
    "a123",
    "very_long_column_name_123",
]


@pytest.mark.parametrize("name", VALID_COLUMN_NAMES)
def test_valid_field_names(name):
    Filter(field_name=name, relation=Relation.INCLUDES, value=[1])


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
        Filter(field_name=name, relation=Relation.INCLUDES, value=[1])


VALID_BETWEEN = [
    ([1, 2], "integers"),
    ([1.0, 2.0], "floats"),
    (["a", "b"], "strings"),
    (["2025-01-01T00:00:00Z", "2025-10-01 00:00:00+00:00"], "iso8601 strings"),
    ([1, None], "with right None"),
    ([None, 1], "with left None"),
    ([1.0, 2], "float and int"),  # pydantic coerces to [1.0, 2.0]
    ([1.5, 2.5], "floats again"),
    ([0, 0, None], "two zeros and None"),
    ([1, 2, None], "between integers or None"),
    ([1, None, None], "greater than integer or None"),
    ([None, 1, None], "less than integer or None"),
]


@pytest.mark.parametrize("value,descr", VALID_BETWEEN)
def test_between_relation(value, descr):
    filter_spec = Filter(field_name="col", relation=Relation.BETWEEN, value=value)
    assert filter_spec.value == value, f"Failed for case: {descr}"


INVALID_BETWEEN = [
    ([1], "single value"),
    ([1, 2, 3], "three values"),
    ([None, None], "both None"),
    ([None, None, None], "three None"),
    ([None, None, "None"], "bad 3rd value"),
    (["a", 1, None], "string and int and None"),
    ([1, "2"], "int and string int"),
    (["1", 2.0], "string int and float"),
    (["1.0", 2], "string float and int"),
    (["1.0", 2.0], "string float and float"),
    ([], "empty list"),
]


@pytest.mark.parametrize("value,descr", INVALID_BETWEEN)
def test_between_relation_invalid(value, descr):
    # The third case occurs when Pydantic backtracks internally to solve the constraints on `value`.
    with pytest.raises(ValidationError, match=r"(BETWEEN relation|same type| validation errors )"):
        Filter(field_name="col", relation=Relation.BETWEEN, value=value)


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
    Filter(field_name="col", relation=relation, value=value)


def test_empty_value_list():
    for relation in (Relation.INCLUDES, Relation.EXCLUDES):
        with pytest.raises(ValidationError, match="value must be a non-empty list"):
            Filter(field_name="col", relation=relation, value=[])


EXPERIMENT_IDS_FILTER_BAD = [
    (Relation.BETWEEN, ["a", "b"], "must have relations of type excludes, includes"),
    (Relation.INCLUDES, ["a,b", "b"], "commas"),
    (Relation.INCLUDES, [" a", "b"], "whitespace"),
]


@pytest.mark.parametrize("relation,value,descr", EXPERIMENT_IDS_FILTER_BAD)
def test_experiment_ids_hack_validators_invalid(relation, value, descr):
    with pytest.raises(ValidationError, match=descr):
        print(Filter(field_name="_experiment_ids", relation=relation, value=value))


EXPERIMENT_IDS_FILTER_GOOD = [
    (Relation.INCLUDES, ["ab", "b"], "strings"),
    (Relation.INCLUDES, ["ab", None], "None"),
    (Relation.EXCLUDES, ["a"], "mixed"),
]


@pytest.mark.parametrize("relation,value,descr", EXPERIMENT_IDS_FILTER_GOOD)
def test_experiment_ids_hack_validators_valid(relation, value, descr):
    Filter(field_name="_experiment_ids", relation=relation, value=value)


def test_arm_weights_validation():
    """Test validation of arm_weights in BaseFrequentistDesignSpec"""
    # Test: weights sum to 100 and match number of arms
    valid_spec = {
        "participant_type": "test_participant",
        "experiment_type": "freq_preassigned",
        "experiment_name": "test",
        "description": "test",
        "start_date": "2024-01-01T00:00:00+00:00",
        "end_date": "2024-12-31T00:00:00+00:00",
        "arms": [
            {"arm_name": "C", "arm_description": "C", "arm_weight": 20.0},
            {"arm_name": "T", "arm_description": "T", "arm_weight": 80.0},
        ],
        "strata": [],
        "metrics": [{"field_name": "metric1", "metric_pct_change": 0.1}],
        "filters": [],
    }
    spec = TypeAdapter(PreassignedFrequentistExperimentSpec).validate_python(valid_spec)
    assert [arm.arm_weight for arm in spec.arms] == [20.0, 80.0]

    # Test: three arms with weights summing to 100
    valid_spec_3arms = valid_spec.copy()
    valid_spec_3arms["arms"] = [
        {"arm_name": "C", "arm_description": "C", "arm_weight": 20.1},
        {"arm_name": "T", "arm_description": "T", "arm_weight": 19.9},
        {"arm_name": "T2", "arm_description": "T2", "arm_weight": 60.0},
    ]
    spec = TypeAdapter(PreassignedFrequentistExperimentSpec).validate_python(valid_spec_3arms)
    assert spec.get_validated_arm_weights() == [20.1, 19.9, 60.0]

    # Invalid case: weights don't sum to 100
    invalid_sum = valid_spec.copy()
    invalid_sum["arms"] = [
        {"arm_name": "C", "arm_description": "C", "arm_weight": 30.0},
        {"arm_name": "T", "arm_description": "T", "arm_weight": 80.0},
    ]
    with pytest.raises(ValidationError, match="arm_weights must sum to 100"):
        TypeAdapter(PreassignedFrequentistExperimentSpec).validate_python(invalid_sum)

    # Invalid case: number of weights doesn't match number of arms
    invalid_count = valid_spec.copy()
    invalid_count["arms"] = [
        {"arm_name": "C", "arm_description": "C", "arm_weight": 50.0},
        {"arm_name": "T", "arm_description": "T", "arm_weight": 50.0},
        {"arm_name": "T2", "arm_description": "T2"},  # missing arm_weight
    ]
    with pytest.raises(ValidationError, match=r"Number of arm weights \(2\) must match number of arms \(3\)"):
        TypeAdapter(PreassignedFrequentistExperimentSpec).validate_python(invalid_count)

    # Invalid case: weights too small and large
    invalid_negative = valid_spec.copy()
    invalid_negative["arms"] = [
        {"arm_name": "C", "arm_description": "C", "arm_weight": -20.0},
        {"arm_name": "T", "arm_description": "T", "arm_weight": 120.0},
    ]
    with pytest.raises(
        ValidationError,
        match=r"(?s)Input should be greater than 0.*Input should be less than 100",
    ):
        TypeAdapter(PreassignedFrequentistExperimentSpec).validate_python(invalid_negative)

    # Invalid case: zero weight
    invalid_zero = valid_spec.copy()
    invalid_zero["arms"] = [
        {"arm_name": "C", "arm_description": "C", "arm_weight": 0.0},
        {"arm_name": "T", "arm_description": "T", "arm_weight": 100.0},
    ]
    with pytest.raises(
        ValidationError,
        match=r"(?s)Input should be greater than 0.*Input should be less than 100",
    ):
        TypeAdapter(PreassignedFrequentistExperimentSpec).validate_python(invalid_zero)

    invalid_inf = valid_spec.copy()
    invalid_inf["arms"] = [
        {"arm_name": "C", "arm_description": "C", "arm_weight": float("inf")},
        {"arm_name": "T", "arm_description": "T", "arm_weight": float("nan")},
    ]
    with pytest.raises(
        ValidationError,
        match=r"(?s)Input should be a finite number.*Input should be a finite number",
    ):
        TypeAdapter(PreassignedFrequentistExperimentSpec).validate_python(invalid_inf)
