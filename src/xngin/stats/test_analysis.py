import math
import random
import uuid
from typing import Any

import pytest

from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.apiserver.sqla import tables
from xngin.stats.analysis import analyze_experiment


@pytest.fixture
def test_assignments(n=1000, seed=42):  # noqa: PT028
    rand = random.Random(seed)
    # Use fixed UUIDs instead of randomly generated ones
    arm_ids = [
        uuid.UUID("0ffe0995-6404-4622-934a-0d5cccfe3a59"),
        uuid.UUID("b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"),
        uuid.UUID("df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"),
    ]
    assignments = []
    for i in range(n):
        arm_id = rand.choice(arm_ids)
        assignments.append(
            # TODO: test Assignment for old stateless api
            # re: https://github.com/agency-fund/xngin/pull/306 since arm_id is a
            #  sqlalchemy.Uuid(as_uuid=False), must assign to it with a string
            tables.ArmAssignment(participant_id=str(i), arm_id=str(arm_id), strata=[])
        )
    return assignments


@pytest.fixture
def test_outcomes(n=1000, seed=43):  # noqa: PT028
    rand = random.Random(seed)
    return [
        ParticipantOutcome(
            participant_id=str(i),
            metric_values=[MetricValue(metric_name="bool_field", metric_value=rand.choice([0, 1]))],
        )
        for i in range(n)
    ]


def test_analysis(test_assignments, test_outcomes):
    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result.keys()) == 1  # One metric
    assert len(next(iter(result.values())).keys()) == 3  # Three arms

    bool_field_results = result["bool_field"]
    # Test using the fixed UUIDs
    assert bool_field_results["0ffe0995-6404-4622-934a-0d5cccfe3a59"].is_baseline is True
    assert bool_field_results["b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"].is_baseline is False
    assert bool_field_results["df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"].is_baseline is False

    # Test approximate values since floating point math may have small variations
    assert (
        pytest.approx(
            bool_field_results["0ffe0995-6404-4622-934a-0d5cccfe3a59"].estimate,
            abs=1e-4,
        )
        == 0.4793
    )
    assert (
        pytest.approx(
            bool_field_results["b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"].estimate,
            abs=1e-4,
        )
        == 0.0222
    )
    assert (
        pytest.approx(
            bool_field_results["df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"].estimate,
            abs=1e-4,
        )
        == 0.0431
    )


def test_analysis_with_custom_baseline(test_assignments, test_outcomes):
    result = analyze_experiment(
        test_assignments,
        test_outcomes,
        baseline_arm_id="b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5",
    )
    assert len(result.keys()) == 1  # One metric
    assert len(next(iter(result.values())).keys()) == 3  # Three arms

    bool_field_results = result["bool_field"]
    # Test using the fixed UUIDs
    assert bool_field_results["0ffe0995-6404-4622-934a-0d5cccfe3a59"].is_baseline is False
    assert bool_field_results["b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"].is_baseline is True
    assert bool_field_results["df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"].is_baseline is False

    # Test approximate values since floating point math may have small variations
    assert (
        pytest.approx(
            bool_field_results["0ffe0995-6404-4622-934a-0d5cccfe3a59"].estimate,
            abs=1e-4,
        )
        == -0.0222  # c.f.: 0.4793 (above) - 0.5015 (new baseline) = -0.0222
    )
    assert (
        pytest.approx(
            bool_field_results["b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"].estimate,
            abs=1e-4,
        )
        == 0.5015
    )
    assert (
        pytest.approx(
            bool_field_results["df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"].estimate,
            abs=1e-4,
        )
        == 0.0209
    )


def test_analysis_with_missing_outcomes(test_assignments, test_outcomes):
    # Replace a portion of the outcomes with None
    none: Any = None
    for i in range(200):
        test_outcomes[i] = ParticipantOutcome(
            participant_id=test_outcomes[i].participant_id,
            metric_values=[MetricValue(metric_name="bool_field", metric_value=none)],
        )
    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result.keys()) == 1  # One metric
    assert len(next(iter(result.values())).keys()) == 3  # Three arms

    bool_field_results = result["bool_field"]
    # Test using the fixed UUIDs
    assert bool_field_results["0ffe0995-6404-4622-934a-0d5cccfe3a59"].is_baseline is True
    assert bool_field_results["b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"].is_baseline is False
    assert bool_field_results["df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"].is_baseline is False

    # Test approximate values since floating point math may have small variations
    assert (
        pytest.approx(
            bool_field_results["0ffe0995-6404-4622-934a-0d5cccfe3a59"].estimate,
            abs=1e-4,
        )
        == 0.4888
    ), bool_field_results
    assert (
        pytest.approx(
            bool_field_results["b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"].estimate,
            abs=1e-4,
        )
        == 0.0149
    )
    assert (
        pytest.approx(
            bool_field_results["df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"].estimate,
            abs=1e-4,
        )
        == 0.0453
    )

    for arm_id in bool_field_results:
        assert (
            pytest.approx(
                bool_field_results[arm_id].num_missing_values,
                abs=10,
            )
            == 200 // 3
        )

    assert sum(arm_results.num_missing_values for arm_results in bool_field_results.values()) == 200


def test_analysis_with_one_arm_missing_all_outcomes(test_assignments, test_outcomes):
    # Make all outcomes missing for one arm. Should still be processed, but result in NaNs
    arm_map = {a.participant_id: a.arm_id for a in test_assignments}
    num_missing_values = 0
    for i in range(len(test_outcomes)):
        if arm_map[test_outcomes[i].participant_id] == "b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5":
            test_outcomes[i].metric_values[0].metric_value = None
            num_missing_values += 1

    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result) == 1  # One metric
    metric_results = result["bool_field"]
    assert len(metric_results) == 3  # Three arms
    for arm_id in metric_results:
        if arm_id == "b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5":
            assert metric_results[arm_id].estimate == 0
            assert math.isnan(metric_results[arm_id].p_value)
            assert math.isnan(metric_results[arm_id].t_stat)
            assert metric_results[arm_id].std_error == 0
            assert metric_results[arm_id].num_missing_values == num_missing_values
        else:
            assert metric_results[arm_id].estimate is not None
            assert metric_results[arm_id].p_value is not None
            assert metric_results[arm_id].t_stat is not None
            assert metric_results[arm_id].std_error is not None
            assert metric_results[arm_id].num_missing_values == 0

    # But when *all* arms are missing values, we should get a dict with the metrics but no arm
    # analyses since no regression was performed.
    for i in range(len(test_outcomes)):
        test_outcomes[i].metric_values[0].metric_value = None
    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result) == 1  # One metric
    assert len(result["bool_field"]) == 0  # No arm analyses
