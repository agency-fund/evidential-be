# import dataclasses
# from decimal import Decimal
# from typing import Any
import uuid
import pytest
import numpy as np
from xngin.apiserver.models.tables import ArmAssignment
from xngin.stats.analysis import analyze_experiment
from xngin.apiserver.api_types import ParticipantOutcome, MetricValue


@pytest.fixture
def test_assignments(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    # Use fixed UUIDs instead of randomly generated ones
    arm_ids = [
        uuid.UUID("0ffe0995-6404-4622-934a-0d5cccfe3a59"),
        uuid.UUID("b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"),
        uuid.UUID("df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"),
    ]
    assignments = []
    for i in range(n):
        arm_id = rng.choice(arm_ids, size=1, replace=True)[0]
        assignments.append(
            # TODO: test Assignment for old stateless api
            ArmAssignment(participant_id=str(i), arm_id=arm_id, strata=[])
        )
    return assignments


@pytest.fixture
def test_outcomes(n=1000, seed=43):
    rng = np.random.default_rng(seed)
    return [
        ParticipantOutcome(
            participant_id=str(i),
            metric_values=[
                MetricValue(metric_name="bool_field", metric_value=rng.choice([0, 1]))
            ],
        )
        for i in range(n)
    ]


def test_analysis(test_assignments, test_outcomes):
    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result.keys()) == 1  # One metric
    assert len(next(iter(result.values())).keys()) == 3  # Three arms

    bool_field_results = result["bool_field"]
    # Test using the fixed UUIDs
    assert (
        bool_field_results[
            uuid.UUID("0ffe0995-6404-4622-934a-0d5cccfe3a59")
        ].is_baseline
        is True
    )
    assert (
        bool_field_results[
            uuid.UUID("b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5")
        ].is_baseline
        is False
    )
    assert (
        bool_field_results[
            uuid.UUID("df84e3ae-f5df-4dc8-9ba6-fa0743e1c895")
        ].is_baseline
        is False
    )

    # Test approximate values since floating point math may have small variations
    assert (
        pytest.approx(
            bool_field_results[
                uuid.UUID("0ffe0995-6404-4622-934a-0d5cccfe3a59")
            ].estimate,
            abs=1e-4,
        )
        == 0.5120
    )
    assert (
        pytest.approx(
            bool_field_results[
                uuid.UUID("b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5")
            ].estimate,
            abs=1e-4,
        )
        == -0.0302
    )
    assert (
        pytest.approx(
            bool_field_results[
                uuid.UUID("df84e3ae-f5df-4dc8-9ba6-fa0743e1c895")
            ].estimate,
            abs=1e-4,
        )
        == 0.0059
    )
