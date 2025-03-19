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
def test_assignments(n=1000, num_arms=3, seed=42):
    np.random.seed(seed)
    arm_ids = [uuid.uuid4() for _ in range(num_arms)]
    assignments = []
    for i in range(n):
        arm_id = np.random.choice(arm_ids, size=1, replace=True)[0]
        assignments.append(
            # TODO: test Assignment for old stateless api
            ArmAssignment(participant_id=str(i), arm_id=arm_id, strata=[])
        )
    return assignments


@pytest.fixture
def test_outcomes(n=1000, seed=42):
    return [
        ParticipantOutcome(
            participant_id=str(i),
            metric_values=[
                MetricValue(
                    metric_name="bool_field", metric_value=np.random.choice([0, 1])
                )
            ],
        )
        for i in range(n)
    ]


def test_analysis(test_assignments, test_outcomes):
    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result.metric_analyses) == 1
    assert len(result.metric_analyses[0].arm_analyses) == 3
    # assert set(assignment.arm_id for assignment in test_assignments) == set(
    #     result[0].arm_ids
    # )
    # assert len(result[0].arm_ids) == 3
    # assert len(result[0].pvalues) == 3
    # assert len(result[0].tstats) == 3
    # assert len(result[0].std_errors) == 3
    # assert result[0].pvalues[0] < 0.01
    # assert result[0].pvalues[1] > 0.01
    # assert result[0].pvalues[2] > 0.01
