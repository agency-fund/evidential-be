from dataclasses import dataclass, asdict

# import dataclasses
# from decimal import Decimal
# from typing import Any
import uuid
import pytest
import pandas as pd
import numpy as np
from xngin.stats.analysis import analyze_experiment
from xngin.apiserver.api_types import Assignment, ParticipantOutcome


@pytest.fixture
def test_assignments(n=1000, num_arms=3, seed=42):
    np.random.seed(seed)
    arm_ids = [uuid.uuid4() for _ in range(num_arms)]
    arm_names = ["arm" + str(i) for i in range(1, num_arms + 1)]
    arms = {arm_id: arm_name for arm_id, arm_name in zip(arm_ids, arm_names)}
    assignments = []
    for i in range(n):
        arm_id = np.random.choice(arm_ids, size=1, replace=True)[0]
        arm_name = "arm" + str(arm_ids.index(arm_id))
        assignments.append(
            Assignment(
                participant_id=str(i), arm_id=arm_id, arm_name=arm_name, strata=[]
            )
        )
    return assignments


@pytest.fixture
def test_outcomes(n=1000, seed=42):
    outcomes = [
        ParticipantOutcome(participant_id=str(i), metric_value=np.random.choice([0, 1]))
        for i in range(n)
    ]

    return outcomes


def test_analysis(test_assignments, test_outcomes):
    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result.arm_ids) == 3
    assert len(result.pvalues) == 3
    assert len(result.tstats) == 3
    assert result.pvalues[0] < 0.01
    assert result.pvalues[1] > 0.01
    assert result.pvalues[2] > 0.01
