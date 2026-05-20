import math
import random
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest

from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.stats.analysis import analyze_experiment


@pytest.fixture(name="test_assignments")
def fixture_assignments(n=1000, seed=42):
    rand = random.Random(seed)
    # Use fixed UUIDs instead of randomly generated ones
    arm_ids = [
        uuid.UUID("0ffe0995-6404-4622-934a-0d5cccfe3a59"),
        uuid.UUID("b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5"),
        uuid.UUID("df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"),
    ]
    assignments = [
        {
            "participant_id": str(i),
            "arm_id": str(rand.choice(arm_ids)),
        }
        for i in range(n)
    ]
    return pd.DataFrame(assignments)


@pytest.fixture(name="test_outcomes")
def fixture_outcomes(n=1000, seed=43):
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
    baseline_id = "0ffe0995-6404-4622-934a-0d5cccfe3a59"
    treatment_ids = ["b1d90769-6e6e-4973-a7eb-d9da1c6ddcd5", "df84e3ae-f5df-4dc8-9ba6-fa0743e1c895"]
    assert bool_field_results[baseline_id].is_baseline is True
    for treatment_id in treatment_ids:
        assert bool_field_results[treatment_id].is_baseline is False

    # Now check the analysis results for the bool_field metric.
    baseline_results = bool_field_results[baseline_id]
    intercept = baseline_results.estimate
    assert pytest.approx(intercept, abs=1e-4) == 0.4793
    # Check CIs exist and by default represent a 95% confidence interval wrapping the estimate
    assert pytest.approx(baseline_results.ci_lower, abs=1e-4) == intercept - 1.96 * baseline_results.std_error
    assert pytest.approx(baseline_results.ci_upper, abs=1e-4) == intercept + 1.96 * baseline_results.std_error
    # CI for the baseline arm's coefficient is the same as the CI for the arm's mean
    assert baseline_results.ci_lower == baseline_results.mean_ci_lower
    assert baseline_results.ci_upper == baseline_results.mean_ci_upper

    for treatment_id, treatment_estimate in zip(treatment_ids, [0.0222, 0.0431], strict=True):
        results = bool_field_results[treatment_id]
        treatment_effect = results.estimate
        assert pytest.approx(treatment_effect, abs=1e-4) == treatment_estimate
        # The arm's treatment effect is an offset from the intercept; check that its CIs bound this offset.
        assert results.ci_lower < treatment_effect < results.ci_upper
        # CIs for the treatment arm's mean estimate
        assert results.mean_ci_lower < intercept + treatment_effect < results.mean_ci_upper

    # Lastly, redo the analysis with a looser alpha (90% CIs) to ensure CIs are *narrower*
    result_a10 = analyze_experiment(test_assignments, test_outcomes, alpha=0.1)
    baseline_results_a10 = result_a10["bool_field"][baseline_id]
    assert baseline_results_a10.estimate == intercept
    assert baseline_results_a10.ci_lower > baseline_results.ci_lower
    assert baseline_results_a10.ci_upper < baseline_results.ci_upper
    assert baseline_results_a10.mean_ci_lower > baseline_results.mean_ci_lower
    assert baseline_results_a10.mean_ci_upper < baseline_results.mean_ci_upper


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
    arm_map = test_assignments.set_index("participant_id")["arm_id"].to_dict()
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
            assert metric_results[arm_id].ci_lower == 0
            assert metric_results[arm_id].ci_upper == 0
            assert metric_results[arm_id].mean_ci_lower is not None
            assert metric_results[arm_id].mean_ci_upper is not None
            assert metric_results[arm_id].num_missing_values == num_missing_values
        else:
            assert metric_results[arm_id].estimate is not None
            assert metric_results[arm_id].p_value is not None
            assert metric_results[arm_id].t_stat is not None
            assert metric_results[arm_id].std_error is not None
            assert metric_results[arm_id].ci_lower is not None
            assert metric_results[arm_id].ci_upper is not None
            assert metric_results[arm_id].mean_ci_lower is not None
            assert metric_results[arm_id].mean_ci_upper is not None
            assert metric_results[arm_id].num_missing_values == 0

    # But when *all* arms are missing values, we should get a dict with the metrics but no arm
    # analyses since no regression was performed.
    for i in range(len(test_outcomes)):
        test_outcomes[i].metric_values[0].metric_value = None
    result = analyze_experiment(test_assignments, test_outcomes)
    assert len(result) == 1  # One metric
    assert len(result["bool_field"]) == 0  # No arm analyses


def test_analysis_rejects_assignments_with_extra_columns(test_assignments, test_outcomes):
    invalid_assignments = test_assignments.assign(strata=[[] for _ in range(len(test_assignments))])

    with pytest.raises(ValueError, match="assignments_df shape is wrong"):
        analyze_experiment(invalid_assignments, test_outcomes)


def test_analysis_with_cluster_col():
    """Clustered SEs differ from HC1 SEs; estimates are identical."""
    rng = np.random.default_rng(42)
    n_clusters = 50
    cluster_ids = np.repeat(range(n_clusters), 20)
    treatment = rng.binomial(1, 0.5, n_clusters)[cluster_ids]
    outcome = 10 + 3 * treatment + rng.normal(0, 2, n_clusters)[cluster_ids] + rng.normal(0, 1, 1000)

    assignments_df = pd.DataFrame({
        "participant_id": [str(i) for i in range(1000)],
        "arm_id": ["control" if t == 0 else "treatment" for t in treatment],
        "cluster": cluster_ids,
    })
    participant_outcomes = [
        ParticipantOutcome(
            participant_id=str(i),
            metric_values=[MetricValue(metric_name="revenue", metric_value=float(outcome[i]))],
        )
        for i in range(1000)
    ]

    result_hc1 = analyze_experiment(assignments_df.drop(columns=["cluster"]), participant_outcomes)
    result_clustered = analyze_experiment(assignments_df, participant_outcomes, cluster_col="cluster")

    hc1_treatment = result_hc1["revenue"]["treatment"]
    clustered_treatment = result_clustered["revenue"]["treatment"]

    # Point estimates are the same
    assert pytest.approx(hc1_treatment.estimate, abs=1e-4) == clustered_treatment.estimate
    assert pytest.approx(clustered_treatment.estimate, abs=1e-4) == 2.9898
    # But clustered SE & p-value are greater than HC1's
    assert pytest.approx(hc1_treatment.std_error, abs=1e-4) == 0.1141
    assert pytest.approx(clustered_treatment.std_error, abs=1e-4) == 0.4327
    assert hc1_treatment.p_value == pytest.approx(0.0, abs=1e-150)
    assert clustered_treatment.p_value == pytest.approx(4.845e-12, abs=1e-13)


def test_analysis_with_cluster_col_missing_from_df_raises():
    """analyze_experiment raises ValueError if cluster_col is missing from assignments_df."""
    assignments_df = pd.DataFrame({
        "participant_id": [str(i) for i in range(100)],
        "arm_id": ["arm1", "arm2"] * 50,
    })
    unused_participant_outcomes: list[ParticipantOutcome] = []
    with pytest.raises(ValueError, match=r"assignments_df shape is wrong: expected=.*missing_cluster_column"):
        analyze_experiment(assignments_df, unused_participant_outcomes, cluster_col="missing_cluster_column")


def test_analysis_with_cluster_col_unequal_sizes():
    """Clustered analysis works correctly with unequal cluster sizes."""
    rng = np.random.default_rng(42)
    cluster_sizes = rng.integers(5, 50, 20)
    cluster_ids = np.repeat(range(20), cluster_sizes)
    n = len(cluster_ids)
    treatment = rng.binomial(1, 0.5, 20)[cluster_ids]
    outcome = 10 + 3 * treatment + rng.normal(0, 2, 20)[cluster_ids] + rng.normal(0, 1, n)

    assignments_df = pd.DataFrame({
        "participant_id": [str(i) for i in range(n)],
        "arm_id": ["control" if t == 0 else "treatment" for t in treatment],
        "cluster": cluster_ids,
    })
    participant_outcomes = [
        ParticipantOutcome(
            participant_id=str(i),
            metric_values=[MetricValue(metric_name="revenue", metric_value=float(outcome[i]))],
        )
        for i in range(n)
    ]

    result = analyze_experiment(assignments_df, participant_outcomes, cluster_col="cluster")
    treatment_result = result["revenue"]["treatment"]

    assert treatment_result.is_baseline is False
    assert pytest.approx(treatment_result.estimate, abs=1e-4) == 1.6612
    assert pytest.approx(treatment_result.std_error, abs=1e-4) == 0.5535
    assert pytest.approx(treatment_result.p_value, abs=1e-4) == 0.0027


def test_analysis_with_cluster_col_missing_values():
    """Clustered analysis correctly handles missing outcome values."""
    rng = np.random.default_rng(42)
    n_clusters = 50
    cluster_ids = np.repeat(range(n_clusters), 20)
    treatment = rng.binomial(1, 0.5, n_clusters)[cluster_ids]
    n_indv = len(cluster_ids)
    outcome = 10 + 3 * treatment + rng.normal(0, 2, n_clusters)[cluster_ids] + rng.normal(0, 1, n_indv)

    assignments_df = pd.DataFrame({
        "participant_id": [str(i) for i in range(n_indv)],
        "arm_id": ["control" if t == 0 else "treatment" for t in treatment],
        "cluster": cluster_ids,
    })
    # Add an extra 100 participants with missing outcomes
    none_df = pd.DataFrame({
        "participant_id": [str(i) for i in range(n_indv, n_indv + 100)],
        "arm_id": ["control"] * 50 + ["treatment"] * 50,
        "cluster": np.repeat(range(n_clusters), 2),
    })
    assignments__with_none_df = pd.concat([assignments_df, none_df])
    none: Any = None
    participant_outcomes = [
        ParticipantOutcome(
            participant_id=str(i),
            metric_values=[MetricValue(metric_name="revenue", metric_value=none if i >= n_indv else float(outcome[i]))],
        )
        for i in range(len(assignments__with_none_df))
    ]

    result = analyze_experiment(assignments__with_none_df, participant_outcomes, cluster_col="cluster")
    treatment_result = result["revenue"]["treatment"]

    # Estimate & SE should match values in test_analysis_with_cluster_col since rows with missing outcomes are dropped.
    assert treatment_result.is_baseline is False
    assert pytest.approx(treatment_result.estimate, abs=1e-4) == 2.9898
    assert pytest.approx(treatment_result.std_error, abs=1e-4) == 0.4327
    assert sum(r.num_missing_values for r in result["revenue"].values()) == len(none_df)


def test_analysis_with_cluster_col_three_arms():
    """Clustered analysis works correctly with three arms."""
    rng = np.random.default_rng(42)
    n_clusters = 100
    cluster_ids = np.repeat(range(n_clusters), 10)
    arm_assignment = rng.integers(0, 3, n_clusters)[cluster_ids]
    n_indv = len(cluster_ids)
    outcome = 10 + arm_assignment * 2 + rng.normal(0, 2, n_clusters)[cluster_ids] + rng.normal(0, 1, n_indv)

    assignments_df = pd.DataFrame({
        "participant_id": [str(i) for i in range(n_indv)],
        "arm_id": ["control" if a == 0 else f"treatment_{a}" for a in arm_assignment],
        "cluster": cluster_ids,
    })
    participant_outcomes = [
        ParticipantOutcome(
            participant_id=str(i),
            metric_values=[MetricValue(metric_name="revenue", metric_value=float(outcome[i]))],
        )
        for i in range(n_indv)
    ]

    result = analyze_experiment(assignments_df, participant_outcomes, cluster_col="cluster")

    revenue_results = result["revenue"]
    assert set(revenue_results.keys()) == {"control", "treatment_1", "treatment_2"}
    assert pytest.approx(revenue_results["control"].estimate, abs=1e-4) == 9.6418
    # Estimates reflect the arm_assignment*2 effect + cluster-level & individual-level noise
    assert pytest.approx(revenue_results["treatment_1"].estimate, abs=1e-4) == 2.3073
    assert pytest.approx(revenue_results["treatment_2"].estimate, abs=1e-4) == 3.9717
    assert pytest.approx(revenue_results["treatment_1"].std_error, abs=1e-4) == 0.4464
    assert pytest.approx(revenue_results["treatment_2"].std_error, abs=1e-4) == 0.4300
