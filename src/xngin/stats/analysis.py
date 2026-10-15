import dataclasses

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy.eval import EvalFactor

from xngin.apiserver.dwh.analysis_types import ParticipantOutcome
from xngin.stats.stats_errors import StatsAnalysisError


@dataclasses.dataclass(slots=True)  # slots=True for performance
class ArmAnalysisResult:
    is_baseline: bool
    estimate: float
    p_value: float
    t_stat: float
    std_error: float
    # Confidence intervals for the estimated coefficient
    ci_lower: float
    ci_upper: float
    # Confidence intervals for the mean of the arm
    mean_ci_lower: float
    mean_ci_upper: float
    num_missing_values: int


def analyze_experiment(
    assignments_df: pd.DataFrame,
    participant_outcomes: list[ParticipantOutcome],
    *,
    unit_col: str = "participant_id",
    arm_col: str = "arm_id",
    cluster_col: str | None = None,
    baseline_arm_id: str | None = None,
    alpha: float | None = None,
) -> dict[str, dict[str, ArmAnalysisResult]]:
    """
    Perform statistical analysis with DesignSpec metrics and their values

    Args:
    assignments_df: DataFrame of {unit_col, arm_col[, cluster_col]} strings containing ids
            corresponding to a unit of analysis (a participant), its treatment assignment, and
            optionally the cluster id this unit is a part of for cluster-randomized experiments.
    participant_outcomes: list of outcomes for each unit in assignments_df, in the same order.
    unit_col: Name of column in assignments_df to use for participant identifiers.
    arm_col: Name of column in assignments_df to use for arm identifiers.
    cluster_col: Name of column in assignments_df to use as cluster identifiers.
            If provided, the analysis uses clustered SEs instead of HC SEs.
    baseline_arm_id: which arm to use as baseline; if not provided, uses the first arm seen
    alpha: significance level for confidence intervals (defaults to 0.05 if None for a 95% CI).

    Returns:
        map of metric name => map of arm_id (may be partial or empty!) => analysis results
        - If an arm is missing for a metric, it's because it had no valid data to process.
        - If *zero* arm analyses exist for a metric (e.g. since zero non-null outcomes were found
        across all arms), the name will exist, but the inner dict will be empty.
    """
    expected_id_columns = {unit_col, arm_col}
    if cluster_col is not None:
        expected_id_columns |= {cluster_col}
    actual_columns = set(assignments_df.columns)
    if actual_columns != expected_id_columns:
        raise ValueError(
            f"assignments_df shape is wrong: expected={','.join(sorted(expected_id_columns))},"
            f" got={','.join(sorted(actual_columns))}"
        )
    if cluster_col is not None and assignments_df[cluster_col].isna().any():
        raise StatsAnalysisError(
            f"One or more participants in a clustered experiment have a null {cluster_col}, which is disallowed."
        )

    if alpha is None:
        alpha = 0.05

    rows = []
    for outcome in participant_outcomes:
        data_row: dict[str, float | str | None] = {unit_col: outcome.participant_id}
        for metric_value in outcome.metric_values:
            data_row[metric_value.metric_name] = metric_value.metric_value
        rows.append(data_row)
    outcomes_df = pd.DataFrame(rows)

    merged_df = assignments_df.merge(outcomes_df, on=unit_col, how="left")

    # Make arm_id categorical and ensure baseline_arm_id is first in the categories
    merged_df[arm_col] = pd.Categorical(merged_df[arm_col])
    if baseline_arm_id in merged_df[arm_col].cat.categories:
        arm_ids = merged_df[arm_col].cat.categories.tolist()
        arm_ids.remove(baseline_arm_id)
        arm_ids.insert(0, baseline_arm_id)
        merged_df[arm_col] = merged_df[arm_col].cat.reorder_categories(arm_ids)

    # Exclude various id columns
    metric_columns = [col for col in merged_df.columns if col not in expected_id_columns]

    # Calculate NaN counts for all metrics. Since assignments_df may have participants that are not
    # yet in the dwh (e.g. in an online experiment) we're also counting missing participants as having NaN as well.
    # count() ignores NaN, so rows-per-arm minus non-null-values-per-arm is the number of missing values.
    grouped = merged_df.groupby(arm_col, observed=False)
    nan_counts_df = grouped[metric_columns].count().rsub(grouped.size(), axis=0)

    # Prep our dict of analyses to return.
    metric_analyses: dict[str, dict[str, ArmAnalysisResult]] = {}
    for metric_name in metric_columns:
        # First init this metric's empty dict of arm analyses.
        arm_analyses: dict[str, ArmAnalysisResult] = {}
        metric_analyses[metric_name] = arm_analyses
        # If all arms have missing values for this metric, we can't perform the analysis.
        # Let callers deal with this case, since even treatment_assignments may not have all arms yet.
        if sum(nan_counts_df[metric_name]) == len(merged_df):
            continue

        # Remove empty arms before fitting so OLS cannot assign them arbitrary coefficients.
        # Without baseline observations, no treatment-v-control comparisons are available.
        merged_df_dropna = merged_df.dropna(subset=[metric_name]).copy()
        merged_df_dropna[arm_col] = merged_df_dropna[arm_col].cat.remove_unused_categories()
        baseline_id = baseline_arm_id if baseline_arm_id is not None else merged_df[arm_col].cat.categories[0]
        if baseline_id not in merged_df_dropna[arm_col].cat.categories:
            continue
        # Fit separate arm means: disjoint indicator columns avoid covariance cancellation
        # when an arm has zero individual or between-cluster variation.
        formula = f"{metric_name} ~ 0 + {arm_col}"
        model = smf.ols(formula, data=merged_df_dropna).fit(
            cov_type="HC1" if cluster_col is None else "cluster",
            cov_kwds=None if cluster_col is None else {"groups": merged_df_dropna[cluster_col]},
        )
        arm_ids = model.model.data.design_info.factor_infos[EvalFactor(arm_col)].categories

        # Preserve the API's baseline mean and treatment-minus-baseline estimates.
        contrasts = np.eye(len(arm_ids))
        contrasts[1:, 0] = -1
        comparisons = model.t_test(contrasts)
        p_values = np.asarray(comparisons.pvalue).reshape(-1)
        t_stats = np.asarray(comparisons.tvalue).reshape(-1)
        standard_errors = np.asarray(comparisons.sd).reshape(-1)
        # statsmodels t_test maps zero standard errors to a zero statistic; retain
        # infinite statistics for nonzero effects and undefined 0/0 inference.
        zero_error = standard_errors == 0
        zero_effect = comparisons.effect[zero_error] == 0
        t_stats[zero_error] = np.where(zero_effect, np.nan, np.copysign(np.inf, comparisons.effect[zero_error]))
        p_values[zero_error] = np.where(zero_effect, np.nan, 0)
        confidence_intervals = comparisons.conf_int(alpha=alpha)
        mean_confidence_intervals = model.conf_int(alpha=alpha)

        for i, arm_id in enumerate(arm_ids):
            arm_analyses[arm_id] = ArmAnalysisResult(
                is_baseline=i == 0 if baseline_arm_id is None else arm_id == baseline_arm_id,
                estimate=float(comparisons.effect[i]),
                p_value=float(p_values[i]),
                t_stat=float(t_stats[i]),
                std_error=float(standard_errors[i]),
                ci_lower=float(confidence_intervals[i, 0]),
                ci_upper=float(confidence_intervals[i, 1]),
                mean_ci_lower=float(mean_confidence_intervals.iloc[i, 0]),
                mean_ci_upper=float(mean_confidence_intervals.iloc[i, 1]),
                num_missing_values=nan_counts_df.loc[arm_id, metric_name],
            )
    return metric_analyses
