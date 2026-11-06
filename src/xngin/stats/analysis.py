import dataclasses
import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy.eval import EvalFactor

from xngin.apiserver.dwh.analysis_types import ParticipantOutcome
from xngin.apiserver.dwh.field_aliases import FieldAliases
from xngin.stats.stats_errors import StatsAnalysisError

# Patsy formulas refer to columns in the dataframes. The input dataframes have customer-provided column names,
# and they could conflict with Patsy operators or built-ins. To avoid that, we compose a dataframe where the
# arms and metrics are mapped to constant column names.
MODEL_ARM_COL = "arm"
MODEL_OUTCOME_COL = "outcome"
MODEL_FORMULA = f"{MODEL_OUTCOME_COL} ~ 0 + {MODEL_ARM_COL}"


@dataclasses.dataclass(slots=True)  # slots=True for performance
class ArmAnalysisResult:
    is_baseline: bool
    # The fields from estimate to ci_upper compare the arm to the baseline (for the baseline
    # itself, they describe its mean). They are NaN when the baseline has no values. All but the estimate
    # are also NaN when the arm or the baseline lacks a variance of its own (see analyze_experiment).
    estimate: float
    p_value: float
    t_stat: float
    std_error: float
    # Confidence intervals for the estimated coefficient
    ci_lower: float
    ci_upper: float
    # The mean of the arm and its confidence interval
    mean: float
    mean_ci_lower: float
    mean_ci_upper: float
    num_missing_values: int


def analyze_experiment(  # noqa: PLR0915
    assignments_df: pd.DataFrame,
    participant_outcomes: list[ParticipantOutcome],
    metric_names: list[str],
    *,
    unit_col: str = "participant_id",
    arm_col: str = "arm_id",
    cluster_col: str | None = None,
    baseline_arm_id: str | None = None,
    alpha: float | None = None,
) -> dict[str, dict[str, ArmAnalysisResult]]:
    """
    Perform statistical analysis with DesignSpec metrics and their values

    Fit separate arm means and use explicit control contrasts to avoid covariance cancellation.
    A zero effect with zero standard error retains undefined t-statistics and p-values.

    Args:
    assignments_df: DataFrame of {unit_col, arm_col[, cluster_col]} strings containing ids
            corresponding to a unit of analysis (a participant), its treatment assignment, and
            optionally the cluster id this unit is a part of for cluster-randomized experiments.
    participant_outcomes: list of outcomes for each unit in assignments_df, in the same order.
    metric_names: names of the metrics to analyze; every metric value in participant_outcomes must be one of them.
    unit_col: Name of column in assignments_df to use for participant identifiers.
    arm_col: Name of column in assignments_df to use for arm identifiers.
    cluster_col: Name of column in assignments_df to use as cluster identifiers.
            If provided, the analysis uses clustered SEs instead of HC SEs.
    baseline_arm_id: which arm to use as baseline; if not provided, uses the first arm seen
    alpha: significance level for confidence intervals (defaults to 0.05 if None for a 95% CI).

    Returns:
        map of metric name => map of arm_id (may be partial or empty!) => analysis results
        - If an arm is missing for a metric, it's because it had no valid data to process.
        - If the baseline arm is missing, the other arms' results only include their means and
        mean CIs; the fields comparing them to the baseline are NaN.
        - An arm with fewer than 2 values (or, when clustered, values from fewer than 2 clusters) has no
        variance of its own to estimate. It keeps its mean and estimate, but its mean CIs and the inference
        fields comparing it to the baseline are NaN. If that arm is the baseline, every arm's comparison
        inference is NaN.
        - Every name in metric_names has an entry. If *zero* arm analyses exist for a metric (e.g. since zero
        non-null outcomes were found across all arms), the inner dict will be empty.
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

    # The frames hold each metric's values under its alias, so that no metric name can collide with an id column.
    aliases = FieldAliases(metric_names)
    rows = []
    for outcome in participant_outcomes:
        data_row: dict[str, float | str | None] = {unit_col: outcome.participant_id}
        for metric_value in outcome.metric_values:
            data_row[aliases.alias(metric_value.metric_name)] = metric_value.metric_value
        rows.append(data_row)
    outcomes_df = pd.DataFrame(rows, columns=[unit_col, *aliases.aliases()])

    merged_df = assignments_df.merge(outcomes_df, on=unit_col, how="left")

    # Make arm_id categorical and ensure baseline_arm_id is first in the categories
    merged_df[arm_col] = pd.Categorical(merged_df[arm_col])
    if baseline_arm_id in merged_df[arm_col].cat.categories:
        arm_ids = merged_df[arm_col].cat.categories.tolist()
        arm_ids.remove(baseline_arm_id)
        arm_ids.insert(0, baseline_arm_id)
        merged_df[arm_col] = merged_df[arm_col].cat.reorder_categories(arm_ids)

    # Calculate NaN counts for all metrics. Since assignments_df may have participants that are not
    # yet in the dwh (e.g. in an online experiment) we're also counting missing participants as having NaN as well.
    # count() ignores NaN, so rows-per-arm minus non-null-values-per-arm is the number of missing values.
    grouped = merged_df.groupby(arm_col, observed=False)
    nan_counts_df = grouped[aliases.aliases()].count().rsub(grouped.size(), axis=0)

    # Prep our dict of analyses to return.
    metric_analyses: dict[str, dict[str, ArmAnalysisResult]] = {}
    for metric_name, metric_col in aliases.name_to_aliases():
        # First init this metric's empty dict of arm analyses.
        arm_analyses: dict[str, ArmAnalysisResult] = {}
        metric_analyses[metric_name] = arm_analyses
        # If all arms have missing values for this metric, we can't perform the analysis.
        # Let callers deal with this case, since even treatment_assignments may not have all arms yet.
        if sum(nan_counts_df[metric_col]) == len(merged_df):
            continue

        observed = merged_df[metric_col].notna()
        # Drop arms left without any values: the model would otherwise keep an empty arm as a level and report
        # fabricated results for it.
        model_df = pd.DataFrame(
            {
                MODEL_ARM_COL: merged_df.loc[observed, arm_col].cat.remove_unused_categories(),
                MODEL_OUTCOME_COL: merged_df.loc[observed, metric_col],
            },
            copy=False,
        )
        # If the baseline has no values, the other arms can't be compared to it, so only their own means are reported.
        baseline = merged_df[arm_col].cat.categories[0] if baseline_arm_id is None else baseline_arm_id
        has_baseline_values = baseline in model_df[MODEL_ARM_COL].cat.categories

        clusters = None if cluster_col is None else merged_df.loc[observed, cluster_col]
        estimable = _arms_with_estimable_variance(model_df[MODEL_ARM_COL], clusters)
        if not estimable.any():
            # A fit would only divide by zero residual degrees of freedom (HC1) or by a degenerate cluster
            # correction, so report the point estimates alone.
            means = model_df.groupby(MODEL_ARM_COL, observed=True)[MODEL_OUTCOME_COL].mean()
            for arm_id in means.index:
                arm_analyses[arm_id] = _point_estimate_only(
                    mean=float(means[arm_id]),
                    baseline_mean=None if arm_id == baseline else float(means.get(baseline, math.nan)),
                    num_missing_values=nan_counts_df.loc[arm_id, metric_col],
                )
            continue

        # Disjoint arm indicators avoid covariance cancellation for zero-variance arm means.
        model = smf.ols(MODEL_FORMULA, data=model_df).fit(
            cov_type="HC1" if clusters is None else "cluster",
            cov_kwds=None if clusters is None else {"groups": clusters},
        )
        arm_ids = model.model.data.design_info.factor_infos[EvalFactor(MODEL_ARM_COL)].categories

        # Preserve the baseline mean and treatment-minus-baseline coefficient convention.
        contrasts = np.eye(len(arm_ids))
        contrasts[1:, 0] = -1
        comparisons = model.t_test(contrasts)
        p_values = np.asarray(comparisons.pvalue).reshape(-1)
        t_stats = np.asarray(comparisons.tvalue).reshape(-1)
        standard_errors = np.asarray(comparisons.sd).reshape(-1)
        # t_test maps zero standard errors to a zero statistic; preserve nonzero/0 and 0/0 inference.
        zero_error = standard_errors == 0
        zero_effect = comparisons.effect[zero_error] == 0
        t_stats[zero_error] = np.where(zero_effect, np.nan, np.copysign(np.inf, comparisons.effect[zero_error]))
        p_values[zero_error] = np.where(zero_effect, np.nan, 0)
        confidence_intervals = comparisons.conf_int(alpha=alpha)
        mean_confidence_intervals = model.conf_int(alpha=alpha)

        for i, arm_id in enumerate(arm_ids):
            result = ArmAnalysisResult(
                is_baseline=arm_id == baseline,
                estimate=float(comparisons.effect[i]),
                p_value=float(p_values[i]),
                t_stat=float(t_stats[i]),
                std_error=float(standard_errors[i]),
                ci_lower=float(confidence_intervals[i, 0]),
                ci_upper=float(confidence_intervals[i, 1]),
                mean=float(model.params.iloc[i]),
                mean_ci_lower=float(mean_confidence_intervals.iloc[i, 0]),
                mean_ci_upper=float(mean_confidence_intervals.iloc[i, 1]),
                num_missing_values=nan_counts_df.loc[arm_id, metric_col],
            )
            # Robust variances would report an arm without a variance of its own as exact, so it keeps its point
            # estimates without inference, as does every comparison against such a baseline.
            if not estimable[i]:
                result.mean_ci_lower = result.mean_ci_upper = math.nan
            if not (has_baseline_values and estimable[i] and estimable[0]):
                result.p_value = result.t_stat = math.nan
                result.std_error = result.ci_lower = result.ci_upper = math.nan
            if not has_baseline_values:
                result.estimate = math.nan
            arm_analyses[arm_id] = result
    return metric_analyses


def _arms_with_estimable_variance(arms: pd.Series, clusters: pd.Series | None) -> np.ndarray:
    """Which arms, in model order, have a variance of their own to estimate.

    Robust variances are built from deviations around each arm's mean. A single observation (or, with
    cluster-robust variances, a single cluster) has none, so its arm's variance would come out as exactly zero.
    """
    sizes = (
        arms.groupby(arms, observed=True).size()
        if clusters is None
        else clusters.groupby(arms, observed=True).nunique()
    )
    return sizes.to_numpy() >= 2


def _point_estimate_only(*, mean: float, baseline_mean: float | None, num_missing_values: int) -> ArmAnalysisResult:
    """An arm's result without inference. baseline_mean is None for the baseline itself, and NaN if it has no values."""
    return ArmAnalysisResult(
        is_baseline=baseline_mean is None,
        estimate=mean if baseline_mean is None else mean - baseline_mean,
        p_value=math.nan,
        t_stat=math.nan,
        std_error=math.nan,
        ci_lower=math.nan,
        ci_upper=math.nan,
        mean=mean,
        mean_ci_lower=math.nan,
        mean_ci_upper=math.nan,
        num_missing_values=num_missing_values,
    )
