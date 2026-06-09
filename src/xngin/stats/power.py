from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    MetricPowerAnalysis,
)
from xngin.apiserver.routers.common_enums import MetricPowerAnalysisMessageType, MetricType
from xngin.stats.cluster_power import solve_for_mde_cluster, solve_for_sample_size_cluster
from xngin.stats.individual_power import (
    power_analysis_error,
    solve_for_mde_individual,
    solve_for_sample_size_individual,
)
from xngin.stats.stats_errors import StatsPowerError


def analyze_metric_power(
    metric: DesignSpecMetric,
    *,
    n_arms: int,
    arm_weights: list[float] | None = None,
    desired_n: int | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> MetricPowerAnalysis:
    """
    Analyze power for a single metric.

    Args:
        metric: DesignSpecMetric containing metric details (including optional descriptive statistics)
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level
        arm_weights: Optional list of weights (summing to 100) for unbalanced arms, else assumes equal allocation.
        desired_n: Optional desired sample size. If provided, calculates MDE for the desired_n
            instead of the minimum sample size for a desired effect. Applies to all metrics.

    Returns:
        MetricPowerAnalysis containing power analysis results
    """
    is_cluster = metric.icc is not None and metric.avg_cluster_size is not None and metric.cv is not None

    # Sample size mode:
    if desired_n is None:
        if is_cluster:
            return solve_for_sample_size_cluster(
                metric=metric,
                n_arms=n_arms,
                power=power,
                alpha=alpha,
                arm_weights=arm_weights,
            )
        return solve_for_sample_size_individual(
            metric=metric,
            n_arms=n_arms,
            power=power,
            alpha=alpha,
            arm_weights=arm_weights,
        )

    # else MDE mode:

    # Validate baseline is present
    if metric.metric_baseline is None:
        return power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.NO_BASELINE,
            (
                "Could not calculate metric baseline with given specification. "
                "Provide a metric baseline or adjust filters."
            ),
        )

    # Validate stddev for NUMERIC metrics
    if metric.metric_type == MetricType.NUMERIC and (metric.metric_stddev is None or metric.metric_stddev <= 0):
        return power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.ZERO_STDDEV,
            (
                "There is no variation in the metric with the given filters. Standard deviation must be "
                "positive to do a sample size calculation."
            ),
        )

    if is_cluster:
        return solve_for_mde_cluster(
            metric=metric,
            desired_n=desired_n,
            arm_weights=arm_weights,
            n_arms=n_arms,
            power=power,
            alpha=alpha,
        )

    return solve_for_mde_individual(
        metric=metric,
        desired_n=desired_n,
        arm_weights=arm_weights,
        n_arms=n_arms,
        power=power,
        alpha=alpha,
    )


def check_power(
    metrics: list[DesignSpecMetric],
    *,
    n_arms: int,
    arm_weights: list[float] | None = None,
    desired_n: int | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> list[MetricPowerAnalysis]:
    """
    Check power for multiple metrics.

    Descriptive statistics of the data (icc, avg_cluster_size, cv) are read per-metric from
    each DesignSpecMetric, allowing each to have their own power calculation.

    Args:
        metrics: List of DesignSpecMetric objects to analyze
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level
        arm_weights: Optional list of weights (summing to 100) for unbalanced arms, else assumes equal allocation.
        desired_n: Optional desired sample size. If provided, calculates MDE for the desired_n
            instead of the minimum sample size for a desired effect. Applies to all metrics.

    Returns:
        List of `MetricPowerAnalysis` results, one item per metric in the same order as the input `metrics`.

        Each analysis object if successful should always include `target_n`, i.e. the minimum sample
        size for a desired effect.  If the analysis is not successful, see its `msg` for the reason.

        Optionally, if `desired_n` is provided, it will also include `pct_change_with_desired_n`,
        i.e. the corresponding MDE as a % change from baseline for the desired sample size. This MDE
        calculation is done in a best-effort manner and assumes there are enough units to meet the
        desired_n, so it can still succeed despite insufficient available units. If it fails,
        `pct_change_with_desired_n` will be None.
    """
    analyses = []
    for metric in metrics:
        try:
            analysis = analyze_metric_power(
                metric=metric, n_arms=n_arms, arm_weights=arm_weights, power=power, alpha=alpha
            )

            # Optional desired_n MDE calculation added to the min sample size analysis:
            if desired_n is not None:
                mde_analysis = analyze_metric_power(
                    metric=metric,
                    n_arms=n_arms,
                    arm_weights=arm_weights,
                    desired_n=desired_n,
                    power=power,
                    alpha=alpha,
                )
                # pct_change_possible may be None if there was any error created with
                # power_analysis_error().
                if mde_analysis.pct_change_possible is not None:
                    analysis.pct_change_with_desired_n = mde_analysis.pct_change_possible

            analyses.append(analysis)

        except ZeroDivisionError as zde:
            raise StatsPowerError("Zero division - You likely have too few samples.") from zde
        except ValueError as verr:
            if "f(a) and f(b) must have different signs" in str(verr):
                raise StatsPowerError(
                    "Unable to solve. Check your sample size, effect size, or significance parameters."
                ) from verr

            raise StatsPowerError.from_error(verr, metric.field_name) from verr

    return analyses
