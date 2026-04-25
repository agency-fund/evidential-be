from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    MetricPowerAnalysis,
    MetricPowerAnalysisMessage,
)
from xngin.apiserver.routers.common_enums import MetricPowerAnalysisMessageType, MetricType
from xngin.stats.cluster_power import analyze_metric_power_cluster, calculate_mde_cluster
from xngin.stats.individual_power import (
    _analyze_power_mde_mode,  # noqa: PLC2701
    _analyze_power_sample_size_mode,  # noqa: PLC2701
    _power_analysis_error,  # noqa: PLC2701
)
from xngin.stats.stats_errors import StatsPowerError


def analyze_metric_power(
    metric: DesignSpecMetric,
    n_arms: int,
    power: float = 0.8,
    alpha: float = 0.05,
    arm_weights: list[float] | None = None,
    desired_n: int | None = None,
) -> MetricPowerAnalysis:
    """
    Analyze power for a single metric.

    Args:
        metric: DesignSpecMetric containing metric details (including optional descriptive statistics)
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level
        arm_weights: Optional list of weights (summing to 100) for unbalanced arms.
                     If None, assumes equal allocation.
        desired_n: Optional desired sample size. If provided, calculates the minimum
               detectable effect (MDE) for this sample size instead of calculating
               required sample size for the specified effect.

    Returns:
        MetricPowerAnalysis containing power analysis results
    """
    icc = metric.icc
    avg_cluster_size = metric.avg_cluster_size
    cv = metric.cv or 0.0
    is_cluster = icc is not None and avg_cluster_size is not None

    if desired_n is None:
        # Sample size mode
        if is_cluster:
            assert icc is not None and avg_cluster_size is not None
            return analyze_metric_power_cluster(
                metric=metric,
                n_arms=n_arms,
                icc=icc,
                avg_cluster_size=avg_cluster_size,
                cv=cv,
                power=power,
                alpha=alpha,
                arm_weights=arm_weights,
            )
        return _analyze_power_sample_size_mode(metric, n_arms, power, alpha, arm_weights)

    # MDE mode

    # Validate baseline is present
    if metric.metric_baseline is None:
        return _power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.NO_BASELINE,
            (
                "Could not calculate metric baseline with given specification. "
                "Provide a metric baseline or adjust filters."
            ),
        )

    # Validate stddev for NUMERIC metrics
    if metric.metric_type == MetricType.NUMERIC and (metric.metric_stddev is None or metric.metric_stddev <= 0):
        return _power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.ZERO_STDDEV,
            (
                "There is no variation in the metric with the given filters. Standard deviation must be "
                "positive to do a sample size calculation."
            ),
        )

    if not is_cluster:
        return _analyze_power_mde_mode(metric, n_arms, desired_n, power, alpha, arm_weights)

    # Cluster MDE
    assert icc is not None and avg_cluster_size is not None
    target_possible, pct_change_possible = calculate_mde_cluster(
        available_n=desired_n,
        metric=metric,
        n_arms=n_arms,
        icc=icc,
        avg_cluster_size=avg_cluster_size,
        cv=cv,
        power=power,
        alpha=alpha,
        arm_weights=arm_weights,
    )

    # Build response object for MDE calculation
    analysis = MetricPowerAnalysis(metric_spec=metric)
    analysis.target_n = desired_n
    analysis.target_possible = target_possible
    analysis.pct_change_possible = pct_change_possible
    analysis.sufficient_n = None  # Not applicable in MDE mode

    # Create message
    values_map: dict[str, float | int] = {
        "desired_n": desired_n,
        "metric_baseline": round(metric.metric_baseline, 4),
        "target_possible": round(target_possible, 4),
    }
    msg_type = MetricPowerAnalysisMessageType.SUFFICIENT
    msg_body = (
        "With a desired sample size of {desired_n} units and a metric baseline of "  # noqa: RUF027
        "{metric_baseline}, the minimum detectable effect (MDE) is {target_possible}."
    )
    analysis.msg = MetricPowerAnalysisMessage(
        type=msg_type,
        msg=msg_body.format_map(values_map),
        source_msg=msg_body,
        values=values_map,
    )
    return analysis


def check_power(
    metrics: list[DesignSpecMetric],
    n_arms: int,
    power: float = 0.8,
    alpha: float = 0.05,
    arm_weights: list[float] | None = None,
    desired_n: int | None = None,
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
        arm_weights: Optional list of weights (summing to 100) for unbalanced arms
        desired_n: Optional desired sample size. If provided, calculates MDE
                   instead of required sample size. Applies to all metrics.

    Returns:
        List of MetricPowerAnalysis results
    """
    analyses = []
    for metric in metrics:
        try:
            analyses.append(analyze_metric_power(metric, n_arms, power, alpha, arm_weights, desired_n))
        except ValueError as verr:
            raise StatsPowerError(verr, metric) from verr
    return analyses
