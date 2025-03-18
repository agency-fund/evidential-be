def analyze_metric_power(
    metric: DesignSpecMetric, n_arms: int, power: float = 0.8, alpha: float = 0.05
) -> MetricPowerAnalysis:
    """
    Analyze power for a single metric.

    Args:
        metric: DesignSpecMetric containing metric details
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level

    Returns:
        MetricPowerAnalysis containing power analysis results
    """
    # ... in the function ...
    if metric.available_n is None or metric.available_n <= 0:
        return _power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.NO_AVAILABLE_N,
            (
                "You have no available units to run your experiment. "
                "Adjust your filters to target more units."
            ),
        )

    # ... later in the function ...
    analysis = MetricPowerAnalysis(metric_spec=metric)
