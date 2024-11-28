import numpy as np
import statsmodels.stats.api as sms

from xngin.apiserver.api_types import (
    DesignSpecMetric,
    MetricType,
    PowerAnalysis,
    MetricAnalysis,
    MetricAnalysisMessage,
    MetricAnalysisMessageType
)

def analyze_metric_power(
    metric: DesignSpecMetric,
    n_arms: int,
    power: float = 0.8,
    alpha: float = 0.05
) -> MetricAnalysis:
    """
    Analyze power for a single metric.
    
    Args:
        metric: DesignSpecMetric containing metric details
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level
    
    Returns:
        MetricAnalysis containing power analysis results
    """
    analysis = MetricAnalysis(
        metric_spec = metric,
        available_n = metric.available_n
    )

    if metric.metric_target is None:
        metric.metric_target = metric.metric_baseline * (1 + metric.metric_pct_change)

    # Case A: Both target and baseline defined - calculate required n
    if metric.metric_target is not None and metric.metric_baseline is not None:
        if metric.metric_type == MetricType.NUMERIC:
            power_analysis = sms.TTestIndPower()
            target_n = np.ceil(
                power_analysis.solve_power(
                    effect_size=(metric.metric_target - metric.metric_baseline) / metric.metric_stddev,
                    alpha=alpha,
                    power=power,
                    ratio=1
                )
            ) * n_arms
        else:  # BINARY
            power_analysis = sms.TTestIndPower()
            effect_size = sms.proportion_effectsize(
                metric.metric_baseline,
                metric.metric_target
            )
            target_n = np.ceil(
                power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1
                )
            ) * n_arms

        analysis.target_n = int(target_n)
        analysis.sufficient_n = bool(target_n <= metric.available_n)

        if analysis.sufficient_n:
            analysis.msg = MetricAnalysisMessage(
                type=MetricAnalysisMessageType.SUFFICIENT,
                msg=(f"There are {metric.available_n} units available to run your experiment and"
                     f" {target_n} units are needed to meet your experimental design specs."
                     f" There are enough units available, you only need {target_n}"
                     f" of the {metric.available_n} units to meet your experimental design specs."),
                values = {}
            )
        else:
            # Calculate needed target if insufficient sample
            if metric.metric_type == MetricType.NUMERIC:
                power_analysis = sms.TTestIndPower()
                needed_delta = power_analysis.solve_power(
                    nobs1=metric.available_n // n_arms,
                    effect_size=None,
                    alpha=alpha,
                    power=power
                ) * metric.metric_stddev
                needed_target = needed_delta + metric.metric_baseline
            else:  # BINARY
                power_analysis = sms.NormalIndPower()
                # Calculate minimum detectable effect size given sample size
                min_effect_size = power_analysis.solve_power(
                    nobs1=metric.available_n // n_arms,
                    alpha=alpha,
                    power=power,
                    ratio=1
                )
                
                # Convert Cohen's h back to proportion
                # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
                # where p1 is baseline and p2 is target
                p1 = metric.metric_baseline
                arcsin_p2 = 2 * np.arcsin(np.sqrt(p1)) - min_effect_size
                needed_target = np.sin(arcsin_p2 / 2) ** 2

            analysis.needed_target = needed_target
            # TODO(roboton): Consider this instead:
            # # note: not an f-string
            # msg="There are {available_n} units available... {target_n} units are needed... {target_n} ...",
            # values = {"available_n": 123, "target_n": 456, ...}
            # This allows the frontend to provide structured/enriched UX experiences based on the message type (e.g. SUFFICIENT) and the variables, and also allows translation to a local language.
            analysis.msg = MetricAnalysisMessage(
                type=MetricAnalysisMessageType.INSUFFICIENT,
                msg=(f"there are {metric.available_n} units available to run your experiment and {target_n} units are needed to meet your experimental design specs."
                     f" there are not enough units available, you need {target_n - metric.available_n} more units"
                     f" to meet your experimental design specifications. in order to meet your specification with the available"
                     f" {metric.available_n} units and a baseline metric value of {metric.metric_baseline:.4f}, your metric"
                     f" target value needs to be {needed_target:.4f}, the current target is {metric.metric_target:.4f}."),
                values = {}

            )
    else:
        analysis.msg = "Could not calculate metric baseline with given specification. Provide metric baseline or adjust filters."

    return analysis

def check_power(
    metrics: list[DesignSpecMetric],
    n_arms: int,
    power: float = 0.8,
    alpha: float = 0.05
) -> PowerAnalysis:
    """
    Check power for multiple metrics.
    
    Args:
        metrics: List of DesignSpecMetric objects to analyze
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level
    
    Returns:
        List of MetricAnalysis results
    """
    return [
        analyze_metric_power(metric, n_arms, power, alpha)
        for metric in metrics
    ] 