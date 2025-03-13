# ruff: noqa: RUF027
import numpy as np
import statsmodels.stats.api as sms

from xngin.apiserver.api_types import (
    DesignSpecMetric,
    MetricType,
    PowerResponse,
    MetricAnalysis,
    MetricAnalysisMessage,
    MetricAnalysisMessageType,
)
from xngin.stats.stats_errors import StatsPowerError


def analyze_metric_power(
    metric: DesignSpecMetric, n_arms: int, power: float = 0.8, alpha: float = 0.05
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
    if metric.metric_type is None:
        raise ValueError("Unknown metric_type.")

    if metric.metric_target is None and metric.metric_baseline is not None:
        metric.metric_target = metric.metric_baseline * (1 + metric.metric_pct_change)

    analysis = MetricAnalysis(metric_spec=metric)

    # Validate we have usable input to do the analysis.
    # TODO? To support more general power calculation functionality by implementing Case B:
    # target is not defined (need to also relax request constraints), but baseline is
    # => calculate effect size. (Case A does this only when there's insufficient available_n.)
    if metric.available_n <= 0:
        msg_body = (
            "You have no available units to run your experiment. "
            "Adjust your filters to target more units."
        )
        analysis.msg = MetricAnalysisMessage(
            type=MetricAnalysisMessageType.NO_AVAILABLE_N,
            msg=msg_body,
            source_msg=msg_body,
        )
        return analysis

    if metric.metric_target is None or metric.metric_baseline is None:
        msg_body = (
            "Could not calculate metric baseline with given specification. "
            "Provide a metric baseline or adjust filters."
        )
        analysis.msg = MetricAnalysisMessage(
            type=MetricAnalysisMessageType.NO_BASELINE,
            msg=msg_body,
            source_msg=msg_body,
        )
        return analysis

    # Case A: Both target and baseline defined - calculate required n
    if metric.metric_type == MetricType.NUMERIC:
        effect_size = (
            metric.metric_target - metric.metric_baseline
        ) / metric.metric_stddev
    elif metric.metric_type == MetricType.BINARY:
        effect_size = sms.proportion_effectsize(
            metric.metric_baseline, metric.metric_target
        )
    else:
        raise ValueError("metric_type must be NUMERIC or BINARY.")

    if effect_size == 0.0:
        raise ValueError(
            "Cannot detect an effect-size of 0. Try changing your effect-size."
        )

    power_analysis = sms.TTestIndPower()
    target_n = (
        np.ceil(
            power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                ratio=1,
            )
        )
        * n_arms
    )

    analysis.target_n = int(target_n)
    analysis.sufficient_n = bool(target_n <= metric.available_n)

    # Construct potential components of the MetricAnalysisMessage
    has_nulls = metric.available_nonnull_n != metric.available_n
    values_map = {
        "available_n": metric.available_n,
        "target_n": analysis.target_n,
        "available_nonnull_n": metric.available_nonnull_n,
    }

    msg_base_stats = (
        "There are {available_n} units available to run your experiment and a "
        "minimum of {target_n} units are needed to meet your experimental design specs."
    )
    msg_null_warning = (
        (
            "WARNING: There are {available_nonnull_n} units with a non-null value out of the "
            "{available_n} available.  The power calculation was done with only units with a "
            "value present, but random assignment is performed over all available units "
            "meeting your filters, including those with a missing value. If you do not want "
            "that, add a filter on this metric to exclude nulls."
        )
        if has_nulls
        else ""
    )

    if analysis.sufficient_n:
        msg_type = MetricAnalysisMessageType.SUFFICIENT
        msg_body = "There are enough units available."
    else:
        msg_type = MetricAnalysisMessageType.INSUFFICIENT
        # Calculate needed target if insufficient sample
        if metric.metric_type == MetricType.NUMERIC:
            power_analysis = sms.TTestIndPower()
            needed_delta = (
                power_analysis.solve_power(
                    nobs1=metric.available_n // n_arms,
                    effect_size=None,
                    alpha=alpha,
                    power=power,
                )
                * metric.metric_stddev
            )
            target_possible = needed_delta + metric.metric_baseline
        else:  # BINARY
            power_analysis = sms.NormalIndPower()
            # Calculate minimum detectable effect size given sample size
            min_effect_size = power_analysis.solve_power(
                nobs1=metric.available_n // n_arms,
                alpha=alpha,
                power=power,
                ratio=1,
            )

            # Convert Cohen's h back to proportion
            # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
            # where p1 is baseline and p2 is target
            p1 = metric.metric_baseline
            arcsin_p2 = 2 * np.arcsin(np.sqrt(p1)) - min_effect_size
            target_possible = np.sin(arcsin_p2 / 2) ** 2

        analysis.target_possible = target_possible
        analysis.pct_change_possible = target_possible / metric.metric_baseline - 1.0

        values_map["additional_n_needed"] = target_n - metric.available_n
        values_map["metric_baseline"] = round(metric.metric_baseline, 4)
        values_map["target_possible"] = round(target_possible, 4)
        values_map["metric_target"] = round(metric.metric_target, 4)
        msg_body = (
            "There are not enough units available. "
            "You need {additional_n_needed} more units to meet your experimental design "
            "specifications. In order to meet your specification with the available "
            "{available_n} units and a metric baseline value of {metric_baseline}, your metric "
            "target value needs to be {target_possible} or further from the baseline. Your "
            "current desired target is {metric_target}."
        )

    # Construct our response from the parts above
    source_msg = " ".join([msg_base_stats, msg_body, msg_null_warning])
    analysis.msg = MetricAnalysisMessage(
        type=msg_type,
        msg=source_msg.format_map(values_map),
        source_msg=source_msg,
        values=values_map,
    )
    return analysis


def check_power(
    metrics: list[DesignSpecMetric],
    n_arms: int,
    power: float = 0.8,
    alpha: float = 0.05,
) -> PowerResponse:
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
    analyses = []
    for metric in metrics:
        try:
            analyses.append(analyze_metric_power(metric, n_arms, power, alpha))
        except ValueError as verr:
            raise StatsPowerError(verr, metric) from verr
    return analyses
