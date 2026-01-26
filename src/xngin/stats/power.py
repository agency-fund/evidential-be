import numpy as np
import statsmodels.stats.api as sms

from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    MetricPowerAnalysis,
    MetricPowerAnalysisMessage,
)
from xngin.apiserver.routers.common_enums import (
    MetricPowerAnalysisMessageType,
    MetricType,
)
from xngin.stats.stats_errors import StatsPowerError


def _calculate_arm_ratio_and_control_prob_from_weights(
    arm_weights: list[float] | None, n_arms: int
) -> tuple[float, float]:
    # Calculate sample size based on arm allocation
    arm_ratio = 1.0  # default represents equal allocation
    control_prob = 1.0 / n_arms
    if arm_weights is not None:
        # For unbalanced arms, we need to calculate based on the ratio of treatment to control
        # Convert weights (sum to 100) to probabilities
        sum_weights = sum(arm_weights)
        weights = [w / sum_weights for w in arm_weights]
        # We always assume the first arm is control.
        control_prob = weights[0]
        # Use the smallest treatment arm for a conservative estimate.
        # (this ensures even the smallest arm has at least the desired statistical power)
        min_treatment_prob = min(weights[1:])
        arm_ratio = min_treatment_prob / control_prob

    return arm_ratio, control_prob


def _power_analysis_error(
    metric: DesignSpecMetric, msg_type: MetricPowerAnalysisMessageType, msg_body: str
) -> MetricPowerAnalysis:
    return MetricPowerAnalysis(
        metric_spec=metric,
        msg=MetricPowerAnalysisMessage(type=msg_type, msg=msg_body, source_msg=msg_body, values=None),
    )


def calculate_mde_with_desired_n(
    desired_n: int,
    metric: DesignSpecMetric,
    n_arms: int,
    alpha: float = 0.05,
    power: float = 0.8,
    arm_weights: list[float] | None = None,
) -> tuple[float, float]:
    """
    Calculate the Minimum Detectable Effect (MDE) for a given metric and sample size.

    Args:
        desired_n: Total sample size to be used in the calculation
        metric: DesignSpecMetric containing metric details
        n_arms: Number of treatment arms
        alpha: Significance level
        power: Desired statistical power
        arm_weights: Optional list of weights (summing to 100) for unbalanced arms
    Returns:
        Minimum Detectable Effect (MDE) as a float
    """
    if desired_n <= 0:
        raise ValueError("Chosen sample size must be positive.")

    if metric.metric_baseline is None:
        raise ValueError("metric_baseline is required for MDE calculation.")

    if metric.metric_type == MetricType.NUMERIC and metric.metric_stddev is None:
        raise ValueError("metric_stddev is required for NUMERIC metrics.")

    # Calculate sample size based on arm allocation
    arm_ratio, control_prob = _calculate_arm_ratio_and_control_prob_from_weights(arm_weights, n_arms)

    control_n_available = int(desired_n * control_prob)

    match metric.metric_type:
        case MetricType.NUMERIC:
            power_analysis = sms.TTestIndPower()
            needed_delta = (
                power_analysis.solve_power(
                    nobs1=control_n_available,
                    effect_size=None,
                    alpha=alpha,
                    power=power,
                    ratio=arm_ratio,
                )
                * metric.metric_stddev
            )
            # need this because solve_power can return array depending on special handling from edge cases
            needed_delta = float(np.atleast_1d(needed_delta)[0])
            target_possible = needed_delta + metric.metric_baseline
        case MetricType.BINARY:
            power_analysis = sms.NormalIndPower()
            # Calculate minimum detectable effect size given sample size
            min_effect_size = power_analysis.solve_power(
                nobs1=control_n_available,
                alpha=alpha,
                power=power,
                ratio=arm_ratio,
            )
            # need this because solve_power can return array depending on special handling from edge cases
            min_effect_size = float(np.atleast_1d(min_effect_size)[0])

            # Convert Cohen's h back to proportion
            # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
            # where p1 is baseline and p2 is target
            p1 = metric.metric_baseline
            arcsin_p2 = 2 * np.arcsin(np.sqrt(p1)) - min_effect_size
            target_possible = np.sin(arcsin_p2 / 2) ** 2
        case _:
            raise ValueError(f"metric_type must be one of {list(MetricType)}.")

    pct_change_possible = target_possible / metric.metric_baseline - 1.0
    return target_possible, pct_change_possible


def _analyze_power_sample_size_mode(
    metric: DesignSpecMetric,
    n_arms: int,
    power: float,
    alpha: float,
    arm_weights: list[float] | None,
) -> MetricPowerAnalysis:
    """
    Calculate required sample size given target effect.

    This function validates its own inputs and returns appropriate errors.
    """

    # Validate metric type
    if metric.metric_type is None:
        raise ValueError("Unknown metric_type.")

    # Validate available_n is present
    if metric.available_n is None or metric.available_n <= 0:
        return _power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.NO_AVAILABLE_N,
            "You have no available units to run your experiment. Adjust your filters to target more units.",
        )

    # Calculate target from pct_change if needed
    if metric.metric_target is None and metric.metric_baseline is not None and metric.metric_pct_change is not None:
        metric.metric_target = metric.metric_baseline * (1 + metric.metric_pct_change)

    # Validate baseline and target are present
    if metric.metric_target is None or metric.metric_baseline is None:
        return _power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.NO_BASELINE,
            (
                "Could not calculate metric baseline with given specification. "
                "Provide a metric baseline or adjust filters."
            ),
        )

    # Calculate effect size
    if metric.metric_type == MetricType.NUMERIC:
        # Validate stddev for NUMERIC metrics
        if metric.metric_stddev is None or metric.metric_stddev <= 0:
            return _power_analysis_error(
                metric,
                MetricPowerAnalysisMessageType.ZERO_STDDEV,
                (
                    "There is no variation in the metric with the given filters. Standard deviation must be "
                    "positive to do a sample size calculation."
                ),
            )
        effect_size = (metric.metric_target - metric.metric_baseline) / metric.metric_stddev
    elif metric.metric_type == MetricType.BINARY:
        effect_size = sms.proportion_effectsize(metric.metric_baseline, metric.metric_target)
    else:
        raise ValueError("metric_type must be NUMERIC or BINARY.")

    if effect_size == 0.0:
        return _power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.ZERO_EFFECT_SIZE,
            "Cannot detect an effect-size of 0. Try changing your effect-size.",
        )

    arm_ratio, control_prob = _calculate_arm_ratio_and_control_prob_from_weights(arm_weights, n_arms)

    # solve_power returns the required sample size for the control group
    power_analysis = sms.TTestIndPower()
    control_n = np.ceil(
        power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=arm_ratio,
        )
    )
    target_n = int(np.ceil(control_n / control_prob))

    # Prep the response object
    analysis = MetricPowerAnalysis(metric_spec=metric)
    analysis.target_n = int(target_n)
    analysis.sufficient_n = bool(target_n <= metric.available_n)

    # Construct potential components of the MetricPowerAnalysisMessage
    has_nulls = metric.available_nonnull_n is not None and metric.available_nonnull_n != metric.available_n
    values_map: dict[str, float | int] = {
        "available_n": metric.available_n,
        "target_n": analysis.target_n,
        "available_nonnull_n": metric.available_nonnull_n or 0,
    }

    msg_base_stats = (
        "There are {available_n} units available to run your experiment and a "
        "minimum of {target_n} units are needed to meet your experimental design specs."  # noqa: RUF027
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
        msg_type = MetricPowerAnalysisMessageType.SUFFICIENT
        msg_body = "There are enough units available."
    else:
        msg_type = MetricPowerAnalysisMessageType.INSUFFICIENT
        # Calculate the Minimum Detectable Effect that meets the power spec with the available subjects.
        target_possible, pct_change_possible = calculate_mde_with_desired_n(
            desired_n=metric.available_n,
            metric=metric,
            n_arms=n_arms,
            alpha=alpha,
            power=power,
            arm_weights=arm_weights,
        )

        analysis.target_possible = target_possible
        analysis.pct_change_possible = pct_change_possible

        values_map["additional_n_needed"] = target_n - metric.available_n
        values_map["metric_baseline"] = round(metric.metric_baseline, 4)
        values_map["target_possible"] = round(target_possible, 4)
        values_map["metric_target"] = round(metric.metric_target, 4)
        msg_body = (
            "There are not enough units available. "
            "You need {additional_n_needed} more units to meet your experimental design "
            "specifications. In order to meet your specification with the available "
            "{available_n} units and a metric baseline value of {metric_baseline}, your metric "
            "target value needs to be {target_possible} or further from the baseline. Your "  # noqa: RUF027
            "current desired target is {metric_target}."
        )

    # Construct our response from the parts above
    source_msg = " ".join([msg_base_stats, msg_body, msg_null_warning])
    analysis.msg = MetricPowerAnalysisMessage(
        type=msg_type,
        msg=source_msg.format_map(values_map),
        source_msg=source_msg,
        values=values_map,
    )
    return analysis


def _analyze_power_mde_mode(
    metric: DesignSpecMetric,
    n_arms: int,
    desired_n: int,
    power: float,
    alpha: float,
    arm_weights: list[float] | None,
) -> MetricPowerAnalysis:
    """
    Calculate MDE given desired sample size.

    This function validates its own inputs and returns appropriate errors.
    """

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

    # Use the helper function to calculate MDE
    target_possible, pct_change_possible = calculate_mde_with_desired_n(
        desired_n=desired_n,
        metric=metric,
        n_arms=n_arms,
        alpha=alpha,
        power=power,
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
        metric: DesignSpecMetric containing metric details
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

    # Dispatch to appropriate helper function based on mode
    if desired_n is None:
        # Sample size mode: calculate required sample size for target effect
        return _analyze_power_sample_size_mode(metric, n_arms, power, alpha, arm_weights)
    # MDE mode: calculate minimum detectable effect for desired sample size
    return _analyze_power_mde_mode(metric, n_arms, desired_n, power, alpha, arm_weights)


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
