"""
Power analysis for individually-randomized designs.
"""

import dataclasses
import math

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


def requested_direction_is_down(metric: DesignSpecMetric) -> bool:
    """Whether the metric's requested change is a decrease from the baseline.

    Defaults to the improvement (positive) direction when the spec carries no target.
    """
    if metric.metric_target is not None and metric.metric_baseline is not None:
        return metric.metric_target < metric.metric_baseline
    if metric.metric_pct_change is not None:
        return metric.metric_pct_change < 0
    return False


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class MdeIndividualResult:
    """Both bounds of the two-sided minimum detectable effect around the baseline.

    target_possible / pct_change_possible are the bound in the improvement (positive)
    direction; the _lower fields are the detectable change below the baseline. NUMERIC
    bounds are symmetric around the baseline. BINARY bounds are symmetric in Cohen's h
    space, but the conversion back to probability space is not, so their magnitudes differ.
    """

    target_possible: float
    pct_change_possible: float
    target_possible_lower: float
    pct_change_possible_lower: float


def power_analysis_error(
    metric: DesignSpecMetric, msg_type: MetricPowerAnalysisMessageType, msg_body: str
) -> MetricPowerAnalysis:
    return MetricPowerAnalysis(
        metric_spec=metric,
        msg=MetricPowerAnalysisMessage(type=msg_type, msg=msg_body, source_msg=msg_body, values=None),
    )


def solve_for_mde_individual_impl(
    metric: DesignSpecMetric,
    *,
    desired_n: int,
    n_arms: int,
    arm_weights: list[float] | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> MdeIndividualResult:
    """
    Calculate MDE for individual randomization.

    Args:
        desired_n: Total sample size across all arms to be used in the calculation
        metric: DesignSpecMetric containing metric descriptive stats
        n_arms: Number of treatment arms
        alpha: Significance level
        power: Desired statistical power
        arm_weights: Optional list of weights (summing to 100) for unbalanced arms
    Returns:
        MdeIndividualResult with both bounds of the Minimum Detectable Effect (MDE)
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
            target_possible = metric.metric_baseline + needed_delta
            target_possible_lower = metric.metric_baseline - needed_delta
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

            # Convert Cohen's h back to proportions:
            # h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1)), where p1 is baseline and p2 is target.
            # solve_power returns |h|, and the test is two-sided, so a change of h in either direction
            # is detectable. Clamp to the arcsine domain [0, pi/2]: past it, not even a proportion of
            # 1.0 (or 0.0) is a large enough change in that direction, so report the boundary.
            baseline_arcsine = 2 * np.arcsin(np.sqrt(metric.metric_baseline))
            arcsin_up = np.clip((baseline_arcsine + min_effect_size) / 2.0, 0.0, np.pi / 2)
            arcsin_down = np.clip((baseline_arcsine - min_effect_size) / 2.0, 0.0, np.pi / 2)
            target_possible = float(np.sin(arcsin_up) ** 2)
            target_possible_lower = float(np.sin(arcsin_down) ** 2)
        case _:
            raise ValueError(f"metric_type must be one of {list(MetricType)}.")

    return MdeIndividualResult(
        target_possible=target_possible,
        pct_change_possible=target_possible / metric.metric_baseline - 1.0,
        target_possible_lower=target_possible_lower,
        pct_change_possible_lower=target_possible_lower / metric.metric_baseline - 1.0,
    )


def solve_for_sample_size_individual(
    metric: DesignSpecMetric,
    *,
    n_arms: int,
    arm_weights: list[float] | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> MetricPowerAnalysis:
    """
    Calculate required sample size for individual randomization.
    """

    # Validate metric type
    if metric.metric_type is None:
        raise ValueError("Unknown metric_type.")

    # Validate available_n is present
    if metric.available_n is None or metric.available_n <= 0:
        return power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.NO_AVAILABLE_N,
            "You have no available units to run your experiment. Adjust your filters to target more units.",
        )

    # Use nonnull count for power calculations (only users with data count toward power)
    effective_n = metric.available_nonnull_n if metric.available_nonnull_n is not None else metric.available_n

    # Check for zero effective_n (no non-null values)
    if effective_n <= 0:
        return power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.INSUFFICIENT,
            (
                "You have no units with non-null values for this metric. "
                "Adjust your filters to target units with non-null values."
            ),
        )

    # Calculate target from pct_change if needed
    if metric.metric_target is None and metric.metric_baseline is not None and metric.metric_pct_change is not None:
        metric.metric_target = metric.metric_baseline * (1 + metric.metric_pct_change)

    # Validate baseline and target are present
    if metric.metric_target is None or metric.metric_baseline is None:
        return power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.NO_BASELINE,
            (
                "Could not calculate metric baseline with given specification. "
                "Provide a metric baseline or adjust filters."
            ),
        )

    # Derive the user's desired effect size from the metric inputs
    if metric.metric_type == MetricType.NUMERIC:
        # Validate stddev for NUMERIC metrics
        if metric.metric_stddev is None or metric.metric_stddev <= 0:
            return power_analysis_error(
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

    if math.isclose(effect_size, 0.0):
        return power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.ZERO_EFFECT_SIZE,
            "Cannot detect an effect-size of 0. Try changing your effect-size.",
        )

    arm_ratio, control_prob = _calculate_arm_ratio_and_control_prob_from_weights(arm_weights, n_arms)

    # Finally, calculate the minimum sample size for the desired effect size.
    # solve_power returns the required sample size for the control, from which we derive the total n needed.
    # Cohen's h is defined for the two-sample proportions z-test, so BINARY metrics use NormalIndPower
    # (consistent with the MDE calculation); NUMERIC metrics keep the t-test.
    power_analysis = sms.NormalIndPower() if metric.metric_type == MetricType.BINARY else sms.TTestIndPower()
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
    # Use nonnull count for power check (only users with data count toward power)
    analysis.sufficient_n = bool(target_n <= effective_n)

    # Construct potential components of the MetricPowerAnalysisMessage
    values_map: dict[str, float | int] = {
        "available_n": metric.available_n,
        "target_n": analysis.target_n,
        "available_nonnull_n": effective_n,
    }

    # Check for nulls only if nonnull_n is provided
    has_nulls = metric.available_nonnull_n is not None and metric.available_nonnull_n != metric.available_n

    msg_base_stats = (
        "There are {available_n} units available. You need at least {target_n} units to satisfy your design specs."
    )
    msg_null_warning = (
        (
            "WARNING: Of the available units, {available_nonnull_n} have a non-null value. "
            "These calculations only used units with a real value present, but random assignment "
            "samples from *all* units that meet your filters, including those missing a value. If "
            "you do not want that, add a filter on this metric to exclude nulls."
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
        mde_result = solve_for_mde_individual_impl(
            metric=metric,
            desired_n=effective_n,
            n_arms=n_arms,
            arm_weights=arm_weights,
            alpha=alpha,
            power=power,
        )

        analysis.target_possible = mde_result.target_possible
        analysis.pct_change_possible = mde_result.pct_change_possible
        analysis.target_possible_lower = mde_result.target_possible_lower
        analysis.pct_change_possible_lower = mde_result.pct_change_possible_lower

        # The message reports the bound in the direction of the user's requested change;
        # the analysis fields keep their fixed upper/lower semantics.
        reported_target_possible = (
            mde_result.target_possible_lower if requested_direction_is_down(metric) else mde_result.target_possible
        )

        values_map["additional_n_needed"] = target_n - effective_n
        values_map["metric_baseline"] = round(metric.metric_baseline, 4)
        values_map["target_possible"] = round(reported_target_possible, 4)
        values_map["metric_target"] = round(metric.metric_target, 4)
        msg_body = (
            "There are not enough non-null valued units available. "
            "You need {additional_n_needed} more units to meet your specified "
            "metric target of {metric_target}. "
            "Alternatively, with the available {available_nonnull_n} non-null units "
            "and a metric baseline of {metric_baseline}, your metric target should be "
            "{target_possible} or further from the baseline. "
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


def solve_for_mde_individual(
    metric: DesignSpecMetric,
    n_arms: int,
    desired_n: int,
    power: float,
    alpha: float,
    arm_weights: list[float] | None,
) -> MetricPowerAnalysis:
    """
    Calculate MDE given desired sample size for individual randomization.
    """
    assert metric.metric_baseline is not None

    mde_result = solve_for_mde_individual_impl(
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
    analysis.target_possible = mde_result.target_possible
    analysis.pct_change_possible = mde_result.pct_change_possible
    analysis.target_possible_lower = mde_result.target_possible_lower
    analysis.pct_change_possible_lower = mde_result.pct_change_possible_lower
    analysis.sufficient_n = None  # Not applicable in MDE mode

    # Create message. It reports the bound in the direction of the user's requested change;
    # the analysis fields keep their fixed upper/lower semantics.
    reported_target_possible = (
        mde_result.target_possible_lower if requested_direction_is_down(metric) else mde_result.target_possible
    )
    values_map: dict[str, float | int] = {
        "desired_n": desired_n,
        "metric_baseline": round(metric.metric_baseline, 4),
        "target_possible": round(reported_target_possible, 4),
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
