"""
Power analysis for individually-randomized designs.
"""

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
) -> tuple[float, float]:
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
        Minimum Detectable Effect (MDE) as a tuple of absolute value and % from baseline
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
            # NOTE: typically the target proportion is > baseline, so h will be negative. But when
            # solving for an effect size given n, it will always be positive, meaning the target
            # will be *smaller* than baseline.
            p1 = metric.metric_baseline
            arcsin_p2 = (2 * np.arcsin(np.sqrt(p1)) - min_effect_size) / 2.0
            target_possible = np.sin(arcsin_p2) ** 2
        case _:
            raise ValueError(f"metric_type must be one of {list(MetricType)}.")

    pct_change_possible = target_possible / metric.metric_baseline - 1.0
    return target_possible, pct_change_possible


def _one_time_insufficient_msg_body(
    analysis: MetricPowerAnalysis,
    values_map: dict[str, float | int],
    *,
    metric: DesignSpecMetric,
    null_n: int,
    n_arms: int,
    arm_weights: list[float] | None,
    alpha: float,
    power: float,
) -> str:
    """Builds the insufficient message for one-time mode, where only the null rows will be assigned.

    Computes the MDE for that null pool, filling in analysis.target_possible / pct_change_possible and the
    message placeholders in values_map. Returns the message body (a format string over values_map).
    """
    assert analysis.target_n is not None
    assert metric.metric_baseline is not None
    assert metric.metric_target is not None

    values_map["additional_n_needed"] = analysis.target_n - null_n
    values_map["metric_baseline"] = round(metric.metric_baseline, 4)
    values_map["metric_target"] = round(metric.metric_target, 4)
    msg_body = (
        "There are not enough units without a value for this metric. "
        "You need {additional_n_needed} more to meet your specified metric target of {metric_target}. "
    )

    # MDE for the pool that will actually be assigned: the null rows only.
    try:
        target_possible, pct_change_possible = solve_for_mde_individual_impl(
            metric=metric,
            desired_n=null_n,
            n_arms=n_arms,
            arm_weights=arm_weights,
            alpha=alpha,
            power=power,
        )
    except ValueError, ZeroDivisionError:
        # Too few null rows for the solver (e.g. under 2 per arm); report without an MDE.
        return msg_body + "There are too few units to estimate a detectable effect. Adjust your filters."

    analysis.target_possible = target_possible
    analysis.pct_change_possible = pct_change_possible
    values_map["target_possible"] = round(target_possible, 4)
    return msg_body + (
        "Alternatively, with the {available_null_n} units without a value "
        "and a metric baseline of {metric_baseline}, your metric target should be "
        "{target_possible} or further from the baseline. "  # noqa: RUF027
    )


def solve_for_sample_size_individual(
    metric: DesignSpecMetric,
    *,
    n_arms: int,
    arm_weights: list[float] | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
    is_primary: bool = True,
) -> MetricPowerAnalysis:
    """
    Calculate required sample size for individual randomization.

    is_primary: whether this is the design's primary metric. One-time mode is only allowed there, so only the
    primary metric is marked one-time eligible or has one-time mode suggested in its message.
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
        # Only show error when ALL values are null (100%)
        error_msg = (
            "Cannot estimate the minimum sample size. All participants have null for this metric. "
            "Adjust your filters to target units with non-null values."
        )
        return power_analysis_error(
            metric,
            MetricPowerAnalysisMessageType.INSUFFICIENT,
            error_msg,
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

    # Check for nulls only if nonnull_n is provided
    has_nulls = metric.available_nonnull_n is not None and metric.available_nonnull_n != metric.available_n
    # Only meaningful when the column actually has nulls (flags are reset otherwise).
    one_time_mode = metric.use_one_time_metric and has_nulls
    # The baseline and stddev always come from the non-null rows, but the pool that will actually be
    # assigned differs: one-time mode assigns only the null rows; otherwise we compare against the
    # non-null rows (only units with data count toward power).
    null_n = metric.available_n - effective_n

    # Prep the response object
    analysis = MetricPowerAnalysis(metric_spec=metric)
    analysis.target_n = int(target_n)
    analysis.sufficient_n = bool(target_n <= (null_n if one_time_mode else effective_n))

    # Construct potential components of the MetricPowerAnalysisMessage
    values_map: dict[str, float | int] = {
        "available_n": metric.available_n,
        "target_n": analysis.target_n,
        "available_nonnull_n": effective_n,
        **({"available_null_n": null_n} if one_time_mode else {}),
    }

    # The primary metric is one-time eligible when the power calc succeeds and its column has nulls. With no nulls
    # left, one-time mode would match nobody, so turn it off.
    analysis.metric_spec = metric.model_copy(
        update={"is_one_time_eligible": is_primary}
        if has_nulls
        else {"is_one_time_eligible": False, "use_one_time_metric": False}
    )

    msg_base_stats = (
        (
            "One-time mode: {available_null_n} of {available_n} units have no value for this metric yet and can be "
            "assigned. You need at least {target_n} units to satisfy your design specs."  # noqa: RUF027
        )
        if one_time_mode
        else "There are {available_n} units available. You need at least {target_n} units to satisfy your design specs."
    )

    msg_null_warning = (
        (
            "NOTE: Only {available_nonnull_n} of {available_n} units have non-null values. "
            "Consider assigning only participants with nulls to the experiment, or adding a filter to "
            "explicitly exclude participants with a null value. Otherwise we will sample from all matching "
            "participants, including those with a null value."
        )
        # Skip the suggestion when the user already chose one-time mode.
        if has_nulls and not one_time_mode and is_primary
        else (
            # Secondary metrics can't use one-time mode, so only suggest the filter.
            "NOTE: Only {available_nonnull_n} of {available_n} units have non-null values. "
            "Consider adding a filter to explicitly exclude participants with a null value. Otherwise we will "
            "sample from all matching participants, including those with a null value."
        )
        if has_nulls and not is_primary
        else ""
    )

    if analysis.sufficient_n:
        msg_type = MetricPowerAnalysisMessageType.SUFFICIENT
        msg_body = "There are enough units available."
    elif one_time_mode:
        msg_type = MetricPowerAnalysisMessageType.INSUFFICIENT
        msg_body = _one_time_insufficient_msg_body(
            analysis,
            values_map,
            metric=metric,
            null_n=null_n,
            n_arms=n_arms,
            arm_weights=arm_weights,
            alpha=alpha,
            power=power,
        )
    else:
        msg_type = MetricPowerAnalysisMessageType.INSUFFICIENT
        # Calculate the Minimum Detectable Effect that meets the power spec with the available subjects.
        target_possible, pct_change_possible = solve_for_mde_individual_impl(
            metric=metric,
            desired_n=effective_n,
            n_arms=n_arms,
            arm_weights=arm_weights,
            alpha=alpha,
            power=power,
        )

        analysis.target_possible = target_possible
        analysis.pct_change_possible = pct_change_possible

        values_map["additional_n_needed"] = target_n - effective_n
        values_map["metric_baseline"] = round(metric.metric_baseline, 4)
        values_map["target_possible"] = round(target_possible, 4)
        values_map["metric_target"] = round(metric.metric_target, 4)
        msg_body = (
            "There are not enough non-null valued units available. "
            "You need {additional_n_needed} more units to meet your specified "
            "metric target of {metric_target}. "
            "Alternatively, with the available {available_nonnull_n} non-null units "
            "and a metric baseline of {metric_baseline}, your metric target should be "
            "{target_possible} or further from the baseline. "  # noqa: RUF027
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

    # One-time mode assigns only the null rows, so a desired size beyond them can't be reached. This includes a
    # column with no nulls at all (null_n == 0): creation would match nobody, so don't report an MDE for it.
    if metric.use_one_time_metric and metric.available_n is not None and metric.available_nonnull_n is not None:
        null_n = metric.available_n - metric.available_nonnull_n
        if desired_n > null_n:
            return power_analysis_error(
                metric,
                MetricPowerAnalysisMessageType.INSUFFICIENT,
                (
                    f"One-time mode: only {null_n} units have no value for this metric yet, which is fewer "
                    f"than the desired sample size of {desired_n}. Lower the desired sample size or adjust "
                    "your filters."
                ),
            )

    target_possible, pct_change_possible = solve_for_mde_individual_impl(
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
