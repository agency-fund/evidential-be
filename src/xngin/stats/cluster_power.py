"""
Power analysis for cluster-randomized designs.
"""

import math

from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    MetricPowerAnalysis,
    MetricPowerAnalysisMessage,
)
from xngin.apiserver.routers.common_enums import MetricPowerAnalysisMessageType
from xngin.stats.individual_power import (
    solve_for_mde_individual_impl,
    solve_for_sample_size_individual,
)


def calculate_design_effect(
    icc: float,
    avg_cluster_size: float,
    cv: float = 0.0,
) -> float:
    """
    Calculate design effect (DEFF >= 1) for cluster-randomized designs.

    The design effect is the ratio of the cluster-randomized design variance to the variance of an
    individually randomized design w/ the same total sample size and no within-cluster correlation.

    Formula: DEFF = 1 + (m - 1) * icc          (when cv=0)
    Formula from Eldridge et al. (2006):
    Formula: DEFF = 1 + icc * [(m-1) + m*CV^2] (with CV^2 adjustment)

    Args:
        icc: Intracluster correlation coefficient (0 to 1)
        avg_cluster_size: Average individuals per cluster (m in the formulas)
        cv: Coefficient of variation in cluster sizes (default 0.0)

    """
    if not 0 <= icc <= 1:
        raise ValueError(f"ICC must be between 0 and 1, got {icc}")
    if avg_cluster_size < 1:
        raise ValueError(f"Cluster size must be >= 1, got {avg_cluster_size}")
    if cv < 0:
        raise ValueError(f"CV must be >= 0, got {cv}")

    return 1 + icc * ((avg_cluster_size - 1) + avg_cluster_size * (cv**2))


def calculate_effective_sample_size(
    total_n: int,
    deff: float,
) -> int:
    """
    Calculate effective sample size for cluster-randomized design.

    Args:
        total_n: Total number of participants across all clusters
        deff: Design effect from calculate_design_effect()

    """
    return int(total_n / deff)


def calculate_num_clusters_needed(
    n_individual: float,
    avg_cluster_size: float,
    deff: float,
) -> int:
    """
    Calculate clusters needed per arm for cluster-randomized design.

    Formula: J = ceil((n_individual / cluster_size) * DEFF)

    Args:
        n_individual: Required sample size for individual randomization
        avg_cluster_size: Average individuals per cluster
        deff: Design effect from calculate_design_effect()

    """
    clusters_needed = (n_individual / avg_cluster_size) * deff
    return math.ceil(clusters_needed)


def _build_cluster_sample_size_message(
    *,
    metric: DesignSpecMetric,
    target_n: int,
    sufficient_n: bool,
    available_nonnull_n: int,
    num_clusters_total: int,
    target_possible: float | None,
) -> MetricPowerAnalysisMessage:
    assert metric.available_n is not None

    values_map: dict[str, float | int] = {
        "available_n": metric.available_n,
        "available_nonnull_n": available_nonnull_n,
        "target_n": target_n,
        "num_clusters_total": num_clusters_total,
    }

    has_nulls = metric.available_nonnull_n is not None and metric.available_nonnull_n != metric.available_n

    msg_base_stats = (
        "There are {available_n} units available. You need at least {target_n} units across "
        "{num_clusters_total} clusters to satisfy your design specs."  # noqa: RUF027
    )
    msg_null_warning = (
        (
            "WARNING: Of the available units, {available_nonnull_n} have a non-null value. "  # noqa: RUF027
            "These calculations only used units with a real value present, but random assignment "
            "samples from *all* units that meet your filters, including those missing a value. If "
            "you do not want that, add a filter on this metric to exclude nulls."
        )
        if has_nulls
        else ""
    )
    msg_cv_warning = (
        (
            "Warning: High cluster size variation (CV={cluster_size_cv:.2f}).  "
            "Number of clusters estimates are approximate."
        )
        if metric.cv is not None and metric.cv > 1.0
        else ""
    )
    if metric.cv is not None and metric.cv > 1.0:
        values_map["cluster_size_cv"] = metric.cv

    if sufficient_n:
        msg_type = MetricPowerAnalysisMessageType.SUFFICIENT
        msg_body = "There are enough non-null valued units available."
    else:
        assert metric.metric_baseline is not None
        assert metric.metric_target is not None
        assert target_possible is not None

        msg_type = MetricPowerAnalysisMessageType.INSUFFICIENT
        values_map["additional_n_needed"] = target_n - available_nonnull_n
        values_map["metric_baseline"] = round(metric.metric_baseline, 4)
        values_map["target_possible"] = round(target_possible, 4)
        values_map["metric_target"] = round(metric.metric_target, 4)
        msg_body = (
            "There are not enough non-null valued units available after accounting for clustering. "
            "You need {additional_n_needed} more units to meet your specified "
            "metric target of {metric_target}. "
            "Alternatively, with the available {available_nonnull_n} non-null units "  # noqa: RUF027
            "and a metric baseline of {metric_baseline}, your metric target should be "
            "{target_possible} or further from the baseline. "  # noqa: RUF027
        )

    source_msg = " ".join(part for part in [msg_base_stats, msg_body, msg_null_warning, msg_cv_warning] if part)
    return MetricPowerAnalysisMessage(
        type=msg_type,
        msg=source_msg.format_map(values_map),
        source_msg=source_msg,
        values=values_map,
    )


def solve_for_mde_cluster_impl(
    metric: DesignSpecMetric,
    *,
    desired_n: int,
    n_arms: int,
    arm_weights: list[float] | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> tuple[float, float]:
    """
    Given desired sample size, calculate minimum detectable effect (MDE) for a
    cluster-randomized design. Cluster parameters are read from the metric.

    Args:
        desired_n: Total sample size across all arms to be used in the calculation
        metric: DesignSpecMetric containing metric descriptive stats
                (icc, avg_cluster_size, cv must be set)
        n_arms: Number of treatment arms
        alpha: Significance level (default 0.05)
        power: Desired statistical power (default 0.8)
        arm_weights: Optional allocation weights for unbalanced designs

    Returns:
        Tuple of (target_value, pct_change):
        - target_value: The minimum detectable effect in absolute terms
        - pct_change: The minimum detectable effect as percent change from baseline

    """
    assert metric.icc is not None and metric.avg_cluster_size is not None
    deff = calculate_design_effect(metric.icc, metric.avg_cluster_size, metric.cv or 0.0)

    effective_n = calculate_effective_sample_size(desired_n, deff)

    return solve_for_mde_individual_impl(
        desired_n=effective_n,
        metric=metric,
        n_arms=n_arms,
        alpha=alpha,
        power=power,
        arm_weights=arm_weights,
    )


def solve_for_mde_cluster(
    metric: DesignSpecMetric,
    *,
    desired_n: int,
    n_arms: int,
    arm_weights: list[float] | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> MetricPowerAnalysis:
    """Calculate the minimum detectable effect (MDE) for a cluster-randomized design."""

    target_possible, pct_change_possible = solve_for_mde_cluster_impl(
        metric=metric,
        desired_n=desired_n,
        arm_weights=arm_weights,
        n_arms=n_arms,
        alpha=alpha,
        power=power,
    )

    # Build response object for MDE calculation
    analysis = MetricPowerAnalysis(metric_spec=metric)
    analysis.target_n = desired_n
    analysis.target_possible = target_possible
    analysis.pct_change_possible = pct_change_possible
    analysis.sufficient_n = None  # Not applicable in MDE mode

    # Create message
    assert metric.metric_baseline is not None
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


def solve_for_sample_size_cluster(
    metric: DesignSpecMetric,
    *,
    n_arms: int,
    arm_weights: list[float] | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
) -> MetricPowerAnalysis:
    """
    Calculate required sample size for cluster-randomized design.

    Args:
        metric: Metric specification with baseline, target, variance, and cluster design fields
        n_arms: Number of treatment arms
        power: Desired statistical power (default 0.8)
        alpha: Significance level (default 0.05)
        arm_weights: Allocation weights for unbalanced designs

    """
    assert metric.icc is not None and metric.avg_cluster_size is not None and metric.cv is not None
    # 1) Solve as if there were no clustering
    individual_analysis = solve_for_sample_size_individual(
        metric=metric,
        n_arms=n_arms,
        power=power,
        alpha=alpha,
        arm_weights=arm_weights,
    )

    if individual_analysis.target_n is None:
        cluster_analysis = MetricPowerAnalysis(
            metric_spec=metric,
            target_n=None,
            sufficient_n=individual_analysis.sufficient_n,
            target_possible=individual_analysis.target_possible,
            pct_change_possible=individual_analysis.pct_change_possible,
            msg=individual_analysis.msg,
        )
    else:
        # 2) Calculate the "design effect" due to clustering, which reduces the effective sample size.
        available_nonnull_n = (
            metric.available_nonnull_n if metric.available_nonnull_n is not None else metric.available_n
        )
        assert available_nonnull_n is not None

        if arm_weights is None:
            arm_probs = [1.0 / n_arms] * n_arms
        else:
            total_weight = sum(arm_weights)
            arm_probs = [w / total_weight for w in arm_weights]

        deff = calculate_design_effect(metric.icc, metric.avg_cluster_size, metric.cv)

        # 3) Compute the number of clusters we need for each arm accounting for the design effect
        # inflating the actual number of total units due to the correlation members of the cluster.
        clusters_per_arm_list = []
        n_per_arm_list = []
        for prob in arm_probs:
            # Get the number of units we need for this arm assuming no clustering.
            n_individual_this_arm = individual_analysis.target_n * prob
            clusters_this_arm = calculate_num_clusters_needed(
                n_individual=n_individual_this_arm,
                avg_cluster_size=metric.avg_cluster_size,
                deff=deff,
            )
            # This is the number we'd actually need given the clusters required.
            n_actual_this_arm = math.ceil(clusters_this_arm * metric.avg_cluster_size)

            clusters_per_arm_list.append(clusters_this_arm)
            n_per_arm_list.append(n_actual_this_arm)

        # 4) Assemble the final values we need for our analysis response.
        # Note that we're using the total inflated individual units needed given the clustering
        # present to determine whether or not we have enough units to sample from!
        clusters_total = sum(clusters_per_arm_list)
        cluster_adj_target_n = sum(n_per_arm_list)
        effective_n = int(cluster_adj_target_n / deff)
        sufficient_n = bool(cluster_adj_target_n <= available_nonnull_n)
        target_possible = None
        pct_change_possible = None
        if not sufficient_n:
            # Let the user know: Given the clustering, what could you detect with what's available?
            target_possible, pct_change_possible = solve_for_mde_cluster_impl(
                metric=metric,
                desired_n=available_nonnull_n,
                n_arms=n_arms,
                arm_weights=arm_weights,
                alpha=alpha,
                power=power,
            )

        final_msg = _build_cluster_sample_size_message(
            metric=metric,
            target_n=cluster_adj_target_n,
            sufficient_n=sufficient_n,
            available_nonnull_n=available_nonnull_n,
            num_clusters_total=clusters_total,
            target_possible=target_possible,
        )

        cluster_analysis = MetricPowerAnalysis(
            metric_spec=metric,
            target_n=cluster_adj_target_n,
            sufficient_n=sufficient_n,
            target_possible=target_possible,
            pct_change_possible=pct_change_possible,
            msg=final_msg,
            num_clusters_total=clusters_total,
            clusters_per_arm=clusters_per_arm_list,
            n_per_arm=n_per_arm_list,
            design_effect=deff,
            effective_sample_size=effective_n,
        )

    return cluster_analysis
