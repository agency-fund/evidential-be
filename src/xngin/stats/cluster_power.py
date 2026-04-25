"""
Power analysis for cluster-randomized designs.
"""

import math

from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetric,
    MetricPowerAnalysis,
    MetricPowerAnalysisMessage,
)
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
        if arm_weights is None:
            arm_probs = [1.0 / n_arms] * n_arms
        else:
            total_weight = sum(arm_weights)
            arm_probs = [w / total_weight for w in arm_weights]

        deff = calculate_design_effect(metric.icc, metric.avg_cluster_size, metric.cv)

        clusters_per_arm_list = []
        n_per_arm_list = []

        for prob in arm_probs:
            n_individual_this_arm = individual_analysis.target_n * prob

            clusters_this_arm = calculate_num_clusters_needed(
                n_individual=n_individual_this_arm,
                avg_cluster_size=metric.avg_cluster_size,
                deff=deff,
            )

            n_actual_this_arm = math.ceil(clusters_this_arm * metric.avg_cluster_size)

            clusters_per_arm_list.append(clusters_this_arm)
            n_per_arm_list.append(n_actual_this_arm)

        clusters_total = sum(clusters_per_arm_list)
        new_target_n = sum(n_per_arm_list)
        effective_n = int(new_target_n / deff)

        final_msg = individual_analysis.msg
        if metric.cv > 1.0 and final_msg:
            warning = (
                f"\n\nWarning: High cluster size variation (CV={metric.cv:.2f}).  "
                f"Number of clusters estimates are approximate."
            )

            final_msg = MetricPowerAnalysisMessage(
                type=final_msg.type,
                msg=final_msg.msg + warning,
                source_msg=final_msg.source_msg,
                values=final_msg.values,
            )

        cluster_analysis = MetricPowerAnalysis(
            metric_spec=metric,
            target_n=new_target_n,
            sufficient_n=individual_analysis.sufficient_n,
            target_possible=individual_analysis.target_possible,
            pct_change_possible=individual_analysis.pct_change_possible,
            msg=final_msg,
            num_clusters_total=clusters_total,
            clusters_per_arm=clusters_per_arm_list,
            n_per_arm=n_per_arm_list,
            design_effect=deff,
            effective_sample_size=effective_n,
        )

    return cluster_analysis
