"""
Power analysis for cluster-randomized designs.
"""

import math

from xngin.apiserver.routers.common_api_types import (
    ClusterMetricPowerAnalysis,
    DesignSpecMetric,
)
from xngin.stats.power import (
    _analyze_power_sample_size_mode,  # noqa: PLC2701
    calculate_mde_with_desired_n,
)


def calculate_design_effect(
    icc: float,
    avg_cluster_size: float,
    cv: float = 0.0,
) -> float:
    """
    Calculate design effect (DEFF) for cluster randomization.

    Formula from Eldridge et al. (2006):
    DEFF = 1 + icc[(m - 1) + m*CV²]

    Where:
    - icc = ICC (intracluster correlation)
    - m = average cluster size
    - CV = coefficient of variation of cluster sizes

    When CV=0, reduces to: DEFF = 1 + icc(m - 1)

    """
    if not 0 <= icc <= 1:
        raise ValueError(f"ICC must be between 0 and 1, got {icc}")
    if avg_cluster_size < 1:
        raise ValueError(f"Cluster size must be >= 1, got {avg_cluster_size}")
    if cv < 0:
        raise ValueError(f"CV must be >= 0, got {cv}")

    m = avg_cluster_size

    # Correct formula from Eldridge et al. (2006)
    return 1 + icc * ((m - 1) + m * (cv**2))


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


def calculate_mde_cluster(
    available_n: int,
    metric: DesignSpecMetric,
    n_arms: int,
    icc: float,
    avg_cluster_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    arm_weights: list[float] | None = None,
) -> tuple[float, float]:
    """
    Given available sample size,
    calculate minimum detectable effect (MDE) for cluster-randomized design.

    Args:
        available_n: Total number of participants available across all arms
        metric: DesignSpecMetric with baseline and variance information
        n_arms: Number of treatment arms
        icc: Intracluster correlation coefficient, range [0, 1]
        avg_cluster_size: Average number of individuals per cluster
        power: Desired statistical power (default 0.8)
        alpha: Significance level (default 0.05)
        arm_weights: Optional allocation weights for unbalanced designs

    Returns:
        Tuple of (target_value, pct_change):
        - target_value: The minimum detectable effect in absolute terms
        - pct_change: The minimum detectable effect as percent change from baseline

    """
    deff = calculate_design_effect(icc, avg_cluster_size)

    effective_n = calculate_effective_sample_size(available_n, deff)

    return calculate_mde_with_desired_n(
        desired_n=effective_n,
        metric=metric,
        n_arms=n_arms,
        alpha=alpha,
        power=power,
        arm_weights=arm_weights,
    )


def analyze_metric_power_cluster(
    metric: DesignSpecMetric,
    n_arms: int,
    icc: float,
    avg_cluster_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    arm_weights: list[float] | None = None,
) -> ClusterMetricPowerAnalysis:
    """
    Calculate required sample size for cluster-randomized design.

    Args:
        metric: Metric specification with baseline, target, and variance
        n_arms: Number of treatment arms
        icc: Intracluster correlation coefficient (0 to 1)
        avg_cluster_size: Average individuals per cluster
        power: Desired statistical power (default 0.8)
        alpha: Significance level (default 0.05)
        arm_weights: Allocation weights for unbalanced designs

    """
    individual_analysis = _analyze_power_sample_size_mode(
        metric=metric,
        n_arms=n_arms,
        power=power,
        alpha=alpha,
        arm_weights=arm_weights,
    )

    if individual_analysis.target_n is None:
        cluster_analysis = ClusterMetricPowerAnalysis(
            metric_spec=individual_analysis.metric_spec,
            target_n=None,
            sufficient_n=individual_analysis.sufficient_n,
            target_possible=individual_analysis.target_possible,
            pct_change_possible=individual_analysis.pct_change_possible,
            msg=individual_analysis.msg,
            icc=icc,
            avg_cluster_size=avg_cluster_size,
        )
    else:
        if arm_weights is None:
            arm_probs = [1.0 / n_arms] * n_arms
        else:
            total_weight = sum(arm_weights)
            arm_probs = [w / total_weight for w in arm_weights]

        deff = calculate_design_effect(icc, avg_cluster_size)

        clusters_per_arm_list = []
        n_per_arm_list = []

        for prob in arm_probs:
            n_individual_this_arm = individual_analysis.target_n * prob

            clusters_this_arm = calculate_num_clusters_needed(
                n_individual=n_individual_this_arm,
                avg_cluster_size=avg_cluster_size,
                deff=deff,
            )

            n_actual_this_arm = int(clusters_this_arm * avg_cluster_size)

            clusters_per_arm_list.append(clusters_this_arm)
            n_per_arm_list.append(n_actual_this_arm)

        clusters_total = sum(clusters_per_arm_list)
        new_target_n = sum(n_per_arm_list)
        effective_n = int(new_target_n / deff)

        cluster_analysis = ClusterMetricPowerAnalysis(
            metric_spec=individual_analysis.metric_spec,
            target_n=new_target_n,
            sufficient_n=individual_analysis.sufficient_n,
            target_possible=individual_analysis.target_possible,
            pct_change_possible=individual_analysis.pct_change_possible,
            msg=individual_analysis.msg,
            num_clusters_total=clusters_total,
            clusters_per_arm=clusters_per_arm_list,
            n_per_arm=n_per_arm_list,
            design_effect=deff,
            icc=icc,
            avg_cluster_size=avg_cluster_size,
            effective_sample_size=effective_n,
        )

    return cluster_analysis
