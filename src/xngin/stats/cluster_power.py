"""
Power analysis for cluster-randomized designs.

Handles experiments where randomization occurs at the cluster level
(schools, hospitals, clinics) rather than individual level.
"""

import math

from xngin.apiserver.routers.common_api_types import (
    ClusterMetricPowerAnalysis,
    DesignSpecMetric,
)
from xngin.stats.power import _analyze_power_sample_size_mode  # noqa: PLC2701


def calculate_design_effect(icc: float, avg_cluster_size: float) -> float:
    """
    Calculate design effect (DEFF) for cluster-randomized designs.

    The design effect quantifies loss of statistical power due to clustering.
    When participants are grouped in clusters (schools, hospitals, etc.),
    observations within clusters are more similar than between clusters.

    Formula: DEFF = 1 + (m - 1) * icc

    Args:
        icc: Intracluster correlation coefficient, range [0, 1]
             Example: 0.15 means 15% of variance is between clusters
        avg_cluster_size: Average number of individuals per cluster (m)

    Returns:
        Design effect (DEFF). Always >= 1.

    Raises:
        ValueError: If icc not in [0, 1] or avg_cluster_size < 1

    Examples:
        >>> calculate_design_effect(icc=0.15, avg_cluster_size=30)
        5.35
    """
    if not 0 <= icc <= 1:
        raise ValueError(f"ICC must be between 0 and 1, got {icc}")

    if avg_cluster_size < 1:
        raise ValueError(f"Cluster size must be >= 1, got {avg_cluster_size}")

    return 1 + (avg_cluster_size - 1) * icc


def calculate_num_clusters_needed(
    n_individual: float,
    avg_cluster_size: float,
    deff: float,
) -> int:
    """
    Calculate number of clusters needed per arm for cluster-randomized design.

    When randomizing at the cluster level (e.g., schools, hospitals), you need
    to determine how many clusters are required to achieve the same power as
    individual randomization.

    Formula: J = ceil((n_individual / cluster_size) * DEFF)

    Args:
        n_individual: Sample size per arm from individual randomization power analysis.
                     This is what you'd need if randomizing individuals instead of clusters.
        avg_cluster_size: Average number of individuals per cluster (m)
        deff: Design effect from calculate_design_effect()

    Returns:
        Number of clusters needed per arm (always rounds up to ensure power)

    Examples:
        >>> calculate_num_clusters_needed(
        ...     n_individual=63,
        ...     avg_cluster_size=30,
        ...     deff=5.35
        ... )
        12
    """
    clusters_needed = (n_individual / avg_cluster_size) * deff
    return math.ceil(clusters_needed)


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
    Analyze statistical power for cluster-randomized design.

    Calculates the number of clusters needed per arm to detect a target effect
    when randomizing at the cluster level (e.g., schools, hospitals, clinics).
    Handles both balanced and unbalanced arm allocation.

    Args:
        metric: DesignSpecMetric containing target effect specification.
                Must have metric_baseline, metric_target (or metric_pct_change),
                and metric_stddev for numeric metrics.
        n_arms: Number of treatment arms (e.g., 2 for treatment vs control)
        icc: Intracluster correlation coefficient, range [0, 1].
             Typical values by domain:
             - Schools (students): 0.10 - 0.20
             - Hospitals (patients): 0.02 - 0.10
             - Households (individuals): 0.20 - 0.30
             - Clinics (patients): 0.05 - 0.15
        avg_cluster_size: Average number of individuals per cluster
        power: Desired statistical power (default 0.8)
        alpha: Significance level (default 0.05)
        arm_weights: Optional allocation weights for unbalanced designs.
                    Example: [20, 80] means 20% control, 80% treatment.
                    If None, uses balanced allocation.

    Returns:
        ClusterMetricPowerAnalysis with:
        - target_n: Total participants needed across all arms
        - num_clusters_total: Total clusters needed
        - clusters_per_arm: List of clusters per arm
        - n_per_arm: List of participants per arm
        - design_effect: DEFF value
        - icc: ICC value used
        - avg_cluster_size: Cluster size used
        - effective_sample_size: Statistical information equivalent

    Examples:
        >>> # Balanced 2-arm design
        >>> metric = DesignSpecMetric(
        ...     field_name="reading_score",
        ...     metric_type=MetricType.NUMERIC,
        ...     metric_baseline=100,
        ...     metric_target=110,
        ...     metric_stddev=20
        ... )
        >>> result = analyze_metric_power_cluster(
        ...     metric=metric,
        ...     n_arms=2,
        ...     icc=0.15,
        ...     avg_cluster_size=30
        ... )
        >>> result.num_clusters_total
        24
        >>> result.clusters_per_arm
        [12, 12]
    """
    # Step 1: Get individual randomization requirements using existing code
    individual_analysis = _analyze_power_sample_size_mode(
        metric=metric,
        n_arms=n_arms,
        power=power,
        alpha=alpha,
        arm_weights=arm_weights,
    )

    # Step 2: Check if individual analysis succeeded
    if individual_analysis.target_n is None:
        # Error case - cannot calculate cluster requirements
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
        # Success case - calculate cluster requirements

        # Calculate arm allocation probabilities
        if arm_weights is None:
            arm_probs = [1.0 / n_arms] * n_arms
        else:
            total_weight = sum(arm_weights)
            arm_probs = [w / total_weight for w in arm_weights]

        # Calculate design effect (same for all arms)
        deff = calculate_design_effect(icc, avg_cluster_size)

        # Calculate clusters needed for each arm
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

        # Calculate totals
        clusters_total = sum(clusters_per_arm_list)
        new_target_n = sum(n_per_arm_list)
        effective_n = int(new_target_n / deff)

        # Build cluster analysis result
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
