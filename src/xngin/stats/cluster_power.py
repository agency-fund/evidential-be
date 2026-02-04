"""
Power analysis for cluster-randomized designs.

Handles experiments where randomization occurs at the cluster level
(schools, hospitals, clinics) rather than individual level.
"""

from xngin.apiserver.routers.common_api_types import DesignSpecMetric
from xngin.stats.power import calculate_mde_with_desired_n


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


def calculate_effective_sample_size(
    total_n: int,
    deff: float,
) -> int:
    """
    Calculate effective sample size accounting for clustering.

    When participants are grouped in clusters, the effective sample size
    is reduced due to intracluster correlation. This function converts
    the actual sample size to its statistical information equivalent.

    Formula: n_effective = n_total / DEFF

    Args:
        total_n: Total number of participants across all clusters
        deff: Design effect from calculate_design_effect()

    Returns:
        Effective sample size (statistical information equivalent)

    Examples:
        >>> # 720 parti        >>> # 720 parti        >>> # 720 parti        >>> # 720 psize(total_n=720, deff=5.35)
        134
        >>> # This means 720 clustered observations have the same
        >>> # statistical power as 134 independent observations
    """
    return int(total_n / deff)


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
    Calculate minimum detectable effect (MDE) for cluster-randomized design.

    Given available sample size in a cluster design, calculates the smallest
    effect that can be reliably detected. This is the cluster-randomized
    equivalent of calculate_mde_with_desired_n().

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

    Examples:
        >>> # 20 clusters * 30 students = 600 participants
        >>> metric = DesignSpecMetric(
        ...     field_name="reading_score",
        ...     metric_type=MetricType.NUMERIC,
        ...     metric_baseline=100,
        ...     metric_stddev=20
        ... )
        >>> target, pct = calculate_mde_cluster(
        ...     available_n=600,
        ...     metric=metric,
        ...     n_arms=2,
        ...     icc=0.15,
        ...     avg_cluster_size=30
        ... )
        >>> target
        108.5
        >>> pct
        0.085
    """
    # Calculate design effect
    deff = calculate_design_effect(icc, avg_cluster_size)

    # Calculate effective sample size accounting for clustering
    effective_n = calculate_effective_sample_size(available_n, deff)

    return calculate_mde_with_desired_n(
        desired_n=effective_n,
        metric=metric,
        n_arms=n_arms,
        alpha=alpha,
        power=power,
        arm_weights=arm_weights,
    )
