"""
Power analysis for cluster-randomized designs.
Calculate minimum detectable effect (MDE) for cluster-randomized design.
"""

from xngin.apiserver.routers.common_api_types import DesignSpecMetric
from xngin.stats.power import calculate_mde_with_desired_n


def calculate_design_effect(icc: float, avg_cluster_size: float) -> float:
    """
    Calculate design effect DEFF (>= 1)
    Args:
        icc: Intracluster correlation coefficient, range [0, 1]
        avg_cluster_size: Average number of individuals per cluster
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
    Args:
        total_n: Total number of participants across all clusters
        deff: Design effect from calculate_design_effect()
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
