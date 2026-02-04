"""
Power analysis for cluster-randomized designs.

Handles experiments where randomization occurs at the cluster level
(schools, hospitals, clinics) rather than individual level.
"""

import math


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
