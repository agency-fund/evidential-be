import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

from xngin.apiserver.routers.common_api_types import PriorTypes


def bandit_weights_to_beta_prior(expected_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bandit weights to Beta prior parameters (alpha, beta) for each arm.

    We simply rescale the alpha parameters based on the expected probabilities and a regularization term.

    Args:
        expected_probabilities (np.ndarray): Array of shape (n_arms,) containing the expected
            probabilities for each arm.

    Returns:
        alpha (np.ndarray): Array of shape (n_arms,) containing the alpha
            parameters for the Beta distribution.
        beta (np.ndarray): Array of shape (n_arms,) containing the beta
            parameters for the Beta distribution.
    """
    normalized_expected_probabilities = np.asarray(expected_probabilities, dtype=np.float64)

    if not math.isclose(expected_probabilities.sum(), 100.0, rel_tol=1e-9):
        raise ValueError("Expected probabilities must sum to 100.")

    normalized_expected_probabilities *= 0.01  # Normalize to sum to 1

    if (
        np.abs(
            (normalized_expected_probabilities.max() - normalized_expected_probabilities.min())
            / (normalized_expected_probabilities.min() + 1e-4)
        )
        < 1e-2
    ).all():
        return np.ones_like(normalized_expected_probabilities), np.ones_like(normalized_expected_probabilities)

    regularization = 1.0 / (10 * (normalized_expected_probabilities.min() + 1e-4))

    alpha_params = regularization * normalized_expected_probabilities
    beta_params = np.ones_like(normalized_expected_probabilities)  # Initialize beta parameters to 1

    return alpha_params, beta_params


def bandit_weights_to_normal_prior(expected_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bandit weights to Normal prior parameters (mu, sigma) for each arm.

    For multi-dimensional Normal distributions (CMABs), the mean parameters are optimized
    to minimize the squared error between the expected probabilities and the probabilities derived
    from the univariate Normal cdf -- this is a simplification in order to avoid precision errors
    from computing the multivariate Normal cdf.
    As the number of dimensions increases, the approximation diverges from the true probabilities.
    However, this is a reasonable error tolerance for the purposes of setting prior parameters for CMABs.

    We note that the problem is also underdetermined (i.e. N parameters, but only N-1 degrees of freedom,
    because the probabilities must sum to 1). Pinning one of the mu values warps the solution space, so we
    elect to regularize mu values to arrive at an approximate solution instead.

    Args:
        expected_probabilities (np.ndarray): Array of shape (n_arms,) containing the expected
            probabilities for each arm.
    Returns:
        mu (np.ndarray): Array of shape (n_arms,) containing the mean parameters for the Normal
            distribution.
        sigma (np.ndarray): Array of shape (n_arms,) containing the standard deviation parameters
            for the Normal distribution.
    """
    normalized_expected_probabilities = np.asarray(expected_probabilities, dtype=np.float64)

    if not math.isclose(normalized_expected_probabilities.sum(), 100.0, rel_tol=1e-9):
        raise ValueError("Expected probabilities must sum to 100.")

    normalized_expected_probabilities *= 0.01  # Normalize to sum to 1
    sigma_params = np.ones_like(normalized_expected_probabilities)  # Initialize beta parameters to 1
    mu_params = np.zeros_like(normalized_expected_probabilities)  # Initialize alpha parameters to 1

    def objective(params: np.ndarray) -> float:
        mus = np.array(params.tolist())

        def prob_n_is_max(n: int) -> float:
            def integrand(x: float) -> float:
                pdf_n = norm.pdf(x, loc=mus, scale=sigma_params)
                cdf_n = norm.cdf(x, loc=mus, scale=sigma_params)
                return float((np.prod(cdf_n) / (cdf_n[n] + 0.00001)) * pdf_n[n])  # type: ignore

            result, _ = quad(integrand, -np.inf, np.inf)
            return float(result)

        computed_probabilities = np.array([prob_n_is_max(n) for n in range(len(normalized_expected_probabilities))])
        return float(np.sum((computed_probabilities - normalized_expected_probabilities) ** 2 + 0.01 * mus**2))

    if (
        np.abs(
            (normalized_expected_probabilities.max() - normalized_expected_probabilities.min())
            / (normalized_expected_probabilities.min() + 1e-4)
        )
        < 1e-2
    ).all():
        return mu_params, sigma_params
    result = minimize(objective, mu_params)
    return result.x, sigma_params


def convert_arm_weights_to_prior_params(
    arm_weights: list[float], prior_type: PriorTypes
) -> tuple[list[float], list[float]]:
    expected_probabilities = np.array(arm_weights, dtype=np.float64)

    if prior_type == PriorTypes.BETA:
        alpha, beta = bandit_weights_to_beta_prior(expected_probabilities)
        return alpha.tolist(), beta.tolist()
    if prior_type == PriorTypes.NORMAL:
        mu, sigma = bandit_weights_to_normal_prior(expected_probabilities)
        return mu.tolist(), sigma.tolist()
    raise ValueError(f"Unsupported prior type: {prior_type}")
