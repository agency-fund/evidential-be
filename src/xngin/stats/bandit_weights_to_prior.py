import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

from xngin.apiserver.routers.common_api_types import PriorTypes


def bandit_weights_to_beta_prior(
    expected_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bandit weights to Beta prior parameters (alpha, beta) for each arm.

    Args:
        expected_probabilities (np.ndarray): Array of shape (n_arms,) containing the expected
            probabilities for each arm.

    Returns:
        alpha (np.ndarray): Array of shape (n_arms,) containing the alpha
            parameters for the Beta distribution.
        beta (np.ndarray): Array of shape (n_arms,) containing the beta
            parameters for the Beta distribution.
    """
    expected_probabilities = np.asarray(expected_probabilities, dtype=np.float64)
    expected_probabilities *= 0.01  # Normalize to sum to 1
    beta_params = np.ones_like(expected_probabilities)  # Initialize beta parameters to 1
    alpha_params = np.ones_like(expected_probabilities)  # Initialize alpha parameters to 1

    def objective(params: np.ndarray) -> float:
        r"""
        The objective function to minimize, which calculates the squared error between the
        expected probabilities and the probabilities derived from the Beta cdf.

        i.e. for each arm i, the arm weight represent the probability that a
        sample from the Beta distribution of arm i is greater than samples from
        the Beta distributions of all other arms.
        This can be calculated using the cumulative distribution function (CDF)
        of the Beta distribution (given its parameters $\alpha_i$ and $\beta_i$).

        $p(\theta_n | \theta_i)_{i=1, i \neq n}^{N}$ = \prod_{i=1, i \neq n}^{N}
        \mathcal{E}_{\theta_n}[(1 - CDF(\theta_i, \alpha_i, \beta_i))]$

        If we assume that $\beta_{i} = 1, \forall i$, then the above equation can
        be simplified to:
        $p(\theta_n | \theta_i)_{i=1, i \neq n}^{N}$ = 2 * \frac{(\alpha_n)^N}
        {\prod_{i=1}^{N} (\alpha_i + \alpha_n)} $

        This can be reduced to the following optimization problem:
        $\min_{\alpha_1, \alpha_2, ..., \alpha_{N-1}} \sum_{n=1}^{N} (2 *
        \frac{(\alpha_n)^N}{\prod_{i=1}^{N} (\alpha_i + \alpha_n)} - p_n)^2$

        Args:
            params (np.ndarray): Array of shape (n_arms-1,) containing the alpha
                parameters for the Beta distribution, excluding the last arm.

        Returns:
            float: The squared error between the expected probabilities and
            the probabilities derived from the Beta cdf.
        """
        alphas = np.abs(np.array([*params.tolist(), 1.0]))

        alpha_mesh_1, alpha_mesh_2 = np.meshgrid(alphas, alphas)
        pairwise_sums = alpha_mesh_1 + alpha_mesh_2

        numerator = 2 * alphas ** len(expected_probabilities)
        denominator = np.prod(pairwise_sums, axis=1)
        return float(
            np.sum(
                ((numerator / denominator) - expected_probabilities) ** 2 + 0.01 * (alphas - np.ones_like(alphas)) ** 2
            )
        )

    if (expected_probabilities == expected_probabilities[0]).all():
        return alpha_params, beta_params

    result = minimize(objective, alpha_params[:-1])
    return np.array([*np.abs(result.x).tolist(), 1.0]), beta_params


def bandit_weights_to_normal_prior(
    expected_probabilities: np.ndarray, num_dimensions: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bandit weights to Normal prior parameters (mu, sigma) for each arm.

    Args:
        expected_probabilities (np.ndarray): Array of shape (n_arms,) containing the expected
            probabilities for each arm.
        num_dimensions (int): The number of dimensions for the Normal distribution. Default is 1.
    Returns:
        mu (np.ndarray): Array of shape (n_arms,) containing the mean parameters for the Normal
            distribution.
        sigma (np.ndarray): Array of shape (n_arms,) containing the standard deviation parameters
            for the Normal distribution.
    """
    expected_probabilities = np.asarray(expected_probabilities, dtype=np.float64)
    expected_probabilities *= 0.01  # Normalize to sum to 1
    sigma_params = np.ones_like(expected_probabilities)  # Initialize beta parameters to 1
    mu_params = np.zeros_like(expected_probabilities)  # Initialize alpha parameters to 1

    def objective(params: np.ndarray) -> float:
        mus = np.array([*params.tolist(), 0.0])

        def prob_n_is_max(n: int) -> float:
            def integrand(x: float) -> float:
                pdf_n = norm.pdf(x, loc=mus, scale=sigma_params)
                cdf_n = norm.cdf(x, loc=mus, scale=sigma_params)
                return float((np.prod(cdf_n) / (cdf_n[n] + 0.00001)) * pdf_n[n])  # type: ignore

            result, _ = quad(integrand, -np.inf, np.inf)
            return float(result)

        computed_probabilities = np.array([prob_n_is_max(n) for n in range(len(expected_probabilities))])
        return float(
            np.sum(
                (computed_probabilities**num_dimensions - expected_probabilities) ** 2 + 0.01 * num_dimensions * mus**2
            )
        )

    if (expected_probabilities.round(1) == expected_probabilities[0].round(1)).all():
        return mu_params, sigma_params
    result = minimize(objective, mu_params[:-1])
    return np.array([*result.x.tolist(), 0.0]), sigma_params


def convert_arm_weights_to_prior_params(
    arm_weights: list[float], prior_type: PriorTypes, num_contexts: int = 1
) -> tuple[list[float], list[float]]:
    expected_probabilities = np.array(arm_weights, dtype=np.float64)

    if prior_type == PriorTypes.BETA:
        alpha, beta = bandit_weights_to_beta_prior(expected_probabilities)
        return alpha.tolist(), beta.tolist()
    if prior_type == PriorTypes.NORMAL:
        mu, sigma = bandit_weights_to_normal_prior(expected_probabilities, num_dimensions=num_contexts)
        return mu.tolist(), sigma.tolist()
    raise ValueError(f"Unsupported prior type: {prior_type}")
