import numpy as np
from scipy.optimize import minimize

from xngin.apiserver.routers.common_enums import (
    ContextLinkFunctions,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)
from xngin.apiserver.sqla import tables


# ------------- Utilities for sampling and updating arms ----------------
# --- Sampling functions for Thompson Sampling ---
def _sample_beta_binomial(alphas: np.ndarray, betas: np.ndarray, random_state: int = 66) -> int:
    """
    Thompson Sampling with Beta-Binomial distribution.

    Parameters
    ----------
    alphas: alpha parameter of Beta distribution for each arm
    betas: beta parameter of Beta distribution for each arm
    random_state : seed for random number generator
    """
    rng = np.random.default_rng(random_state)
    samples = rng.beta(alphas, betas)
    return int(samples.argmax())


def _sample_normal(
    mus: list[np.ndarray],
    covariances: list[np.ndarray],
    context: np.ndarray,
    link_function: ContextLinkFunctions,
    random_state: int = 66,
) -> int:
    """
    Thompson Sampling with normal prior.

    Parameters
    ----------
    mus: mean of Normal distribution for each arm
    covariances: covariance matrix of Normal distribution for each arm
    context: context vector
    link_function: link function for the context
    random_state: seed for random number generator
    """
    rng = np.random.default_rng(random_state)
    samples = np.array([
        rng.multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covariances, strict=False)
    ]).reshape(-1, len(context))

    probs = link_function(samples @ context)
    return int(probs.argmax())


# --- Arm update functions ---
def _update_arm_beta_binomial(alpha: float, beta: float, reward: bool) -> tuple[float, float]:
    """
    Update the alpha and beta parameters of the Beta distribution.

    Parameters
    ----------
    alpha : int
        The alpha parameter of the Beta distribution.
    beta : int
        The beta parameter of the Beta distribution.
    reward : bool
        The reward of the arm.
    """
    if reward:
        return alpha + 1, beta
    return alpha, beta + 1


def _update_arm_normal(
    current_mu: np.ndarray,
    current_covariance: np.ndarray,
    reward: float,
    llhood_sigma: float,
    context: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Update the mean and standard deviation of the Normal distribution.

    Parameters
    ----------
    current_mu : The mean of the Normal distribution.
    current_covariance : The covariance of the Normal distribution.
    reward : The reward of the arm.
    llhood_sigma : The standard deviation of the likelihood.
    context : The context vector.
    """
    # Likelihood covariance matrix inverse
    llhood_covariance_inv = np.eye(len(current_mu)) / llhood_sigma**2
    llhood_covariance_inv *= context.T @ context

    # Prior covariance matrix inverse
    prior_covariance_inv = np.linalg.inv(current_covariance)

    # New covariance
    new_covariance = np.linalg.inv(prior_covariance_inv + llhood_covariance_inv)

    # New mean
    llhood_term: np.ndarray | float = reward / llhood_sigma**2
    if context is not None:
        llhood_term = (context * llhood_term).squeeze()

    new_mu = new_covariance @ ((prior_covariance_inv @ current_mu) + llhood_term)
    return new_mu.tolist(), new_covariance.tolist()


def _update_arm_laplace(
    current_mu: np.ndarray,
    current_covariance: np.ndarray,
    reward: np.ndarray,
    context: np.ndarray,
    link_function: ContextLinkFunctions,
    reward_likelihood: LikelihoodTypes,
    prior_type: PriorTypes,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the mean and covariance using the Laplace approximation.

    Parameters
    ----------
    current_mu : The mean of the normal distribution.
    current_covariance : The covariance matrix of the normal distribution.
    reward : The list of rewards for the arm.
    context : The list of contexts for the arm.
    link_function : The link function for parameters to rewards.
    reward_likelihood : The likelihood function of the reward.
    prior_type : The prior type of the arm.
    """

    def objective(theta: np.ndarray) -> float:
        """
        Objective function for the Laplace approximation.

        Parameters
        ----------
        theta : The parameters of the arm.
        """
        # Log prior
        log_prior = prior_type(theta, mu=current_mu, covariance=current_covariance)

        # Log likelihood
        log_likelihood = reward_likelihood(reward, link_function(context @ theta))

        return float(-log_prior - log_likelihood)

    result = minimize(objective, x0=np.zeros_like(current_mu), method="L-BFGS-B", hess="2-point")
    new_mu = result.x
    hess_inv = result.hess_inv
    if not hasattr(hess_inv, "todense"):
        raise TypeError(f"unexpected type: {type(result.hess_inv)}")
    covariance = hess_inv.todense()

    new_covariance = 0.5 * (covariance + covariance.T)
    return new_mu.tolist(), new_covariance.tolist()


# ------------- Import functions ----------------
# --- Choose arm function ---
def choose_arm(
    experiment: tables.Experiment,
    context: list[float] | None = None,
    random_state: int = 66,
) -> tables.Arm:
    """
    Choose arm based on posterior using Thompson Sampling.

    Parameters
    ----------
    experiment: The experiment data containing priors and rewards for each arm.
    context: Optional context vector for the experiment.
    """
    # TODO: Only supported for MAB and CMAB experiments
    if experiment.experiment_type == ExperimentsType.BAYESAB_ONLINE.value:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")

    sorted_arms = sorted(experiment.arms, key=lambda a: a.name)
    if experiment.prior_type == PriorTypes.BETA.value:
        if experiment.reward_type != LikelihoodTypes.BERNOULLI.value:
            raise ValueError("Beta prior is only supported for Bernoulli rewards.")
        alphas = np.array([arm.alpha for arm in sorted_arms])
        betas = np.array([arm.beta for arm in sorted_arms])

        arm_index = _sample_beta_binomial(alphas=alphas, betas=betas, random_state=random_state)

    elif experiment.prior_type == PriorTypes.NORMAL.value:
        mus = [np.array(arm.mu) for arm in sorted_arms]
        covariances = [np.array(arm.covariance) for arm in sorted_arms]

        context_array = np.ones_like(mus[0]) if context is None else np.array(context)
        arm_index = _sample_normal(
            mus=mus,
            covariances=covariances,
            context=context_array,
            link_function=(
                ContextLinkFunctions.NONE
                if experiment.reward_type == LikelihoodTypes.NORMAL.value
                else ContextLinkFunctions.LOGISTIC
            ),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported prior type: {experiment.prior_type}")
    return sorted_arms[arm_index]


# --- Update arm parameters ---
def update_arm(
    experiment: tables.Experiment,
    arm_to_update: tables.Arm,
    outcomes: list[float],
    context: list[list[float]] | None = None,
) -> tuple:
    """
    Update the arm parameters based on the experiment type and reward.

    Parameters
    ----------
    experiment: The experiment data containing arms, prior type and reward
        type information.
    outcomes: The rewards received from the arm.
    context: The context vector for the arm.
    treatments: The treatments applied to the arm, for a Bayesian A/B test.
    """
    # TODO: Does not support Bayes A/B experiments
    if experiment.experiment_type == ExperimentsType.BAYESAB_ONLINE.value:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")
    if not experiment.prior_type or not experiment.reward_type:
        raise ValueError("Experiment must have prior and reward types defined.")

    # Beta-binomial priors
    if experiment.prior_type == PriorTypes.BETA.value:
        assert arm_to_update.alpha and arm_to_update.beta, "Arm must have alpha and beta parameters."
        return _update_arm_beta_binomial(alpha=arm_to_update.alpha, beta=arm_to_update.beta, reward=bool(outcomes[0]))

    # Normal priors
    if experiment.prior_type == PriorTypes.NORMAL.value:
        assert arm_to_update.mu and arm_to_update.covariance, "Arm must have mu and covariance parameters."

        if context is None:
            context = [[1.0] * len(arm_to_update.mu)]  # Default context if not provided
        # Normal likelihood
        if experiment.reward_type == LikelihoodTypes.NORMAL.value:
            return _update_arm_normal(
                current_mu=np.array(arm_to_update.mu),
                current_covariance=np.array(arm_to_update.covariance),
                reward=outcomes[0],
                llhood_sigma=1.0,  # TODO: Assuming a fixed likelihood sigma
                context=np.array(context[0]),
            )
        # TODO: currently only supports Bernoulli likelihood
        return _update_arm_laplace(
            current_mu=np.array(arm_to_update.mu),
            current_covariance=np.array(arm_to_update.covariance),
            reward=np.array(outcomes),
            context=np.array(context),
            link_function=ContextLinkFunctions.LOGISTIC,
            reward_likelihood=LikelihoodTypes(experiment.reward_type),
            prior_type=PriorTypes(experiment.prior_type),
        )
    raise ValueError("Unsupported prior type for arm update.")
