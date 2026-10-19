import numpy as np
from scipy.optimize import minimize

from xngin.apiserver.limits import TOP_TWO_BETA, TOP_TWO_MIN_ARMS
from xngin.apiserver.routers.common_enums import (
    ContextLinkFunctions,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
    PriorUpdateType,
    UpdateTypeBeta,
    UpdateTypeNormal,
)
from xngin.apiserver.sqla import tables


# ------------- Utilities for sampling and updating arms ----------------
# --- Sampling functions for Thompson Sampling ---
def _sample_beta_binomial(
    alphas: np.ndarray, betas: np.ndarray, random_state: int | None = None, use_top_two: bool = False
) -> int:
    """
    Thompson Sampling with Beta-Binomial distribution.

    When use_top_two is True, takes the top two arms from a single posterior
    sample and flips a coin (probability TOP_TWO_BETA) to pick between the
    leader and challenger.

    Parameters
    ----------
    alphas: alpha parameter of Beta distribution for each arm
    betas: beta parameter of Beta distribution for each arm
    random_state: Seed for the random number generator. None for true randomness.
    use_top_two: Whether to use Top-Two selection.
    """
    rng = np.random.default_rng(random_state)
    samples = rng.beta(alphas, betas)
    if use_top_two:
        sorted_idx = np.argsort(-samples)
        leader, challenger = int(sorted_idx[0]), int(sorted_idx[1])
        return leader if rng.random() < TOP_TWO_BETA else challenger
    return int(samples.argmax())


def _sample_normal(
    mus: list[np.ndarray],
    covariances: list[np.ndarray],
    context: np.ndarray | None,
    link_function: ContextLinkFunctions,
    random_state: int | None = None,
    use_top_two: bool = False,
) -> int:
    """
    Thompson Sampling with normal prior.

    When use_top_two is True, takes the top two arms from a single posterior
    sample and flips a coin (probability TOP_TWO_BETA) to pick between the
    leader and challenger.

    Parameters
    ----------
    mus: mean of Normal distribution for each arm
    covariances: covariance matrix of Normal distribution for each arm
    context: context vector of shape (1, d), or None for a non-contextual bandit
    link_function: link function for the context
    random_state: Seed for the random number generator. None for true randomness.
    use_top_two: Whether to use Top-Two selection.
    """

    rng = np.random.default_rng(random_state)
    num_dimensions = len(mus[0])

    if context is not None and context.shape != (1, num_dimensions):
        raise ValueError(f"Context has unexpected dimensions: {context.shape}; expected {(1, num_dimensions)}")

    # Sample model parameters for each arm relating context to outcome
    samples = np.array([rng.multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covariances, strict=True)])
    samples = samples.reshape(-1, num_dimensions)
    # No context is equivalent to an all-ones context vector, i.e. summing the parameters.
    scores = link_function(samples.sum(-1)) if context is None else link_function(samples @ context.T)

    if use_top_two:
        sorted_idx = np.argsort(-scores.squeeze())
        leader, challenger = int(sorted_idx[0]), int(sorted_idx[1])
        return leader if rng.random() < TOP_TWO_BETA else challenger
    return int(scores.argmax())


# --- Arm update functions ---
def _update_arm_beta_binomial(alpha: float, beta: float, rewards: np.ndarray) -> UpdateTypeBeta:
    """
    Update the alpha and beta parameters of the Beta distribution.

    Parameters
    ----------
    alpha : float
        The alpha parameter of the Beta distribution.
    beta : float
        The beta parameter of the Beta distribution.
    rewards : np.ndarray
        The binary (0/1) rewards of the arm, of length T.

    Description
    -----------
    Given a beta prior with parameters α_0 and β_0:
    p(θ) = Beta(α_0, β_0)
    and binary valued rewards r:
    p(r) = Bernoulli(θ)

    The posterior distribution of the arm parameters is also Beta, with parameters
    α_1 = α_0 + r
    β_1 = β_0 + (1 - r)

    If r is a binary-valued vector of outcomes r = [r_1, r_2, ... r_T] for T timesteps,
    then:
    α_1 = α_0 + Σ_{i=1}^T r_i
    β_1 = β_0 + Σ_{i=1}^T (1 - r_i)
    """
    num_positive = float((rewards == 1).sum())
    num_negative = float((rewards == 0).sum())

    return UpdateTypeBeta(alpha=alpha + num_positive, beta=beta + num_negative)


def _update_arm_normal(
    current_mu: np.ndarray,
    current_covariance: np.ndarray,
    rewards: np.ndarray,
    llhood_sigma: float,
    contexts: np.ndarray | None,
) -> UpdateTypeNormal:
    """
    Update the mean and standard deviation of the Normal distribution.

    Parameters
    ----------
    current_mu : The mean of the Normal distribution (of length d, where d is
        the number of dimensions / context inputs).
    current_covariance : The covariance of the Normal distribution (of size d x d).
    rewards : The array of rewards of the arm (of length T).
    llhood_sigma : The standard deviation of the likelihood.
    contexts : The array of context vectors (T x d), or None for a non-contextual bandit.

    Description
    -----------
    Given a normal prior on the arm parameters (of dimension d):
    p(θ) = Ν(μ_0, Σ_0)

    and a likelihood function for the rewards (where X is context):
    p(r | θ, X) = Ν(X.θ, σ^2)

    The posterior distribution of the arm parameters is also normal, and the mean and covariance
    can be computed in closed form, as follows:
    Σ_1 = (Σ_0 ^ (-1) + X^T.X / σ^2)^(-1)
    μ_1 = Σ_1 (Σ_0 ^ (-1) μ_0 + X^T.r / σ^2)

    Now, if r = [r_1, r_2, ..., r_T] and X = [x_1, x_2, ..., x_T]^T (i.e. we have multiple rewards and
    corresponding contexts observed over multiple time steps), the above formulas hold as long as r is
    a vector of dimension (T x 1) and X is a matrix of dimension (T x d).
    We also assume that NO updates have been made to the prior parameters between timestep 0 and
    timestep T.
    """
    num_dimensions = current_mu.shape[0]
    num_rewards = len(rewards)
    rewards = rewards.reshape(-1, 1)

    if contexts is None:
        # No context is equivalent to an all-ones context, matching _sample_normal and _update_arm_laplace.
        contexts = np.ones((num_rewards, num_dimensions))
    elif contexts.shape != (num_rewards, num_dimensions):
        raise ValueError(
            f"Context has unexpected dimensions: {contexts.shape}; expected {(num_rewards, num_dimensions)}"
        )

    llhood_covariance_inv = contexts.T @ contexts / llhood_sigma**2

    # Prior covariance matrix inverse
    prior_covariance_inv = np.linalg.inv(current_covariance)

    # New covariance
    new_covariance = np.linalg.inv(prior_covariance_inv + llhood_covariance_inv)

    # New mean
    llhood_term = (contexts.T @ rewards / llhood_sigma**2).squeeze(-1)
    new_mu = new_covariance @ ((prior_covariance_inv @ current_mu) + llhood_term)

    return UpdateTypeNormal(mu=new_mu.tolist(), covariance=new_covariance.tolist())


def _update_arm_laplace(
    current_mu: np.ndarray,
    current_covariance: np.ndarray,
    rewards: np.ndarray,
    contexts: np.ndarray | None,
    link_function: ContextLinkFunctions,
    reward_likelihood: LikelihoodTypes,
    prior_type: PriorTypes,
) -> UpdateTypeNormal:
    """
    Update the mean and covariance using the Laplace approximation.

    Parameters
    ----------
    current_mu : The mean of the normal distribution (of size d, where d is the dimension
        / number of context inputs).
    current_covariance : The covariance matrix of the normal distribution (of size d x d).
    rewards : The list of rewards for the arm (of size T, where T is the number of observed timesteps).
    contexts : The list of contexts for the arm (of size T x d), or None for a non-contextual bandit.
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
        predicted = link_function(np.sum(theta)) if contexts is None else link_function(contexts @ theta)
        log_likelihood = reward_likelihood(rewards, predicted)

        return float(-log_prior - log_likelihood)

    num_dimensions = current_mu.shape[0]
    num_rewards = rewards.shape[0]

    if contexts is not None and contexts.shape != (num_rewards, num_dimensions):
        raise ValueError(
            f"Context has unexpected dimensions: {contexts.shape}; expected {(num_rewards, num_dimensions)}"
        )

    result = minimize(objective, x0=np.zeros_like(current_mu), method="L-BFGS-B")
    new_mu = result.x
    hess_inv = result.hess_inv
    if not hasattr(hess_inv, "todense"):
        raise TypeError(f"unexpected type: {type(result.hess_inv)}")
    covariance = hess_inv.todense()

    new_covariance = 0.5 * (covariance + covariance.T)
    return UpdateTypeNormal(mu=new_mu.tolist(), covariance=new_covariance.tolist())


# ------------- Import functions ----------------
# --- Choose arm function ---
def choose_arm(
    experiment: tables.Experiment,
    sorted_context_vals: list[float] | None = None,
    random_state: int | None = None,
) -> tables.Arm:
    """
    Choose arm based on posterior using Thompson Sampling.

    For experiments with ≥TOP_TWO_MIN_ARMS arms, uses Top-Two Thompson
    Sampling: takes the top two arms from a single posterior sample and
    flips a coin (probability TOP_TWO_BETA) to pick between the leader
    and challenger. This prevents premature lock-in when many arms have
    similar rewards.

    Parameters
    ----------
    experiment: The experiment data containing priors and rewards for each arm.
    sorted_context_vals: Optional context vector for the experiment.
    random_state: Seed for the random number generator. None for true randomness.
    """
    supported_types = {
        ExperimentsType.MAB_ONLINE.value,
        ExperimentsType.MAB_ONLINE_DWH.value,
        ExperimentsType.CMAB_ONLINE.value,
    }
    if experiment.experiment_type not in supported_types:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")

    sorted_arms = sorted(experiment.arms, key=lambda a: a.id)
    use_top_two = len(sorted_arms) >= TOP_TWO_MIN_ARMS

    if experiment.prior_type == PriorTypes.BETA.value:
        if experiment.reward_type != LikelihoodTypes.BERNOULLI.value:
            raise ValueError("Beta prior is only supported for Bernoulli rewards.")
        alphas = np.array([arm.alpha for arm in sorted_arms])
        betas = np.array([arm.beta for arm in sorted_arms])
        arm_index = _sample_beta_binomial(
            alphas=alphas,
            betas=betas,
            random_state=random_state,
            use_top_two=use_top_two,
        )

    elif experiment.prior_type == PriorTypes.NORMAL.value:
        mus = [np.array(arm.mu) for arm in sorted_arms]
        covariances = [np.array(arm.covariance) for arm in sorted_arms]
        context_array = None if sorted_context_vals is None else np.array(sorted_context_vals).reshape(1, -1)
        link_function = (
            ContextLinkFunctions.NONE
            if experiment.reward_type == LikelihoodTypes.NORMAL.value
            else ContextLinkFunctions.LOGISTIC
        )
        arm_index = _sample_normal(
            mus=mus,
            covariances=covariances,
            context=context_array,
            link_function=link_function,
            random_state=random_state,
            use_top_two=use_top_two,
        )
    else:
        raise ValueError(f"Unsupported prior type: {experiment.prior_type}")
    return sorted_arms[arm_index]


# --- Update arm parameters ---
def update_arm(
    experiment: tables.Experiment,
    arm_to_update: tables.Arm,
    outcomes: list[float],
    contexts: list[list[float]] | None = None,
) -> PriorUpdateType:
    """
    Update the arm parameters based on the experiment type and reward.

    Parameters
    ----------
    experiment: The experiment data containing arms, prior type and reward
        type information.
    arm_to_update: The arm to update based on the reward received.
    outcomes: The rewards received from the arm.
    contexts: The context vectors for the arm.
    """
    supported_types = {
        ExperimentsType.MAB_ONLINE.value,
        ExperimentsType.MAB_ONLINE_DWH.value,
        ExperimentsType.CMAB_ONLINE.value,
    }
    if experiment.experiment_type not in supported_types:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")
    if not experiment.prior_type or not experiment.reward_type:
        raise ValueError("Experiment must have prior and reward types defined.")

    if contexts is not None and len(outcomes) != len(contexts):
        raise ValueError("Number of context and outcome values do not match")

    # Beta-binomial priors
    if experiment.prior_type == PriorTypes.BETA.value:
        if not (arm_to_update.alpha and arm_to_update.beta):
            raise ValueError("Arm must have alpha and beta parameters.")
        rewards = np.array([outcomes[0]])
        return _update_arm_beta_binomial(alpha=arm_to_update.alpha, beta=arm_to_update.beta, rewards=rewards)

    # Normal priors
    if experiment.prior_type == PriorTypes.NORMAL.value:
        if arm_to_update.mu is None and arm_to_update.covariance is None:
            raise ValueError("Arm must have mu and covariance parameters.")

        rewards = np.array(outcomes)
        context_array = np.array(contexts).reshape(len(rewards), -1) if contexts is not None else None
        # Normal likelihood
        if experiment.reward_type == LikelihoodTypes.NORMAL.value:
            return _update_arm_normal(
                current_mu=np.array(arm_to_update.mu),
                current_covariance=np.array(arm_to_update.covariance),
                rewards=rewards[:1],
                llhood_sigma=1.0,  # TODO: Assuming a fixed likelihood sigma
                contexts=context_array[:1, :] if context_array is not None else None,
            )
        # TODO: currently only supports Bernoulli likelihood
        return _update_arm_laplace(
            current_mu=np.array(arm_to_update.mu),
            current_covariance=np.array(arm_to_update.covariance),
            rewards=rewards,
            contexts=context_array,
            link_function=ContextLinkFunctions.LOGISTIC,
            reward_likelihood=LikelihoodTypes(experiment.reward_type),
            prior_type=PriorTypes(experiment.prior_type),
        )
    raise ValueError("Unsupported prior type for arm update.")
