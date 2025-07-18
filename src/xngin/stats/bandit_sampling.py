import numpy as np

from xngin.apiserver.models import tables
from xngin.apiserver.routers.common_enums import (
    ContextLinkFunctions,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)


# ------------- Utilities for sampling and updating arms ----------------
# --- Sampling functions for Thompson Sampling ---
def _sample_beta_binomial(
    alphas: np.ndarray, betas: np.ndarray, random_state: int = 66
) -> int:
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
        rng.multivariate_normal(mean=mu, cov=cov)
        for mu, cov in zip(mus, covariances, strict=False)
    ]).reshape(-1, len(context))
    probs = link_function(samples @ context)
    return int(probs.argmax())


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
    # Only supported for Bayesian experiments
    if experiment.experiment_type not in {
        ExperimentsType.MAB_ONLINE.value,
        ExperimentsType.CMAB_ONLINE.value,
        ExperimentsType.BAYESAB_ONLINE.value,
    }:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")

    sorted_arms = sorted(experiment.arms, key=lambda a: a.name)
    if experiment.prior_type == PriorTypes.BETA.value:
        if experiment.reward_type != LikelihoodTypes.BERNOULLI.value:
            raise ValueError("Beta prior is only supported for Bernoulli rewards.")
        alphas = np.array([arm.alpha for arm in sorted_arms])
        betas = np.array([arm.beta for arm in sorted_arms])

        arm_index = _sample_beta_binomial(
            alphas=alphas, betas=betas, random_state=random_state
        )

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
