import numpy as np

from xngin.apiserver.routers.common_api_types import BanditArmAnalysis
from xngin.apiserver.routers.common_enums import (
    ContextLinkFunctions,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)
from xngin.apiserver.sqla import tables


def estimate_outcome_std_deviation(draws: list[tables.Draw]) -> float:
    outcomes = np.array([draw.outcome for draw in draws if draw.outcome is not None])
    return np.std(outcomes) if len(outcomes) > 1 else 1.0


def _analyze_beta_binomial(
    alpha_init: float, alpha: float, beta: float, beta_init: float
) -> tuple[float, float, float, float]:
    """
    Analyze a single arm with Beta-Binomial model.
    Args:
        alpha_init: The initial alpha parameter of the arm.
        alpha: The posterior alpha parameter of the arm.
        beta: The posterior beta parameter of the arm.
        beta_init: The initial beta parameter of the arm.
    """
    prior_pred_mean = alpha_init / (alpha_init + beta_init)
    prior_pred_sttdev = np.sqrt(alpha_init * beta_init) / (alpha_init + beta_init)
    post_pred_mean = alpha / (alpha + beta)
    post_pred_stdev = np.sqrt(alpha * beta) / (alpha + beta)
    return prior_pred_mean, prior_pred_sttdev, post_pred_mean, post_pred_stdev


def _analyze_normal(
    mu_init: float, sigma_init: float, mu: np.ndarray, covariance: np.ndarray, outcome_std_dev: float
) -> tuple[float, float, float, float]:
    """
    Analyze a single arm with Normal model.
    Args:
        arm: The arm to analyze.
        outcome_std_dev: Standard deviation of the outcomes.
    """
    prior_pred_mean = mu_init
    prior_pred_stdev = np.sqrt(sigma_init**2 + outcome_std_dev**2)
    post_pred_mean = mu[0]
    post_pred_stdev = np.sqrt(covariance.flatten()[0] ** 2 + outcome_std_dev**2)
    return prior_pred_mean, prior_pred_stdev, post_pred_mean, post_pred_stdev


def _analyse_normal_binary(
    mu_init: float,
    sigma_init: float,
    mu: np.ndarray,
    covariance: np.ndarray,
    context_link_functions: ContextLinkFunctions,
) -> tuple[float, float, float, float]:
    """
    Analyze a single arm with Normal model for binary outcomes.
    Args:
        arm: The arm to analyze.
        context_link_functions: The link function to use.
        num_samples: Number of samples to draw from the posterior.

    """
    random_state = 66  # TODO: Make this configurable
    rng = np.random.default_rng(random_state)
    num_samples = 10000  # TODO: Make this configurable

    prior_samples = rng.multivariate_normal(mean=np.array([mu_init]), cov=np.array([[sigma_init]]), size=num_samples)
    posterior_samples = rng.multivariate_normal(mean=mu, cov=covariance, size=num_samples)

    transformed_prior_samples = context_link_functions(prior_samples)
    transformed_posterior_samples = context_link_functions(posterior_samples)
    return (
        transformed_prior_samples.mean(),
        transformed_prior_samples.std(),
        transformed_posterior_samples.mean(),
        transformed_posterior_samples.std(),
    )


def analyze_experiment(
    experiment: tables.Experiment,
) -> list[BanditArmAnalysis]:
    """
    Analyze a bandit experiment. Assumes arms and draws are preloaded.

    Args:
        experiment: The bandit experiment to analyze.
    """
    # TODO: Does not support Bayes A/B or CMAB experiments
    if not experiment.experiment_type == ExperimentsType.MAB_ONLINE.value:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")
    if not experiment.prior_type or not experiment.reward_type:
        raise ValueError("Experiment must have prior and reward types defined.")

    likelihood_type = LikelihoodTypes(experiment.reward_type)
    prior_type = PriorTypes(experiment.prior_type)

    arm_analyses: list[BanditArmAnalysis] = []
    for arm in experiment.arms:
        prior_pred_mean: float
        prior_pred_stdev: float
        post_pred_stdev: float
        post_pred_mean: float
        match prior_type, likelihood_type:
            case PriorTypes.BETA, LikelihoodTypes.BERNOULLI:
                assert (
                    arm.alpha_init is not None
                    and arm.beta_init is not None
                    and arm.alpha is not None
                    and arm.beta is not None
                ), "Arm must have alpha and beta parameters."
                (prior_pred_mean, prior_pred_stdev, post_pred_mean, post_pred_stdev) = _analyze_beta_binomial(
                    alpha_init=(arm.alpha_init), alpha=arm.alpha, beta=arm.beta, beta_init=arm.beta_init
                )
            case PriorTypes.NORMAL, LikelihoodTypes.NORMAL:
                assert (
                    arm.mu_init is not None
                    and arm.sigma_init is not None
                    and arm.mu is not None
                    and arm.covariance is not None
                ), "Arm must have mu and sigma parameters."
                outcome_std_dev = estimate_outcome_std_deviation(experiment.draws)
                (prior_pred_mean, prior_pred_stdev, post_pred_mean, post_pred_stdev) = _analyze_normal(
                    arm.mu_init, arm.sigma_init, np.array(arm.mu), np.array(arm.covariance), outcome_std_dev
                )
            case PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI:
                assert (
                    arm.mu_init is not None
                    and arm.sigma_init is not None
                    and arm.mu is not None
                    and arm.covariance is not None
                ), "Arm must have mu and sigma parameters."
                (prior_pred_mean, prior_pred_stdev, post_pred_mean, post_pred_stdev) = _analyse_normal_binary(
                    arm.mu_init,
                    arm.sigma_init,
                    np.array(arm.mu),
                    np.array(arm.covariance),
                    context_link_functions=ContextLinkFunctions.LOGISTIC,
                )
            case _:
                raise ValueError(f"Unsupported prior and likelihood combination: {prior_type}, {likelihood_type}")
        arm_analyses.append(
            BanditArmAnalysis(
                arm_id=arm.id,
                arm_name=arm.name,
                arm_description=arm.description,
                prior_pred_mean=prior_pred_mean,
                prior_pred_stdev=prior_pred_stdev,
                post_pred_mean=post_pred_mean,
                post_pred_stdev=post_pred_stdev,
                alpha_init=arm.alpha_init,
                beta_init=arm.beta_init,
                alpha=arm.alpha,
                beta=arm.beta,
                mu_init=arm.mu_init,
                sigma_init=arm.sigma_init,
                mu=arm.mu,
                covariance=arm.covariance,
            )
        )
    return arm_analyses
