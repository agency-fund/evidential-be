import numpy as np

from xngin.apiserver.routers.common_api_types import BanditArmAnalysis
from xngin.apiserver.routers.common_enums import (
    ContextLinkFunctions,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)
from xngin.apiserver.sqla import tables


def _analyze_beta_binomial(alpha: float, beta: float) -> tuple[float, float]:
    """
    Analyze a single arm with Beta-Binomial model.
    Args:
        alpha: The posterior alpha parameter of the arm.
        beta: The posterior beta parameter of the arm.
    """
    predictive_mean = alpha / (alpha + beta)
    predictive_stdev = np.sqrt(alpha * beta) / (alpha + beta)
    return predictive_mean, predictive_stdev


def _analyze_normal(
    mu: np.ndarray, covariance: np.ndarray, outcome_std_dev: float, context: np.ndarray | None
) -> tuple[float, float]:
    """
    Analyze a single arm with Normal model.
    Args:
        mu: The posterior mean vector of the arm.
        covariance: The posterior covariance matrix of the arm.
        outcome_std_dev: Standard deviation of the outcomes.
        context: Optional context vector.
    """
    if context is None:
        predictive_mean = mu[0]
        predictive_mean_stdev = np.sqrt(covariance.flatten()[0] ** 2 + outcome_std_dev**2)
    else:
        predictive_mean = context @ mu
        predictive_mean_stdev = np.sqrt(context @ covariance @ context + outcome_std_dev**2)
    return predictive_mean, predictive_mean_stdev


def _analyze_normal_binary(
    mu: np.ndarray,
    covariance: np.ndarray,
    context_link_functions: ContextLinkFunctions,
    context: np.ndarray | None = None,
    random_state: int | None = None,
) -> tuple[float, float]:
    """
    Analyze a single arm with Normal model for binary outcomes.
    Args:
        mu: The posterior mean vector of the arm.
        covariance: The posterior covariance matrix of the arm.
        context_link_functions: The link function to use.
        context: Optional context vector.
        random_state: Use a fixed int for deterministic behavior in tests.
    """
    rng = np.random.default_rng(random_state)
    num_samples = 10000  # TODO: Make this configurable

    samples = rng.multivariate_normal(mean=mu, cov=covariance, size=num_samples)
    if context is not None:
        parameter_samples = samples @ context
    else:
        parameter_samples = samples
    transformed_parameter_samples = context_link_functions(parameter_samples)
    outcome_samples = rng.binomial(n=1, p=transformed_parameter_samples)
    return (
        outcome_samples.mean(),
        outcome_samples.std(),
    )


def analyze_experiment(
    experiment: tables.Experiment,
    outcome_std_dev: float = 1.0,
    contexts: list | None = None,
    random_state: int | None = None,
) -> list[BanditArmAnalysis]:
    """
    Analyze a bandit experiment. Assumes arms and draws are preloaded.

    Args:
        experiment: The bandit experiment to analyze.
        outcome_std_dev: Standard deviation of the outcomes. Only used for Normal likelihood.
        contexts: Optional context values for CMAB experiments.
        random_state: Use a fixed int for deterministic behavior in tests.
    """
    # TODO: Does not support Bayes A/B experiments
    if experiment.experiment_type == ExperimentsType.BAYESAB_ONLINE.value:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")
    if not experiment.prior_type or not experiment.reward_type:
        raise ValueError("Experiment must have prior and reward types defined.")
    if (experiment.experiment_type == ExperimentsType.CMAB_ONLINE.value) and (contexts is None):
        raise ValueError("Contexts must be provided for CMAB experiment analysis.")

    likelihood_type = LikelihoodTypes(experiment.reward_type)
    prior_type = PriorTypes(experiment.prior_type)

    arm_analyses: list[BanditArmAnalysis] = []
    for arm in experiment.arms:
        match prior_type, likelihood_type:
            case PriorTypes.BETA, LikelihoodTypes.BERNOULLI:
                assert arm.alpha_init is not None and arm.beta_init is not None, (
                    "Arm must have initial alpha and beta parameters."
                )
                prior_pred_mean, prior_pred_stdev = _analyze_beta_binomial(arm.alpha_init, arm.beta_init)

                assert arm.alpha is not None and arm.beta is not None, (
                    "Arm must have initial alpha and beta parameters."
                )
                post_pred_mean, post_pred_stdev = _analyze_beta_binomial(arm.alpha, arm.beta)

            case PriorTypes.NORMAL, LikelihoodTypes.NORMAL:
                assert arm.mu_init is not None and arm.sigma_init is not None, (
                    "Arm must have initial mu and sigma parameters."
                )
                prior_pred_mean, prior_pred_stdev = _analyze_normal(
                    np.array([arm.mu_init] * max(len(experiment.contexts), 1)),
                    np.diag([arm.sigma_init] * max(len(experiment.contexts), 1)),
                    outcome_std_dev,
                    context=np.array(contexts) if contexts else None,
                )

                assert arm.mu is not None and arm.covariance is not None, "Arm must have mu and covariance parameters."
                post_pred_mean, post_pred_stdev = _analyze_normal(
                    np.array(arm.mu),
                    np.array(arm.covariance),
                    outcome_std_dev,
                    context=np.array(contexts) if contexts else None,
                )

            case PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI:
                assert arm.mu_init is not None and arm.sigma_init is not None, (
                    "Arm must have initial mu and sigma parameters."
                )
                prior_pred_mean, prior_pred_stdev = _analyze_normal_binary(
                    np.array([arm.mu_init] * max(len(experiment.contexts), 1)),
                    np.diag([arm.sigma_init] * max(len(experiment.contexts), 1)),
                    ContextLinkFunctions.LOGISTIC,
                    context=np.array(contexts) if contexts else None,
                    random_state=random_state,
                )

                assert arm.mu is not None and arm.covariance is not None, "Arm must have mu and covariance parameters."
                post_pred_mean, post_pred_stdev = _analyze_normal_binary(
                    np.array(arm.mu),
                    np.array(arm.covariance),
                    ContextLinkFunctions.LOGISTIC,
                    context=np.array(contexts) if contexts else None,
                    random_state=random_state,
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
