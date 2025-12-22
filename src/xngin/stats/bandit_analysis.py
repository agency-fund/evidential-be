import numpy as np

from xngin.apiserver.routers.common_api_types import BanditArmAnalysis
from xngin.apiserver.routers.common_enums import (
    ContextLinkFunctions,
    ExperimentsType,
    LikelihoodTypes,
    PriorTypes,
)
from xngin.apiserver.sqla import tables


def _analyze_beta_binomial(
    alpha: float, beta: float, random_state: int | None = None, n_random_samples: int = 100000
) -> tuple[float, float, float, float]:
    """
    Analyze a single arm with Beta-Binomial model.
    Args:
        alpha: The posterior alpha parameter of the arm.
        beta: The posterior beta parameter of the arm.
        random_state: Use a fixed int for deterministic behavior in tests.
        n_random_samples: Number of samples to draw for estimation.
    """
    predictive_mean = alpha / (alpha + beta)
    predictive_stdev = np.sqrt(alpha * beta) / (alpha + beta)

    rng = np.random.default_rng(random_state)
    samples = rng.beta(a=alpha, b=beta, size=n_random_samples)
    alpha_level = 0.025  # 95% credible interval = (1 - 0.95) / 2
    ci_lower = np.percentile(samples, alpha_level * 100)
    ci_upper = np.percentile(samples, (1 - alpha_level) * 100)

    return predictive_mean, float(predictive_stdev), float(ci_upper), float(ci_lower)


def _analyze_normal(
    mu: np.ndarray, covariance: np.ndarray, outcome_std_dev: float, context: np.ndarray | None
) -> tuple[float, float, float, float]:
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
        # Variance of our estimate of the mean
        var_of_mean = covariance.flatten()[0]
    else:
        predictive_mean = context @ mu
        var_of_mean = context @ covariance @ context

    # Compute 95% Credible Interval bounds on our estimate of the mean
    stderr_of_mean = np.sqrt(var_of_mean)
    ci_upper = predictive_mean + 1.96 * stderr_of_mean
    ci_lower = predictive_mean - 1.96 * stderr_of_mean

    # Standard deviation of the predictive distribution (includes outcome noise)
    predictive_stdev = np.sqrt(var_of_mean + outcome_std_dev**2)

    return float(predictive_mean), float(predictive_stdev), float(ci_upper), float(ci_lower)


def _analyze_normal_binary(
    mu: np.ndarray,
    covariance: np.ndarray,
    context_link_functions: ContextLinkFunctions,
    context: np.ndarray | None = None,
    num_samples: int = 1000,
    random_state: int | None = None,
) -> tuple[float, float, float, float]:
    """
    Analyze a single arm with Normal model for binary outcomes.
    Args:
        mu: The posterior mean vector of the arm.
        covariance: The posterior covariance matrix of the arm.
        context_link_functions: The link function to use.
        context: Optional context vector.
        num_samples: Number of samples to draw for estimation.
        random_state: Use a fixed int for deterministic behavior in tests.
    """
    # First derive the Credible Interval in latent space
    if context is None:
        latent_mean = float(mu[0])
        latent_var = float(covariance[0, 0])
    else:
        latent_mean = float(context @ mu)
        latent_var = float(context @ covariance @ context)

    # Calculate 95% CI bounds
    latent_stderr = np.sqrt(latent_var)
    latent_ci_upper = latent_mean + 1.96 * latent_stderr
    latent_ci_lower = latent_mean - 1.96 * latent_stderr
    # Transform back into to probability space
    latent_bounds = np.array([latent_ci_lower, latent_ci_upper])
    ci_lower, ci_upper = context_link_functions(latent_bounds).tolist()

    # Estimate the mean via MC integration given the non-linear link function.
    # First sample in latent space
    rng = np.random.default_rng(random_state)
    samples = rng.multivariate_normal(mean=mu, cov=covariance, size=num_samples)
    if context is not None:
        parameter_samples = samples @ context
    else:
        parameter_samples = samples
    # Convert to probabilities to compute the posterior predictive mean
    prob_samples = context_link_functions(parameter_samples)
    predictive_mean = prob_samples.mean()

    # Analytical standard deviation for bernoulli outcomes
    predictive_stdev = np.sqrt(predictive_mean * (1 - predictive_mean))

    return float(predictive_mean), float(predictive_stdev), ci_upper, ci_lower


def analyze_experiment(
    experiment: tables.Experiment,
    outcome_std_dev: float = 1.0,
    context_vals: list | None = None,
    random_state: int | None = None,
) -> list[BanditArmAnalysis]:
    """
    Analyze a bandit experiment. Assumes arms and draws are preloaded.

    Args:
        experiment: The bandit experiment to analyze.
        outcome_std_dev: Standard deviation of the outcomes. Only used for Normal likelihood.
        context_vals: Optional context values for CMAB experiments.
        random_state: Use a fixed int for deterministic behavior in tests.
    """
    # TODO: Does not support Bayes A/B experiments
    if experiment.experiment_type == ExperimentsType.BAYESAB_ONLINE.value:
        raise ValueError(f"Invalid experiment type: {experiment.experiment_type}.")
    if not experiment.prior_type or not experiment.reward_type:
        raise ValueError("Experiment must have prior and reward types defined.")
    if (experiment.experiment_type == ExperimentsType.CMAB_ONLINE.value) and (context_vals is None):
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
                (prior_pred_mean, prior_pred_stdev, prior_pred_ci_upper, prior_pred_ci_lower) = _analyze_beta_binomial(
                    arm.alpha_init, arm.beta_init, random_state=random_state
                )

                assert arm.alpha is not None and arm.beta is not None, (
                    "Arm must have initial alpha and beta parameters."
                )
                (post_pred_mean, post_pred_stdev, post_pred_ci_upper, post_pred_ci_lower) = _analyze_beta_binomial(
                    arm.alpha, arm.beta, random_state=random_state
                )

            case PriorTypes.NORMAL, LikelihoodTypes.NORMAL:
                assert arm.mu_init is not None and arm.sigma_init is not None, (
                    "Arm must have initial mu and sigma parameters."
                )
                (prior_pred_mean, prior_pred_stdev, prior_pred_ci_upper, prior_pred_ci_lower) = _analyze_normal(
                    np.array([arm.mu_init] * max(len(experiment.contexts), 1)),
                    np.diag([arm.sigma_init] * max(len(experiment.contexts), 1)),
                    outcome_std_dev,
                    context=np.array(context_vals) if context_vals else None,
                )

                assert arm.mu is not None and arm.covariance is not None, "Arm must have mu and covariance parameters."
                (post_pred_mean, post_pred_stdev, post_pred_ci_upper, post_pred_ci_lower) = _analyze_normal(
                    np.array(arm.mu),
                    np.array(arm.covariance),
                    outcome_std_dev,
                    context=np.array(context_vals) if context_vals else None,
                )

            case PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI:
                assert arm.mu_init is not None and arm.sigma_init is not None, (
                    "Arm must have initial mu and sigma parameters."
                )
                (prior_pred_mean, prior_pred_stdev, prior_pred_ci_upper, prior_pred_ci_lower) = _analyze_normal_binary(
                    np.array([arm.mu_init] * max(len(experiment.contexts), 1)),
                    np.diag([arm.sigma_init] * max(len(experiment.contexts), 1)),
                    ContextLinkFunctions.LOGISTIC,
                    context=np.array(context_vals) if context_vals else None,
                    random_state=random_state,
                )

                assert arm.mu is not None and arm.covariance is not None, "Arm must have mu and covariance parameters."
                (post_pred_mean, post_pred_stdev, post_pred_ci_upper, post_pred_ci_lower) = _analyze_normal_binary(
                    np.array(arm.mu),
                    np.array(arm.covariance),
                    ContextLinkFunctions.LOGISTIC,
                    context=np.array(context_vals) if context_vals else None,
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
                prior_pred_ci_upper=prior_pred_ci_upper,
                prior_pred_ci_lower=prior_pred_ci_lower,
                post_pred_mean=post_pred_mean,
                post_pred_stdev=post_pred_stdev,
                post_pred_ci_upper=post_pred_ci_upper,
                post_pred_ci_lower=post_pred_ci_lower,
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
