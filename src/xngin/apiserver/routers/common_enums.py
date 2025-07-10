import decimal
import enum
from typing import Any

import numpy as np


class MetricType(enum.StrEnum):
    """Classifies metrics by their value type."""

    BINARY = "binary"
    NUMERIC = "numeric"

    @classmethod
    def from_python_type(cls, python_type: type) -> "MetricType":
        """Maps Python types to metric types."""

        if python_type in {int, float, decimal.Decimal}:
            return MetricType.NUMERIC
        if python_type is bool:
            return MetricType.BINARY
        raise ValueError(f"Unsupported type: {python_type}")


class MetricPowerAnalysisMessageType(enum.StrEnum):
    """Classifies metric power analysis results."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    NO_BASELINE = "no baseline"
    NO_AVAILABLE_N = "no available n"
    ZERO_EFFECT_SIZE = "zero effect size"
    ZERO_STDDEV = "zero variation"


class ExperimentsType(enum.StrEnum):
    """
    Enum for the experiment types.
    """

    MAB_ONLINE = "mab_online"
    CMAB_ONLINE = "cmab_online"
    BAYESAB_ONLINE = "bayes_ab_online"
    FREQ_ONLINE = "freq_online"
    FREQ_PREASSIGNED = "freq_preassigned"


class PriorTypes(enum.StrEnum):
    """
    Enum for the prior distribution of the arm.
    """

    BETA = "beta"
    NORMAL = "normal"

    def __call__(self, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Return the log pdf of the input param.
        """
        if self == PriorTypes.BETA:
            alpha = kwargs.get("alpha", np.ones_like(theta))
            beta = kwargs.get("beta", np.ones_like(theta))
            pdf = (alpha - 1) * np.log(theta) + (beta - 1) * np.log(1 - theta)
            return np.array(pdf)

        if self == PriorTypes.NORMAL:
            mu = kwargs.get("mu", np.zeros_like(theta))
            covariance = kwargs.get("covariance", np.diag(np.ones_like(theta)))
            inv_cov = np.linalg.inv(covariance)
            x = theta - mu
            pdf = -0.5 * x @ inv_cov @ x
            return np.array(pdf)
        raise ValueError(f"Unsupported prior type: {self}.")


class LikelihoodTypes(enum.StrEnum):
    """
    Enum for the likelihood distribution of the reward.
    """

    BERNOULLI = "binary"
    NORMAL = "real-valued"

    def __call__(self, reward: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """
        Calculate the log likelihood of the reward.

        Parameters
        ----------
        reward : The reward.
        probs : The probability of the reward.
        """
        if self == LikelihoodTypes.NORMAL:
            llhood = -0.5 * np.sum((reward - probs) ** 2)
            return np.array(llhood)
        if self == LikelihoodTypes.BERNOULLI:
            llhood = np.sum(reward * np.log(probs) + (1 - reward) * np.log(1 - probs))
            return np.array(llhood)
        raise ValueError(f"Unsupported likelihood type: {self}.")


class ContextType(enum.StrEnum):
    """
    Enum for the type of context.
    """

    BINARY = "binary"
    REAL_VALUED = "real-valued"


class ContextLinkFunctions(enum.StrEnum):
    """
    Enum for the link function of the arm params and context.
    """

    NONE = "none"
    LOGISTIC = "logistic"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the link function to the input param.

        Parameters
        ----------
        x : The input param.
        """
        if self == ContextLinkFunctions.NONE:
            return x
        if self == ContextLinkFunctions.LOGISTIC:
            return np.array(1.0 / (1.0 + np.exp(-x)))
        raise ValueError(f"Unsupported link function: {self}.")
