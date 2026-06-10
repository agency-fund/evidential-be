"""Custom exceptions for statistical procedures in the xngin.stats package."""


class StatsError(Exception):
    """Wrapper for ValueErrors raised by our xngin.stats package."""

    @property
    def type(self):
        """Friendlier class of error for user-facing error callouts."""
        return self.__class__.__name__


class StatsPowerError(StatsError):
    """Errors that arose while doing a power calculation for a metric.

    Examples:
    - metric_type must be NUMERIC or BINARY.
    """

    def __init__(self, message: str):
        super().__init__(message)

    @classmethod
    def from_error(cls, verr: Exception, metric_field_name: str):
        return cls(f"Power calc error for metric {metric_field_name}: {verr}")

    @property
    def type(self):
        return "power calc"


class StatsBalanceError(StatsError):
    """Errors arising from steps in the balance check, e.g. not having usable covariates."""

    @property
    def type(self):
        return "balance check"


class StatsAssignmentError(StatsError):
    """Errors arising from steps in the assignment, e.g. invalid strata."""

    @property
    def type(self):
        return "random assignment"


class StatsAnalysisError(StatsError):
    """Errors arising from steps in the analysis, e.g. not having data."""

    @property
    def type(self):
        return "analysis"
