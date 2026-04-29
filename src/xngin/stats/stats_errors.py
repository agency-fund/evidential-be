"""Custom exceptions for statistical procedures in the xngin.stats package."""


class StatsError(Exception):
    """Wrapper for ValueErrors raised by our xngin.stats package."""


class StatsPowerError(StatsError):
    """Errors that arose while doing a power calculation for a metric.

    Examples:
    - metric_type must be NUMERIC or BINARY.
    """

    def __init__(self, verr: ValueError, metric_field_name: str):
        super().__init__(f"Power calc error for metric {metric_field_name}: {verr}")


class StatsBalanceError(StatsError):
    """Errors arising from steps in the balance check, e.g. not having usable covariates."""


class StatsAnalysisError(StatsError):
    """Errors arising from steps in the analysis, e.g. not having data."""
