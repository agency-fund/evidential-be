from xngin.apiserver.stateless_api_types import DesignSpecMetric


class StatsError(Exception):
    """Wrapper for ValueErrors raised by our xngin.stats package."""


class StatsPowerError(StatsError):
    """Errors that arose while doing a power calculation for a metric.

    Examples:
    - metric_type must be NUMERIC or BINARY.
    """

    def __init__(self, verr: ValueError, metric: DesignSpecMetric):
        self.metric = metric
        super().__init__(f"Power calc error for metric {metric.field_name}: {verr}")


class StatsBalanceError(StatsError):
    """Errors arising from steps in the balance check, e.g. not having usable covariates."""
