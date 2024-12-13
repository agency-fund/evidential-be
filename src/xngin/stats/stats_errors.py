from xngin.apiserver.api_types import DesignSpecMetric


class StatsError(Exception):
    """Wrapper for ValueErrors raised by our xngin.stats package."""

    def __init__(self, error_message: str):
        super().__init__(error_message)


class StatsPowerError(StatsError):
    """Errors that arose while doing a power calculation for a metric.

    Examples:
    - Cannot detect an effect-size of 0. Try changing your effect-size.
    """

    def __init__(self, verr: ValueError, metric: DesignSpecMetric):
        self.metric = metric
        super().__init__(f"Power calc error for metric {metric.metric_name}: {verr}")
