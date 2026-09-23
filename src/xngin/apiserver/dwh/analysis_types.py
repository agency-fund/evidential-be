from dataclasses import dataclass


@dataclass(slots=True)
class MetricValue:
    metric_name: str
    metric_value: float | None


@dataclass(slots=True)
class ParticipantOutcome:
    participant_id: str
    metric_values: list[MetricValue]
