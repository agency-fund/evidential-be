from dataclasses import dataclass

from xngin.apiserver.common_field_types import FieldName


@dataclass(slots=True)
class MetricValue:
    metric_name: FieldName
    metric_value: float | None


@dataclass(slots=True)
class ParticipantOutcome:
    participant_id: str
    metric_values: list[MetricValue]
