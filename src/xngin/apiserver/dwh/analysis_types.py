from typing import Annotated

from pydantic import BaseModel, Field

from xngin.apiserver.common_field_types import FieldName
from xngin.apiserver.limits import (
    MAX_LENGTH_OF_PARTICIPANT_ID_VALUE,
    MAX_NUMBER_OF_FIELDS,
)


class MetricValue(BaseModel):
    metric_name: Annotated[
        FieldName,
        Field(
            description="The field_name from the datasource which this analysis models as the dependent variable (y)."
        ),
    ]
    metric_value: Annotated[float | None, Field(description="The queried value for this field_name.")]


class ParticipantOutcome(BaseModel):
    participant_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_PARTICIPANT_ID_VALUE)]
    metric_values: Annotated[list[MetricValue], Field(max_length=MAX_NUMBER_OF_FIELDS)]
