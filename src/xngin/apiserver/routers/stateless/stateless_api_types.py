import datetime
import enum
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
)

from xngin.apiserver.common_field_types import FieldName
from xngin.apiserver.limits import (
    MAX_LENGTH_OF_DESCRIPTION_VALUE,
    MAX_LENGTH_OF_PARTICIPANT_ID_VALUE,
    MAX_NUMBER_OF_FIELDS,
    MAX_NUMBER_OF_FILTERS,
)
from xngin.apiserver.routers.common_api_types import (
    ApiBaseModel,
    Assignment,
    BalanceCheck,
    DataType,
    DesignSpec,
    GetMetricsResponseElement,
    GetStrataResponseElement,
    PowerResponse,
    Relation,
)

VALID_SQL_COLUMN_REGEX = r"^[a-zA-Z_][a-zA-Z0-9_]*$"


class StrataType(enum.StrEnum):
    """Classifies strata by their value type."""

    BINARY = "binary"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"

    @classmethod
    def from_python_type(cls, python_type: type):
        """ "Maps Python types to strata types."""

        if python_type in {int, float}:
            return StrataType.NUMERIC
        if python_type is bool:
            return StrataType.BINARY
        if python_type is str:
            return StrataType.CATEGORICAL

        raise ValueError(f"Unsupported type: {python_type}")


class MetricValue(ApiBaseModel):
    metric_name: Annotated[
        FieldName,
        Field(
            description="The field_name from the datasource which this analysis models as the dependent variable (y)."
        ),
    ]
    metric_value: Annotated[
        float | None, Field(description="The queried value for this field_name.")
    ]


class ParticipantOutcome(ApiBaseModel):
    participant_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_PARTICIPANT_ID_VALUE)]
    metric_values: Annotated[list[MetricValue], Field(max_length=MAX_NUMBER_OF_FIELDS)]


class AssignResponse(ApiBaseModel):
    """Describes assignments for all participants and balance test results."""

    balance_check: Annotated[
        BalanceCheck | None,
        Field(
            description="Result of checking that the arms are balanced. May not be present if we are not able to stratify on any design metrics or other fields specified for stratification. (Fields used must be supported data types whose values are NOT all unique or all the same)."
        ),
    ] = None

    experiment_id: str
    sample_size: Annotated[
        int,
        Field(description="The number of participants across all arms in total."),
    ]
    unique_id_field: Annotated[
        str,
        Field(
            description="Name of the datasource field used as the unique identifier for the participant_id value stored in each Assignment, as configured in the datasource settings. Included for frontend convenience."
        ),
    ]
    # TODO(qixotic): Consider lifting up Assignment.arm_id & arm_name to the AssignResponse level
    # and organize assignments into lists by arm. Be less bulky and arm sizes come naturally.
    assignments: Annotated[list[Assignment], Field()]


class AnalysisRequest(ApiBaseModel):
    design: DesignSpec
    assignment: AssignResponse


class GetFiltersResponseBase(ApiBaseModel):
    field_name: Annotated[FieldName, Field(..., description="Name of the field.")]
    data_type: DataType
    relations: Annotated[
        list[Relation], Field(..., min_length=1, max_length=MAX_NUMBER_OF_FILTERS)
    ]
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]


class GetFiltersResponseNumericOrDate(GetFiltersResponseBase):
    """Describes a numeric or date filter variable."""

    min: datetime.datetime | datetime.date | float | int | None = Field(
        ...,
        description="The minimum observed value.",
    )
    max: datetime.datetime | datetime.date | float | int | None = Field(
        ...,
        description="The maximum observed value.",
    )


class GetFiltersResponseDiscrete(GetFiltersResponseBase):
    """Describes a discrete filter variable."""

    distinct_values: Annotated[
        list[str] | None, Field(..., description="Sorted list of unique values.")
    ]


type GetFiltersResponseElement = (
    GetFiltersResponseNumericOrDate | GetFiltersResponseDiscrete
)


class GetFiltersResponse(ApiBaseModel):
    """Response model for the /filters endpoint."""

    results: list[GetFiltersResponseElement]


class GetMetricsResponse(ApiBaseModel):
    """Response model for the /metrics endpoint."""

    results: Annotated[list[GetMetricsResponseElement], Field()]


class GetStrataResponse(BaseModel):
    """Response model for the /strata endpoint."""

    results: Annotated[list[GetStrataResponseElement], Field()]


class AssignRequest(ApiBaseModel):
    design_spec: DesignSpec


class CommitRequest(ApiBaseModel):
    """The complete experiment configuration to persist in an experiment registry."""

    design_spec: DesignSpec
    power_analyses: Annotated[
        PowerResponse | None,
        Field(
            description="Optionally include the power analyses of your tracking metrics if performed."
        ),
    ] = None
    experiment_assignment: AssignResponse
