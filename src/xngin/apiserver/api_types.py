import decimal
import enum
import re
import uuid
from datetime import datetime
from typing import Annotated, Self
from collections.abc import Sequence

import sqlalchemy.sql.sqltypes
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_serializer,
    ConfigDict,
    BeforeValidator,
)
from pydantic_core.core_schema import ValidationInfo

VALID_SQL_COLUMN_REGEX = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

EXPERIMENT_IDS_SUFFIX = "experiment_ids"

# An experiment is comprised of two primary components:
# 1. An AudienceSpec which defines the pool of potential participants
# 2. A DesignSpec specifying:
#   a. the treatment arms
#   b. outcome metrics,
#   c. strata
#   d. experiment meta data (start/end date, name, description)
#   e. statistical parameters (power, significance, balance test threshold)

# The goal is to produce an ExperimentAssignment from the AudienceSpec and DesignSpec.
# We go through two steps to enable this:
# 0. Baseline data retrieval -
# 1. Power analysis - Given and AudienceSpec and DesignSpec we analyze the
#    statistical power for each metric in the DesignSpec, along with the statistical
#    parameters. This occurs as follows for each metric:
#   a. If there is no baseline value (and std dev for numeric metrics), go to the
#      data source to fetch these values.
#   b. Use the declared metric_target or compute this target based on
#      metric_pct_change and the baseline value.
#   c. Use the number of arms, metric baseline and target, statistical parameters
#      and the number of participants available using the Audience filter to determine
#      if we're sufficiently powered. If we are not, compute the effect size needed to
#      be powered. This power information can be added to the metrics in the DesignSpec.
# 2. Assignment - the Power analysis computes the minimum number of participants needed
#    to be statistically powered "n" which is left to the user to choose during assignment.
#    Assignment takes the same inputs, the AudienceSpec and DesignSpec to generate a list
#    of "n" users randomly assigned using the set of treatment arms and the strata. This
#    should return a list of objects containing a participant id, treatment assignment,
#    and strata values.
# 3. Analysis - TBD


def validate_can_be_used_as_column_name(value: str, info: ValidationInfo) -> str:
    """Validates value is usable as a SQL column name."""
    if not isinstance(value, str):
        raise ValueError(f"{info.field_name} must be a string")  # noqa: TRY004
    if not re.match(VALID_SQL_COLUMN_REGEX, value):
        raise ValueError(
            f"{info.field_name} must start with letter/underscore and contain only letters, numbers, underscores"
        )
    return value


FieldName = Annotated[
    str,
    BeforeValidator(validate_can_be_used_as_column_name),
    Field(
        json_schema_extra={"pattern": VALID_SQL_COLUMN_REGEX}, examples=["field_name"]
    ),
]


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataType(enum.StrEnum):
    """Defines the supported data types for fields in the data source."""

    BOOLEAN = "boolean"
    CHARACTER_VARYING = "character varying"
    DATE = "date"
    INTEGER = "integer"
    DOUBLE_PRECISION = "double precision"
    NUMERIC = "numeric"
    TIMESTAMP_WITHOUT_TIMEZONE = "timestamp without time zone"
    BIGINT = "bigint"

    @classmethod
    def match(cls, value):
        """Maps a Python or SQLAlchemy type or value's type to the corresponding DataType.

        Value may be a Python type or a SQLAlchemy type.
        """
        if value in DataType:
            return DataType[value]
        if value is str:
            return DataType.CHARACTER_VARYING
        if isinstance(value, sqlalchemy.sql.sqltypes.String):
            return DataType.CHARACTER_VARYING
        if isinstance(value, sqlalchemy.sql.sqltypes.Boolean):
            return DataType.BOOLEAN
        if isinstance(value, sqlalchemy.sql.sqltypes.BigInteger):
            return DataType.BIGINT
        if isinstance(value, sqlalchemy.sql.sqltypes.Integer):
            return DataType.INTEGER
        if isinstance(value, sqlalchemy.sql.sqltypes.Double):
            return DataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.Float):
            return DataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.Numeric):
            return DataType.NUMERIC
        if isinstance(value, sqlalchemy.sql.sqltypes.Date):
            return DataType.DATE
        if isinstance(value, sqlalchemy.sql.sqltypes.DateTime):
            return DataType.TIMESTAMP_WITHOUT_TIMEZONE
        if value is int:
            return DataType.INTEGER
        if value is float:
            return DataType.DOUBLE_PRECISION
        raise ValueError(f"Unmatched type: {value}.")

    def filter_class(self, field_name):
        """Classifies a DataType into a filter class."""
        match self:
            # TODO: is this customer specific?
            case _ if field_name.lower().endswith("_id"):
                return DataTypeClass.DISCRETE
            case DataType.BOOLEAN | DataType.CHARACTER_VARYING:
                return DataTypeClass.DISCRETE
            case (
                DataType.DATE
                | DataType.INTEGER
                | DataType.DOUBLE_PRECISION
                | DataType.NUMERIC
                | DataType.TIMESTAMP_WITHOUT_TIMEZONE
                | DataType.BIGINT
            ):
                return DataTypeClass.NUMERIC
            case _:
                return DataTypeClass.UNKNOWN


class DataTypeClass(enum.StrEnum):
    DISCRETE = "discrete"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"

    def valid_relations(self):
        """Gets the valid relation operators for this data type class."""
        match self:
            case DataTypeClass.DISCRETE:
                return [Relation.INCLUDES, Relation.EXCLUDES]
            case DataTypeClass.NUMERIC:
                return [Relation.BETWEEN]
        raise ValueError(f"{self} has no valid defined relations.")


class Relation(enum.StrEnum):
    """Defines operators for filtering values.

    INCLUDES matches when the value matches any of the provided values. For CSV fields
    (i.e. experiment_ids), any value in the CSV that matches the provided values will match.

    EXCLUDES matches when the value does not match any of the provided values. For CSV fields
    (i.e. experiment_ids), the match will fail if any of the provided values are present in the value.

    BETWEEN matches when the value is between the two provided values. Not allowed for CSV fields.
    """

    INCLUDES = "includes"
    EXCLUDES = "excludes"
    BETWEEN = "between"


class AudienceSpecFilter(ApiBaseModel):
    """Defines criteria for filtering rows by value.

    ## Examples

    | Relation | Value      | Result                       |
    |----------|------------|------------------------------|
    | INCLUDES | ["a"]      | Match when `x IN ("a")`      |
    | INCLUDES | ["a", "b"] | Match when `x IN ("a", "b")` |
    | EXCLUDES | ["a","b"]  | Match `x NOT IN ("a", "b")`  |

    String comparisons are case-sensitive.

    ## Special Handling for Comma-Separated Fields

    When the filter name ends in "experiment_ids", the filter is interpreted as follows:

    | Value | Filter         | Result   |
    |-------|----------------|----------|
    | "a,b" | INCLUDES ["a"] | Match    |
    | "a,b" | INCLUDES ["d"] | No match |
    | "a,b" | EXCLUDES ["d"] | Match    |
    | "a,b" | EXCLUDES ["b"] | No match |

    Note: The BETWEEN relation is not supported for comma-separated values.

    Note: CSV field comparisons are case-insensitive.
    """

    field_name: FieldName
    relation: Relation
    value: (
        Sequence[Annotated[int, Field(strict=True)] | None]
        | Sequence[Annotated[float, Field(strict=True, allow_inf_nan=False)] | None]
        | Sequence[str | None]
    )

    @model_validator(mode="after")
    def ensure_experiment_ids_hack_compatible(self) -> "AudienceSpecFilter":
        """Ensures that the filter is compatible with the "experiment_ids" hack."""
        if not self.field_name.endswith(EXPERIMENT_IDS_SUFFIX):
            return self
        allowed_relations = (Relation.INCLUDES, Relation.EXCLUDES)
        if self.relation not in allowed_relations:
            raise ValueError(
                f"filters on experiment_id fields must have relations of type {', '.join(sorted(allowed_relations))}"
            )
        for v in self.value:
            if not isinstance(v, str):
                continue
            if "," in v:
                raise ValueError(
                    "values in an experiment_id filter may not contain commas"
                )
            if v.strip() != v:
                raise ValueError(
                    "values in an experiment_id filter may not contain leading or trailing whitespace"
                )
        return self

    @model_validator(mode="after")
    def ensure_value(self) -> "AudienceSpecFilter":
        """Ensures that the `value` field is an unambiguous filter and correct for the relation.

        Note this happens /after/ Pydantic does its type coercion, so we control some of the
        built-in type coercion using the strict=True annotations on the value field. There
        are probably some bugs in this.
        """
        if self.relation == Relation.BETWEEN:
            if len(self.value) != 2:
                raise ValueError("BETWEEN relation requires exactly 2 values")

            none_count = sum(1 for v in self.value if v is None)
            if none_count > 1:
                raise ValueError("BETWEEN relation can have at most one None value")
            if none_count == 0 and type(self.value[0]) is not type(self.value[1]):
                raise ValueError(
                    "BETWEEN relation requires same values to be of the same type"
                )
        else:
            if not self.value:
                raise ValueError("value must be a non-empty list")

        return self


class AudienceSpec(ApiBaseModel):
    """Defines target participants for an experiment using filters."""

    participant_type: str
    filters: list[AudienceSpecFilter]


class MetricType(enum.StrEnum):
    """Classifies metrics by their value type."""

    BINARY = "binary"
    NUMERIC = "numeric"

    @classmethod
    def from_python_type(cls, python_type: type) -> "MetricType":
        """Maps Python types to metric types."""

        if python_type in (int, float, decimal.Decimal):
            return MetricType.NUMERIC
        if python_type is bool:
            return MetricType.BINARY
        raise ValueError(f"Unsupported type: {python_type}")


class DesignSpecMetricBase(ApiBaseModel):
    """Base class for defining a metric to measure in the experiment."""

    metric_name: FieldName
    metric_pct_change: Annotated[
        float | None,
        Field(description="Percent change target relative to the metric_baseline."),
    ] = None
    metric_target: Annotated[
        float | None,
        Field(
            description="Absolute target value = metric_baseline*(1 + metric_pct_change)"
        ),
    ] = None


class DesignSpecMetric(DesignSpecMetricBase):
    """Defines a metric to measure in an experiment with its baseline stats."""

    metric_type: Annotated[
        MetricType | None, Field(description="Inferred from dwh type.")
    ] = None
    metric_baseline: float | None = None
    # TODO(roboton): we should only set this value if metric_type is NUMERIC
    metric_stddev: float | None = None
    available_n: int | None = None


class DesignSpecMetricRequest(DesignSpecMetricBase):
    """Defines a request to look up baseline stats for a metric to measure in an experiment."""

    # TODO: consider supporting {metric_baseline, metric_stddev, available_n} as inputs when the metric may not exist or
    # be usable yet in the dwh, so that it it can be used as a general power/sizing calculator.

    # Override the descriptions from above:
    metric_pct_change: Annotated[
        float | None,
        Field(
            description="Specify a meaningful min percent change relative to the metric_baseline "
            "you want to detect. Cannot be set if you set metric_target."
        ),
    ] = None
    metric_target: Annotated[
        float | None,
        Field(
            description="Specify the absolute value you want to detect. "
            "Cannot be set if you set metric_pct_change."
        ),
    ] = None

    @model_validator(mode="after")
    def check_has_only_one_of_pct_change_or_target(self) -> Self:
        if self.metric_pct_change is not None and self.metric_target is not None:
            raise ValueError("Cannot set both metric_pct_change and metric_target")
        if self.metric_pct_change is None and self.metric_target is None:
            raise ValueError("Must set one of metric_pct_change or metric_target")
        return self


class Arm(ApiBaseModel):
    """Describes an experiment treatment arm."""

    # generally should not let users set this, auto-generated uuid by default
    arm_id: uuid.UUID
    arm_name: str
    arm_description: str | None = None


class DesignSpecBase(ApiBaseModel):
    """Describes the experiment design parameters excluding metrics."""

    experiment_id: uuid.UUID
    experiment_name: str
    description: str
    start_date: datetime
    end_date: datetime

    # arms (at least two)
    arms: Annotated[list[Arm], Field(..., min_length=2)]

    # strata as strings
    # TODO(roboton): rename as strata_names?
    strata_cols: list[FieldName]

    # stat parameters
    power: Annotated[float, Field(0.8, ge=0, le=1)]
    alpha: Annotated[float, Field(0.05, ge=0, le=1)]
    fstat_thresh: Annotated[float, Field(0.6, ge=0, le=1)]

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


# TODO? Consider making this the one and only DesignSpec model, and if the user wants to store DesignSpecMetric details,
# it should be done as part of storing PowerResponse in the CommitRequest, rather than assuming the user will fish out
# the DesignSpecMetric details from the response just to put them back into the original DesignSpecForPower to create a
# DesignSpec.
class DesignSpecForPower(DesignSpecBase):
    """Experiment design parameters for power calculations."""

    metrics: Annotated[
        list[DesignSpecMetricRequest],
        Field(
            ...,
            description="Primary and optional secondary metrics to target.",
            min_length=1,
        ),
    ]


class DesignSpec(DesignSpecBase):
    """Describes the experiment design parameters."""

    metrics: Annotated[list[DesignSpecMetric], Field(..., min_length=1)]


class MetricAnalysisMessageType(enum.StrEnum):
    """Classifies metric analysis results."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class MetricAnalysisMessage(ApiBaseModel):
    """Describes interpretation of analysis results."""

    type: MetricAnalysisMessageType
    msg: str
    values: dict[str, float | int] | None = None


class MetricAnalysis(ApiBaseModel):
    """Describes analysis results of a single metric."""

    # Store the original request+baseline info here
    metric_spec: DesignSpecMetric
    # TODO: Remove available_n as it's redundant with the metric_spec.
    available_n: int

    # The initial result of the power calculation
    target_n: Annotated[
        int | None,
        Field(description="Minimum sample size needed to meet the design specs."),
    ] = None
    sufficient_n: Annotated[
        bool | None,
        Field(
            description="Whether or not there are enough available units to sample from to meet target_n."
        ),
    ] = None

    # If insufficient sample size, tell the user what metric value their n does let them possibly detect as an absolute
    # value and % change from baseline.
    # TODO? Rename target_possible and pct_change_possible
    needed_target: float | None = None
    # TODO: add compute the equivalent % change
    # metric_pct_change_possible: float | None = None

    msg: Annotated[
        MetricAnalysisMessage | None,
        Field(description="Human friendly message about the above results."),
    ] = None


class StrataType(enum.StrEnum):
    """Classifies strata by their value type."""

    BINARY = "binary"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"

    @classmethod
    def from_python_type(cls, python_type: type):
        """ "Maps Python types to strata types."""

        if python_type in (int, float):
            return StrataType.NUMERIC
        if python_type is bool:
            return StrataType.BINARY
        if python_type is str:
            return StrataType.CATEGORICAL

        raise ValueError(f"Unsupported type: {python_type}")


class Strata(ApiBaseModel):
    """Describes stratification for an experiment participant."""

    strata_name: FieldName
    # TODO(roboton): Add in strata type, update tests to reflect this field, should be derived
    # from data warehouse.
    # strata_type: Optional[StrataType]
    strata_value: str | None = None


class Assignment(ApiBaseModel):
    """Describes treatment assignment for an experiment participant."""

    # this references the column marked is_unique_id == TRUE in the configuration spreadsheet
    participant_id: str
    treatment_assignment: str
    strata: list[Strata]


class BalanceCheck(ApiBaseModel):
    """Describes balance test results for treatment assignment."""

    f_stat: float
    numerator_df: int
    denominator_df: int
    p_value: float
    balance_ok: bool


class AssignResponse(ApiBaseModel):
    """Describes assignments for all participants and balance test results."""

    # TODO(roboton): remove next 5 fields in favor of BalanceCheck object
    f_statistic: float
    numerator_df: int
    denominator_df: int
    p_value: float
    balance_ok: bool

    # TODO(roboton): should we include design_spec and audience_spec in this object
    experiment_id: uuid.UUID
    # TODO(roboton): drop description since it will be in included design_spec
    description: str
    sample_size: int
    id_col: str
    assignments: list[Assignment]


class GetStrataResponseElement(ApiBaseModel):
    """Describes a stratification variable."""

    data_type: DataType
    column_name: FieldName
    description: str
    # Extra fields will be stored here in case a user configured their worksheet with extra metadata for their own
    # downstream use, e.g. to group strata with a friendly identifier.
    extra: dict[str, str] | None = None


class GetFiltersResponseBase(ApiBaseModel):
    field_name: Annotated[FieldName, Field(..., description="Name of the field.")]
    data_type: DataType
    relations: list[Relation] = Field(..., min_length=1)
    description: str


class GetFiltersResponseNumeric(GetFiltersResponseBase):
    """Describes a numeric filter variable."""

    min: float | int | None = Field(
        ...,
        description="The minimum observed value.",
    )
    max: float | int | None = Field(
        ...,
        description="The maximum observed value.",
    )


class GetFiltersResponseDiscrete(GetFiltersResponseBase):
    """Describes a discrete filter variable."""

    distinct_values: list[str] | None = Field(
        ...,
        description="Sorted list of unique values.",
    )


class GetMetricsResponseElement(ApiBaseModel):
    """Describes a metric."""

    column_name: FieldName
    data_type: DataType
    description: str


type GetFiltersResponseElement = GetFiltersResponseNumeric | GetFiltersResponseDiscrete
type GetFiltersResponse = list[GetFiltersResponseElement]
type GetMetricsResponse = list[GetMetricsResponseElement]
type GetStrataResponse = list[GetStrataResponseElement]
type PowerResponse = list[MetricAnalysis]


class AssignRequest(ApiBaseModel):
    design_spec: DesignSpec
    audience_spec: AudienceSpec


class CommitRequest(ApiBaseModel):
    design_spec: DesignSpec
    audience_spec: AudienceSpec
    experiment_assignment: AssignResponse


class PowerRequest(ApiBaseModel):
    design_spec: DesignSpecForPower
    audience_spec: AudienceSpec
