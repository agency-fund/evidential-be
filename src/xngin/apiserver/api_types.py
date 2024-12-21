import enum
import re
import uuid
from datetime import datetime
from typing import Annotated

import sqlalchemy.sql.sqltypes
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    field_serializer,
    ConfigDict,
)

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


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataType(enum.StrEnum):
    """Defines the supported data types for columns in the data source."""

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

    def filter_class(self, column_name):
        """Classifies a DataType into a filter class."""
        match self:
            # TODO: is this customer specific?
            case _ if column_name.lower().endswith("_id"):
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

    # TODO(qixotic): rename this to column_name?
    filter_name: str
    relation: Relation
    value: (
        list[Annotated[int, Field(strict=True)] | None]
        | list[Annotated[float, Field(strict=True, allow_inf_nan=False)] | None]
        | list[str | None]
    )

    @field_validator("filter_name")
    @classmethod
    def ensure_filter_name_is_sql_compatible(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError("filter_name must be a valid SQL column name")
        return v

    @model_validator(mode="after")
    def ensure_experiment_ids_hack_compatible(self) -> "AudienceSpecFilter":
        """Ensures that the filter is compatible with the "experiment_ids" hack."""
        if not self.filter_name.endswith(EXPERIMENT_IDS_SUFFIX):
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

        if python_type in (int, float):
            return MetricType.NUMERIC
        if python_type is bool:
            return MetricType.BINARY
        raise ValueError(f"Unsupported type: {python_type}")


class DesignSpecMetric(ApiBaseModel):
    """Defines a metric to measure in the experiment."""

    metric_name: str
    # TODO(roboton): metric_type should be inferred by name from db when missing
    metric_type: MetricType | None = None
    # TODO(roboton): metric_baseline should be drawn from dwh when missing
    metric_baseline: float | None = None
    # TODO(roboton): we should only set this value if metric_type is NUMERIC
    metric_stddev: float | None = None
    # TOOD(roboton): if target is set, metric_pct_change is ignored, but we
    # should display a warning
    metric_pct_change: float | None = None
    # TODO(roboton): metric_target will be computed from metric_baseline and
    # TODO(roboton): metric_pct_change if missing
    # TODO(roboton): metric_target = 1 + metric_pct_change * metric_baseline
    metric_target: float | None = None
    # TODO(roboton): available_n should probably be in another structure related to power_analysis?
    available_n: int | None = None


class Arm(ApiBaseModel):
    """Describes an experiment treatment arm."""

    # generally should not let users set this, auto-generated uuid by default
    arm_id: uuid.UUID
    arm_name: str
    arm_description: str | None = None


class DesignSpec(ApiBaseModel):
    """Describes the experiment design parameters."""

    experiment_id: uuid.UUID
    experiment_name: str
    description: str
    start_date: datetime
    end_date: datetime

    # arms (at least two)
    arms: list[Arm]

    # strata as strings
    # TODO(roboton): rename as strata_names?
    strata_cols: list[str]

    # metric specs (at least one)
    metrics: list[DesignSpecMetric]

    # stat parameters
    power: float = 0.8
    alpha: float = 0.05
    fstat_thresh: float = 0.6

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()

    @field_validator("power", "alpha", "fstat_thresh")
    @classmethod
    def check_values_between_0_and_1(cls, value, field):
        """Ensure that power, alpha, and fstat_thresh are between 0 and 1."""
        if not (0 <= value <= 1):
            raise ValueError(f"{field.name} must be between 0 and 1.")
        return value

    @field_validator("arms")
    @classmethod
    def check_arms_length(cls, value):
        """Ensure that arms list has at least two elements."""
        if len(value) < 2:
            raise ValueError("The arms list must contain at least two elements.")
        return value

    @field_validator("metrics")
    @classmethod
    def check_metrics_length(cls, value):
        """Ensure that metrics list has at least one element."""
        if len(value) < 1:
            raise ValueError("The metrics list must contain at least one element.")
        return value


type PowerAnalysis = list[MetricAnalysis]


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

    metric_spec: DesignSpecMetric
    available_n: int
    target_n: int | None = None
    sufficient_n: bool | None = None
    needed_target: float | None = None
    metric_target_possible: float | None = None
    metric_pct_change_possible: float | None = None
    delta: float | None = None
    msg: MetricAnalysisMessage | None = None


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


class ExperimentStrata(ApiBaseModel):
    """Describes stratification for an experiment participant."""

    strata_name: str
    # TODO(roboton): Add in strata type, update tests to reflect this field, should be derived
    # from data warehouse.
    # strata_type: Optional[StrataType]
    strata_value: str | None = None


class ExperimentParticipant(ApiBaseModel):
    """Describes treatment assignment for an experiment participant."""

    # this references the column marked is_unique_id == TRUE in the configuration spreadsheet
    participant_id: str
    treatment_assignment: str
    strata: list[ExperimentStrata]


class BalanceCheck(ApiBaseModel):
    """Describes balance test results for treatment assignment."""

    f_stat: float
    numerator_df: int
    denominator_df: int
    p_value: float
    balance_ok: bool


class ExperimentAssignment(ApiBaseModel):
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
    assignments: list[ExperimentParticipant]


class GetStrataResponseElement(ApiBaseModel):
    """Describes a stratification variable."""

    data_type: DataType
    column_name: str
    description: str
    # Extra fields will be stored here in case a user configured their worksheet with extra metadata for their own
    # downstream use, e.g. to group strata with a friendly identifier.
    extra: dict[str, str] | None = None


class GetFiltersResponseBase(ApiBaseModel):
    # TODO: Can we rename this to column_name for consistency with GetStrataResponseElement?
    filter_name: str = Field(..., description="Name of the column.")
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


type GetFiltersResponseElement = GetFiltersResponseNumeric | GetFiltersResponseDiscrete


class GetMetricsResponseElement(ApiBaseModel):
    """Describes a metric."""

    column_name: str
    data_type: DataType
    description: str
