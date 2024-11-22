import enum
import re
import uuid
from datetime import datetime
from typing import Literal, Annotated, Self

import sqlalchemy.sql.sqltypes
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    field_serializer,
)

EXPERIMENT_IDS_SUFFIX = "experiment_ids"


class DataType(enum.StrEnum):
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
        """Attempt to infer the appropriate DataType for a value.

        Value may be a Python type or a SQLAlchemy type.
        """
        if value in DataType:
            return DataType[value]
        if value is str:
            return DataType.CHARACTER_VARYING
        if value is int:
            return DataType.INTEGER
        if value is float:
            return DataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.String):
            return DataType.CHARACTER_VARYING
        if isinstance(value, sqlalchemy.sql.sqltypes.Integer):
            return DataType.INTEGER
        if isinstance(value, sqlalchemy.sql.sqltypes.Float):
            return DataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.Boolean):
            return DataType.BOOLEAN
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
        match self:
            case DataTypeClass.DISCRETE:
                return [Relation.INCLUDES, Relation.EXCLUDES]
            case DataTypeClass.NUMERIC:
                return [Relation.BETWEEN]
        raise ValueError(f"{self} has no valid defined relations..")


class Relation(enum.StrEnum):
    """Relation defines the operator to apply in this filter.

    INCLUDES matches when the database value matches any of the provided values. For CSV fields
    (i.e. experiment_ids), any value in the CSV that matches the provided values will match.

    EXCLUDES matches when the database value does not match any of the provided values. For CSV fields
    (i.e. experiment_ids), the match will fail if any of the provided values are present in the database value.

    BETWEEN matches when the database value is between the two provided values. Not allowed for CSV fields.
    """

    INCLUDES = "includes"
    EXCLUDES = "excludes"
    BETWEEN = "between"


class AudienceSpecFilter(BaseModel):
    """Defines a filter on the rows in the database.

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


class AudienceSpec(BaseModel):
    """Audience specification."""

    participant_type: str
    filters: list[AudienceSpecFilter]


class DesignSpecArm(BaseModel):
    arm_name: str
    arm_id: uuid.UUID


class DesignSpecMetric(BaseModel):
    metric_name: str
    metric_pct_change: float


class DesignSpec(BaseModel):
    """Design specification."""

    experiment_id: uuid.UUID
    experiment_name: str
    description: str
    arms: list[DesignSpecArm]
    start_date: datetime
    end_date: datetime
    strata_cols: list[str]
    power: float
    alpha: float
    fstat_thresh: float
    metrics: list[DesignSpecMetric]

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


class ExperimentStrata(BaseModel):
    strata_name: str
    strata_value: str


class ExperimentAssignmentUnit(BaseModel):
    # Name of the experiment arm this unit was assigned to
    treatment_assignment: str
    strata: list[ExperimentStrata]

    # Allow extra fields so that we can support dwh-specific ids
    model_config = {"extra": "allow"}
    # And require the extra field to be an integer;
    # can loosen this if string ids are used in the future
    __pydantic_extra__: dict[str, int] = Field(init=False)

    @model_validator(mode="after")
    def validate_single_id_field(self) -> Self:
        num_extra = len(self.__pydantic_extra__)
        if num_extra != 1:
            raise ValueError(f"Model must have exactly one id field. Found {num_extra}")
        return self


class ExperimentAssignment(BaseModel):
    """Experiment assignment details including balance statistics and group assignments."""

    f_stat: float = Field(alias="f.stat")
    numerator_df: int = Field(alias="numerator.df")
    denominator_df: int = Field(alias="denominator.df")
    p_value: float = Field(alias="p.value")
    balance_ok: bool
    experiment_id: uuid.UUID
    description: str
    sample_size: int
    assignments: list[ExperimentAssignmentUnit]


class UnimplementedResponse(BaseModel):
    todo: Literal["TODO"] = "TODO"


class GetStrataResponseElement(BaseModel):
    data_type: DataType
    column_name: str
    description: str
    strata_group: str


class GetFiltersResponseElement(BaseModel):
    data_type: DataType
    description: str
    distinct_values: list[str] | None = Field(
        ...,
        description="If the type of the column is non-numeric, contains sorted list of unique values.",
    )
    min: float | int | None = Field(
        ...,
        description="If the type of the column is numeric, this will contain the minimum observed value.",
    )
    max: float | int | None = Field(
        ...,
        description="If the type of the column is numeric, this will contain the maximum observed value.",
    )
    # TODO: Can we rename this to column_name for consistency with GetStrataResponseElement?
    filter_name: str = Field(..., description="Name of the column.")
    relations: list[Relation] = Field(..., min_length=1)


class GetMetricsResponseElement(BaseModel):
    data_type: DataType
    column_name: str
    description: str


type GetPowerResponse = list[GetPowerResponseElement]


class MetricType(enum.StrEnum):
    BINARY = "binary"
    CHARACTER = "character"
    CONTINUOUS = "continuous"

    @classmethod
    def from_python_type(cls, python_type: type) -> "MetricType":
        """Given a Python type, return an appropriate MetricType."""

        if python_type is str:
            return MetricType.CHARACTER
        if python_type in (int, float):
            return MetricType.CONTINUOUS
        if python_type is bool:
            return MetricType.BINARY
        raise ValueError(f"Unsupported type: {python_type}")


class Stats(BaseModel):
    mean: float
    stddev: float
    available_n: int


class GetPowerResponseElement(BaseModel):
    """Response for the /power endpoint."""

    metric_name: str
    metric_pct_change: float
    metric_type: MetricType
    stats: Stats
    metric_target: float
    target_n: int
    sufficient_n: bool
    msg: str  # TODO: replace with structured message
    needed_target: float | None = None
