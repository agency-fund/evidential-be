import decimal
import enum
import re
import uuid
import datetime
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
                | DataType.TIMESTAMP_WITHOUT_TIMEZONE
                | DataType.INTEGER
                | DataType.DOUBLE_PRECISION
                | DataType.NUMERIC
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
                return [
                    Relation.BETWEEN,
                    Relation.EXCLUDES,
                    Relation.INCLUDES,
                ]
        raise ValueError(f"{self} has no valid defined relations.")


class Relation(enum.StrEnum):
    """Defines operators for filtering values.

    INCLUDES matches when the value matches any of the provided values, including null if explicitly
    specified. For CSV fields (i.e. experiment_ids), any value in the CSV that matches the provided
    values will match, but nulls are unsupported. This is equivalent to NOT(EXCLUDES(values)).

    EXCLUDES matches when the value does not match any of the provided values, including null if
    explicitly specified. If null is not explicitly excluded, we include nulls in the result.  CSV
    fields (i.e. experiment_ids), the match will fail if any of the provided values are present
    in the value, but nulls are unsupported.

    BETWEEN matches when the value is between the two provided values (inclusive). Not allowed for CSV fields.
    """

    INCLUDES = "includes"
    EXCLUDES = "excludes"
    BETWEEN = "between"


type FilterValueTypes = (
    Sequence[Annotated[int, Field(strict=True)] | None]
    | Sequence[Annotated[float, Field(strict=True, allow_inf_nan=False)] | None]
    | Sequence[str | None]
    | Sequence[bool | None]
)


class AudienceSpecFilter(ApiBaseModel):
    """Defines criteria for filtering rows by value.

    ## Examples

    | Relation | Value       | logical Result                                    |
    |----------|-------------|---------------------------------------------------|
    | INCLUDES | [None]      | Match when `x IS NULL`                            |
    | INCLUDES | ["a"]       | Match when `x IN ("a")`                           |
    | INCLUDES | ["a", None] | Match when `x IS NULL OR x IN ("a")`              |
    | INCLUDES | ["a", "b"]  | Match when `x IN ("a", "b")`                      |
    | EXCLUDES | [None]      | Match `x IS NOT NULL`                             |
    | EXCLUDES | ["a", None] | Match `x IS NOT NULL AND x NOT IN ("a")`          |
    | EXCLUDES | ["a", "b"]  | Match `x IS NULL OR (x NOT IN ("a", "b"))`        |
    | BETWEEN  | ["a", "z"]  | Match `"a" <= x <= "z"`                           |
    | BETWEEN  | ["a", None] | Match `x >= "a"`                                  |

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

    ## Handling of datetime and timestamp values

    DATETIME or TIMESTAMP-type columns support INCLUDES/EXCLUDES/BETWEEN, similar to numerics.

    Values must be expressed as ISO8601 datetime strings compatible with Python's datetime.fromisoformat()
    (https://docs.python.org/3/library/datetime.html#datetime.datetime.fromisoformat).

    If a timezone is provided, it must be UTC.
    """

    field_name: FieldName
    relation: Relation
    value: FilterValueTypes

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
        elif not self.value:
            raise ValueError("value must be a non-empty list")

        return self

    @model_validator(mode="after")
    def ensure_sane_bool_list(self) -> "AudienceSpecFilter":
        """Ensures that the `value` field does not include redundant or nonsencial items."""
        n_values = len(self.value)
        # First check if we're dealing with a list of more than one boolean:
        if n_values > 1 and all([v is None or isinstance(v, bool) for v in self.value]):
            # First two technically would also catch non-bool [None, None]
            if self.relation == Relation.BETWEEN:
                raise ValueError("Values do not support BETWEEN.")
            if n_values != len(set(self.value)):
                raise ValueError("Duplicate values detected.")
            if n_values == 3 and self.relation == Relation.INCLUDES:
                raise ValueError("Boolean filter allows all possible values.")
            if n_values == 3 and self.relation == Relation.EXCLUDES:
                raise ValueError("Boolean filter rejects all possible values.")

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

    field_name: FieldName
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
    metric_baseline: Annotated[
        float | None, Field(description="Mean of the tracked metric.")
    ] = None
    metric_stddev: Annotated[
        float | None,
        Field(
            description="Standard deviation is set only for metric_type.NUMERIC metrics."
        ),
    ] = None
    available_nonnull_n: Annotated[
        int | None,
        Field(
            description="The number of participants meeting the filtering criteria with a *non-null* value for this metric."
        ),
    ] = None
    available_n: Annotated[
        int | None,
        Field(
            description="The number of participants meeting the filtering criteria regardless of whether or not this metric's value is NULL. NOTE: Assignments are made from the targeted aviailable_n population, so be sure you are ok with participants potentially having this value missing during assignment if available_n != available_nonnull_n."
        ),
    ] = None

    @model_validator(mode="after")
    def stddev_only_if_numeric(self):
        """Enforce that metric_stddev is present for NUMERICs"""
        if self.metric_type == MetricType.NUMERIC and self.metric_stddev is None:
            raise ValueError("missing stddev")
        if (
            self.metric_type is not MetricType.NUMERIC
            and self.metric_stddev is not None
        ):
            raise ValueError("should not have stddev")
        return self


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

    arm_id: Annotated[
        uuid.UUID | None,
        Field(
            description="UUID of the arm. If using the /experiments/with-assignment endpoint, this is generated for you and available in the response; you should NOT set this. Only generate ids of your own if using the stateless Experiment Design API as you will do your own persistence."
        ),
    ]
    arm_name: str  # TODO: add naming constraints
    arm_description: str | None = None


class DesignSpec(ApiBaseModel):
    """Experiment design parameters for power calculations and treatment assignment."""

    experiment_id: Annotated[
        uuid.UUID | None,
        Field(
            description="UUID of the experiment. If using the /experiments/with-assignment endpoint, this is generated for you and available in the response; you should NOT set this. Only generate ids of your own if using the stateless Experiment Design API as you will do your own persistence."
        ),
    ]
    experiment_name: str
    description: str
    start_date: datetime.datetime
    end_date: datetime.datetime

    # arms (at least two)
    arms: Annotated[list[Arm], Field(..., min_length=2)]

    # strata as strings
    # TODO? If we ever need to accept other metadata about strata, migrate to a new "strata:"
    #       field that takes a list of Stratum objects, akin to filters: and metrics:.
    strata_field_names: list[FieldName]

    metrics: Annotated[
        list[DesignSpecMetricRequest],
        Field(
            ...,
            description="Primary and optional secondary metrics to target.",
            min_length=1,
        ),
    ]

    # stat parameters
    power: Annotated[
        float,
        Field(
            0.8,
            ge=0,
            le=1,
            description="The chance of detecting a real non-null effect, i.e. 1 - false negative rate.",
        ),
    ]
    alpha: Annotated[
        float,
        Field(
            0.05,
            ge=0,
            le=1,
            description="The chance of a false positive, i.e. there is no real non-null effect, but we mistakenly think there is one.",
        ),
    ]
    fstat_thresh: Annotated[
        float,
        Field(
            0.6,
            ge=0,
            le=1,
            description='Threshold on the p-value of joint significance in doing the omnibus balance check, above which we declare the data to be "balanced".',
        ),
    ]

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime.datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()

    def uuids_are_present(self) -> bool:
        """True if the any UUIDs are present."""
        return self.experiment_id is not None or any([
            arm.arm_id is not None for arm in self.arms
        ])


class MetricAnalysisMessageType(enum.StrEnum):
    """Classifies metric analysis results."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    NO_BASELINE = "no baseline"


class MetricAnalysisMessage(ApiBaseModel):
    """Describes interpretation of analysis results."""

    type: MetricAnalysisMessageType
    msg: Annotated[
        str, Field(description="Main analysis result stated in human-friendly English.")
    ]
    source_msg: Annotated[
        str,
        Field(
            description="Analysis result formatted as a template string with curly-braced {} named placeholders. Use with the dictionary of values to support localization of messages."
        ),
    ]
    values: dict[str, float | int] | None = None


class MetricAnalysis(ApiBaseModel):
    """Describes analysis results of a single metric."""

    # Store the original request+baseline info here
    metric_spec: DesignSpecMetric

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

    target_possible: Annotated[
        float | None,
        Field(
            description="If there is an insufficient sample size to meet the desired metric_target, we report what is possible given the available_n. This value is equivalent to the relative pct_change_possible. This is None when there is a sufficient sample size to detect the desired change."
        ),
    ] = None
    pct_change_possible: Annotated[
        float | None,
        Field(
            description="If there is an insufficient sample size to meet the desired metric_pct_change, we report what is possible given the available_n. This value is equivalent to the absolute target_possible. This is None when there is a sufficient sample size to detect the desired change."
        ),
    ] = None

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

    field_name: FieldName
    # TODO(roboton): Add in strata type, update tests to reflect this field, should be derived
    # from data warehouse.
    # strata_type: Optional[StrataType]
    strata_value: str | None = None


class Assignment(ApiBaseModel):
    """Describes treatment assignment for an experiment participant."""

    # this references the field marked is_unique_id == TRUE in the configuration spreadsheet
    participant_id: str
    arm_id: Annotated[
        uuid.UUID,
        Field(
            description="UUID of the arm this participant was assigned to. Same as Arm.arm_id."
        ),
    ]
    arm_name: Annotated[
        str,
        Field(
            description="The arm this participant was assigned to. Same as Arm.arm_name."
        ),
    ]
    strata: Annotated[
        list[Strata],
        Field(
            description="List of properties and their values for this participant used for stratification or tracking metrics."
        ),
    ]


class BalanceCheck(ApiBaseModel):
    """Describes balance test results for treatment assignment."""

    f_statistic: float
    numerator_df: int
    denominator_df: int
    p_value: float
    balance_ok: bool


class AssignResponse(ApiBaseModel):
    """Describes assignments for all participants and balance test results."""

    balance_check: BalanceCheck

    experiment_id: uuid.UUID
    sample_size: int
    unique_id_field: Annotated[
        str,
        Field(
            description="Name of the datasource field used as the unique identifier for the participant_id value stored in each Assignment, as configured in the datasource settings. Included for frontend convenience."
        ),
    ]
    assignments: list[Assignment]


class GetStrataResponseElement(ApiBaseModel):
    """Describes a stratification variable."""

    data_type: DataType
    field_name: FieldName
    description: str
    # Extra fields will be stored here in case a user configured their worksheet with extra metadata for their own
    # downstream use, e.g. to group strata with a friendly identifier.
    extra: dict[str, str] | None = None


class GetFiltersResponseBase(ApiBaseModel):
    field_name: Annotated[FieldName, Field(..., description="Name of the field.")]
    data_type: DataType
    relations: list[Relation] = Field(..., min_length=1)
    description: str


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

    distinct_values: list[str] | None = Field(
        ...,
        description="Sorted list of unique values.",
    )


class GetMetricsResponseElement(ApiBaseModel):
    """Describes a metric."""

    field_name: FieldName
    data_type: DataType
    description: str


type GetFiltersResponseElement = (
    GetFiltersResponseNumericOrDate | GetFiltersResponseDiscrete
)


class GetFiltersResponse(ApiBaseModel):
    """Response model for the /filters endpoint."""

    results: list[GetFiltersResponseElement]


class GetMetricsResponse(ApiBaseModel):
    """Response model for the /metrics endpoint."""

    results: list[GetMetricsResponseElement]


class GetStrataResponse(BaseModel):
    """Response model for the /strata endpoint."""

    results: list[GetStrataResponseElement]


class AssignRequest(ApiBaseModel):
    design_spec: DesignSpec
    audience_spec: AudienceSpec


class PowerRequest(ApiBaseModel):
    design_spec: DesignSpec
    audience_spec: AudienceSpec


class PowerResponse(ApiBaseModel):
    analyses: list[MetricAnalysis]


class CommitRequest(ApiBaseModel):
    """The complete experiment configuration to persist in an experiment registry."""

    design_spec: DesignSpec
    audience_spec: AudienceSpec
    power_analyses: Annotated[
        PowerResponse | None,
        Field(
            description="Optionally include the power analyses of your tracking metrics if performed."
        ),
    ] = None
    experiment_assignment: AssignResponse
