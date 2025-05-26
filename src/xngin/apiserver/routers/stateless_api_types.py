import datetime
import decimal
import enum
import math
import uuid
from collections.abc import Sequence
from typing import Annotated, Literal, Self, get_args

import sqlalchemy.sql.sqltypes
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from xngin.apiserver.common_field_types import FieldName
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.limits import (
    MAX_LENGTH_OF_DESCRIPTION_VALUE,
    MAX_LENGTH_OF_NAME_VALUE,
    MAX_LENGTH_OF_PARTICIPANT_ID_VALUE,
    MAX_NUMBER_OF_ARMS,
    MAX_NUMBER_OF_FIELDS,
    MAX_NUMBER_OF_FILTERS,
)

VALID_SQL_COLUMN_REGEX = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

EXPERIMENT_IDS_SUFFIX = "experiment_ids"


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataType(enum.StrEnum):
    """Defines the supported data types for fields in the data source."""

    BOOLEAN = "boolean"
    CHARACTER_VARYING = "character varying"
    UUID = "uuid"
    DATE = "date"
    INTEGER = "integer"
    DOUBLE_PRECISION = "double precision"
    NUMERIC = "numeric"
    TIMESTAMP_WITHOUT_TIMEZONE = "timestamp without time zone"
    TIMESTAMP_WITH_TIMEZONE = "timestamp with time zone"
    BIGINT = "bigint"
    JSONB = "jsonb (unsupported)"
    JSON = "json (unsupported)"
    UNKNOWN = "unsupported"
    # NOTE: If adding types, the frontend (e.g. data-type-badge.tsx when viewing participant type
    # details) should also be updated to badge appropriately in the UI.

    @classmethod
    def match(cls, value):
        """Maps a Python or SQLAlchemy type or value's type to the corresponding DataType.

        Value may be a Python type or a SQLAlchemy type.
        """
        if value in DataType:
            return DataType[value]
        if value is str:
            return DataType.CHARACTER_VARYING
        if isinstance(value, sqlalchemy.sql.sqltypes.UUID):
            return DataType.UUID
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
        if isinstance(value, sqlalchemy.sql.sqltypes.DateTime) and value.timezone:
            return DataType.TIMESTAMP_WITH_TIMEZONE
        if isinstance(value, sqlalchemy.sql.sqltypes.DateTime) and not value.timezone:
            return DataType.TIMESTAMP_WITHOUT_TIMEZONE
        if isinstance(value, sqlalchemy.dialects.postgresql.json.JSONB):
            return DataType.JSONB
        if isinstance(value, sqlalchemy.dialects.postgresql.json.JSON):
            return DataType.JSON
        if value is int:
            return DataType.INTEGER
        if value is float:
            return DataType.DOUBLE_PRECISION
        logger.warning("Unmatched type: {}", type(value))
        return DataType.UNKNOWN

    @classmethod
    def supported_participant_id_types(cls) -> list["DataType"]:
        """Returns the list of data types that are supported as participant IDs."""
        return [
            DataType.INTEGER,
            DataType.BIGINT,
            DataType.UUID,
            DataType.CHARACTER_VARYING,
        ]

    @classmethod
    def is_supported_type(cls, data_type: Self):
        """Returns True if the type is supported as a strata, filter, and/or metric."""
        return data_type not in {DataType.JSONB, DataType.JSON, DataType.UNKNOWN}

    def is_supported(self):
        """Returns True if the type is supported as a strata, filter, and/or metric."""
        return DataType.is_supported_type(self)

    def filter_class(self, field_name):
        """Classifies a DataType into a filter class."""
        match self:
            # TODO: is this customer specific?
            case _ if field_name.lower().endswith("_id"):
                return FilterClass.DISCRETE
            case DataType.BOOLEAN | DataType.CHARACTER_VARYING | DataType.UUID:
                return FilterClass.DISCRETE
            case (
                DataType.DATE
                | DataType.TIMESTAMP_WITHOUT_TIMEZONE
                | DataType.TIMESTAMP_WITH_TIMEZONE
                | DataType.INTEGER
                | DataType.DOUBLE_PRECISION
                | DataType.NUMERIC
                | DataType.BIGINT
            ):
                return FilterClass.NUMERIC
            case _:
                return FilterClass.UNKNOWN


class FilterClass(enum.StrEnum):
    """Internal helper for grouping our supported data types by what filter relations they can use."""

    DISCRETE = "discrete"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"

    def valid_relations(self):
        """Gets the valid relation operators for this data type class."""
        match self:
            case FilterClass.DISCRETE:
                return [Relation.INCLUDES, Relation.EXCLUDES]
            case FilterClass.NUMERIC:
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


class Filter(ApiBaseModel):
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

    @classmethod
    def cast_participant_id(
        cls, pid: str, column_type: sqlalchemy.sql.sqltypes.TypeEngine
    ) -> int | uuid.UUID | str:
        """Casts a participant ID string to an appropriate type based on the column type.

        Only supports INTEGER, BIGINT, UUID and STRING types as defined in DataType.supported_participant_id_types().
        """
        if isinstance(
            column_type,
            sqlalchemy.sql.sqltypes.Integer | sqlalchemy.sql.sqltypes.BigInteger,
        ):
            return int(pid)
        if isinstance(
            column_type, sqlalchemy.sql.sqltypes.UUID | sqlalchemy.sql.sqltypes.String
        ):
            return pid
        raise LateValidationError(f"Unsupported participant ID type: {column_type}")

    @model_validator(mode="after")
    def ensure_experiment_ids_hack_compatible(self) -> "Filter":
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
    def ensure_value(self) -> "Filter":
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
    def ensure_sane_bool_list(self) -> "Filter":
        """Ensures that the `value` field does not include redundant or nonsensical items."""
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


class MetricType(enum.StrEnum):
    """Classifies metrics by their value type."""

    BINARY = "binary"
    NUMERIC = "numeric"

    @classmethod
    def from_python_type(cls, python_type: type) -> "MetricType":
        """Maps Python types to metric types."""

        if python_type in {int, float, decimal.Decimal}:
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
            description="Standard deviation is set only for metric_type.NUMERIC metrics. Must be set for numeric metrics when available_n > 0."
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
        if (
            self.metric_type == MetricType.NUMERIC
            and self.available_n
            and self.metric_stddev is None
        ):
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
        str | None,
        Field(
            description="ID of the arm. If creating a new experiment (POST /datasources/{datasource_id}/experiments), this is generated for you and made available in the response; you should NOT set this. Only generate ids of your own if using the stateless Experiment Design API as you will do your own persistence."
        ),
    ] = None
    arm_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    arm_description: Annotated[
        str | None, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)
    ] = None


class ArmAnalysis(Arm):
    is_baseline: Annotated[
        bool,
        Field(
            description="Whether this arm is the baseline/control arm for comparison."
        ),
    ]
    estimate: Annotated[
        float,
        Field(
            description="The estimated treatment effect relative to the baseline arm."
        ),
    ]
    p_value: Annotated[
        float | None,
        Field(
            description="The p-value indicating statistical significance of the treatment effect. Value may be None if the t-stat is not available, e.g. due to inability to calculate the standard error."
        ),
    ]
    t_stat: Annotated[
        float | None,
        Field(
            description="The t-statistic from the statistical test. If the value is actually NaN, e.g. due to inability to calculate the standard error, we return None."
        ),
    ]
    std_error: Annotated[
        float, Field(description="The standard error of the treatment effect estimate.")
    ]

    @field_serializer("t_stat", "p_value", when_used="json")
    def serialize_float(self, v: float, _info):
        """Serialize floats to None when they are NaN, which becomes null in JSON."""
        if math.isnan(v):
            return None
        return v


class MetricAnalysis(ApiBaseModel):
    """Describes the change in a single metric for each arm of an experiment."""

    metric_name: str | None = None
    metric: DesignSpecMetricRequest | None = None
    arm_analyses: Annotated[
        list[ArmAnalysis],
        Field(
            description="The results of the analysis for each arm (coefficient) for this specific metric."
        ),
    ]

    @model_validator(mode="after")
    def validate_single_baseline(self) -> Self:
        """Ensure that if is_baseline is set to True, it is the only baseline arm."""
        baseline_arms = [arm for arm in self.arm_analyses if arm.is_baseline]
        if len(baseline_arms) != 1:
            raise ValueError(
                f"Exactly one arm must be designated as the baseline arm. Found {len(baseline_arms)} baseline arms."
            )
        return self


class ExperimentAnalysis(ApiBaseModel):
    """Describes the change if any in metrics targeted by an experiment."""

    experiment_id: Annotated[
        str,
        Field(description="ID of the experiment."),
    ]
    metric_analyses: Annotated[
        list[MetricAnalysis],
        Field(
            description="Contains one analysis per metric targeted by the experiment."
        ),
    ]


ExperimentType = Literal["online", "preassigned"]


class Stratum(ApiBaseModel):
    """Describes a variable used for stratification."""

    field_name: FieldName


class BaseDesignSpec(ApiBaseModel):
    """Experiment design metadata and target metrics common to all experiment types."""

    participant_type: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]

    experiment_id: Annotated[
        str | None,
        Field(
            description="ID of the experiment. If creating a new experiment (POST /datasources/{datasource_id}/experiments), this is generated for you and made available in the response; you should NOT set this. Only generate ids of your own if using the stateless Experiment Design API as you will do your own persistence."
        ),
    ] = None
    experiment_type: Annotated[
        str,
        Field(description="This type determines how we do assignment and analyses."),
    ]
    experiment_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]
    start_date: datetime.datetime
    end_date: datetime.datetime

    # arms (at least two)
    arms: Annotated[list[Arm], Field(..., min_length=2, max_length=MAX_NUMBER_OF_ARMS)]

    strata: Annotated[
        list[Stratum],
        Field(
            description="Optional participant_type fields to use for stratified assignment.",
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ]

    metrics: Annotated[
        list[DesignSpecMetricRequest],
        Field(
            ...,
            description="Primary and optional secondary metrics to target.",
            min_length=1,
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ]

    filters: Annotated[
        list[Filter],
        Field(
            description="Optional filters that constrain a general participant_type to a specific subset who can participate in an experiment.",
            max_length=MAX_NUMBER_OF_FILTERS,
        ),
    ]

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime.datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()

    @field_validator("experiment_type")
    @classmethod
    def validate_experiment_type(cls, v):
        """Validate that the experiment type is one of the supported ExperimentTypes."""
        if v not in get_args(ExperimentType):
            raise ValueError(f"Invalid experiment type: {v}")
        return v

    def ids_are_present(self) -> bool:
        """True if any IDs are present."""
        return self.experiment_id is not None or any(
            arm.arm_id is not None for arm in self.arms
        )


class FrequentistExperimentSpec(BaseDesignSpec):
    """Experiment design parameters for power calculations and analysis."""

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


class PreassignedExperimentSpec(FrequentistExperimentSpec):
    """Use this type to randomly select and assign from existing participants at design time."""

    experiment_type: Literal["preassigned"] = "preassigned"


class OnlineExperimentSpec(FrequentistExperimentSpec):
    """Use this type to randomly assign participants into arms during live experiment execution.

    For example, you may wish to experiment on new users. Assignments are issued via API request.
    """

    experiment_type: Annotated[
        Literal["online"],
        Field(description="Experiment type identifier for online experiments"),
    ] = "online"


type DesignSpec = Annotated[
    PreassignedExperimentSpec | OnlineExperimentSpec,
    Field(
        discriminator="experiment_type",
        description="Concrete type of experiment to run.",
    ),
]


class MetricPowerAnalysisMessageType(enum.StrEnum):
    """Classifies metric power analysis results."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    NO_BASELINE = "no baseline"
    NO_AVAILABLE_N = "no available n"
    ZERO_EFFECT_SIZE = "zero effect size"
    ZERO_STDDEV = "zero variation"


class MetricPowerAnalysisMessage(ApiBaseModel):
    """Describes interpretation of power analysis results."""

    type: MetricPowerAnalysisMessageType
    msg: Annotated[
        str,
        Field(
            description="Main power analysis result stated in human-friendly English."
        ),
    ]
    source_msg: Annotated[
        str,
        Field(
            description="Power analysis result formatted as a template string with curly-braced {} named placeholders. Use with the dictionary of values to support localization of messages."
        ),
    ]
    values: dict[str, float | int] | None = None


class MetricPowerAnalysis(ApiBaseModel):
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
        MetricPowerAnalysisMessage | None,
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

        if python_type in {int, float}:
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
    participant_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_PARTICIPANT_ID_VALUE)]
    arm_id: Annotated[
        str,
        Field(
            description="ID of the arm this participant was assigned to. Same as Arm.arm_id."
        ),
    ]
    arm_name: Annotated[
        str,
        Field(
            description="The arm this participant was assigned to. Same as Arm.arm_name.",
            max_length=MAX_LENGTH_OF_NAME_VALUE,
        ),
    ]
    created_at: Annotated[
        datetime.datetime | None,
        Field(description="The date and time the assignment was created."),
    ] = None
    strata: Annotated[
        list[Strata] | None,
        Field(
            description="List of properties and their values for this participant used for stratification or tracking metrics. If stratification is not used, this will be None.",
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None


class MetricValue(ApiBaseModel):
    metric_name: Annotated[
        FieldName,
        Field(
            description="The field_name from the datasource which this analysis models as the dependent variable (y)."
        ),
    ]
    metric_value: Annotated[
        float, Field(description="The queried value for this field_name.")
    ]


class ParticipantOutcome(ApiBaseModel):
    participant_id: Annotated[str, Field(max_length=MAX_LENGTH_OF_PARTICIPANT_ID_VALUE)]
    metric_values: Annotated[list[MetricValue], Field(max_length=MAX_NUMBER_OF_FIELDS)]


class BalanceCheck(ApiBaseModel):
    """Describes balance test results for treatment assignment."""

    f_statistic: Annotated[
        float,
        Field(
            description="F-statistic testing the overall significance of the model predicting treatment assignment."
        ),
    ]
    numerator_df: Annotated[
        int,
        Field(
            description="The numerator degrees of freedom for the f-statistic related to number of dependent variables."
        ),
    ]
    denominator_df: Annotated[
        int,
        Field(
            description="Denominator degrees of freedom related to the number of observations."
        ),
    ]
    p_value: Annotated[
        float,
        Field(
            description="Probability of observing these data if strata do not predict treatment assignment, i.e. our randomization is balanced."
        ),
    ]
    balance_ok: Annotated[
        bool,
        Field(
            description="Whether the p-value for our observed f_statistic is greater than the f-stat threshold specified in our design specification. (See DesignSpec.fstat_thresh)"
        ),
    ]


class ArmSize(ApiBaseModel):
    """Describes the number of participants assigned to each arm."""

    arm: Arm
    size: int = 0


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


class GetStrataResponseElement(ApiBaseModel):
    """Describes a stratification variable."""

    data_type: DataType
    field_name: FieldName
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]
    # Extra fields will be stored here in case a user configured their worksheet with extra metadata for their own
    # downstream use, e.g. to group strata with a friendly identifier.
    extra: Annotated[dict[str, str] | None, Field(max_length=MAX_NUMBER_OF_FIELDS)] = (
        None
    )


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


class GetMetricsResponseElement(ApiBaseModel):
    """Describes a metric."""

    field_name: FieldName
    data_type: DataType
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]


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


class PowerRequest(ApiBaseModel):
    design_spec: DesignSpec


class AssignRequest(ApiBaseModel):
    design_spec: DesignSpec


class PowerResponse(ApiBaseModel):
    analyses: Annotated[
        list[MetricPowerAnalysis], Field(max_length=MAX_NUMBER_OF_FIELDS)
    ]


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
