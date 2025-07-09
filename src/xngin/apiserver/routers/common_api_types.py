import datetime
import math
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

import sqlalchemy.sql
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    RootModel,
    Tag,
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
from xngin.apiserver.models.enums import (
    DataType,
    ExperimentState,
    Relation,
    StopAssignmentReason,
)
from xngin.apiserver.routers.common_enums import (
    ArmPriors,
    AssignmentType,
    ContextType,
    ExperimentsType,
    MetricPowerAnalysisMessageType,
    MetricType,
    OutcomeLikelihood,
)

type StrictInt = Annotated[int | None, Field(strict=True)]
type StrictFloat = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]
type FilterValueTypes = (
    Sequence[StrictInt]
    | Sequence[StrictFloat]
    | Sequence[str | None]
    | Sequence[bool | None]
)


class ApiBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class Context(ApiBaseModel):
    """
    Pydantic model for context of the experiment.
    """

    context_id: Annotated[
        int | None,
        Field(
            description="Unique identifier for the context, you should NOT set this when creating a new context.",
            examples=[1],
        ),
    ]
    context_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    context_description: Annotated[
        str | None, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)
    ] = None
    value_type: Annotated[
        ContextType,
        Field(
            description="Type of value the context can take", default=ContextType.BINARY
        ),
    ]


class ContextInput(ApiBaseModel):
    """
    Pydantic model for a context input
    """

    context_id: Annotated[
        int,
        Field(
            description="Unique identifier for the context.",
            examples=[1],
        ),
    ]
    context_value: Annotated[
        float,
        Field(
            description="Value of the context",
            examples=[2.5],
        ),
    ]


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
    num_missing_values: Annotated[
        int,
        Field(
            description="The number of participants assigned to this arm with missing values (NaNs) for this metric. These rows are excluded from the analysis."
        ),
    ]
    is_baseline: Annotated[
        bool,
        Field(
            description="Whether this arm is the baseline/control arm for comparison."
        ),
    ]

    @field_serializer("t_stat", "p_value", when_used="json")
    def serialize_float(self, v: float, _info):
        """Serialize floats to None when they are NaN, which becomes null in JSON."""
        if math.isnan(v):
            return None
        return v


class ArmBandit(Arm):
    """Describes an experiment arm for bandit experiments."""

    # Prior variables
    alpha_init: Annotated[
        float | None,
        Field(
            default=None,
            examples=[None, 1.0],
            description="Initial alpha parameter for Beta prior",
        ),
    ]
    beta_init: Annotated[
        float | None,
        Field(
            default=None,
            examples=[None, 1.0],
            description="Initial beta parameter for Beta prior",
        ),
    ]
    mu_init: Annotated[
        float | None,
        Field(
            default=None,
            examples=[None, 0.0],
            description="Initial mean parameter for Normal prior",
        ),
    ]
    sigma_init: Annotated[
        float | None,
        Field(
            default=None,
            examples=[None, 1.0],
            description="Initial standard deviation parameter for Normal prior",
        ),
    ]
    n_outcomes: Annotated[
        int, Field(default=0, description="The number of outcomes for this arm.")
    ]
    alpha: Annotated[
        float | None,
        Field(
            default=None,
            examples=[None, 1.0],
            description="Updated alpha parameter for Beta prior",
        ),
    ]
    beta: Annotated[
        float | None,
        Field(
            default=None,
            examples=[None, 1.0],
            description="Updated beta parameter for Beta prior",
        ),
    ]
    mu: Annotated[
        list[float] | None,
        Field(
            default=None,
            examples=[None, [0.0]],
            description="Updated mean vector for Normal prior",
        ),
    ]
    covariance: Annotated[
        list[list[float]] | None,
        Field(
            default=None,
            examples=[None, [[1.0]]],
            description="Updated covariance matrix for Normal prior",
        ),
    ]
    is_baseline: Annotated[
        bool | None,
        Field(
            description="Whether this arm is the baseline/control arm for comparison."
        ),
    ]

    @model_validator(mode="after")
    def check_values(self) -> Self:
        """
        Check if the values are unique.
        """
        alpha = self.alpha_init
        beta = self.beta_init
        sigma = self.sigma_init
        if alpha is not None and alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")
        if beta is not None and beta <= 0:
            raise ValueError("Beta must be greater than 0.")
        if sigma is not None and sigma <= 0:
            raise ValueError("Sigma must be greater than 0.")
        return self

    @field_serializer(
        "alpha_init", "beta_init", "mu_init", "sigma_init", when_used="json"
    )
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
    num_participants: Annotated[
        int,
        Field(
            description="The number of participants assigned to the experiment pulled from the dwh across all arms. Metric outcomes are not guaranteed to be present for all participants."
        ),
    ]
    num_missing_participants: Annotated[
        int | None,
        Field(
            description="The number of participants assigned to the experiment across all arms that are not found in the data warehouse when pulling metrics."
        ),
    ] = None
    created_at: Annotated[
        datetime.datetime,
        Field(description="The date and time the experiment analysis was created."),
    ]


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


class GetMetricsResponseElement(ApiBaseModel):
    """Describes a metric."""

    field_name: FieldName
    data_type: DataType
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]


EXPERIMENT_IDS_SUFFIX = "experiment_ids"


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
    assignment_type: Annotated[
        AssignmentType,
        Field(
            description="This type determines how we do assignment and analyses.",
            default="online",
        ),
    ]
    experiment_type: Annotated[
        ExperimentsType,
        Field(
            description="The type of experiment, e.g. A/B, CMAB, etc. This is used to determine how the experiment is analyzed and what metrics are available.",
            default=ExperimentsType.MAB,
        ),
    ]

    experiment_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]
    start_date: datetime.datetime
    end_date: datetime.datetime

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime.datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


class FrequentistExperimentSpecBase(BaseDesignSpec):
    """Base class for Frequentist experiment design parameters."""

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

    def ids_are_present(self) -> bool:
        """True if any IDs are present."""
        return self.experiment_id is not None or any(
            arm.arm_id is not None for arm in self.arms
        )


class FrequentistExperimentSpec(FrequentistExperimentSpecBase):
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


class BanditExperimentSpec(BaseDesignSpec):
    """Experiment design parameters for bandit experiments."""

    # arms (at least two)
    arms: Annotated[
        list[ArmBandit], Field(..., min_length=2, max_length=MAX_NUMBER_OF_ARMS)
    ]
    contexts: Annotated[
        list[Context] | None,
        Field(
            default=None,
            description="Optional list of contexts that can be used to condition the bandit assignment. Required for contextual bandit experiments.",
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ]

    # Experiment config
    prior_type: Annotated[
        ArmPriors,
        Field(
            description="The type of prior distribution for the arms.",
            default=ArmPriors.BETA,
        ),
    ]
    reward_type: Annotated[
        OutcomeLikelihood,
        Field(
            description="The type of reward we observe from the experiment.",
            default=OutcomeLikelihood.BERNOULLI,
        ),
    ]

    @model_validator(mode="after")
    def check_arm_missing_params(self) -> Self:
        """
        Check if the arm reward type is same as the experiment reward type.
        """
        prior_type = self.prior_type
        arms = self.arms

        prior_params = {
            ArmPriors.BETA: ("alpha_init", "beta_init"),
            ArmPriors.NORMAL: ("mu_init", "sigma_init"),
        }

        for arm in arms:
            arm_dict = arm.model_dump()
            if prior_type in prior_params:
                missing_params = []
                for param in prior_params[prior_type]:
                    if param not in arm_dict or arm_dict[param] is None:
                        missing_params.append(param)

                if missing_params:
                    val = prior_type.value
                    raise ValueError(f"{val} prior needs {','.join(missing_params)}.")
        return self

    @model_validator(mode="after")
    def check_treatment_info(self) -> Self:
        """
        Validate that the treatment arm information is set correctly.
        """
        arms = self.arms
        if self.experiment_type == ExperimentsType.BAYESAB:
            if not any(arm.is_baseline for arm in arms):
                raise ValueError("At least one arm must be a baseline/control arm.")
            if all(arm.is_baseline for arm in arms):
                raise ValueError("At least one arm must be a treatment arm.")
        return self

    @model_validator(mode="after")
    def check_prior_reward_type_combo(self) -> Self:
        """
        Validate that the prior and reward type combination is allowed.
        """
        if self.prior_type == ArmPriors.BETA:
            if not self.reward_type == OutcomeLikelihood.BERNOULLI:
                raise ValueError(
                    "Beta prior can only be used with binary-valued rewards."
                )
            if self.experiment_type != ExperimentsType.MAB:
                raise ValueError(
                    f"Experiments of type {self.experiment_type} can only use Gaussian priors."
                )

        return self

    @model_validator(mode="after")
    def check_contexts(self) -> Self:
        """
        Validate that the contexts inputs are valid.
        """
        if self.experiment_type == ExperimentsType.CMAB and not self.contexts:
            raise ValueError("Contextual MAB experiments require at least one context.")
        if self.experiment_type != ExperimentsType.CMAB and self.contexts:
            raise ValueError(
                "Contexts are only applicable for contextual MAB experiments."
            )
        return self


class PreassignedExperimentSpec(FrequentistExperimentSpec):
    """Use this type to randomly select and assign from existing participants at design time with frequentist A/B experiments."""

    assignment_type: Literal["preassigned"] = "preassigned"
    experiment_type: Literal[ExperimentsType.FREQ_AB] = ExperimentsType.FREQ_AB


class OnlineFrequentistExperimentSpec(FrequentistExperimentSpec):
    """Use this type to randomly assign participants into arms during live experiment execution with frequentist A/B experiments.

    For example, you may wish to experiment on new users. Assignments are issued via API request.
    """

    assignment_type: Literal["online"] = "online"
    experiment_type: Literal[ExperimentsType.FREQ_AB] = ExperimentsType.FREQ_AB


type DesignSpec = Annotated[
    PreassignedExperimentSpec | OnlineFrequentistExperimentSpec,
    Field(
        discriminator="assignment_type",
        description="The type of assignment and experiment design.",
    ),
]


class PowerRequest(ApiBaseModel):
    design_spec: DesignSpec


class PowerResponse(ApiBaseModel):
    analyses: Annotated[
        list[MetricPowerAnalysis], Field(max_length=MAX_NUMBER_OF_FIELDS)
    ]


class Strata(ApiBaseModel):
    """Describes stratification for an experiment participant."""

    field_name: FieldName
    # TODO(roboton): Add in strata type, update tests to reflect this field, should be derived
    # from data warehouse.
    # strata_type: Optional[StrataType]
    strata_value: str | None = None


class AssignmentBase(ApiBaseModel):
    """Base class for treatment assignment in experiments."""

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


class FrequentistAssignment(AssignmentBase):
    """Describes treatment assignment for an experiment participant."""

    strata: Annotated[
        list[Strata] | None,
        Field(
            description="List of properties and their values for this participant used for stratification or tracking metrics. If stratification is not used, this will be None.",
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None


class BanditAssignment(AssignmentBase):
    """Describes treatment assignment for a bandit experiment participant."""

    draw_id: Annotated[
        str | None,
        Field(
            description="Unique identifier for the draw.",
            examples=["draw_123", None],
        ),
    ]

    observed_at: Annotated[
        datetime.datetime | None,
        Field(description="The date and time the outcome was recorded."),
    ] = None

    outcome: Annotated[
        float, Field(description="The observed outcome for this assignment.")
    ]
    context_values: Annotated[
        list[ContextInput] | None,
        Field(
            description="List of context values for this assignment. If no contexts are used, this will be None.",
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None


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


class ExperimentsBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateExperimentRequestBandit(ExperimentsBaseModel):
    design_spec: BanditExperimentSpec


class CreateExperimentRequestFrequentist(ExperimentsBaseModel):
    design_spec: DesignSpec
    webhooks: Annotated[
        list[str],
        Field(
            default=[],
            description="List of webhook IDs to associate with this experiment. When the experiment is committed, these webhooks will be triggered with experiment details. Must contain unique values.",
        ),
    ]

    @field_validator("webhooks")
    @classmethod
    def validate_unique_webhooks(cls, v: list[str]) -> list[str]:
        """Ensure all webhook IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Webhook IDs must be unique")
        return v

    power_analyses: PowerResponse | None = None


def experiment_request_discriminator(value: dict[str, Any]) -> str:
    """Discriminator function for CreateExperimentRequest."""

    if isinstance(value, dict):
        try:
            experiment_type = value["design_spec"].experiment_type
        except AttributeError:
            experiment_type = value["design_spec"].get("experiment_type")
    elif isinstance(value, ExperimentConfigBase):
        experiment_type = value.design_spec.experiment_type

    if experiment_type == ExperimentsType.FREQ_AB:
        return "frequentist"
    if experiment_type in ExperimentsType:
        return "bandit"
    raise ValueError(
        f"Unknown experiment type: {experiment_type}. Expected one of {ExperimentsType}."
    )


class CreateExperimentRequest(RootModel):
    root: Annotated[
        Annotated[CreateExperimentRequestFrequentist, Tag("frequentist")]
        | Annotated[CreateExperimentRequestBandit, Tag("bandit")],
        Field(
            discriminator=Discriminator(experiment_request_discriminator),
            description="Concrete type of experiment to create, determined by the experiment_type in the design_spec.",
        ),
    ]


class AssignSummary(ExperimentsBaseModel):
    """Key pieces of an AssignResponse without the assignments."""

    balance_check: Annotated[
        BalanceCheck | None,
        Field(
            description="Balance test results if available. 'online' experiments do not have balance checks."
        ),
    ] = None
    sample_size: Annotated[
        int, Field(description="The number of participants across all arms in total.")
    ]
    arm_sizes: Annotated[
        list[ArmSize] | None,
        Field(
            description="For each arm, the number of participants assigned. "
            "TODO: make required once development has stabilized. May be None if unknown due to persisting prior versions of an AssignSummary.",
            max_length=MAX_NUMBER_OF_ARMS,
        ),
    ] = None


class ExperimentConfigBase(ExperimentsBaseModel):
    """Representation of our stored Experiment information."""

    datasource_id: str
    state: Annotated[
        ExperimentState, Field(description="Current state of this experiment.")
    ]
    stopped_assignments_at: Annotated[
        datetime.datetime | None,
        Field(
            description="The date and time assignments were stopped. Null if assignments are still allowed to be made."
        ),
    ]
    stopped_assignments_reason: Annotated[
        StopAssignmentReason | None,
        Field(
            description="The reason assignments were stopped. Null if assignments are still allowed to be made."
        ),
    ]


class ExperimentConfigFrequentist(ExperimentConfigBase):
    """Experiment configuration for Frequentist experiments."""

    design_spec: DesignSpec
    power_analyses: PowerResponse | None
    assign_summary: AssignSummary
    webhooks: Annotated[
        list[str],
        Field(
            default=[],
            description="List of webhook IDs associated with this experiment. These webhooks are triggered when the experiment is committed.",
        ),
    ]


class ExperimentConfigBandit(ExperimentConfigBase):
    """Experiment configuration for Bandit experiments."""

    design_spec: BanditExperimentSpec


ExperimentConfig = Annotated[
    Annotated[ExperimentConfigFrequentist, Tag("frequentist")]
    | Annotated[ExperimentConfigBandit, Tag("bandit")],
    Field(
        discriminator=Discriminator(experiment_request_discriminator),
        description="Concrete type of experiment configuration, determined by the experiment_type in the design_spec.",
    ),
]


class GetExperimentResponse(ExperimentsBaseModel):
    """An experiment configuration capturing all info at design time when assignment was made."""

    config: Annotated[ExperimentConfig, Field(description="Experiment configuration.")]


class ListExperimentsResponse(ExperimentsBaseModel):
    items: list[ExperimentConfig]


class GetParticipantAssignmentResponse(ExperimentsBaseModel):
    """Describes assignment for a single <experiment, participant> pair."""

    experiment_id: str
    participant_id: str
    assignment: Annotated[
        FrequentistAssignment | None,
        Field(description="Null if no assignment. assignment.strata are not included."),
    ]


class CreateExperimentResponse(ExperimentsBaseModel):
    """Same as the request but with ids filled for the experiment and arms, and summary info on the assignment."""

    config: ExperimentConfig


class GetExperimentAssignmentsResponse(ExperimentsBaseModel):
    """Describes assignments for all participants and balance test results if available."""

    balance_check: Annotated[
        BalanceCheck | None,
        Field(
            description="Balance test results if available. 'online' experiments do not have balance checks."
        ),
    ] = None

    experiment_id: str
    sample_size: int
    assignments: list[FrequentistAssignment]


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
