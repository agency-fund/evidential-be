import datetime
import json
import math
import uuid
from collections.abc import Sequence
from typing import Annotated, Literal, Self

import sqlalchemy.sql
from annotated_types import MaxLen, MinLen
from pydantic import (
    AfterValidator,
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
    MAX_GCP_SERVICE_ACCOUNT_LEN,
    MAX_LENGTH_OF_DESCRIPTION_VALUE,
    MAX_LENGTH_OF_NAME_VALUE,
    MAX_LENGTH_OF_PARTICIPANT_ID_VALUE,
    MAX_NUMBER_OF_ARMS,
    MAX_NUMBER_OF_CONTEXTS,
    MAX_NUMBER_OF_FIELDS,
    MAX_NUMBER_OF_FILTERS,
)
from xngin.apiserver.routers.common_enums import (
    ContextType,
    DataType,
    ExperimentAnalysisType,
    ExperimentState,
    ExperimentsType,
    LikelihoodTypes,
    MetricPowerAnalysisMessageType,
    MetricType,
    PriorTypes,
    Relation,
    StopAssignmentReason,
)

type StrictInt = Annotated[int | None, Field(strict=True)]
type StrictFloat = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]
type FilterValueTypes = Sequence[StrictInt] | Sequence[StrictFloat] | Sequence[str | None] | Sequence[bool | None]


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
        Field(description="Absolute target value = metric_baseline*(1 + metric_pct_change)"),
    ] = None


class DesignSpecMetric(DesignSpecMetricBase):
    """Defines a metric to measure in an experiment with its baseline stats."""

    metric_type: Annotated[MetricType | None, Field(description="Inferred from dwh type.")] = None
    metric_baseline: Annotated[float | None, Field(description="Mean of the tracked metric.")] = None
    metric_stddev: Annotated[
        float | None,
        Field(
            description=(
                "Standard deviation is set only for metric_type.NUMERIC metrics. Must be set for "
                "numeric metrics when available_n > 0."
            )
        ),
    ] = None
    available_nonnull_n: Annotated[
        int | None,
        Field(
            description=(
                "The number of participants meeting the filtering criteria with a *non-null* value for this metric."
            )
        ),
    ] = None
    available_n: Annotated[
        int | None,
        Field(
            description=(
                "The number of participants meeting the filtering criteria regardless of whether or "
                "not this metric's value is NULL. NOTE: Assignments are made from the targeted "
                "aviailable_n population, so be sure you are ok with participants potentially having "
                "this value missing during assignment if available_n != available_nonnull_n."
            )
        ),
    ] = None

    @model_validator(mode="after")
    def stddev_check(self):
        """Enforce that metric_stddev is empty for non-NUMERICs. FE will handle numerics without stddev
        (due to all nulls)"""
        if self.metric_type is not MetricType.NUMERIC and self.metric_stddev is not None:
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
        Field(description="Specify the absolute value you want to detect. Cannot be set if you set metric_pct_change."),
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
        str | None,
        Field(
            description="Unique identifier for the context, you should NOT set this when creating a new context.",
            examples=["1"],
        ),
    ] = None
    context_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    context_description: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)] = None
    value_type: Annotated[
        ContextType,
        Field(description="Type of value the context can take", default=ContextType.BINARY),
    ]


class ContextInput(ApiBaseModel):
    """
    Pydantic model for a context input
    """

    context_id: Annotated[
        str,
        Field(
            description="Unique identifier for the context.",
            examples=["1"],
        ),
    ]
    context_value: Annotated[
        float,
        Field(
            description="Value of the context",
            examples=[2.5],
        ),
    ]


class CreateCMABAssignmentRequest(ApiBaseModel):
    """Request model for creating a new CMAB assignment.

    When submitting context values for a CMAB experiment, the following rules apply:
    1. Each context_input must reference a valid context_id from the experiment's defined contexts
    2. The order of context_inputs does not need to match the order of contexts in the experiment
    3. You must provide values for all contexts defined in the experiment
    4. Number of input context values must match the number of contexts defined in the experiment
    5. The context value input can be None, but only in the case of retrieving a pre-existing assignment.

    Example:
        If an experiment defines contexts with IDs ["ctx_1", "ctx_2"], your request must include
        both of these context_ids in the context_inputs list, but they can be in any order.
    """

    type: Literal["cmab_assignment"] = (
        "cmab_assignment"  # Adding type field to allow for type-discriminated unions in future
    )
    context_inputs: Annotated[
        list[ContextInput] | None,
        Field(
            description="""
            List of context values for the assignment.
            Must include exactly the same number contexts defined in the experiment.
            The values are matched to the experiment's contexts by context_id, not by position in the list.
            Each context_id must correspond to one of the IDs of the contexts defined in the experiment.
            Can be None, when simply retrieving pre-existing assignments; must have valid inputs otherwise.
            """
        ),
    ]


class Arm(ApiBaseModel):
    """Describes an experiment treatment arm."""

    arm_id: Annotated[
        str | None,
        Field(
            description=(
                "ID of the arm. If creating a new experiment (POST /datasources/{datasource_id}/experiments), "
                "this is generated for you and made available in the response; you should NOT set this. "
                "Only generate ids of your own if using the stateless Experiment Design API as you will "
                "do your own persistence."
            )
        ),
    ] = None
    arm_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    arm_description: Annotated[str | None, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)] = None


class ArmAnalysis(Arm):
    estimate: Annotated[
        float,
        Field(description="The estimated treatment effect relative to the baseline arm."),
    ]
    p_value: Annotated[
        float | None,
        Field(
            description=(
                "The p-value indicating statistical significance of the treatment effect. Value may be "
                "None if the t-stat is not available, e.g. due to inability to calculate the standard error."
            )
        ),
    ]
    t_stat: Annotated[
        float | None,
        Field(
            description=(
                "The t-statistic from the statistical test. If the value is actually NaN, e.g. due to "
                "inability to calculate the standard error, we return None."
            )
        ),
    ]
    std_error: Annotated[float, Field(description="The standard error of the treatment effect estimate.")]
    num_missing_values: Annotated[
        int,
        Field(
            description=(
                "The number of participants assigned to this arm with missing values (NaNs) for this "
                "metric. These rows are excluded from the analysis."
            )
        ),
    ]
    is_baseline: Annotated[
        bool,
        Field(description="Whether this arm is the baseline/control arm for comparison."),
    ]

    @field_serializer("t_stat", "p_value", when_used="json")
    def serialize_float(self, v: float | None, _info):
        """Serialize floats to None when they are NaN, which becomes null in JSON."""
        if v is None or math.isnan(v):
            return None
        return v


class ArmBandit(Arm):
    """Describes an experiment arm for bandit experiments."""

    # Prior variables
    alpha_init: Annotated[
        float | None,
        Field(
            examples=[None, 1.0],
            description="Initial alpha parameter for Beta prior",
        ),
    ] = None
    beta_init: Annotated[
        float | None,
        Field(
            examples=[None, 1.0],
            description="Initial beta parameter for Beta prior",
        ),
    ] = None
    mu_init: Annotated[
        float | None,
        Field(
            examples=[None, 0.0],
            description="Initial mean parameter for Normal prior",
        ),
    ] = None
    sigma_init: Annotated[
        float | None,
        Field(
            examples=[None, 1.0],
            description="Initial standard deviation parameter for Normal prior",
        ),
    ] = None
    alpha: Annotated[
        float | None,
        Field(
            examples=[None, 1.0],
            description="Updated alpha parameter for Beta prior",
        ),
    ] = None
    beta: Annotated[
        float | None,
        Field(
            examples=[None, 1.0],
            description="Updated beta parameter for Beta prior",
        ),
    ] = None
    mu: Annotated[
        list[float] | None,
        Field(
            examples=[None, [0.0]],
            description="Updated mean vector for Normal prior",
        ),
    ] = None
    covariance: Annotated[
        list[list[float]] | None,
        Field(
            examples=[None, [[1.0]]],
            description="Updated covariance matrix for Normal prior",
        ),
    ] = None

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


class MetricAnalysis(ApiBaseModel):
    """Describes the change in a single metric for each arm of an experiment."""

    metric_name: str | None = None
    metric: DesignSpecMetricRequest | None = None
    arm_analyses: Annotated[
        list[ArmAnalysis],
        Field(description="The results of the analysis for each arm (coefficient) for this specific metric."),
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


class BanditExperimentAnalysisResponse(ApiBaseModel):
    """Describes changes in arms for a bandit experiment"""

    type: Literal[ExperimentAnalysisType.BANDIT] = ExperimentAnalysisType.BANDIT

    experiment_id: Annotated[
        str,
        Field(description="ID of the experiment."),
    ]
    n_trials: Annotated[
        int,
        Field(description="The number of trials conducted for this experiment."),
    ]
    n_outcomes: Annotated[
        int,
        Field(description="The number of outcomes observed for this experiment."),
    ]
    posterior_means: Annotated[
        list[float],
        Field(description="Posterior means for each arm in the experiment."),
    ]
    posterior_stds: Annotated[
        list[float],
        Field(description="Posterior standard deviations for each arm in the experiment."),
    ]
    volumes: Annotated[
        list[float],
        Field(description="Volume of participants for each arm in the experiment."),
    ]


class FreqExperimentAnalysisResponse(ApiBaseModel):
    """Describes the change if any in metrics targeted by an experiment."""

    type: Literal[ExperimentAnalysisType.FREQ] = ExperimentAnalysisType.FREQ

    experiment_id: Annotated[
        str,
        Field(description="ID of the experiment."),
    ]
    metric_analyses: Annotated[
        list[MetricAnalysis],
        Field(description="Contains one analysis per metric targeted by the experiment."),
    ]
    num_participants: Annotated[
        int,
        Field(
            description=(
                "The number of participants assigned to the experiment pulled from the dwh across all arms. "
                "Metric outcomes are not guaranteed to be present for all participants."
            )
        ),
    ]
    num_missing_participants: Annotated[
        int | None,
        Field(
            description=(
                "The number of participants assigned to the experiment across all arms that are not found "
                "in the data warehouse when pulling metrics."
            )
        ),
    ] = None
    created_at: Annotated[
        datetime.datetime,
        Field(description="The date and time the experiment analysis was created."),
    ]


type ExperimentAnalysisResponse = Annotated[
    FreqExperimentAnalysisResponse | BanditExperimentAnalysisResponse,
    Field(
        discriminator="type",
        description="The type of experiment analysis response.",
    ),
]


class MetricPowerAnalysisMessage(ApiBaseModel):
    """Describes interpretation of power analysis results."""

    type: MetricPowerAnalysisMessageType
    msg: Annotated[
        str,
        Field(description="Main power analysis result stated in human-friendly English."),
    ]
    source_msg: Annotated[
        str,
        Field(
            description=(
                "Power analysis result formatted as a template string with curly-braced {} named placeholders. "
                "Use with the dictionary of values to support localization of messages."
            )
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
        Field(description="Whether or not there are enough available units to sample from to meet target_n."),
    ] = None

    target_possible: Annotated[
        float | None,
        Field(
            description=(
                "If there is an insufficient sample size to meet the desired metric_target, we report what is possible "
                "given the available_n. This value is equivalent to the relative pct_change_possible. "
                "This is None when there is a sufficient sample size to detect the desired change."
            )
        ),
    ] = None

    pct_change_possible: Annotated[
        float | None,
        Field(
            description=(
                "If there is an insufficient sample size to meet the desired metric_pct_change, we report what is "
                "possible given the available_n. This value is equivalent to the absolute target_possible. "
                "This is None when there is a sufficient sample size to detect the desired change."
            )
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
    extra: Annotated[dict[str, str] | None, Field(max_length=MAX_NUMBER_OF_FIELDS)] = None


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
    def cast_participant_id(cls, pid: str, column_type: sqlalchemy.sql.sqltypes.TypeEngine) -> int | uuid.UUID | str:
        """Casts a participant ID string to an appropriate type based on the column type.

        Only supports INTEGER, BIGINT, UUID and STRING types as defined in DataType.supported_participant_id_types().
        """
        if isinstance(
            column_type,
            sqlalchemy.sql.sqltypes.Integer | sqlalchemy.sql.sqltypes.BigInteger,
        ):
            return int(pid)
        if isinstance(column_type, sqlalchemy.sql.sqltypes.UUID | sqlalchemy.sql.sqltypes.String):
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
                raise ValueError("values in an experiment_id filter may not contain commas")
            if v.strip() != v:
                raise ValueError("values in an experiment_id filter may not contain leading or trailing whitespace")
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
                raise ValueError("BETWEEN relation requires same values to be of the same type")
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


# -- Experiment Design Specification --


class BaseDesignSpec(ApiBaseModel):
    """Experiment design metadata and target metrics common to all experiment types."""

    # --- Experiment metadata ---
    participant_type: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]

    experiment_id: Annotated[
        str | None,
        Field(
            deprecated=True,
            description=(
                "ID of the experiment. If creating a new experiment (POST /datasources/{datasource_id}/experiments), "
                "this is generated for you and made available in the response; you should NOT set this. "
                "Only generate ids of your own if using the stateless Experiment Design API as you will "
                "do your own persistence. \n"
                "DEPRECATED: This field is no longer used and will be removed in a future release. "
                "Use the Create/GetExperimentResponse field directly."
            ),
        ),
    ] = None

    experiment_type: Annotated[
        ExperimentsType,
        Field(
            description="This type determines how we do assignment and analyses.",
        ),
    ]

    experiment_name: Annotated[str, Field(max_length=MAX_LENGTH_OF_NAME_VALUE)]
    description: Annotated[str, Field(max_length=MAX_LENGTH_OF_DESCRIPTION_VALUE)]
    start_date: datetime.datetime
    end_date: datetime.datetime

    # arms (at least two)
    arms: Annotated[Sequence[Arm], Field(..., min_length=2, max_length=MAX_NUMBER_OF_ARMS)]

    def ids_are_present(self) -> bool:
        """True if any IDs are present."""
        return self.experiment_id is not None or any(arm.arm_id is not None for arm in self.arms)


class BaseFrequentistDesignSpec(BaseDesignSpec):
    """Experiment design parameters for frequentist experiments."""

    # Frequentist config params
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
            description=(
                "Optional filters that constrain a general participant_type to a specific subset "
                "who can participate in an experiment."
            ),
            max_length=MAX_NUMBER_OF_FILTERS,
        ),
    ]

    # stat parameters
    power: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="The chance of detecting a real non-null effect, i.e. 1 - false negative rate.",
        ),
    ] = 0.8
    alpha: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description=(
                "The chance of a false positive, i.e. there is no real non-null effect, but we "
                "mistakenly think there is one."
            ),
        ),
    ] = 0.05
    fstat_thresh: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description=(
                "Threshold on the p-value of joint significance in doing the omnibus balance check, "
                'above which we declare the data to be "balanced".'
            ),
        ),
    ] = 0.6

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime.datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


class BaseBanditExperimentSpec(BaseDesignSpec):
    """Experiment design parameters for bandit experiments."""

    # Type-narrowing to ArmBandit for type checking, to ensure bandit arms are the correct subtype.
    arms: Annotated[Sequence[ArmBandit], Field(..., min_length=2, max_length=MAX_NUMBER_OF_ARMS)]
    contexts: Annotated[
        list[Context] | None,
        Field(
            description=(
                "Optional list of contexts that can be used to condition the bandit assignment. "
                "Required for contextual bandit experiments."
            ),
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None

    # Experiment config
    prior_type: Annotated[
        PriorTypes,
        Field(
            description="The type of prior distribution for the arms.",
            default=PriorTypes.BETA,
        ),
    ]
    reward_type: Annotated[
        LikelihoodTypes,
        Field(
            description="The type of reward we observe from the experiment.",
            default=LikelihoodTypes.BERNOULLI,
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
            PriorTypes.BETA: ("alpha_init", "beta_init"),
            PriorTypes.NORMAL: ("mu_init", "sigma_init"),
        }

        if prior_type not in prior_params:
            raise ValueError(
                f"Unsupported prior type: {prior_type}. Supported types are: {', '.join(prior_params.keys())}."
            )

        for arm in arms:
            arm_dict = arm.model_dump()
            missing_params = []
            for param in prior_params[prior_type]:
                if param not in arm_dict or arm_dict[param] is None:
                    missing_params.append(param)

            if missing_params:
                val = prior_type.value
                raise ValueError(f"{val} prior needs {','.join(missing_params)}.")
        return self

    @model_validator(mode="after")
    def check_prior_reward_type_combo(self) -> Self:
        """
        Validate that the prior and reward type combination is allowed.
        """
        if self.prior_type == PriorTypes.BETA:
            if not self.reward_type == LikelihoodTypes.BERNOULLI:
                raise ValueError("Beta prior can only be used with binary-valued rewards.")
            if self.experiment_type != ExperimentsType.MAB_ONLINE:
                raise ValueError(f"Experiments of type {self.experiment_type} can only use Gaussian priors.")

        return self

    @model_validator(mode="after")
    def check_contexts(self) -> Self:
        """
        Validate that the contexts inputs are valid.
        """
        if self.experiment_type == ExperimentsType.CMAB_ONLINE and not self.contexts:
            raise ValueError("Contextual MAB experiments require at least one context.")
        if self.experiment_type != ExperimentsType.CMAB_ONLINE and self.contexts:
            raise ValueError("Contexts are only applicable for contextual MAB experiments.")
        return self


class PreassignedFrequentistExperimentSpec(BaseFrequentistDesignSpec):
    """Use this type to randomly select and assign from existing participants at design time with
    frequentist A/B experiments."""

    experiment_type: Literal[ExperimentsType.FREQ_PREASSIGNED] = ExperimentsType.FREQ_PREASSIGNED


class OnlineFrequentistExperimentSpec(BaseFrequentistDesignSpec):
    """Use this type to randomly assign participants into arms during live experiment execution with
    frequentist A/B experiments.

    For example, you may wish to experiment on new users. Assignments are issued via API request.
    """

    experiment_type: Literal[ExperimentsType.FREQ_ONLINE] = ExperimentsType.FREQ_ONLINE


class MABExperimentSpec(BaseBanditExperimentSpec):
    """Use this type to randomly assign participants into arms during live experiment execution with MAB experiments.

    For example, you may wish to experiment on new users. Assignments are issued via API request.
    """

    experiment_type: Literal[ExperimentsType.MAB_ONLINE] = ExperimentsType.MAB_ONLINE


class CMABExperimentSpec(BaseBanditExperimentSpec):
    """Use this type to randomly assign participants into arms during live experiment execution with
    contextual MAB experiments.

    For example, you may wish to experiment on new users. Assignments are issued via API request.
    """

    experiment_type: Literal[ExperimentsType.CMAB_ONLINE] = ExperimentsType.CMAB_ONLINE


class BayesABExperimentSpec(BaseBanditExperimentSpec):
    """Use this type to randomly assign participants into arms during live experiment execution with
    Bayesian A/B experiments.

    For example, you may wish to experiment on new users. Assignments are issued via API request.
    """

    experiment_type: Literal[ExperimentsType.BAYESAB_ONLINE] = ExperimentsType.BAYESAB_ONLINE


type DesignSpec = Annotated[
    PreassignedFrequentistExperimentSpec
    | OnlineFrequentistExperimentSpec
    | MABExperimentSpec
    | CMABExperimentSpec
    | BayesABExperimentSpec,
    Field(
        discriminator="experiment_type",
        description="The type of assignment and experiment design.",
    ),
]


class PowerRequest(ApiBaseModel):
    design_spec: DesignSpec


class PowerResponse(ApiBaseModel):
    analyses: Annotated[list[MetricPowerAnalysis], Field(max_length=MAX_NUMBER_OF_FIELDS)]


class Strata(ApiBaseModel):
    """Describes stratification for an experiment participant."""

    field_name: FieldName
    # TODO(roboton): Add in strata type, update tests to reflect this field, should be derived
    # from data warehouse.
    # strata_type: Optional[StrataType]
    strata_value: str | None = None


class Assignment(ApiBaseModel):
    """Base class for treatment assignment in experiments."""

    # this references the field marked is_unique_id == TRUE in the configuration spreadsheet
    arm_id: Annotated[
        str,
        Field(description="ID of the arm this participant was assigned to. Same as Arm.arm_id."),
    ]
    participant_id: Annotated[
        str,
        Field(
            description=(
                "Unique identifier for the participant. This is the primary key for the participant "
                "in the data warehouse."
            ),
            max_length=MAX_LENGTH_OF_PARTICIPANT_ID_VALUE,
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

    # -- Frequentist-specific fields --
    strata: Annotated[
        list[Strata] | None,
        Field(
            description=(
                "List of properties and their values for this participant used for stratification or "
                "tracking metrics. If stratification is not used, this will be None."
            ),
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None

    # -- Bandit-specific fields --
    observed_at: Annotated[
        datetime.datetime | None,
        Field(description="The date and time the outcome was recorded."),
    ] = None

    outcome: Annotated[float | None, Field(description="The observed outcome for this assignment.")] = None

    context_values: Annotated[
        list[float] | None,
        Field(
            description="List of context values for this assignment. If no contexts are used, this will be None.",
            max_length=MAX_NUMBER_OF_CONTEXTS,
        ),
    ] = None


class BalanceCheck(ApiBaseModel):
    """Describes balance test results for treatment assignment."""

    f_statistic: Annotated[
        float,
        Field(description="F-statistic testing the overall significance of the model predicting treatment assignment."),
    ]
    numerator_df: Annotated[
        int,
        Field(
            description="The numerator degrees of freedom for the f-statistic related to number of dependent variables."
        ),
    ]
    denominator_df: Annotated[
        int,
        Field(description="Denominator degrees of freedom related to the number of observations."),
    ]
    p_value: Annotated[
        float,
        Field(
            description=(
                "Probability of observing these data if strata do not predict treatment assignment, "
                "i.e. our randomization is balanced."
            )
        ),
    ]
    balance_ok: Annotated[
        bool,
        Field(
            description=(
                "Whether the p-value for our observed f_statistic is greater than the f-stat threshold "
                "specified in our design specification. (See DesignSpec.fstat_thresh)"
            )
        ),
    ]


class ArmSize(ApiBaseModel):
    """Describes the number of participants assigned to each arm."""

    arm: Arm
    size: int = 0


class CreateExperimentRequest(ApiBaseModel):
    design_spec: DesignSpec
    power_analyses: PowerResponse | None = None
    webhooks: Annotated[
        list[str],
        Field(
            description=(
                "List of webhook IDs to associate with this experiment. When the experiment is committed, "
                "these webhooks will be triggered with experiment details. Must contain unique values."
            ),
        ),
    ] = []

    @field_validator("webhooks")
    @classmethod
    def validate_unique_webhooks(cls, v: list[str]) -> list[str]:
        """Ensure all webhook IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Webhook IDs must be unique")
        return v


# TODO: make this class work with the Bayesian experiment types and their Draw records.
class AssignSummary(ApiBaseModel):
    """Key pieces of an AssignResponse without the assignments."""

    balance_check: Annotated[
        BalanceCheck | None,
        Field(description="Balance test results if available. 'online' experiments do not have balance checks."),
    ] = None
    sample_size: Annotated[int, Field(description="The number of participants across all arms in total.")]
    arm_sizes: Annotated[
        list[ArmSize] | None,
        Field(
            description="For each arm, the number of participants assigned.",
            max_length=MAX_NUMBER_OF_ARMS,
        ),
    ] = None


class ExperimentConfig(ApiBaseModel):
    """Representation of our stored Experiment information."""

    experiment_id: Annotated[str, Field(description="Server-generated ID of the experiment.")]
    datasource_id: str
    state: Annotated[ExperimentState, Field(description="Current state of this experiment.")]
    stopped_assignments_at: Annotated[
        datetime.datetime | None,
        Field(
            description="The date and time assignments were stopped. Null if assignments are still allowed to be made."
        ),
    ]
    stopped_assignments_reason: Annotated[
        StopAssignmentReason | None,
        Field(description="The reason assignments were stopped. Null if assignments are still allowed to be made."),
    ]
    design_spec: DesignSpec
    power_analyses: PowerResponse | None
    assign_summary: AssignSummary | None
    webhooks: Annotated[
        list[str],
        Field(
            description=(
                "List of webhook IDs associated with this experiment. "
                "These webhooks are triggered when the experiment is committed."
            ),
        ),
    ] = []


class GetExperimentResponse(ExperimentConfig):
    """An experiment configuration capturing all info at design time when assignment was made."""


class ListExperimentsResponse(ApiBaseModel):
    items: list[ExperimentConfig]


class GetParticipantAssignmentResponse(ApiBaseModel):
    """Describes assignment for a single <experiment, participant> pair."""

    experiment_id: str
    participant_id: str
    assignment: Annotated[
        Assignment | None,
        Field(description="Null if no assignment. assignment.strata are not included."),
    ]


class CreateExperimentResponse(ExperimentConfig):
    """Same as the request but with ids filled for the experiment and arms, and summary info on the assignment."""


class GetExperimentAssignmentsResponse(ApiBaseModel):
    """Describes assignments for all participants and balance test results if available."""

    balance_check: Annotated[
        BalanceCheck | None,
        Field(description="Balance test results if available. 'online' experiments do not have balance checks."),
    ] = None

    experiment_id: str
    sample_size: int
    assignments: list[Assignment]


class GetFiltersResponseBase(ApiBaseModel):
    field_name: Annotated[FieldName, Field(..., description="Name of the field.")]
    data_type: DataType
    relations: Annotated[list[Relation], Field(..., min_length=1, max_length=MAX_NUMBER_OF_FILTERS)]
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

    distinct_values: Annotated[list[str] | None, Field(..., description="Sorted list of unique values.")]


type GetFiltersResponseElement = GetFiltersResponseNumericOrDate | GetFiltersResponseDiscrete


class UpdateBanditArmOutcomeRequest(ApiBaseModel):
    """Describes the outcome of a bandit experiment."""

    outcome: float


def validate_gcp_service_account_info_json(serviceaccount_json):
    """Raises a ValueError if decoded does not resemble a JSON string containing GCP Service Account info."""
    try:
        creds = json.loads(serviceaccount_json)
        required_fields = {
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
        }
        if not all(field in creds for field in required_fields):
            raise ValueError("Missing required fields in service account JSON")
        if creds["type"] != "service_account":
            raise ValueError('Service account JSON must have type="service_account"')
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON in service account credentials") from e
    else:
        return serviceaccount_json


type GcpServiceAccountBlob = Annotated[
    str,
    MinLen(4),
    MaxLen(MAX_GCP_SERVICE_ACCOUNT_LEN),
    AfterValidator(validate_gcp_service_account_info_json),
    Field(
        description="The service account info in the canonical JSON form. Required fields: type, project_id, "
        "private_key_id, private_key, client_email."
    ),
]
