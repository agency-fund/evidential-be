"""Internal models stored as json data in our app db. Used to decouple API types from storage."""

from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from xngin.apiserver.common_field_types import FieldName
from xngin.apiserver.limits import MAX_NUMBER_OF_FIELDS, MAX_NUMBER_OF_FILTERS


class StorageBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StorageStratum(StorageBaseModel):
    """Describes a variable used for stratification. See stateless_api_types.Stratum"""

    field_name: FieldName


class StorageFilter(StorageBaseModel):
    """Defines criteria for filtering rows by value. See stateless_api_types.Filter"""

    field_name: FieldName
    # Relaxed type for storage of stateless_api_types.Relation
    relation: str
    # Simplified type for storage of stateless_api_types.FilterValueTypes.
    value: Sequence[Any]


class StorageMetric(StorageBaseModel):
    """Defines a metric to target. See stateless_api_types.DesignSpecMetricRequest"""

    field_name: FieldName
    metric_pct_change: Annotated[
        float | None,
        Field(description="Percent change target relative to the metric_baseline."),
    ] = None
    metric_target: Annotated[
        float | None,
        Field(description="Absolute target value = metric_baseline*(1 + metric_pct_change)"),
    ] = None


class DesignSpecFields(StorageBaseModel):
    """Holds the Participant Type fields specified in an experiment's DesignSpec."""

    strata: Annotated[
        list[StorageStratum] | None,
        Field(
            description="Optional participant_type fields to use for stratified assignment.",
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None

    metrics: Annotated[
        list[StorageMetric] | None,
        Field(
            ...,
            description="Primary and optional secondary metrics to target.",
            min_length=1,
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None

    filters: Annotated[
        list[StorageFilter] | None,
        Field(
            description=(
                "Optional filters that constrain a general participant_type to a specific subset "
                "who can participate in an experiment."
            ),
            max_length=MAX_NUMBER_OF_FILTERS,
        ),
    ] = None

    arm_weights: Annotated[
        list[float] | None,
        Field(
            description="Optional weights for unequal arm sizes. Weights must be floats in (0, 100) and sum to 100.",
        ),
    ] = None
