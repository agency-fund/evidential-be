"""Internal models stored as json data in our app db. Used to decouple API types from storage."""

from typing import Annotated, Any, Self
from collections.abc import Sequence
from pydantic import BaseModel, ConfigDict, Field
from xngin.apiserver.limits import MAX_NUMBER_OF_FIELDS, MAX_NUMBER_OF_FILTERS
from xngin.apiserver.common_field_types import FieldName
from xngin.apiserver.routers.stateless_api_types import Relation, DesignSpec
from xngin.apiserver.routers.stateless_api_types import (
    Stratum as ApiStratum,
    DesignSpecMetricRequest as ApiDesignSpecMetricRequest,
    Filter as ApiFilter,
)


class StorageBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StorageStratum(StorageBaseModel):
    """Describes a variable used for stratification. See stateless_api_types.Stratum"""

    field_name: FieldName


class StorageFilter(StorageBaseModel):
    """Defines criteria for filtering rows by value. See stateless_api_types.Filter"""

    field_name: FieldName
    relation: Relation
    # Storage-specific simplified version of stateless_api_types.FilterValueTypes.
    # Detailed type validation is in the API layer.
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
        Field(
            description="Absolute target value = metric_baseline*(1 + metric_pct_change)"
        ),
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
            description="Optional filters that constrain a general participant_type to a specific subset who can participate in an experiment.",
            max_length=MAX_NUMBER_OF_FILTERS,
        ),
    ] = None

    @classmethod
    def from_design_spec(cls, design_spec: DesignSpec) -> Self:
        """Converts a DesignSpec to a new DesignSpecFields object."""

        storage_strata = None
        if design_spec.strata:
            storage_strata = [
                StorageStratum(field_name=s.field_name) for s in design_spec.strata
            ]

        storage_metrics = None
        if design_spec.metrics:
            storage_metrics = [
                StorageMetric(
                    field_name=m.field_name,
                    metric_pct_change=m.metric_pct_change,
                    metric_target=m.metric_target,
                )
                for m in design_spec.metrics
            ]

        storage_filters = None
        if design_spec.filters:
            storage_filters = [
                StorageFilter(
                    field_name=f.field_name,
                    relation=f.relation,
                    value=f.value,
                )
                for f in design_spec.filters
            ]

        return cls(
            strata=storage_strata,
            metrics=storage_metrics,
            filters=storage_filters,
        )

    def get_api_strata(self) -> list[ApiStratum]:
        """Converts stored strata to API Stratum objects."""
        if self.strata is None:
            return []
        return [ApiStratum(field_name=s.field_name) for s in self.strata]

    def get_api_metrics(self) -> list[ApiDesignSpecMetricRequest]:
        """Converts stored metrics to API DesignSpecMetricRequest objects."""
        if self.metrics is None:
            return []
        return [
            ApiDesignSpecMetricRequest(
                field_name=m.field_name,
                metric_pct_change=m.metric_pct_change,
                metric_target=m.metric_target,
            )
            for m in self.metrics
        ]

    def get_api_filters(self) -> list[ApiFilter]:
        """Converts stored filters to API Filter objects."""
        if self.filters is None:
            return []
        # The `value` field in StorageFilter is Sequence[Any].
        # Pydantic will validate when creating ApiFilter.
        return [
            ApiFilter(
                field_name=f.field_name,
                relation=f.relation,
                value=f.value,
            )
            for f in self.filters
        ]
