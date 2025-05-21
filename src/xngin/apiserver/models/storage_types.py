"""Internal models stored as json data in our app db. Used to decouple API types from storage."""

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field
from xngin.apiserver.limits import MAX_NUMBER_OF_FIELDS, MAX_NUMBER_OF_FILTERS
from xngin.apiserver.routers.stateless_api_types import (
    DesignSpecMetricRequest,
    Filter,
    Stratum,
)


class StorageBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DesignSpecFields(StorageBaseModel):
    """Holds the Participant Type fields specified in an experiment's DesignSpec."""

    strata: Annotated[
        list[Stratum] | None,
        Field(
            description="Optional participant_type fields to use for stratified assignment.",
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None

    metrics: Annotated[
        list[DesignSpecMetricRequest] | None,
        Field(
            ...,
            description="Primary and optional secondary metrics to target.",
            min_length=1,
            max_length=MAX_NUMBER_OF_FIELDS,
        ),
    ] = None

    filters: Annotated[
        list[Filter] | None,
        Field(
            description="Optional filters that constrain a general participant_type to a specific subset who can participate in an experiment.",
            max_length=MAX_NUMBER_OF_FILTERS,
        ),
    ] = None
