"""This defines the various webhook request/response contracts as Pydantic models."""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from xngin.apiserver.routers.stateless_api_types import (
    AssignResponse,
    AudienceSpec,
    DesignSpec,
    PowerResponse,
)


class WebhookBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebhookResponse(WebhookBaseModel):
    """Generic wrapper around upstream webhook responses."""

    status_code: int = Field(
        description="HTTP status code we received from the webhook's server."
    )
    body: str = Field(
        description="HTTP body (if any) we received from the webhook's server. May be empty."
    )

    @classmethod
    def from_httpx(cls, response: httpx.Response):
        """Create WebhookResponse from an httpx.Response object."""
        # No need to parse the response text as json, just pass it through.
        return cls(status_code=response.status_code, body=response.text)


# Dict of extra responses to use with all webhook-related endpoints. See:
# https://fastapi.tiangolo.com/advanced/additional-responses/?h=responses#additional-response-with-model
# for how to use with path operation decorators.
STANDARD_WEBHOOK_RESPONSES: dict[int, dict[str, Any]] = {
    502: {
        "model": WebhookResponse,
        "description": "Webhook service returned a non-200 code.",
    }
}


class WebhookCommitRequest(WebhookBaseModel):
    """Data model for experiment commit webhook payload."""

    experiment_commit_datetime: datetime = Field(
        description="timestamp when the experiment was committed",
        default_factory=lambda: datetime.now(UTC),
    )
    experiment_commit_id: uuid.UUID = Field(
        description="unique identifier for this experiment commit",
        default_factory=uuid.uuid4,
    )
    creator_user_id: str = Field(description="ID of the user creating the experiment")
    design_spec: DesignSpec
    audience_spec: AudienceSpec
    power_analyses: PowerResponse | None = None
    experiment_assignment: AssignResponse

    @field_serializer("experiment_commit_datetime", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


class WebhookUpdateTimestampsRequest(WebhookBaseModel):
    """Describes how to update an experiment's start and/or end dates."""

    experiment_id: uuid.UUID = Field(description="ID of the experiment to update.")
    start_date: datetime = Field(
        description="New or the same start date to update with."
    )
    end_date: datetime = Field(
        description="New or the same end date to update with. Must be later "
        "than start_date."
    )

    @model_validator(mode="after")
    def validate_end_date_after_start(self) -> Self:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


class ArmUpdate(WebhookBaseModel):
    arm_name: str = Field(
        description="New experiment arm name to be updated.", min_length=1
    )
    arm_id: uuid.UUID = Field(
        description="The id originally assigned to this arm by the user."
    )


class WebhookUpdateDescriptionRequest(WebhookBaseModel):
    """Describes how to update an experiment description and/or the names of its arms."""

    experiment_id: uuid.UUID = Field(description="ID of the experiment to update.")
    experiment_name: str = Field(description="New experiment name.", min_length=1)
    description: str = Field(description="New experiment description.", min_length=1)
    arms: list[ArmUpdate] = Field(
        description="All arms as saved in the original DesignSpec must be present here, even if "
        "you don't intend to change the arm_name"
    )


class WebhookUpdateCommitRequest(WebhookBaseModel):
    """Request structure for supported types of experiment updates."""

    update_json: WebhookUpdateTimestampsRequest | WebhookUpdateDescriptionRequest


# TODO: as part of potential API endpoint revisions
class UpdateExperimentStartEndRequest(WebhookBaseModel):
    """WIP to alternate interface to updating an experiment"""

    update_type: Literal["timestamps"]
    start_date: datetime
    end_date: datetime


# TODO: as part of potential API endpoint revisions
class UpdateExperimentDescriptionsRequest(WebhookBaseModel):
    """WIP to alternate interface to updating an experiment name & description."""

    update_type: Literal["description"]
    experiment_name: str = Field(description="New experiment name.", min_length=1)
    description: str = Field(description="New experiment description.", min_length=1)
