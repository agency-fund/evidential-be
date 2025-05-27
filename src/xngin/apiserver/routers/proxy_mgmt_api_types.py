"""This defines the various webhook request/response contracts as Pydantic models."""

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from xngin.apiserver.routers.stateless_api_types import (
    AssignResponse,
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
    power_analyses: PowerResponse | None = None
    experiment_assignment: AssignResponse

    @field_serializer("experiment_commit_datetime", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()
