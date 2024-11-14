"""This defines the various webhook request/response contracts as pydantic models."""

from datetime import datetime
import uuid

import httpx
from pydantic import BaseModel, Field, ConfigDict

from xngin.apiserver.api_types import DesignSpec, AudienceSpec, ExperimentAssignment


class WebhookResponse(BaseModel):
    """Generic wrapper around downstream webhook responses."""

    proxied_response: str

    @classmethod
    def from_httpx(cls, response: httpx.Response):
        """Create WebhookResponse from an httpx.Response object."""
        # No need to parse the response text as json, just pass it through.
        return cls(proxied_response=response.text)


class WebhookRequestCommit(BaseModel):
    """Data model for experiment commit webhook payload."""

    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat()})

    experiment_commit_datetime: datetime = Field(
        description="timestamp when the experiment was committed",
        default_factory=datetime.now,
    )
    experiment_commit_id: uuid.UUID = Field(
        description="unique identifier for this experiment commit",
        default_factory=uuid.uuid4,
    )
    creator_user_id: str = Field(description="ID of the user creating the experiment")
    experiment_assignment: ExperimentAssignment
    design_spec: DesignSpec
    audience_spec: AudienceSpec
