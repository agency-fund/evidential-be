"""This defines the various webhook request/response contracts as pydantic models."""

from datetime import datetime
from typing import List
import uuid

import httpx
from pydantic import BaseModel, Field, field_serializer

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

    @field_serializer("experiment_commit_datetime", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


class WebhookRequestUpdateTimestamps(BaseModel):
    experiment_id: uuid.UUID
    start_date: datetime
    end_date: datetime

    @field_serializer("start_date", "end_date", when_used="json")
    def serialize_dt(self, dt: datetime, _info):
        """Convert dates to iso strings in model_dump_json()/model_dump(mode='json')"""
        return dt.isoformat()


class ExperimentArm(BaseModel):
    arm_name: str
    arm_id: uuid.UUID


class WebhookRequestUpdateDescriptions(BaseModel):
    experiment_id: uuid.UUID
    description: str
    arms: List[ExperimentArm]


class WebhookRequestUpdateContainer(BaseModel):
    """Wrapper around experiment update types."""

    update_json: WebhookRequestUpdateTimestamps | WebhookRequestUpdateDescriptions
