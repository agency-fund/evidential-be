from typing import Literal

from pydantic import BaseModel


class ExperimentCreatedWebhookBody(BaseModel):
    """Defines the HTTP request body for a WebhookOutboundTask when notifying webhook endpoints on experiment commit."""

    type: Literal["experiment.created"] = "experiment.created"
    organization_id: str
    datasource_id: str
    experiment_id: str
    experiment_url: str
