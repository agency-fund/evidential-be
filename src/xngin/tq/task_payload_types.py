"""Pydantic types of the values in the table.Task.payload field."""

from typing import Literal

from pydantic import BaseModel


class WebhookOutboundTask(BaseModel):
    """Defines the payload of a Task of type = "webhook.outbound".

    Consumed by webhook_outbound_handler.
    """

    url: str
    body: dict = dict()
    headers: dict[str, str] = dict()
    organization_id: str
    method: Literal["POST", "GET"] = "POST"
