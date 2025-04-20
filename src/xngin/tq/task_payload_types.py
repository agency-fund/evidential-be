"""Pydantic types of the values in the table.Task.payload field."""

from typing import Literal

from pydantic import BaseModel

WEBHOOK_OUTBOUND_TASK_TYPE = "webhook.outbound"


class WebhookOutboundTask(BaseModel):
    """Defines the payload of a Task of type = WEBHOOK_OUTBOUND_TASK_TYPE.

    Consumed by webhook_outbound_handler.
    """

    url: str
    body: dict = dict()
    headers: dict[str, str] = dict()
    organization_id: str
    method: Literal["POST", "GET"] = "POST"
