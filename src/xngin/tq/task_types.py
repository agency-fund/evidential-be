from typing import Literal

from pydantic import BaseModel


class WebhookOutboundTask(BaseModel):
    """Defines the payload of a Task of type = "webhook.outbound"."""

    url: str
    body: dict = dict()
    headers: dict[str, str] = dict()
    organization_id: str
    method: Literal["POST", "GET"] = "POST"
