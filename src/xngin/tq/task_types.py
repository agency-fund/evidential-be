from typing import Literal

from pydantic import BaseModel


class WebhookOutboundTask(BaseModel):
    """Defines the payload of a Task of type = "webhook.outbound"."""

    url: str
    payload: dict = dict()
    headers: dict[str, str] = dict()
    method: Literal["POST", "GET"] = "POST"
