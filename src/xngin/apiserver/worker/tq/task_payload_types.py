"""Pydantic types of the values in the table.Task.payload field."""

from typing import Literal

from pydantic import BaseModel

from xngin.apiserver.constants import HEADER_WEBHOOK_TOKEN

WEBHOOK_OUTBOUND_TASK_TYPE = "webhook.outbound"
TURN_JOURNEYS_CHANGED_TASK_TYPE = "turn.journeys_changed"
_WEBHOOK_TOKEN_HEADER = HEADER_WEBHOOK_TOKEN.casefold()


class WebhookOutboundTask(BaseModel):
    """Defines the payload understood by webhook_outbound_handler."""

    url: str
    body: dict = dict()
    headers: dict[str, str] = dict()
    organization_id: str
    method: Literal["POST", "GET"] = "POST"

    def sanitize(self):
        sanitized_headers = {
            header: "***" if header.casefold() == _WEBHOOK_TOKEN_HEADER else value
            for header, value in self.headers.items()
        }
        if sanitized_headers == self.headers:
            return self
        return self.model_copy(update={"headers": sanitized_headers})


class TurnJourneysChangedTask(BaseModel):
    """Defines the payload understood by turn_journeys_changed handler."""

    organization_id: str
    webhook_id: str
    webhook_auth_token: str
