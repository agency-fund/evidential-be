from typing import ClassVar, Literal

from xngin.events.common import BaseEventModel
from xngin.tq.task_payload_types import WebhookOutboundTask


class WebhookSentEvent(BaseEventModel):
    """Describes the result of an outbound webhook request."""

    TYPE: ClassVar[Literal["webhook.sent"]] = "webhook.sent"

    type: Literal["webhook.sent"] = "webhook.sent"
    request: WebhookOutboundTask
    success: bool
    response: str

    def summarize(self) -> str:
        summary = f"Sent {self.request.method} to {self.request.url}"
        if self.success:
            summary += " (success)"
        else:
            summary += " (failed)"
        return summary
