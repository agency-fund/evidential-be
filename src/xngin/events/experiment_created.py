from typing import Annotated, Literal

from pydantic import Field
from xngin.events.common import BaseEventModel
from xngin.tq.task_types import WebhookOutboundTask


class ExperimentCreated(BaseEventModel):
    type: Literal["experiment.created"] = "experiment.created"
    experiment_id: Annotated[str, Field(description="The experiment ID.")]

class WebhookSent(BaseEventModel):
    type: Literal["webhook.sent"] = "webhook.sent"
    request: WebhookOutboundTask
    success: bool
    response: str

    def summarize(self)-> str:
        summary = f"Sent {self.request.method} to {self.request.url}"
        if self.success:
            summary += " (success)"
        else:
            summary += " (failed)"
        return summary
