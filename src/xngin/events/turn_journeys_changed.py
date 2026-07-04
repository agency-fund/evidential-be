from typing import Annotated, ClassVar, Literal

from pydantic import Field

from xngin.events.common import BaseEventModel


class TurnJourneysChangedEvent(BaseEventModel):
    """Describes the result of an inbound webhook request."""

    TYPE: ClassVar[Literal["turn.journeys_changed"]] = "turn.journeys_changed"

    type: Literal["turn.journeys_changed"] = "turn.journeys_changed"
    organization_id: Annotated[str, Field(description="The organization ID.")]
    webhook_id: Annotated[str, Field(description="The webhook ID")]
    success: bool
    response: str

    def status_icon(self) -> Literal["success", "failure"]:
        return "success" if self.success else "failure"

    def summarize(self) -> str:
        return f"Journeys changed for {self.organization_id}"
