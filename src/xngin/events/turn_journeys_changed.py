from typing import Annotated, ClassVar, Literal

from pydantic import Field

from xngin.events.common import BaseEventModel


class TurnJourneysChangedEvent(BaseEventModel):
    """Describes the result of an inbound webhook request."""

    TYPE: ClassVar[Literal["turn.journeys_changed"]] = "turn.journeys_changed"

    type: Literal["turn.journeys_changed"] = "turn.journeys_changed"
    organization_id: Annotated[str, Field(description="The organization ID.")]
    success: bool
    response: str

    def summarize(self) -> str:
        return f"Journeys changed for {self.organization_id}"
