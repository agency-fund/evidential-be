from typing import Annotated, Literal

from pydantic import Field
from xngin.events.common import BaseEventModel


class ExperimentCreatedEvent(BaseEventModel):
    """Describes which experiment was created."""

    type: Literal["experiment.created"] = "experiment.created"
    experiment_id: Annotated[str, Field(description="The experiment ID.")]

    def summarize(self) -> str:
        return f"Created experiment {self.experiment_id}"

    def link(self) -> str | None:
        return f"/experiments/view/{self.experiment_id}"
