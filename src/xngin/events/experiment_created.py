from typing import Annotated, ClassVar, Literal

from pydantic import Field

from xngin.events.common import BaseEventModel


class ExperimentCreatedEvent(BaseEventModel):
    """Describes which experiment was created."""

    TYPE: ClassVar[Literal["experiment.created"]] = "experiment.created"

    type: Literal["experiment.created"] = TYPE
    datasource_id: Annotated[str | None, Field(description="The datasource ID.")] = None
    experiment_id: Annotated[str, Field(description="The experiment ID.")]

    def summarize(self) -> str:
        return f"Created experiment {self.experiment_id}"

    def link(self) -> str | None:
        if not self.datasource_id:
            # Some of the initial ExperimentCreatedEvent messages do not have a datasource_id field so we
            # cannot construct an unambiguous URL for them.
            return None
        return (
            f"/datasources/{self.datasource_id}/experiments/view/{self.experiment_id}"
        )
