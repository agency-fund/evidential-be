from typing import Annotated, Literal

from pydantic import Field
from xngin.events.common import BaseEventModel


class ExperimentCreated(BaseEventModel):
    type: Literal["experiment.created"] = "experiment.created"
    experiment_id: Annotated[str, Field(description="The experiment ID.")]
