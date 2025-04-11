from typing import Annotated, Literal

from pydantic import Field
from xngin.events.common import BaseEventModel


class ExperimentCreated(BaseEventModel):
    type: Annotated[Literal["experiment.created"], Field(default="experiment.created")]
    experiment_id: Annotated[str, Field(description="The experiment ID.")]

    @staticmethod
    def create(experiment_id: str):
        return ExperimentCreated(experiment_id=experiment_id)
