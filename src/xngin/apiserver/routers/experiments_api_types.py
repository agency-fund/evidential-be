import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from xngin.apiserver.api_types import (
    ArmSize,
    DesignSpec,
    AudienceSpec,
    PowerResponse,
    BalanceCheck,
    Assignment,
)
from xngin.apiserver.models.enums import ExperimentState


class ExperimentsBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateExperimentRequest(ExperimentsBaseModel):
    design_spec: DesignSpec
    audience_spec: AudienceSpec
    power_analyses: PowerResponse | None = None


class AssignSummary(ExperimentsBaseModel):
    """Key pieces of an AssignResponse without the assignments."""

    balance_check: BalanceCheck
    sample_size: Annotated[
        int, Field(description="The number of participants across all arms in total.")
    ]
    arm_sizes: Annotated[
        list[ArmSize] | None,
        Field(
            description="For each arm, the number of participants assigned. "
            "TODO: make required once development has stabilized. May be None if unknown due to persisting prior versions of an AssignSummary."
        ),
    ] = None


class ExperimentConfig(ExperimentsBaseModel):
    """Representation of our stored Experiment information."""

    datasource_id: str
    state: Annotated[
        ExperimentState, Field(description="Current state of this experiment.")
    ]
    design_spec: DesignSpec
    audience_spec: AudienceSpec
    power_analyses: PowerResponse | None
    assign_summary: AssignSummary


class CreateExperimentWithAssignmentResponse(ExperimentConfig):
    """Same as the request but with uuids filled for the experiment and arms, and summary info on the assignment."""


class GetExperimentResponse(ExperimentConfig):
    """An experiment configuration capturing all info at design time when assignment was made."""


class ListExperimentsResponse(ExperimentsBaseModel):
    items: list[ExperimentConfig]


class GetExperimentAssigmentsResponse(ExperimentsBaseModel):
    """Describes assignments for all participants and balance test results."""

    balance_check: BalanceCheck

    experiment_id: uuid.UUID
    sample_size: int
    assignments: list[Assignment]
