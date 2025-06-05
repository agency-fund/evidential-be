from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from xngin.apiserver.limits import MAX_NUMBER_OF_ARMS
from xngin.apiserver.models.enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.routers.stateless_api_types import (
    ArmSize,
    Assignment,
    BalanceCheck,
    DesignSpec,
    PowerResponse,
)


class ExperimentsBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateExperimentRequest(ExperimentsBaseModel):
    design_spec: DesignSpec
    power_analyses: PowerResponse | None = None


class AssignSummary(ExperimentsBaseModel):
    """Key pieces of an AssignResponse without the assignments."""

    balance_check: Annotated[
        BalanceCheck | None,
        Field(
            description="Balance test results if available. 'online' experiments do not have balance checks."
        ),
    ] = None
    sample_size: Annotated[
        int, Field(description="The number of participants across all arms in total.")
    ]
    arm_sizes: Annotated[
        list[ArmSize] | None,
        Field(
            description="For each arm, the number of participants assigned. "
            "TODO: make required once development has stabilized. May be None if unknown due to persisting prior versions of an AssignSummary.",
            max_length=MAX_NUMBER_OF_ARMS,
        ),
    ] = None


class ExperimentConfig(ExperimentsBaseModel):
    """Representation of our stored Experiment information."""

    datasource_id: str
    state: Annotated[
        ExperimentState, Field(description="Current state of this experiment.")
    ]
    stopped_assignments_at: Annotated[
        datetime | None,
        Field(
            description="The date and time assignments were stopped. Null if assignments are still allowed to be made."
        ),
    ]
    stopped_assignments_reason: Annotated[
        StopAssignmentReason | None,
        Field(
            description="The reason assignments were stopped. Null if assignments are still allowed to be made."
        ),
    ]
    design_spec: DesignSpec
    power_analyses: PowerResponse | None
    assign_summary: AssignSummary


class CreateExperimentResponse(ExperimentConfig):
    """Same as the request but with ids filled for the experiment and arms, and summary info on the assignment."""


class GetExperimentResponse(ExperimentConfig):
    """An experiment configuration capturing all info at design time when assignment was made."""


class ListExperimentsResponse(ExperimentsBaseModel):
    items: list[ExperimentConfig]


class GetExperimentAssignmentsResponse(ExperimentsBaseModel):
    """Describes assignments for all participants and balance test results if available."""

    balance_check: Annotated[
        BalanceCheck | None,
        Field(
            description="Balance test results if available. 'online' experiments do not have balance checks."
        ),
    ] = None

    experiment_id: str
    sample_size: int
    assignments: list[Assignment]


class GetParticipantAssignmentResponse(ExperimentsBaseModel):
    """Describes assignment for a single <experiment, participant> pair."""

    experiment_id: str
    participant_id: str
    assignment: Annotated[
        Assignment | None,
        Field(description="Null if no assignment. assignment.strata are not included."),
    ]
