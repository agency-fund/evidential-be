from typing import Annotated

from pydantic import Field

from xngin.apiserver.routers.admin.admin_api_types import AdminApiBaseModel


class SetConnectionToTurnRequest(AdminApiBaseModel):
    """Request to create or update an organization's Turn.io connection."""

    turn_api_token: Annotated[
        str,
        Field(
            min_length=335,
            description=(
                "The Turn.io API token used to authenticate calls to the Turn API on behalf of the organization."
            ),
        ),
    ]


class GetTurnJourneysResponse(AdminApiBaseModel):
    """Response describing an organization's Turn.io journeys."""

    journeys: Annotated[
        dict[str, str],
        Field(
            description=(
                "Mapping of journey names to their corresponding IDs, retrieved from the Turn API. This allows "
                "admins to reference specific journeys when configuring experiments that integrate with Turn.io."
            )
        ),
    ]


class GetTurnConnectionResponse(AdminApiBaseModel):
    """Response describing an organization's Turn.io connection."""

    token_preview: Annotated[
        str,
        Field(
            description=(
                "The last 4 characters of the configured Turn.io API token, shown so admins can identify "
                "which token is currently configured without exposing the full secret."
            )
        ),
    ]


class SetTurnArmJourneyMappingRequest(AdminApiBaseModel):
    """Request to create or update the mapping between experiment arms and Turn.io journeys."""

    arm_to_journeys: Annotated[
        dict[str, str],
        Field(
            description=(
                "Mapping of experiment arm IDs to Turn.io journey IDs. This configures which Turn.io "
                "journey each arm corresponds to for experiments that integrate with Turn.io."
            )
        ),
    ]


class GetTurnArmJourneyMappingResponse(SetTurnArmJourneyMappingRequest):
    """Response describing the mapping between experiment arms and Turn.io journeys."""
