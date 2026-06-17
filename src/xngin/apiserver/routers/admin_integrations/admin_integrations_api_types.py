from typing import Annotated

from pydantic import Field, field_validator

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

    @field_validator("turn_api_token")
    @classmethod
    def validate_turn_api_token(cls, v: str) -> str:
        """Check for newline or carriage return characters to prevent HTTP header injection."""
        if "\n" in v or "\r" in v:
            raise ValueError("Turn API token must not contain newline or carriage return characters.")
        return v


class Journey(AdminApiBaseModel):
    name: Annotated[str, Field(description=("The name of the Turn.io journey, as defined in the Turn.io platform."))]
    uuid: Annotated[
        str,
        Field(description=("The unique identifier (UUID) of the Turn.io journey, as defined in the Turn.io platform.")),
    ]


class GetTurnJourneysResponse(AdminApiBaseModel):
    """Response describing an organization's Turn.io journeys."""

    journeys: list[Journey]


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

    stale_arm_ids: Annotated[
        list[str],
        Field(
            description=(
                "List of experiment arm IDs that are no longer valid based on the "
                "most recent data from the Turn API. This can be used to identify and clean up stale mappings "
                "after changes to Turn.io journeys or API tokens."
            )
        ),
    ]
