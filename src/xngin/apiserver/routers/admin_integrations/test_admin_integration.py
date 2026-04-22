import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.conftest import expect_status_code
from xngin.apiserver.routers.admin.admin_api_types import CreateOrganizationRequest
from xngin.apiserver.routers.admin_integrations.admin_integration_api_types import (
    SetConnectionToTurnRequest,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.admin_integrations_api_client import AdminIntegrationsAPIClient


async def test_turn_connection_lifecycle(aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient):
    """Test creating, rotating, previewing, and deleting an organization's Turn.io connection."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_turn_connection_lifecycle")).data.id

    # GET before a connection exists -> 404.
    with expect_status_code(404):
        iaclient.get_organization_turn_connection(organization_id=org_id)

    # Create the connection.
    initial_token = "abcde" * 67  # 335 chars
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=initial_token),
    )

    # GET returns a preview of the last 4 chars of the token.
    preview = iaclient.get_organization_turn_connection(organization_id=org_id).data.token_preview
    assert preview == initial_token[-4:]

    # Rotate: PUT with a new token.
    rotated_token = "fghij" * 67  # 335 chars
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=rotated_token),
    )

    # Preview now reflects the new token.
    preview = iaclient.get_organization_turn_connection(organization_id=org_id).data.token_preview
    assert preview == rotated_token[-4:]

    # Delete.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # GET after delete -> 404.
    with expect_status_code(404):
        iaclient.get_organization_turn_connection(organization_id=org_id)

    # Delete again without allow_missing -> 404.
    with expect_status_code(404):
        iaclient.delete_turn_connection_from_organization(organization_id=org_id)

    # Delete again with allow_missing -> 204.
    iaclient.delete_turn_connection_from_organization(organization_id=org_id, allow_missing=True)


async def test_turn_connection_encrypted_at_rest(
    xngin_session: AsyncSession, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """The Turn.io API token must be encrypted at rest and recoverable via get_turn_api_token()."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_turn_connection_encrypted")).data.id

    token = "abcde" * 67  # 335 chars
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token=token),
    )

    row = (
        await xngin_session.execute(
            select(tables.TurnConnection).where(tables.TurnConnection.organization_id == org_id)
        )
    ).scalar_one()

    assert token not in row.encrypted_turn_api_token
    assert row.turn_api_token_preview == token[-4:]
    assert row.get_turn_api_token() == token


async def test_turn_journeys_caching(
    monkeypatch: pytest.MonkeyPatch, aclient: AdminAPIClient, iaclient: AdminIntegrationsAPIClient
):
    """GET /turn-connection/journeys serves from cache until TTL expires or the token is rotated."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="test_turn_journeys_caching")).data.id
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="a" * 335),
    )

    call_log: int = 0
    stacks: list[dict] = [{"name": "Arm A", "uuid": "arm-a-uuid"}]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, headers=None):
            nonlocal call_log
            call_log += 1
            return httpx.Response(
                status_code=200,
                json=list(stacks),
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    # First call: cache miss, one hit on the Turn API
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert call_log == 1
    assert journeys == {"Arm A": "arm-a-uuid"}

    # Second call inside the TTL: served from the DB cache.
    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert call_log == 1
    assert journeys == {"Arm A": "arm-a-uuid"}

    # Rotating the token must invalidate the cache.
    iaclient.set_organization_turn_connection(
        organization_id=org_id,
        body=SetConnectionToTurnRequest(turn_api_token="b" * 335),
    )
    stacks[:] = [{"name": "Arm B", "uuid": "arm-b-uuid"}]

    journeys = iaclient.get_organization_turn_journeys(organization_id=org_id).data.journeys
    assert call_log == 2
    assert journeys == {"Arm B": "arm-b-uuid"}
