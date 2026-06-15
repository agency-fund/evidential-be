import base64
import copy
import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import constants, settings
from xngin.apiserver.conftest import delete_seeded_users
from xngin.apiserver.routers.admin.admin_api_types import (
    ApiOnlyDsn,
    BqDsn,
    CreateDatasourceRequest,
    CreateOrganizationRequest,
    GcpServiceAccount,
    Hidden,
    PostgresDsn,
    RedshiftDsn,
    RevealedStr,
    UpdateDatasourceRequest,
)
from xngin.apiserver.routers.common_api_types import ExperimentsType
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.events.webhook_sent import WebhookSentEvent
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE, WebhookOutboundTask

SAMPLE_GCLOUD_SERVICE_ACCOUNT = {
    "auth_provider_x509_cert_url": "",
    "auth_uri": "",
    "client_email": "",
    "client_id": "",
    "client_x509_cert_url": "",
    "private_key": "",
    "private_key_id": "",
    "project_id": "",
    "token_uri": "",
    "type": "service_account",
    "universe_domain": "googleapis.com",
}
SAMPLE_GCLOUD_SERVICE_ACCOUNT_JSON = json.dumps(SAMPLE_GCLOUD_SERVICE_ACCOUNT)
_PERSISTED_WEBHOOK_TOKEN = "sample-token"


@pytest.mark.parametrize(
    "dsn",
    [
        PostgresDsn(
            host="127.0.0.1",
            user="postgres",
            port=5499,
            password=RevealedStr(value="postgres"),
            dbname="postgres",
            sslmode="disable",
            search_path=None,
        ),
        RedshiftDsn(
            host="foo.redshift.amazonaws.com",
            user="postgres",
            port=5499,
            password=RevealedStr(value="postgres"),
            dbname="postgres",
            search_path=None,
        ),
        RedshiftDsn(
            host="foo.redshift-serverless.amazonaws.com",
            user="postgres",
            port=5499,
            password=RevealedStr(value="postgres"),
            dbname="postgres",
            search_path=None,
        ),
        BqDsn(
            project_id="projectid",
            dataset_id="dataset_id",
            credentials=GcpServiceAccount(content=SAMPLE_GCLOUD_SERVICE_ACCOUNT_JSON),
        ),
    ],
    ids=lambda d: type(d),
)
async def test_datasources_hide_credentials(
    dsn: PostgresDsn | RedshiftDsn | BqDsn,
    xngin_session: AsyncSession,
    aclient: AdminAPIClient,
):
    org_id = aclient.create_organizations(
        body=CreateOrganizationRequest(name="test_datasources_hide_credentials")
    ).data.id

    datasource_id = aclient.create_datasource(
        body=CreateDatasourceRequest(name="test_create_datasource", organization_id=org_id, dsn=dsn)
    ).data.id

    datasource_response = aclient.get_datasource(datasource_id=datasource_id).data
    match datasource_response.dsn:
        case ApiOnlyDsn():
            raise TypeError("unexpected dsn type")
        case PostgresDsn() | RedshiftDsn():
            assert isinstance(datasource_response.dsn.password, Hidden)
        case BqDsn():
            assert isinstance(datasource_response.dsn.credentials, Hidden)

    before_revision = (await xngin_session.get_one(tables.Datasource, datasource_id)).get_config()
    match dsn:
        case PostgresDsn() | RedshiftDsn():
            revised_pg: PostgresDsn | RedshiftDsn = dsn.model_copy(deep=True)
            revised_pg.password = RevealedStr(value="updated")
            update_request = UpdateDatasourceRequest(dsn=revised_pg)
        case BqDsn():
            revised_service_account = copy.deepcopy(SAMPLE_GCLOUD_SERVICE_ACCOUNT)
            revised_service_account["private_key"] = "newprivatekey"
            revised_bq = dsn.model_copy(deep=True)
            revised_bq.credentials = GcpServiceAccount(content=json.dumps(revised_service_account))
            update_request = UpdateDatasourceRequest(dsn=revised_bq)
    aclient.update_datasource(datasource_id=datasource_id, body=update_request)

    after_revision = (await xngin_session.get_one(tables.Datasource, datasource_id)).get_config()
    match after_revision.dwh, before_revision.dwh:
        case settings.Dsn() as after, settings.Dsn() as before:
            assert after.host == before.host
            assert after.password != before.password
            assert after.password == "updated"
        case settings.BqDsn() as after, settings.BqDsn() as before:
            assert after.project_id == before.project_id
            assert after.credentials != before.credentials
            assert isinstance(after.credentials, settings.GcpServiceAccountInfo)
            assert "newprivatekey" in base64.standard_b64decode(after.credentials.content_base64).decode()
        case _:
            raise TypeError("unexpected dwh type")

    match dsn:
        case PostgresDsn() | RedshiftDsn():
            revised_pg = dsn.model_copy(deep=True)
            revised_pg.dbname = "newdatabase"
            revised_pg.password = Hidden()
            update_request = UpdateDatasourceRequest(dsn=revised_pg)
        case BqDsn():
            revised_bq = dsn.model_copy(deep=True)
            revised_bq.project_id = "newprojectid"
            revised_bq.credentials = Hidden()
            update_request = UpdateDatasourceRequest(dsn=revised_bq)
    aclient.update_datasource(datasource_id=datasource_id, body=update_request)

    after_second_revision = (await xngin_session.get_one(tables.Datasource, datasource_id)).get_config()
    match after_second_revision.dwh, after_revision.dwh:
        case settings.Dsn() as second, settings.Dsn() as first:
            assert second.dbname == "newdatabase"
            assert second.host == first.host
            assert second.password == first.password
            assert second.password == "updated"
        case settings.BqDsn() as second, settings.BqDsn() as first:
            assert second.project_id == "newprojectid"
            assert second.dataset_id == first.dataset_id
            assert second.credentials == first.credentials
            assert isinstance(second.credentials, settings.GcpServiceAccountInfo)
            assert "newprivatekey" in base64.standard_b64decode(second.credentials.content_base64).decode()
        case _:
            raise TypeError("unexpected dwh type")


async def _insert_webhook_sent_event(
    xngin_session: AsyncSession,
    organization_id: str,
    *,
    webhook_token_header: str = constants.HEADER_WEBHOOK_TOKEN,
) -> tuple[str, dict]:
    """Create a persisted webhook.sent event for resend tests without running the task queue."""
    outbound = WebhookOutboundTask(
        organization_id=organization_id,
        url="https://example.com/webhook",
        body={"hello": "world"},
        headers={webhook_token_header: _PERSISTED_WEBHOOK_TOKEN},
    )
    event = tables.Event(organization_id=organization_id, type=WebhookSentEvent.TYPE).set_data(
        WebhookSentEvent(request=outbound, success=False, response="boom")
    )
    xngin_session.add(event)
    await xngin_session.commit()
    return event.id, outbound.model_dump()


@pytest.mark.parametrize(
    "webhook_token_header", [constants.HEADER_WEBHOOK_TOKEN, constants.HEADER_WEBHOOK_TOKEN.lower()]
)
async def test_list_organization_events_redacts_webhook_token(
    xngin_session: AsyncSession,
    aclient: AdminAPIClient,
    webhook_token_header: str,
):
    """The webhook.sent event details surfaced by the API mask the webhook token header."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="resend-redact")).data.id
    event_id, _ = await _insert_webhook_sent_event(xngin_session, org_id, webhook_token_header=webhook_token_header)

    events = aclient.list_organization_events(organization_id=org_id).data.items
    event = next(ev for ev in events if ev.id == event_id)
    assert event.details is not None
    assert event.details["request"]["headers"][webhook_token_header] == "***"
    assert event.status_icon == "failure"


async def test_resend_organization_event_enqueues_task(
    xngin_session: AsyncSession,
    aclient: AdminAPIClient,
):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="resend-happy")).data.id
    event_id, expected_payload = await _insert_webhook_sent_event(xngin_session, org_id)

    aclient.resend_organization_event(organization_id=org_id, event_id=event_id)

    tasks = list(
        await xngin_session.scalars(select(tables.Task).where(tables.Task.task_type == WEBHOOK_OUTBOUND_TASK_TYPE))
    )
    assert len(tasks) == 1
    assert tasks[0].status == "pending"
    assert tasks[0].payload == expected_payload
    assert tasks[0].payload["headers"][constants.HEADER_WEBHOOK_TOKEN] == _PERSISTED_WEBHOOK_TOKEN


async def test_first_user_default_experiment_templates_created(
    xngin_session: AsyncSession, aclient_unpriv: AdminAPIClient
):
    await delete_seeded_users(xngin_session)

    organization = aclient_unpriv.list_organizations().data.items[0]
    experiments = aclient_unpriv.list_organization_experiments(organization_id=organization.id).data.items
    assert {experiment.design_spec.experiment_type for experiment in experiments} == {
        ExperimentsType.FREQ_PREASSIGNED,
        ExperimentsType.FREQ_ONLINE,
        ExperimentsType.MAB_ONLINE,
        ExperimentsType.CMAB_ONLINE,
    }
