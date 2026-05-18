import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.routers.admin.admin_api_types import CreateOrganizationRequest
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.tq.handlers import make_webhook_outbound_handler
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE
from xngin.tq.task_queue import TaskQueue
from xngin.tq.tq_test_support import insert_task, tq_runner, wait_for_task_status

pytest_plugins = ("xngin.apiserver.conftest",)


def _create_organization(aclient: AdminAPIClient, name: str) -> str:
    return aclient.create_organizations(body=CreateOrganizationRequest(name=name)).data.id


async def test_webhook_outbound_handler_records_dns_failure_event(
    aclient: AdminAPIClient,
    xngin_session: AsyncSession,
    tq_dsn: str,
):
    org_id = _create_organization(aclient, "test-webhook-dns-failure")

    task_queue = TaskQueue(dsn=tq_dsn, max_retries=0, poll_interval_secs=1)
    task_queue.register_handler(WEBHOOK_OUTBOUND_TASK_TYPE, make_webhook_outbound_handler(tq_dsn))
    with tq_runner(task_queue):
        task = await insert_task(
            xngin_session,
            task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
            payload={
                "url": f"http://{safe_resolve.UNSAFE_IP_FOR_TESTING}/hook",
                "organization_id": org_id,
            },
        )
        dead_task = await wait_for_task_status(task.id, "dead")

    assert dead_task.message == "DNS issue with host: Detected sentinel value of invalid IP used for testing purposes."

    events = aclient.list_organization_events(organization_id=org_id).data.items

    assert len(events) == 1
    event = events[0]
    assert event.type == "webhook.sent"
    assert event.details is not None
    assert event.details["success"] is False
    assert "Failed to resolve hostname" in event.details["response"]


async def test_webhook_outbound_handler_records_success_event(
    aclient: AdminAPIClient,
    xngin_session: AsyncSession,
    tq_dsn: str,
):
    org_id = _create_organization(aclient, "test-webhook-success")

    transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=0, poll_interval_secs=1)
    task_queue.register_handler(WEBHOOK_OUTBOUND_TASK_TYPE, make_webhook_outbound_handler(tq_dsn, transport=transport))
    with tq_runner(task_queue):
        task = await insert_task(
            xngin_session,
            task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
            payload={
                "url": "http://203.0.113.1/hook",
                "organization_id": org_id,
            },
        )
        success_task = await wait_for_task_status(task.id, "success")

    assert success_task.message is None

    events = aclient.list_organization_events(organization_id=org_id).data.items

    assert len(events) == 1
    event = events[0]
    assert event.type == "webhook.sent"
    assert event.details is not None
    assert event.details["success"] is True
    assert event.details["response"] == "200"


async def test_webhook_outbound_handler_preserves_url_credentials(
    aclient: AdminAPIClient,
    xngin_session: AsyncSession,
    tq_dsn: str,
):
    org_id = _create_organization(aclient, "test-webhook-url-credentials")

    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, request=request)

    task_queue = TaskQueue(dsn=tq_dsn, max_retries=0, poll_interval_secs=1)
    task_queue.register_handler(
        WEBHOOK_OUTBOUND_TASK_TYPE,
        make_webhook_outbound_handler(tq_dsn, transport=httpx.MockTransport(handler)),
    )
    with tq_runner(task_queue):
        task = await insert_task(
            xngin_session,
            task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
            payload={
                "url": "http://infra:password@localhost:8000/hook",
                "organization_id": org_id,
            },
        )
        success_task = await wait_for_task_status(task.id, "success")

    assert success_task.message is None
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.url.host == "127.0.0.1"
    assert request.url.port == 8000
    assert request.headers["host"] == "localhost:8000"
    assert request.url.username == "infra"
    assert request.url.password == "password"
    # Credentials in the URL are emitted as a Basic auth header by httpx
    assert request.headers["authorization"] == "Basic aW5mcmE6cGFzc3dvcmQ="
