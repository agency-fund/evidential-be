import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.sqla import tables
from xngin.events.webhook_sent import WebhookSentEvent
from xngin.tq.handlers import make_webhook_outbound_handler
from xngin.tq.task_payload_types import WEBHOOK_OUTBOUND_TASK_TYPE
from xngin.tq.task_queue import TaskQueue
from xngin.tq.tq_test_support import insert_task, tq_runner, wait_for_task_status

pytest_plugins = ("xngin.apiserver.conftest",)


async def test_webhook_outbound_handler_records_dns_failure_event(
    xngin_session: AsyncSession,
    tq_dsn: str,
):
    org = tables.Organization(name="test-org")
    xngin_session.add(org)
    await xngin_session.commit()
    await xngin_session.refresh(org)

    task_queue = TaskQueue(dsn=tq_dsn, max_retries=0, poll_interval_secs=1)
    task_queue.register_handler(WEBHOOK_OUTBOUND_TASK_TYPE, make_webhook_outbound_handler(tq_dsn))
    with tq_runner(task_queue):
        task = await insert_task(
            xngin_session,
            task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
            payload={
                "url": f"http://{safe_resolve.UNSAFE_IP_FOR_TESTING}/hook",
                "organization_id": org.id,
            },
        )
        dead_task = await wait_for_task_status(task.id, "dead")

    assert dead_task.message == "DNS issue with host: Detected sentinel value of invalid IP used for testing purposes."

    events = (
        (await xngin_session.execute(select(tables.Event).where(tables.Event.organization_id == org.id)))
        .scalars()
        .all()
    )

    assert len(events) == 1
    event_data = events[0].get_data()
    assert isinstance(event_data, WebhookSentEvent)
    assert event_data.success is False
    assert "Failed to resolve hostname" in event_data.response


async def test_webhook_outbound_handler_records_success_event(
    xngin_session: AsyncSession,
    tq_dsn: str,
):
    org = tables.Organization(name="test-org")
    xngin_session.add(org)
    await xngin_session.commit()
    await xngin_session.refresh(org)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    task_queue = TaskQueue(dsn=tq_dsn, max_retries=0, poll_interval_secs=1)
    task_queue.register_handler(WEBHOOK_OUTBOUND_TASK_TYPE, make_webhook_outbound_handler(tq_dsn, transport=transport))
    with tq_runner(task_queue):
        task = await insert_task(
            xngin_session,
            task_type=WEBHOOK_OUTBOUND_TASK_TYPE,
            payload={
                "url": "http://203.0.113.1/hook",
                "organization_id": org.id,
            },
        )
        success_task = await wait_for_task_status(task.id, "success")

    assert success_task.message is None

    events = (
        (await xngin_session.execute(select(tables.Event).where(tables.Event.organization_id == org.id)))
        .scalars()
        .all()
    )

    assert len(events) == 1
    event_data = events[0].get_data()
    assert isinstance(event_data, WebhookSentEvent)
    assert event_data.success is True
    assert event_data.response == "200"
