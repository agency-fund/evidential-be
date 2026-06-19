import pytest

from xngin.apiserver import constants
from xngin.tq.task_payload_types import WebhookOutboundTask


@pytest.mark.parametrize(
    "webhook_token_header",
    [
        constants.HEADER_WEBHOOK_TOKEN,
        constants.HEADER_WEBHOOK_TOKEN.lower(),
        constants.HEADER_WEBHOOK_TOKEN.upper(),
    ],
)
def test_webhook_outbound_task_sanitize_redacts_webhook_token_case_insensitively(webhook_token_header: str):
    task = WebhookOutboundTask(
        organization_id="org_123",
        url="https://example.com/webhook",
        body={"hello": "world"},
        headers={
            webhook_token_header: "secret-token",
            "X-Other": "kept",
        },
    )

    sanitized = task.sanitize()

    assert sanitized.headers == {
        webhook_token_header: "***",
        "X-Other": "kept",
    }
    assert task.headers[webhook_token_header] == "secret-token"
