from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import pytest

from xngin.apiserver.exceptionhandlers import XHTTPValidationError
from xngin.apiserver.testing.admin_api_client import AdminAPIClientNotDefaultStatusError
from xngin.apiserver.testing.admin_integrations_api_client import (
    AdminIntegrationsAPIClientNotDefaultStatusError,
)
from xngin.apiserver.testing.integrations_api_client import IntegrationsAPIClientNotDefaultStatusError


def _dates_equal(db_date: datetime, other_date: datetime):
    """Compare dates with or without timezone info, honoring the db_date's timezone."""
    if db_date.tzinfo is None:
        return db_date == other_date.replace(tzinfo=None)
    return db_date == other_date


def assert_dates_equal(db_date: datetime, other_date: datetime):
    """Asserts that the db_date is equal to other_date, honoring the db_date's timezone."""
    assert _dates_equal(db_date, other_date), f"Date {db_date} is not equal to {other_date}"


@dataclass
class StatusCodeMatcher:
    exc: (
        AdminAPIClientNotDefaultStatusError
        | AdminIntegrationsAPIClientNotDefaultStatusError
        | IntegrationsAPIClientNotDefaultStatusError
        | None
    ) = None

    def http_response(self):
        """Returns the httpx2 Response."""
        if self.exc is None:
            raise AssertionError("StatusCodeMatcher has no captured response yet.")
        return self.exc.result.response

    def _detail_messages(self) -> list[str]:
        response = self.http_response()
        parsed_error = XHTTPValidationError.model_validate(response.json())
        return [detail.msg for detail in parsed_error.detail]

    def _has_text(self, text: str) -> bool:
        response = self.http_response()
        body_text = response.text if response.text is not None else str(response.content)
        return text in body_text

    def _has_message(self, msg: str, *, contains: bool = False) -> bool:
        response = self.http_response()
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError(f"Expected JSON object response body, got {type(body).__name__}.")
        actual_msg = body.get("message")
        if not isinstance(actual_msg, str):
            raise TypeError(f'Expected JSON object with string "message", got: {body!r}')
        actual_msg = actual_msg.lower()
        expected_msg = msg.lower()
        return expected_msg in actual_msg if contains else actual_msg == expected_msg


@contextmanager
def expect_status_code(
    status_code: int,
    *,
    message_contains: str | None = None,
    message_eq: str | None = None,
    detail_contains: str | None = None,
    detail_eq: str | None = None,
    text: str | None = None,
) -> Generator[StatusCodeMatcher]:
    """Like pytest.raises(), but for checking the non-default response codes of an AdminAPIClient request."""
    match = StatusCodeMatcher()
    with pytest.raises((
        AdminAPIClientNotDefaultStatusError,
        AdminIntegrationsAPIClientNotDefaultStatusError,
        IntegrationsAPIClientNotDefaultStatusError,
    )) as exc:
        yield match
    match.exc = cast(
        "AdminAPIClientNotDefaultStatusError"
        " | AdminIntegrationsAPIClientNotDefaultStatusError"
        " | IntegrationsAPIClientNotDefaultStatusError",
        exc.value,
    )
    http_response = match.http_response()
    assert http_response.status_code == status_code, (
        f"Expected '{status_code}' response code but got {http_response.status_code}: {http_response.content}"
    )
    if message_eq is not None:
        assert match._has_message(message_eq), (
            f"Expected '{message_eq}' to be equal to the .message field: {http_response.content}"
        )
    if message_contains is not None:
        assert match._has_message(message_contains, contains=True), (
            f"Expected '{message_contains}' to be a substring of the .message field: {http_response.content}"
        )
    if detail_eq is not None:
        assert any(msg == detail_eq for msg in match._detail_messages()), (
            f"Expected '{detail_eq}' to be one of the .msg fields in the response: {http_response.content}"
        )
    if detail_contains is not None:
        messages = match._detail_messages()
        assert any(detail_contains in msg for msg in messages), (
            f"Expected '{detail_contains}' to be in one of the .msg fields in the response: {http_response.content}"
            f"\nmsg list (length {len(messages)}): \n\t{'\n\t'.join(messages)}"
        )
    if text is not None:
        assert match._has_text(text), f"Expected '{text}' to be in the response: {http_response.content}"
