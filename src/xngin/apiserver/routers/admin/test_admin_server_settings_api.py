"""Tests for the server settings and domain settings endpoints in admin_api.py."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xngin.apiserver.limits import MAX_NUMBER_OF_DOMAIN_SETTINGS
from xngin.apiserver.routers.admin.admin_api_types import (
    CreateDomainSettingsRequest,
    SignupSettings,
    UpdateDomainSettingsRequest,
    UpdateServerSettingsRequest,
)
from xngin.apiserver.routers.common_enums import SignupMode
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.assertions import expect_status_code

SERVER_SETTINGS_PATH = "/v1/m/server-settings"
DOMAIN_SETTINGS_PATH = "/v1/m/domain-settings"


def _set_mode(aclient: AdminAPIClient, mode: SignupMode) -> None:
    aclient.update_server_settings(body=UpdateServerSettingsRequest(signup=SignupSettings(mode=mode)))


def _listed_domains(aclient: AdminAPIClient) -> list[str]:
    return [d.domain for d in aclient.list_domain_settings().data.items]


def test_get_server_settings_defaults(aclient: AdminAPIClient):
    response = aclient.get_server_settings().data

    assert response.signup.mode == SignupMode.EXISTING
    assert response.updated_at is None


def test_server_settings_are_stored_in_a_single_row(aclient: AdminAPIClient, xngin_session: Session):
    def settings_row_count() -> int:
        return xngin_session.scalar(select(func.count()).select_from(tables.ServerSettings)) or 0

    aclient.get_server_settings()
    aclient.update_server_settings(body=UpdateServerSettingsRequest())
    assert settings_row_count() == 0

    _set_mode(aclient, SignupMode.EXISTING_AND_DOMAINS)
    _set_mode(aclient, SignupMode.ANYONE)
    assert settings_row_count() == 1
    assert xngin_session.get_one(tables.ServerSettings, tables.SERVER_SETTINGS_ID).signup_mode == "anyone"


def test_settings_require_privileged(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    with expect_status_code(403):
        aclient_unpriv.get_server_settings()
    with expect_status_code(403):
        _set_mode(aclient_unpriv, SignupMode.ANYONE)
    with expect_status_code(403):
        aclient_unpriv.list_domain_settings()
    with expect_status_code(403):
        aclient_unpriv.create_domain_settings(body=CreateDomainSettingsRequest(domain="example.org"))
    with expect_status_code(403):
        aclient_unpriv.update_domain_settings(
            domain_settings_id="d_missing", body=UpdateDomainSettingsRequest(allow_signup=False)
        )
    with expect_status_code(403):
        aclient_unpriv.delete_domain_settings(domain_settings_id="d_missing")

    assert aclient.get_server_settings().data.updated_at is None
    assert _listed_domains(aclient) == []


def test_update_server_settings_records_update_time(aclient: AdminAPIClient, records):
    before = datetime.now(UTC) - timedelta(seconds=5)

    _set_mode(aclient, SignupMode.EXISTING_AND_DOMAINS)
    first = aclient.get_server_settings().data
    _set_mode(aclient, SignupMode.ANYONE)
    second = aclient.get_server_settings().data

    assert first.signup.mode == SignupMode.EXISTING_AND_DOMAINS
    assert first.updated_at is not None
    assert first.updated_at >= before
    assert second.signup.mode == SignupMode.ANYONE
    assert second.updated_at is not None
    assert second.updated_at >= first.updated_at
    logged = [r["message"] for r in records if r["message"].startswith("Sign-up mode changed")]
    assert "existing -> existing_and_domains" in logged[0]
    assert "existing_and_domains -> anyone" in logged[1]


def test_update_server_settings_empty_request_is_noop(aclient: AdminAPIClient):
    _set_mode(aclient, SignupMode.ANYONE)
    stored = aclient.get_server_settings().data

    aclient.update_server_settings(body=UpdateServerSettingsRequest())

    assert aclient.get_server_settings().data == stored


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"signup": {"mode": "everyone"}}, id="invalid-mode"),
        pytest.param({"signup": {"mode": "anyone", "allowed_domains": []}}, id="unknown-signup-field"),
        pytest.param({"unknown": {}}, id="unknown-section"),
    ],
)
def test_update_server_settings_rejects_invalid_requests(aclient: AdminAPIClient, body):
    response = aclient.client.patch(SERVER_SETTINGS_PATH, json=body)

    assert response.status_code == 422, response.content
    assert aclient.get_server_settings().data.updated_at is None


def test_domain_settings_lifecycle(aclient: AdminAPIClient, records):
    created = aclient.create_domain_settings(body=CreateDomainSettingsRequest(domain=" @Example.NET ")).data
    aclient.create_domain_settings(body=CreateDomainSettingsRequest(domain="example.com", allow_signup=False))

    assert created.domain == "example.net"
    assert created.allow_signup is True
    assert created.id.startswith("d_")
    listed = aclient.list_domain_settings().data.items
    assert [(d.domain, d.allow_signup) for d in listed] == [("example.com", False), ("example.net", True)]
    assert listed[1] == created

    aclient.update_domain_settings(domain_settings_id=created.id, body=UpdateDomainSettingsRequest(allow_signup=False))
    aclient.update_domain_settings(domain_settings_id=created.id, body=UpdateDomainSettingsRequest())
    assert aclient.list_domain_settings().data.items[1].allow_signup is False

    aclient.delete_domain_settings(domain_settings_id=created.id)
    assert _listed_domains(aclient) == ["example.com"]

    logged = [r["message"] for r in records if r["message"].startswith("Domain settings for example.net")]
    assert len(logged) == 3
    assert "created" in logged[0]
    assert "allow_signup=False" in logged[1]
    assert "deleted" in logged[2]


def test_create_domain_settings_rejects_duplicates(aclient: AdminAPIClient):
    aclient.create_domain_settings(body=CreateDomainSettingsRequest(domain="example.org"))

    response = aclient.client.post(DOMAIN_SETTINGS_PATH, json={"domain": "EXAMPLE.org"})

    assert response.status_code == 409, response.content
    assert _listed_domains(aclient) == ["example.org"]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"domain": "*.example.org"}, id="wildcard"),
        pytest.param({"domain": "example"}, id="no-dot"),
        pytest.param({"domain": "http://example.org"}, id="url"),
        pytest.param({"domain": "a" * 117 + ".example.org"}, id="too-long"),
        pytest.param({}, id="missing-domain"),
        pytest.param({"domain": "example.org", "extra": 1}, id="unknown-field"),
    ],
)
def test_create_domain_settings_rejects_invalid_requests(aclient: AdminAPIClient, body):
    response = aclient.client.post(DOMAIN_SETTINGS_PATH, json=body)

    assert response.status_code == 422, response.content
    assert _listed_domains(aclient) == []


def test_create_domain_settings_enforces_maximum(aclient: AdminAPIClient):
    for i in range(MAX_NUMBER_OF_DOMAIN_SETTINGS):
        aclient.create_domain_settings(body=CreateDomainSettingsRequest(domain=f"d{i}.example.org"))

    response = aclient.client.post(DOMAIN_SETTINGS_PATH, json={"domain": "one-too-many.example.org"})

    assert response.status_code == 422, response.content
    assert len(_listed_domains(aclient)) == MAX_NUMBER_OF_DOMAIN_SETTINGS


def test_missing_domain_settings(aclient: AdminAPIClient):
    with expect_status_code(404):
        aclient.update_domain_settings(
            domain_settings_id="d_missing", body=UpdateDomainSettingsRequest(allow_signup=False)
        )
    with expect_status_code(404):
        aclient.delete_domain_settings(domain_settings_id="d_missing")
    aclient.delete_domain_settings(domain_settings_id="d_missing", allow_missing=True)
