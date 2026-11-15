import datetime as dt
import hashlib
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from jwt.utils import base64url_encode
from pydantic import ValidationError
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from xngin.apiserver import flags, routes
from xngin.apiserver.routers import devidp
from xngin.apiserver.routers.auth import auth_api
from xngin.apiserver.routers.auth.discovery import OidcDiscovery
from xngin.apiserver.routers.auth.oidc_settings import OidcSettings
from xngin.apiserver.sqla import tables

CLIENT_ID = "evidential"
REDIRECT_URI = "http://localhost:3000/"
BASE_URL = "http://localhost:8000"
CODE_VERIFIER = "a-code-verifier-that-is-long-enough-to-be-realistic-0123456789"


def challenge_for(verifier: str) -> str:
    return base64url_encode(hashlib.sha256(verifier.encode()).digest()).decode()


AUTHORIZE_PARAMS = {
    "response_type": "code",
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "state": "the-state",
    "nonce": "the-nonce",
    "code_challenge": challenge_for(CODE_VERIFIER),
    "code_challenge_method": "S256",
    "scope": "openid email",
}


@pytest.fixture(name="client")
def fixture_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, xngin_session: Session) -> Iterator[TestClient]:
    monkeypatch.setattr(flags, "OIDC_CLIENT_ID", CLIENT_ID)
    monkeypatch.setattr(flags, "OIDC_REDIRECT_URI", REDIRECT_URI)
    monkeypatch.setattr(flags, "XNGIN_PUBLIC_API_BASE_URL", BASE_URL)
    monkeypatch.setattr(devidp, "SIGNING_KEY_PATH", tmp_path / "signing_key")
    devidp._signing_key.cache_clear()
    devidp._public_jwk.cache_clear()
    app = FastAPI()
    app.include_router(devidp.router)
    with TestClient(app, follow_redirects=False) as client:
        yield client
    devidp._signing_key.cache_clear()
    devidp._public_jwk.cache_clear()


def choose(client: TestClient, user: str, **overrides: str) -> dict[str, list[str]]:
    response = client.get("/devidp/choose", params=AUTHORIZE_PARAMS | {"user": user} | overrides)
    assert response.status_code == 303
    return parse_qs(urlsplit(response.headers["location"]).query)


def exchange(client: TestClient, authorization_code: str, **overrides: str):
    form = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "code_verifier": CODE_VERIFIER,
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
    }
    return client.post("/devidp/token", data=form | overrides)


@pytest.mark.parametrize("client_secret", [None, "a-google-secret-left-in-dotenv"])
def test_discovery_document_satisfies_the_relying_party(client: TestClient, client_secret: str | None):
    config = client.get("/devidp/.well-known/openid-configuration").json()
    assert config["issuer"] == f"{BASE_URL}/devidp"
    assert config["authorization_endpoint"] == f"{BASE_URL}/devidp/authorize"
    settings = OidcSettings(
        issuer=f"{BASE_URL}/devidp", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, client_secret=client_secret
    )
    discovery = object.__new__(OidcDiscovery)
    discovery.settings = settings
    discovery._validate_config(config)

    keys = client.get("/devidp/jwks").json()["keys"]
    assert len(keys) == 1
    assert OidcDiscovery._is_usable_signing_key(keys[0])


def card_for(page: str, email: str) -> str:
    return next(card for card in page.split('<a class="user')[1:] if email in card)


def pill_labels(card: str) -> list[str]:
    return [label.split("</span>")[0].split(">")[-1] for label in card.split('<span class="pill')[1:]]


def test_authorize_highlights_and_labels_the_last_used_user(client: TestClient):
    client.cookies.set(devidp.REMEMBERED_USER_COOKIE, "bob", path="/devidp")
    page = client.get("/devidp/authorize", params=AUTHORIZE_PARAMS).text
    assert page.count('class="user remembered"') == 1
    assert card_for(page, "bob@example.com").startswith(' remembered"')
    assert pill_labels(card_for(page, "bob@example.com")) == ["last used"]
    assert pill_labels(card_for(page, "alice@example.com")) == []


def test_authorize_labels_whether_users_are_registered_and_privileged(client: TestClient, xngin_session: Session):
    xngin_session.add_all([
        tables.User(email="Alice@example.com", iss=None, sub=None, is_privileged=True),
        tables.User(email="bob@example.com", iss=f"{BASE_URL}/devidp", sub="sub-bob"),
        tables.User(email="eve@example.net", iss="http://localhost:5556/dex", sub="CiQw"),
    ])
    xngin_session.commit()

    page = client.get("/devidp/authorize", params=AUTHORIZE_PARAMS).text

    assert pill_labels(card_for(page, "alice@example.com")) == ["invited", "privileged"]
    assert pill_labels(card_for(page, "bob@example.com")) == ["registered"]
    assert pill_labels(card_for(page, "eve@example.net")) == ["registered to a different identity"]
    assert 'title="iss=http://localhost:5556/dex sub=CiQw"' in card_for(page, "eve@example.net")


def test_authorize_rereads_the_config_file(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config_path = tmp_path / "devidp_config.yaml"
    monkeypatch.setattr(devidp, "CONFIG_PATH", config_path)
    config_path.write_text("template: {}\nusers:\n  carol: {sub: sub-carol, email: carol@example.org}\n")
    assert "carol@example.org" in client.get("/devidp/authorize", params=AUTHORIZE_PARAMS).text

    config_path.write_text("template: {}\nusers:\n  dave: {sub: sub-dave, email: dave@example.org}\n")
    page = client.get("/devidp/authorize", params=AUTHORIZE_PARAMS).text
    assert "dave@example.org" in page
    assert "carol@example.org" not in page


@pytest.mark.parametrize(
    ("overrides", "status_code"),
    [
        ({"redirect_uri": "https://attacker.example/"}, 400),
        ({"client_id": "someone-else"}, 400),
        ({"code_challenge_method": "plain"}, 422),
    ],
)
def test_authorize_rejects_invalid_requests_without_redirecting(
    client: TestClient, overrides: dict[str, str], status_code: int
):
    for path in ("/devidp/authorize", "/devidp/choose", "/devidp/deny"):
        response = client.get(path, params=AUTHORIZE_PARAMS | {"user": "alice"} | overrides)
        assert response.status_code == status_code
        assert "location" not in response.headers


def test_choose_rejects_unknown_users(client: TestClient):
    assert client.get("/devidp/choose", params=AUTHORIZE_PARAMS | {"user": "mallory"}).status_code == 400


def test_choose_remembers_the_user_and_redirects_with_code_and_state(client: TestClient):
    response = client.get("/devidp/choose", params=AUTHORIZE_PARAMS | {"user": "alice"})
    assert response.status_code == 303
    assert response.headers["location"].startswith(REDIRECT_URI + "?")
    assert response.cookies[devidp.REMEMBERED_USER_COOKIE] == "alice"
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["state"] == ["the-state"]
    assert devidp.AuthorizationCode.decode(query["code"][0]).user == "alice"


def test_deny_redirects_with_access_denied(client: TestClient):
    response = client.get("/devidp/deny", params=AUTHORIZE_PARAMS)
    assert parse_qs(urlsplit(response.headers["location"]).query) == {
        "error": ["access_denied"],
        "state": ["the-state"],
    }


def test_token_issues_an_id_token_the_relying_party_accepts(client: TestClient):
    response = exchange(client, choose(client, "alice")["code"][0])
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    id_token = response.json()["id_token"]
    signing_key = client.get("/devidp/jwks").json()["keys"][0]
    assert jwt.get_unverified_header(id_token)["kid"] == signing_key["kid"]
    settings = OidcSettings(issuer=f"{BASE_URL}/devidp", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI)

    claims = auth_api._validate_idtoken(settings, signing_key, id_token=id_token, nonce="the-nonce")

    configured_claims = devidp._read_config().claims_for("alice")
    assert {name: claims[name] for name in configured_claims} == configured_claims


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"code_verifier": "wrong"}, "invalid_grant"),
        ({"redirect_uri": "http://localhost:3001/"}, "invalid_grant"),
        ({"client_id": "someone-else"}, "invalid_grant"),
        ({"grant_type": "refresh_token"}, "unsupported_grant_type"),
        ({"code": "not-a-code"}, "invalid_grant"),
    ],
)
def test_token_rejects_invalid_exchanges(client: TestClient, overrides: dict[str, str], error: str):
    response = exchange(client, choose(client, "alice")["code"][0], **overrides)
    assert response.status_code == 400
    assert response.json()["error"] == error


def test_token_ignores_client_secrets(client: TestClient):
    response = exchange(client, choose(client, "alice")["code"][0], client_secret="a-google-secret-left-in-dotenv")
    assert response.status_code == 200


def test_token_rejects_expired_codes(client: TestClient):
    expired_code = devidp.AuthorizationCode(
        user="alice",
        nonce="the-nonce",
        code_challenge=challenge_for(CODE_VERIFIER),
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        expires_at=dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1),
    )
    response = exchange(client, expired_code.encode())
    assert response.json() == {"error": "invalid_grant", "error_description": "The code has expired."}


def test_template_claims_are_overridden_per_user():
    users = devidp.DevIdpConfig.model_validate({
        "template": {"email_verified": True, "hd": "example.com"},
        "users": {"carol": {"sub": "1", "email": "carol@example.org", "hd": "example.org"}},
    })
    assert users.claims_for("carol") == {
        "email_verified": True,
        "hd": "example.org",
        "sub": "1",
        "email": "carol@example.org",
    }


def test_users_require_sub_and_email():
    with pytest.raises(ValidationError, match="lacks string claims: email"):
        devidp.DevIdpConfig.model_validate({"template": {}, "users": {"carol": {"sub": "1"}}})


def test_signing_key_is_created_once_and_reused(tmp_path: Path):
    path = tmp_path / "signing_key"
    first = devidp._load_or_create_signing_key(path)
    assert path.stat().st_mode & 0o777 == 0o600
    second = devidp._load_or_create_signing_key(path)
    assert first.private_numbers() == second.private_numbers()
    assert list(tmp_path.iterdir()) == [path]


def ec_private_key_pem() -> bytes:
    return ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


@pytest.mark.parametrize(
    "unusable_contents",
    [b"", b"not a key", ec_private_key_pem()],
    ids=["empty", "garbage", "not-rsa"],
)
def test_unusable_signing_key_file_is_replaced(tmp_path: Path, unusable_contents: bytes):
    path = tmp_path / "signing_key"
    path.write_bytes(unusable_contents)
    replacement = devidp._load_or_create_signing_key(path)
    cached = devidp._read_cached_signing_key(path)
    assert cached is not None
    assert cached.private_numbers() == replacement.private_numbers()
    assert path.stat().st_mode & 0o777 == 0o600


def test_devidp_refuses_to_run_outside_development(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flags, "DEVIDP_ENABLED", True)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(devidp.DevIdpNotAllowedError):
        routes.register(FastAPI())


def test_devidp_is_served_only_when_enabled(monkeypatch: pytest.MonkeyPatch):
    def authorize_status_code() -> int:
        app = FastAPI()
        routes.register(app)
        return TestClient(app).get("/devidp/deny").status_code

    assert authorize_status_code() == 404
    monkeypatch.setattr(flags, "DEVIDP_ENABLED", True)
    assert authorize_status_code() == 422
