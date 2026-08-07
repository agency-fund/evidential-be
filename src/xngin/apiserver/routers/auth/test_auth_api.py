"""Characterization tests for Google ID token validation.

These pin the observable behavior of auth_api._validate_idtoken so that swapping the underlying JWT
library cannot change it silently. Tokens are minted by hand here rather than with the library under
test: if the test signed with the same library that verifies, it would only prove self-consistency.
"""

import base64
import datetime
import hashlib
import hmac
import json

import httpx2
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException
from starlette.testclient import TestClient

from xngin.apiserver import constants, flags
from xngin.apiserver.dependencies import retrying_httpx_dependency
from xngin.apiserver.main import app
from xngin.apiserver.routers.auth import auth_api
from xngin.apiserver.routers.auth.auth_dependencies import (
    GoogleOidcConfig,
    SessionTokenCryptor,
    get_google_configuration,
)
from xngin.apiserver.routers.auth.test_auth_dependencies import temporary_env_var
from xngin.xsecrets.nacl_provider import NaclProviderKeyset

TEST_CLIENT_ID = "test-client-id.apps.googleusercontent.com"
TEST_ISSUER = "https://accounts.google.com"
TEST_KID = "test-key-id"
TEST_NONCE = "test-nonce-value"
TEST_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _int_to_b64url(value: int) -> str:
    return _b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))


@pytest.fixture(scope="session", name="signing_key")
def fixture_signing_key():
    """An RSA keypair standing in for Google's signing key.

    Session-scoped because generating one costs real time, and every test reuses the same key.
    """
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session", name="other_key")
def fixture_other_key():
    """A second keypair, for tokens signed by someone who is not Google."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _make_jwk(private_key, kid: str) -> dict:
    numbers = private_key.public_key().public_numbers()
    return {
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "kid": kid,
        "n": _int_to_b64url(numbers.n),
        "e": _int_to_b64url(numbers.e),
    }


def _mint(private_key, claims: dict, *, kid: str = TEST_KID, alg: str = "RS256") -> str:
    """Builds a signed JWT without using the library under test.

    `alg` is a parameter rather than a constant so the algorithm-confusion cases can be expressed:
    "none" produces an unsigned token, and HS256 signs with the public modulus as the HMAC secret,
    which is the classic attack against a verifier that trusts the header's algorithm.
    """
    header = {"alg": alg, "kid": kid, "typ": "JWT"}
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(claims).encode())}".encode()
    if alg == "none":
        signature = b""
    elif alg == "HS256":
        secret = str(private_key.public_key().public_numbers().n).encode()
        signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
    else:
        signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input.decode()}.{_b64url(signature)}"


def _claims(**overrides) -> dict:
    now = int(datetime.datetime.now(datetime.UTC).timestamp())
    claims = {
        "iss": TEST_ISSUER,
        "aud": TEST_CLIENT_ID,
        "azp": TEST_CLIENT_ID,
        "sub": "1234567890",
        "email": "user@example.com",
        "hd": "example.com",
        "nonce": TEST_NONCE,
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


@pytest.fixture(name="oidc_config")
def fixture_oidc_config(signing_key):
    return GoogleOidcConfig(
        last_refreshed=datetime.datetime.now(datetime.UTC),
        config={"issuer": TEST_ISSUER, "token_endpoint": TEST_TOKEN_ENDPOINT},
        jwks={"keys": [_make_jwk(signing_key, TEST_KID)]},
    )


@pytest.fixture(name="client_id", autouse=True)
def fixture_client_id(monkeypatch: pytest.MonkeyPatch):
    """_validate_idtoken reads flags.CLIENT_ID at call time; it is unset in the test environment."""
    monkeypatch.setattr(flags, "CLIENT_ID", TEST_CLIENT_ID)


def _validate(oidc_config, token: str, nonce: str = TEST_NONCE) -> dict:
    return auth_api._validate_idtoken(oidc_config, id_token=token, nonce=nonce)


def test_accepts_a_valid_token(oidc_config, signing_key):
    decoded = _validate(oidc_config, _mint(signing_key, _claims()))

    assert decoded["email"] == "user@example.com"
    assert decoded["sub"] == "1234567890"
    assert decoded["hd"] == "example.com"
    assert decoded["iss"] == TEST_ISSUER
    assert decoded["nonce"] == TEST_NONCE


def test_rejects_unknown_kid(oidc_config, signing_key):
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, _mint(signing_key, _claims(), kid="some-other-kid"))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Unable to find appropriate key"


@pytest.mark.parametrize("garbage", ["", "not-a-jwt", "a.b", "a.b.c", "...."])
def test_rejects_malformed_tokens(oidc_config, garbage):
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, garbage)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_rejects_tampered_signature(oidc_config, signing_key):
    token = _mint(signing_key, _claims())
    header, payload, signature = token.split(".")
    tampered = f"{header}.{payload}.{signature[:-4]}AAAA"

    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, tampered)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_rejects_token_signed_by_another_key(oidc_config, other_key):
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, _mint(other_key, _claims()))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"aud": "some-other-client-id", "azp": "some-other-client-id"}, id="wrong-aud"),
        pytest.param({"iss": "https://accounts.evil.example"}, id="wrong-iss"),
        pytest.param({"exp": int(datetime.datetime.now(datetime.UTC).timestamp()) - 3600}, id="expired"),
        pytest.param({"iss": None}, id="missing-iss"),
        pytest.param({"aud": None}, id="missing-aud"),
        pytest.param({"iat": None}, id="missing-iat"),
        pytest.param({"exp": None}, id="missing-exp"),
    ],
)
def test_rejects_invalid_claims(oidc_config, signing_key, overrides):
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, _mint(signing_key, _claims(**overrides)))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_rejects_azp_that_does_not_match_aud(oidc_config, signing_key):
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, _mint(signing_key, _claims(azp="a-different-party")))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid azp/aud"


def test_rejects_mismatched_nonce(oidc_config, signing_key):
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, _mint(signing_key, _claims()), nonce="a-different-nonce")

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid nonce"


@pytest.mark.parametrize("alg", ["none", "HS256"])
def test_rejects_algorithm_confusion(oidc_config, signing_key, alg):
    """A token must not be accepted just because its header claims a weaker algorithm."""
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, _mint(signing_key, _claims(), alg=alg))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_tolerates_small_clock_skew_on_iat(oidc_config, signing_key):
    """A slightly future-dated iat is accepted, so a fast clock at Google does not break login."""
    skewed = int(datetime.datetime.now(datetime.UTC).timestamp()) + 10
    decoded = _validate(oidc_config, _mint(signing_key, _claims(iat=skewed)))

    assert decoded["iat"] == skewed


def test_rejects_iat_far_in_the_future(oidc_config, signing_key):
    far_future = int(datetime.datetime.now(datetime.UTC).timestamp()) + 600
    with pytest.raises(HTTPException) as exc:
        _validate(oidc_config, _mint(signing_key, _claims(iat=far_future)))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_auth_callback_exchanges_code_for_a_session_token(oidc_config, signing_key):
    """Covers the wiring around _validate_idtoken: code exchange, validation, session token."""
    id_token = _mint(signing_key, _claims())

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == TEST_TOKEN_ENDPOINT
        return httpx2.Response(200, json={"id_token": id_token}, request=request)

    async def fake_httpx_client():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            yield client

    # conftest's session-scoped fixture already overrides get_google_configuration with a no-op, so
    # restore whatever was there rather than deleting our own override.
    previous = app.dependency_overrides.get(get_google_configuration)
    app.dependency_overrides[get_google_configuration] = lambda: oidc_config
    app.dependency_overrides[retrying_httpx_dependency] = fake_httpx_client
    try:
        with temporary_env_var(flags.ENV_SESSION_TOKEN_KEYSET, NaclProviderKeyset.create().serialize_base64()):
            with TestClient(app) as client:
                response = client.post(
                    f"{constants.API_PREFIX_V1}/a/oidc/callback",
                    json={"code": "an-auth-code", "code_verifier": "v" * 43, "nonce": TEST_NONCE},
                )
            assert response.status_code == 200
            principal = SessionTokenCryptor().decode(response.json()["session_token"])
    finally:
        app.dependency_overrides.pop(retrying_httpx_dependency, None)
        if previous is None:
            del app.dependency_overrides[get_google_configuration]
        else:
            app.dependency_overrides[get_google_configuration] = previous

    assert principal.email == "user@example.com"
    assert principal.iss == TEST_ISSUER
    assert principal.sub == "1234567890"
    assert principal.hd == "example.com"
