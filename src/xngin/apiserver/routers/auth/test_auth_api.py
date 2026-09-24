"""Characterization tests for ID token validation and the token exchange.

These pin the observable behavior of auth_api._validate_idtoken so that swapping the underlying JWT
library cannot change it silently. Tokens are minted by hand here rather than with the library under
test: if the test signed with the same library that verifies, it would only prove self-consistency.
"""

import base64
import datetime
import hashlib
import hmac
import json
import threading
from urllib.parse import parse_qsl

import httpx2
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI, HTTPException

from xngin.apiserver import flags
from xngin.apiserver.dependencies import retrying_httpx_dependency
from xngin.apiserver.main import app
from xngin.apiserver.routers.auth import auth_api
from xngin.apiserver.routers.auth.auth_dependencies import SessionTokenCryptor
from xngin.apiserver.routers.auth.discovery import OidcProviderTimeoutError, OidcUserinfoError, get_oidc_discovery
from xngin.apiserver.routers.auth.oidc_settings import OidcSettings, get_oidc_settings
from xngin.apiserver.routers.auth.principal import Principal
from xngin.xsecrets.nacl_provider import NaclProviderKeyset

TEST_CLIENT_ID = "test-client-id.apps.googleusercontent.com"
TEST_ISSUER = "https://accounts.google.com"
TEST_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TEST_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
TEST_KID = "test-key-id"
TEST_NONCE = "test-nonce-value"
TEST_REDIRECT_URI = "http://localhost:3000/"
TEST_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
TEST_CODE_VERIFIER = "v" * 43
TEST_ACCESS_TOKEN = "test-access-token"
TEST_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _int_to_b64url(value: int) -> str:
    return _b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))


@pytest.fixture(scope="session", name="signing_key")
def fixture_signing_key():
    """An RSA keypair standing in for the identity provider's signing key.

    Session-scoped because generating one costs real time, and every test reuses the same key.
    """
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session", name="other_key")
def fixture_other_key():
    """A second keypair, for tokens signed by someone who is not the identity provider."""
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


def _now() -> int:
    return int(datetime.datetime.now(datetime.UTC).timestamp())


def _claims(**overrides) -> dict:
    """Google-shaped claims; pass None to drop a claim."""
    now = _now()
    claims = {
        "iss": TEST_ISSUER,
        "aud": TEST_CLIENT_ID,
        "azp": TEST_CLIENT_ID,
        "sub": "1234567890",
        "email": "user@example.com",
        "email_verified": True,
        "hd": "example.com",
        "nonce": TEST_NONCE,
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


def _settings(**overrides) -> OidcSettings:
    kwargs: dict = {
        "issuer": TEST_ISSUER,
        "client_id": TEST_CLIENT_ID,
        "redirect_uri": TEST_REDIRECT_URI,
        "claim_map": {"hd": "hd"},
    }
    kwargs.update(overrides)
    return OidcSettings(**kwargs)


@pytest.fixture(name="settings")
def fixture_settings():
    return _settings()


@pytest.fixture(name="discovery")
def fixture_discovery(signing_key):
    return FakeDiscovery([_make_jwk(signing_key, TEST_KID)])


class FakeDiscovery:
    def __init__(self, signing_keys: list[dict], userinfo_endpoint: str | None = TEST_USERINFO_ENDPOINT):
        self.signing_keys = signing_keys
        self._userinfo_endpoint = userinfo_endpoint

    def authorization_endpoint(self) -> str:
        return TEST_AUTHORIZATION_ENDPOINT

    def token_endpoint(self) -> str:
        return TEST_TOKEN_ENDPOINT

    def userinfo_endpoint(self) -> str | None:
        return self._userinfo_endpoint

    def get_signing_key(self, *, kid: object, algorithm: object) -> dict | None:
        if algorithm != "RS256" or not isinstance(kid, str) or not kid:
            return None
        return next((key for key in self.signing_keys if key.get("kid") == kid), None)


def _validate(settings, discovery, token: str, nonce: str = TEST_NONCE) -> dict:
    try:
        header = auth_api.jwt.get_unverified_header(token)
    except auth_api.jwt.PyJWTError:
        key = discovery.signing_keys[0]
    else:
        key = next(
            (candidate for candidate in discovery.signing_keys if candidate.get("kid") == header.get("kid")),
            None,
        )
        if key is None:
            raise HTTPException(status_code=401, detail="Unable to find appropriate key")
    return auth_api._validate_idtoken(settings, key, id_token=token, nonce=nonce)


def test_accepts_a_valid_token(settings, discovery, signing_key):
    decoded = _validate(settings, discovery, _mint(signing_key, _claims()))

    assert decoded["email"] == "user@example.com"
    assert decoded["sub"] == "1234567890"
    assert decoded["hd"] == "example.com"
    assert decoded["iss"] == TEST_ISSUER
    assert decoded["nonce"] == TEST_NONCE


def test_accepts_token_without_azp(settings, discovery, signing_key):
    decoded = _validate(settings, discovery, _mint(signing_key, _claims(azp=None)))

    assert decoded["sub"] == "1234567890"


def test_rejects_unverified_email(settings, discovery, signing_key):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims(email_verified=False)))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Email address is not verified"


def test_leaves_missing_email_verified_to_the_userinfo_check(settings, discovery, signing_key):
    decoded = _validate(settings, discovery, _mint(signing_key, _claims(email_verified=None)))

    assert "email_verified" not in decoded


def test_accepts_string_true_email_verified(settings, discovery, signing_key):
    decoded = _validate(settings, discovery, _mint(signing_key, _claims(email_verified="true")))

    assert decoded["email_verified"] == "true"


@pytest.mark.parametrize("email_verified", ["false", "TRUE", "yes", 1, 0])
def test_rejects_other_email_verified_values(settings, discovery, signing_key, email_verified):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims(email_verified=email_verified)))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Email address is not verified"


def test_rejects_unknown_kid(settings, discovery, signing_key):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims(), kid="some-other-kid"))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Unable to find appropriate key"


@pytest.mark.parametrize(
    ("token", "detail"),
    [
        pytest.param("not-a-jwt", "Invalid authentication credentials", id="malformed"),
        pytest.param(None, "Unable to find appropriate key", id="unknown-kid"),
    ],
)
def test_signing_key_selection_rejects_invalid_tokens(discovery, signing_key, token, detail):
    if token is None:
        token = _mint(signing_key, _claims(), kid="unknown")

    with pytest.raises(HTTPException) as exc:
        auth_api._get_signing_key(discovery, id_token=token)

    assert exc.value.status_code == 401
    assert exc.value.detail == detail


def test_signing_key_selection_rejects_unsupported_algorithm(discovery, signing_key):
    token = _mint(signing_key, _claims(), alg="HS256")

    with pytest.raises(HTTPException) as exc:
        auth_api._get_signing_key(discovery, id_token=token)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


@pytest.mark.parametrize("garbage", ["", "not-a-jwt", "a.b", "a.b.c", "...."])
def test_rejects_malformed_tokens(settings, discovery, garbage):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, garbage)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_rejects_tampered_signature(settings, discovery, signing_key):
    token = _mint(signing_key, _claims())
    header, payload, signature = token.split(".")
    tampered = f"{header}.{payload}.{signature[:-4]}AAAA"

    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, tampered)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_rejects_token_signed_by_another_key(settings, discovery, other_key):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(other_key, _claims()))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"aud": "some-other-client-id", "azp": "some-other-client-id"}, id="wrong-aud"),
        pytest.param({"aud": [TEST_CLIENT_ID, "another-client-id"]}, id="untrusted-additional-audience"),
        pytest.param({"aud": [TEST_CLIENT_ID, "another-client-id"], "azp": None}, id="multi-audience-without-azp"),
        pytest.param({"aud": [TEST_CLIENT_ID]}, id="audience-array"),
        pytest.param({"iss": "https://accounts.evil.example"}, id="wrong-iss"),
        pytest.param({"iss": "accounts.google.com"}, id="schemeless-iss"),
        pytest.param({"exp": int(datetime.datetime.now(datetime.UTC).timestamp()) - 3600}, id="expired"),
        pytest.param({"iss": None}, id="missing-iss"),
        pytest.param({"aud": None}, id="missing-aud"),
        pytest.param({"iat": None}, id="missing-iat"),
        pytest.param({"exp": None}, id="missing-exp"),
        pytest.param({"sub": None}, id="missing-sub"),
        pytest.param({"email": None}, id="missing-email"),
        pytest.param({"email": 42}, id="non-string-email"),
    ],
)
def test_rejects_invalid_claims(settings, discovery, signing_key, overrides):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims(**overrides)))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_rejects_azp_that_does_not_match_aud(settings, discovery, signing_key):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims(azp="a-different-party")))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid azp/aud"


def test_rejects_mismatched_nonce(settings, discovery, signing_key):
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims()), nonce="a-different-nonce")

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid nonce"


@pytest.mark.parametrize("alg", ["none", "HS256"])
def test_rejects_algorithm_confusion(settings, discovery, signing_key, alg):
    """A token must not be accepted just because its header claims a weaker algorithm."""
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims(), alg=alg))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_binds_jwk_algorithm_to_rs256(settings, discovery, signing_key):
    """Untrusted JWK metadata must not choose the signature algorithm."""
    discovery.signing_keys[0]["alg"] = "none"

    decoded = _validate(settings, discovery, _mint(signing_key, _claims()))

    assert decoded["sub"] == "1234567890"


def test_tolerates_small_clock_skew_on_iat(settings, discovery, signing_key):
    """A slightly future-dated iat is accepted, so a fast clock at the identity provider does not break login."""
    skewed = _now() + 10
    decoded = _validate(settings, discovery, _mint(signing_key, _claims(iat=skewed)))

    assert decoded["iat"] == skewed


def test_rejects_iat_far_in_the_future(settings, discovery, signing_key):
    far_future = _now() + 600
    with pytest.raises(HTTPException) as exc:
        _validate(settings, discovery, _mint(signing_key, _claims(iat=far_future)))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def test_principal_applies_claim_map(settings):
    principal = auth_api._principal_from_claims(settings, _claims())

    assert principal.email == "user@example.com"
    assert principal.hd == "example.com"
    assert principal.iss == TEST_ISSUER
    assert principal.sub == "1234567890"


def test_principal_without_claim_map_leaves_auxiliary_fields_blank():
    principal = auth_api._principal_from_claims(_settings(claim_map={}), _claims())

    assert principal.hd == ""


def test_principal_tolerates_missing_mapped_claim(settings):
    principal = auth_api._principal_from_claims(settings, _claims(azp=None, hd=None))

    assert principal.hd == ""


def test_principal_rejects_non_string_mapped_claim(settings):
    with pytest.raises(HTTPException) as exc:
        auth_api._principal_from_claims(settings, _claims(hd=["example.com"]))

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid authentication credentials"


def _token_endpoint_client(requests: list[httpx2.Request], response: httpx2.Response | None = None):
    """An httpx2 client whose transport records requests and answers as a token endpoint would."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if response is not None:
            return response
        return httpx2.Response(200, json={"id_token": "the-id-token", "access_token": "ignored"})

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _form(request: httpx2.Request) -> dict[str, str]:
    return dict(parse_qsl(request.content.decode()))


def test_token_exchange_omits_client_secret_for_public_clients(settings, discovery):
    requests: list[httpx2.Request] = []
    with _token_endpoint_client(requests) as client:
        tokens = auth_api._exchange_code_for_tokens(
            settings, discovery, client, code="the-code", code_verifier=TEST_CODE_VERIFIER
        )

    assert tokens.id_token == "the-id-token"
    assert tokens.access_token == "ignored"
    assert str(requests[0].url) == TEST_TOKEN_ENDPOINT
    assert "authorization" not in requests[0].headers
    assert _form(requests[0]) == {
        "client_id": TEST_CLIENT_ID,
        "code": "the-code",
        "code_verifier": TEST_CODE_VERIFIER,
        "redirect_uri": TEST_REDIRECT_URI,
        "grant_type": "authorization_code",
    }


def test_token_exchange_sends_client_secret_when_configured(discovery):
    requests: list[httpx2.Request] = []
    with _token_endpoint_client(requests) as client:
        auth_api._exchange_code_for_tokens(
            _settings(client_secret="shh"), discovery, client, code="the-code", code_verifier=TEST_CODE_VERIFIER
        )

    assert _form(requests[0])["client_secret"] == "shh"
    assert "authorization" not in requests[0].headers


def test_token_exchange_tolerates_missing_access_token(settings, discovery):
    response = httpx2.Response(200, json={"id_token": "the-id-token"})
    with _token_endpoint_client([], response) as client:
        tokens = auth_api._exchange_code_for_tokens(
            settings, discovery, client, code="the-code", code_verifier=TEST_CODE_VERIFIER
        )

    assert tokens.access_token is None


def _userinfo(**overrides) -> dict:
    """A userinfo response for the user in _claims(); pass None to drop a claim."""
    userinfo = {"sub": "1234567890", "email": "user@example.com", "email_verified": True}
    userinfo.update(overrides)
    return {k: v for k, v in userinfo.items() if v is not None}


def _userinfo_client(requests: list[httpx2.Request], response: httpx2.Response) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return response

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _check_userinfo(discovery, client, access_token: str | None = TEST_ACCESS_TOKEN):
    claims = _claims(email_verified=None)
    auth_api._require_email_verified_by_userinfo(discovery, client, claims=claims, access_token=access_token)


@pytest.mark.parametrize("email_verified", [True, "true"])
def test_userinfo_confirms_verified_email(discovery, email_verified):
    requests: list[httpx2.Request] = []
    with _userinfo_client(requests, httpx2.Response(200, json=_userinfo(email_verified=email_verified))) as client:
        _check_userinfo(discovery, client)

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == TEST_USERINFO_ENDPOINT
    assert requests[0].headers["authorization"] == f"Bearer {TEST_ACCESS_TOKEN}"
    assert requests[0].headers["accept"] == "application/json"


@pytest.mark.parametrize(
    ("userinfo", "detail"),
    [
        pytest.param(_userinfo(email_verified=False), "Email address is not verified", id="unverified"),
        pytest.param(_userinfo(email_verified=None), "Email address is not verified", id="missing-email-verified"),
        pytest.param(_userinfo(email_verified="false"), "Email address is not verified", id="string-false"),
        pytest.param(_userinfo(sub="someone-else"), "Invalid authentication credentials", id="sub-mismatch"),
        pytest.param(_userinfo(sub=None), "Invalid authentication credentials", id="missing-sub"),
        pytest.param(_userinfo(email=None), "Invalid authentication credentials", id="missing-email"),
        pytest.param(_userinfo(email="other@example.com"), "Invalid authentication credentials", id="email-mismatch"),
    ],
)
def test_userinfo_rejects_unverified_or_mismatched_users(discovery, userinfo, detail):
    with (
        _userinfo_client([], httpx2.Response(200, json=userinfo)) as client,
        pytest.raises(HTTPException) as exc,
    ):
        _check_userinfo(discovery, client)

    assert exc.value.status_code == 401
    assert exc.value.detail == detail


@pytest.mark.parametrize(
    ("userinfo_endpoint", "access_token"),
    [
        pytest.param(None, TEST_ACCESS_TOKEN, id="no-userinfo-endpoint"),
        pytest.param(TEST_USERINFO_ENDPOINT, None, id="no-access-token"),
    ],
)
def test_userinfo_unavailable_means_unverified(signing_key, userinfo_endpoint, access_token):
    requests: list[httpx2.Request] = []
    discovery = FakeDiscovery([_make_jwk(signing_key, TEST_KID)], userinfo_endpoint=userinfo_endpoint)
    with (
        _userinfo_client(requests, httpx2.Response(200, json=_userinfo())) as client,
        pytest.raises(HTTPException) as exc,
    ):
        _check_userinfo(discovery, client, access_token=access_token)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Email address is not verified"
    assert requests == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        pytest.param(
            httpx2.Response(
                401,
                headers={"WWW-Authenticate": 'Bearer error="invalid_token", error_description="expired"'},
            ),
            f"Userinfo endpoint {TEST_USERINFO_ENDPOINT} returned status code 401"
            """ (WWW-Authenticate='Bearer error="invalid_token", error_description="expired"').""",
            id="bearer-error",
        ),
        pytest.param(
            httpx2.Response(403, headers={"WWW-Authenticate": "x" * 500}),
            f"Userinfo endpoint {TEST_USERINFO_ENDPOINT} returned status code 403 (WWW-Authenticate='{'x' * 200}').",
            id="truncates-header",
        ),
        pytest.param(
            httpx2.Response(500),
            f"Userinfo endpoint {TEST_USERINFO_ENDPOINT} returned status code 500.",
            id="error-status",
        ),
        pytest.param(
            httpx2.Response(200, headers={"Content-Type": "application/jwt; charset=utf-8"}, content=b"a.b.c"),
            f"Userinfo endpoint {TEST_USERINFO_ENDPOINT} returned a signed or encrypted response, which is not "
            "supported.",
            id="jwt",
        ),
        pytest.param(
            httpx2.Response(200, content=b"not json"),
            f"Userinfo endpoint {TEST_USERINFO_ENDPOINT} returned invalid JSON.",
            id="invalid-json",
        ),
        pytest.param(
            httpx2.Response(200, json=["nope"]),
            f"Userinfo endpoint {TEST_USERINFO_ENDPOINT} returned a non-dictionary response.",
            id="not-an-object",
        ),
    ],
)
def test_userinfo_rejects_unusable_responses(discovery, response, message):
    with _userinfo_client([], response) as client, pytest.raises(OidcUserinfoError) as exc:
        _check_userinfo(discovery, client)

    assert str(exc.value) == message


@pytest.mark.parametrize(
    ("error_type", "expected_exception"),
    [
        pytest.param(httpx2.ConnectError, OidcUserinfoError, id="connection"),
        pytest.param(httpx2.ReadTimeout, OidcProviderTimeoutError, id="timeout"),
    ],
)
def test_userinfo_normalizes_request_failures(discovery, error_type, expected_exception):
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error_type("upstream failure", request=request)

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client, pytest.raises(expected_exception):
        _check_userinfo(discovery, client)


@pytest.fixture(name="issued_claims")
def fixture_issued_claims():
    """The claims inside the ID token the fake token endpoint hands out."""
    return _claims()


@pytest.fixture(name="configured_app")
def fixture_configured_app(settings, discovery, signing_key, issued_claims, monkeypatch: pytest.MonkeyPatch):
    """Points the app at the test settings, discovery document, and a fake token endpoint.

    conftest's overrides are restored afterwards.
    """
    monkeypatch.setenv(flags.ENV_SESSION_TOKEN_KEYSET, NaclProviderKeyset.create().serialize_base64())
    token_endpoint_response = httpx2.Response(200, json={"id_token": _mint(signing_key, issued_claims)})
    token_client = _token_endpoint_client([], token_endpoint_response)
    dependencies = (get_oidc_settings, get_oidc_discovery, retrying_httpx_dependency)
    previous = {dependency: app.dependency_overrides.get(dependency) for dependency in dependencies}
    app.dependency_overrides[get_oidc_settings] = lambda: settings
    app.dependency_overrides[get_oidc_discovery] = lambda: discovery
    app.dependency_overrides[retrying_httpx_dependency] = lambda: token_client
    yield
    for dependency, override in previous.items():
        if override is None:
            app.dependency_overrides.pop(dependency, None)
        else:
            app.dependency_overrides[dependency] = override


def _callback_body(nonce: str = TEST_NONCE) -> dict:
    return {"code": "the-code", "code_verifier": TEST_CODE_VERIFIER, "nonce": nonce}


def test_callback_returns_session_token(client, configured_app, issued_claims):
    response = client.post("/v1/a/oidc/callback", json=_callback_body())

    assert response.status_code == 200, response.text
    principal = SessionTokenCryptor().decode(response.json()["session_token"])
    assert principal == Principal(
        email="user@example.com",
        hd="example.com",
        iat=issued_claims["iat"],
        iss=TEST_ISSUER,
        sub="1234567890",
    )


def test_callback_rejects_token_issued_for_another_login_attempt(client, configured_app):
    response = client.post("/v1/a/oidc/callback", json=_callback_body(nonce="a-different-nonce"))

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid nonce"}


def _idp_client(requests: list[httpx2.Request], *, id_token: str, userinfo: dict) -> httpx2.Client:
    """An httpx2 client whose transport answers as the token and userinfo endpoints would."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if str(request.url) == TEST_USERINFO_ENDPOINT:
            return httpx2.Response(200, json=userinfo)
        return httpx2.Response(200, json={"id_token": id_token, "access_token": TEST_ACCESS_TOKEN})

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_callback_does_not_query_userinfo_when_id_token_has_email_verified(client, configured_app, signing_key):
    requests: list[httpx2.Request] = []
    upstream = _idp_client(requests, id_token=_mint(signing_key, _claims()), userinfo=_userinfo(email_verified=False))
    app.dependency_overrides[retrying_httpx_dependency] = lambda: upstream

    response = client.post("/v1/a/oidc/callback", json=_callback_body())

    assert response.status_code == 200, response.text
    assert [str(request.url) for request in requests] == [TEST_TOKEN_ENDPOINT]


@pytest.mark.parametrize(
    ("email_verified", "status_code"),
    [
        pytest.param(True, 200, id="verified"),
        pytest.param(False, 401, id="unverified"),
    ],
)
def test_callback_checks_userinfo_when_id_token_omits_email_verified(
    client, configured_app, signing_key, email_verified, status_code
):
    requests: list[httpx2.Request] = []
    upstream = _idp_client(
        requests,
        id_token=_mint(signing_key, _claims(email_verified=None)),
        userinfo=_userinfo(email_verified=email_verified),
    )
    app.dependency_overrides[retrying_httpx_dependency] = lambda: upstream

    response = client.post("/v1/a/oidc/callback", json=_callback_body())

    assert response.status_code == status_code, response.text
    assert [str(request.url) for request in requests] == [TEST_TOKEN_ENDPOINT, TEST_USERINFO_ENDPOINT]
    if status_code == 401:
        assert response.json() == {"detail": "Email address is not verified"}


def test_callback_unavailable_when_login_is_disabled(client):
    response = client.post("/v1/a/oidc/callback", json=_callback_body())

    assert response.status_code == 503


async def test_lifespan_constructs_discovery_off_the_event_loop(monkeypatch: pytest.MonkeyPatch):
    loop_thread = threading.get_ident()
    constructed_in: list[int] = []

    class RecordingDiscovery:
        closed = False

        def __init__(self, settings: OidcSettings):
            self.settings = settings
            constructed_in.append(threading.get_ident())

        def close(self):
            self.closed = True

    settings = _settings()
    monkeypatch.setattr(flags, "AIRPLANE_MODE", False)
    monkeypatch.setattr(auth_api.auth_dependencies, "TESTING_TOKENS_ENABLED", False)
    monkeypatch.setattr(auth_api, "get_oidc_settings", lambda: settings)
    monkeypatch.setattr(auth_api, "OidcDiscovery", RecordingDiscovery)
    lifespan_app = FastAPI()

    async with auth_api.lifespan(lifespan_app):
        discovery = lifespan_app.state.oidc_discovery
        assert discovery.settings is settings

    assert constructed_in != [loop_thread]
    assert len(constructed_in) == 1
    assert discovery.closed
    assert not hasattr(lifespan_app.state, "oidc_discovery")
