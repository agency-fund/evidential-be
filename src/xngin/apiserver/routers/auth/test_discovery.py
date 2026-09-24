import base64

import httpx2
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException, Request

from xngin.apiserver.routers.auth.discovery import (
    OidcDiscovery,
    OidcDiscoveryError,
    OidcProviderTimeoutError,
    get_oidc_discovery,
)
from xngin.apiserver.routers.auth.oidc_settings import OidcSettings

DISCOVERY_ISSUER = "https://idp.example.com"
DISCOVERY_PATH = "/.well-known/openid-configuration"


def _b64url_uint(value: int) -> str:
    return base64.urlsafe_b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()


@pytest.fixture(scope="module", name="idp_jwk")
def fixture_idp_jwk() -> dict:
    numbers = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key().public_numbers()
    return {"kty": "RSA", "use": "sig", "kid": "k1", "n": _b64url_uint(numbers.n), "e": _b64url_uint(numbers.e)}


def _settings(issuer: str = DISCOVERY_ISSUER, client_secret: str | None = None) -> OidcSettings:
    return OidcSettings(
        issuer=issuer,
        client_id="client-id",
        redirect_uri="http://localhost:3000/",
        client_secret=client_secret,
    )


def _idp_client(
    jwk: dict,
    discovery_overrides: dict | None = None,
    *,
    keys: list | None = None,
    requests: list[str] | None = None,
    discovery_status: list[int] | None = None,
    jwks_status: list[int] | None = None,
) -> httpx2.Client:
    published_keys = [jwk] if keys is None else keys

    def handler(request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if requests is not None:
            requests.append(url)
        if url.endswith(DISCOVERY_PATH):
            issuer = url.removesuffix(DISCOVERY_PATH)
            document: dict[str, object] = {
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
                "jwks_uri": f"{issuer}/keys",
                "response_types_supported": ["code"],
                "id_token_signing_alg_values_supported": ["RS256"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
            }
            document.update(discovery_overrides or {})
            status = discovery_status[0] if discovery_status else 200
            return httpx2.Response(
                status,
                json={key: value for key, value in document.items() if value is not None},
            )
        if url.endswith("/keys"):
            status = jwks_status[0] if jwks_status else 200
            return httpx2.Response(status, json={"keys": published_keys})
        return httpx2.Response(404)

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_dependency_returns_application_owned_discovery(idp_jwk):
    app = FastAPI()
    with _idp_client(idp_jwk) as client:
        discovery = OidcDiscovery(_settings(), client)
        app.state.oidc_discovery = discovery
        request = Request({"type": "http", "app": app})

        assert get_oidc_discovery(request) is discovery


def test_dependency_reports_login_unavailable_without_discovery():
    request = Request({"type": "http", "app": FastAPI()})

    with pytest.raises(HTTPException) as exc:
        get_oidc_discovery(request)

    assert exc.value.status_code == 503
    assert exc.value.detail == "Login is not configured on this server."


def test_discovery_loads_once_and_exposes_clean_client_api(idp_jwk):
    requests: list[str] = []
    with _idp_client(idp_jwk, requests=requests) as client:
        discovery = OidcDiscovery(_settings(), client)
        key = discovery.get_signing_key(kid="k1", algorithm="RS256")

    assert discovery.authorization_endpoint() == f"{DISCOVERY_ISSUER}/authorize"
    assert discovery.token_endpoint() == f"{DISCOVERY_ISSUER}/token"
    assert key == idp_jwk
    assert requests == [f"{DISCOVERY_ISSUER}{DISCOVERY_PATH}", f"{DISCOVERY_ISSUER}/keys"]


def test_discovery_accepts_issuer_with_trailing_slash(idp_jwk):
    issuer = f"{DISCOVERY_ISSUER}/application/o/evidential/"
    with _idp_client(idp_jwk, {"issuer": issuer}) as client:
        discovery = OidcDiscovery(_settings(issuer), client)

    assert discovery.authorization_endpoint() == f"{issuer}authorize"


def test_discovery_rejects_issuer_mismatch(idp_jwk):
    with (
        _idp_client(idp_jwk, {"issuer": "https://impostor.example.com"}) as client,
        pytest.raises(OidcDiscoveryError, match="does not match XNGIN_OIDC_ISSUER"),
    ):
        OidcDiscovery(_settings(), client)


@pytest.mark.parametrize("missing", ["authorization_endpoint", "token_endpoint", "jwks_uri"])
def test_discovery_requires_endpoints(idp_jwk, missing):
    with (
        _idp_client(idp_jwk, {missing: None}) as client,
        pytest.raises(OidcDiscoveryError, match=f"missing {missing}"),
    ):
        OidcDiscovery(_settings(), client)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"response_types_supported": ["token"]},
            "Discovery document's response_types_supported does not include 'code'.",
            id="no-code-response-type",
        ),
        pytest.param(
            {"response_types_supported": None},
            "Discovery document is missing response_types_supported.",
            id="missing-response-types",
        ),
        pytest.param(
            {"id_token_signing_alg_values_supported": ["ES256"]},
            "Discovery document's id_token_signing_alg_values_supported does not include 'RS256'.",
            id="no-rs256",
        ),
        pytest.param(
            {"id_token_signing_alg_values_supported": None},
            "Discovery document is missing id_token_signing_alg_values_supported.",
            id="missing-signing-algs",
        ),
        pytest.param(
            {"code_challenge_methods_supported": ["plain"]},
            "Discovery document's code_challenge_methods_supported does not include 'S256'.",
            id="no-s256",
        ),
    ],
)
def test_discovery_requires_compatible_capabilities(idp_jwk, overrides, message):
    with _idp_client(idp_jwk, overrides) as client, pytest.raises(OidcDiscoveryError) as exc:
        OidcDiscovery(_settings(), client)

    assert str(exc.value) == message


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"authorization_endpoint": "javascript:alert(1)"},
            "Discovery document's authorization_endpoint must be an https:// URL (got 'javascript:alert(1)').",
            id="javascript-authorization-endpoint",
        ),
        pytest.param(
            {"authorization_endpoint": "/authorize"},
            "Discovery document's authorization_endpoint must be an https:// URL (got '/authorize').",
            id="relative-authorization-endpoint",
        ),
        pytest.param(
            {"authorization_endpoint": "https://idp.example.com/authorize#frag"},
            "Discovery document's authorization_endpoint must not include a fragment "
            "(got 'https://idp.example.com/authorize#frag').",
            id="authorization-endpoint-with-fragment",
        ),
        pytest.param(
            {"token_endpoint": "http://idp.example.com/token"},
            "Discovery document's token_endpoint must be an https:// URL (got 'http://idp.example.com/token').",
            id="http-token-endpoint",
        ),
        pytest.param(
            {"jwks_uri": "https://user:secret@idp.example.com/keys"},
            "Discovery document's jwks_uri must not include credentials "
            "(got 'https://user:secret@idp.example.com/keys').",
            id="jwks-uri-with-credentials",
        ),
    ],
)
def test_discovery_requires_https_endpoints(idp_jwk, overrides, message):
    with _idp_client(idp_jwk, overrides) as client, pytest.raises(OidcDiscoveryError) as exc:
        OidcDiscovery(_settings(), client)

    assert str(exc.value) == message


def test_discovery_ignores_unused_optional_endpoints(idp_jwk):
    with _idp_client(idp_jwk, {"userinfo_endpoint": "not-a-url"}) as client:
        discovery = OidcDiscovery(_settings(), client)

    assert discovery.authorization_endpoint() == f"{DISCOVERY_ISSUER}/authorize"


def test_discovery_allows_http_endpoints_for_a_development_http_issuer(idp_jwk):
    with _idp_client(idp_jwk) as client:
        discovery = OidcDiscovery(_settings("http://localhost:8080/realms/dev"), client)

    assert discovery.authorization_endpoint() == "http://localhost:8080/realms/dev/authorize"


def test_discovery_tolerates_absent_optional_capability_lists(idp_jwk):
    overrides = {"code_challenge_methods_supported": None, "token_endpoint_auth_methods_supported": None}
    with _idp_client(idp_jwk, overrides) as client:
        discovery = OidcDiscovery(_settings(), client)

    assert discovery.authorization_endpoint() == f"{DISCOVERY_ISSUER}/authorize"


def test_discovery_checks_token_auth_method_for_confidential_clients(idp_jwk):
    with _idp_client(idp_jwk, {"token_endpoint_auth_methods_supported": ["client_secret_post"]}) as client:
        OidcDiscovery(_settings(client_secret="secret"), client)

    with (
        _idp_client(idp_jwk, {"token_endpoint_auth_methods_supported": ["none"]}) as client,
        pytest.raises(OidcDiscoveryError, match="does not include 'client_secret_post'"),
    ):
        OidcDiscovery(_settings(client_secret="secret"), client)

    with (
        _idp_client(idp_jwk, {"token_endpoint_auth_methods_supported": "client_secret_post"}) as client,
        pytest.raises(OidcDiscoveryError, match="does not include 'client_secret_post'"),
    ):
        OidcDiscovery(_settings(client_secret="secret"), client)


def test_discovery_does_not_require_none_to_be_advertised_for_public_clients(idp_jwk):
    methods = ["private_key_jwt", "client_secret_basic", "client_secret_post", "client_secret_jwt"]
    with _idp_client(idp_jwk, {"token_endpoint_auth_methods_supported": methods}) as client:
        OidcDiscovery(_settings(), client)


def test_discovery_keeps_only_usable_signing_keys(idp_jwk):
    unusable = [
        {"kty": "EC", "crv": "P-256", "kid": "ec", "x": "AQAB", "y": "AQAB"},
        {key: value for key, value in idp_jwk.items() if key != "kid"},
        {**idp_jwk, "kid": ""},
        {**idp_jwk, "kid": "enc", "use": "enc"},
        {**idp_jwk, "kid": "rs512", "alg": "RS512"},
        {**idp_jwk, "kid": "private", "d": "AQAB"},
        {key: value for key, value in {**idp_jwk, "kid": "no-modulus"}.items() if key != "n"},
        "not-an-object",
    ]
    with _idp_client(idp_jwk, keys=[*unusable, idp_jwk]) as client:
        discovery = OidcDiscovery(_settings(), client)
        key = discovery.get_signing_key(kid="k1", algorithm="RS256")

    assert key == idp_jwk
    assert discovery._snapshot.signing_keys == (idp_jwk,)


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        pytest.param([], "JWKS response does not contain any usable RS256 public signing keys.", id="empty"),
        pytest.param(
            [{"kty": "EC", "crv": "P-256", "kid": "ec", "x": "AQAB", "y": "AQAB"}],
            "JWKS response does not contain any usable RS256 public signing keys.",
            id="only-unusable",
        ),
        pytest.param({"kid": "k1"}, "JWKS response does not contain a list of keys.", id="not-a-list"),
    ],
)
def test_discovery_rejects_jwks_without_usable_keys(idp_jwk, keys, message):
    with _idp_client(idp_jwk, keys=keys) as client, pytest.raises(OidcDiscoveryError) as exc:
        OidcDiscovery(_settings(), client)

    assert str(exc.value) == message


@pytest.mark.parametrize(
    ("response", "exception", "match"),
    [
        pytest.param(httpx2.Response(200, content=b"not json"), OidcDiscoveryError, "invalid JSON", id="json"),
        pytest.param(httpx2.Response(503), OidcDiscoveryError, "status code: 503", id="status"),
    ],
)
def test_discovery_normalizes_unusable_http_responses(response, exception, match):
    with (
        httpx2.Client(transport=httpx2.MockTransport(lambda _request: response)) as client,
        pytest.raises(exception, match=match),
    ):
        OidcDiscovery(_settings(), client)


@pytest.mark.parametrize(
    ("error_type", "exception"),
    [
        pytest.param(httpx2.ConnectError, OidcDiscoveryError, id="connection"),
        pytest.param(httpx2.ReadTimeout, OidcProviderTimeoutError, id="timeout"),
    ],
)
def test_discovery_normalizes_request_failures(error_type, exception):
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error_type("upstream failure", request=request)

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client, pytest.raises(exception):
        OidcDiscovery(_settings(), client)
