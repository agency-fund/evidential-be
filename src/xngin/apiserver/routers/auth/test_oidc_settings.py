import pytest

from xngin.apiserver import flags
from xngin.apiserver.routers.auth.oidc_settings import OidcMisconfiguredError, OidcSettings, parse_claim_map


def _parse(**overrides) -> OidcSettings:
    kwargs: dict = {
        "issuer": "https://accounts.google.com",
        "client_id": "client-id",
        "client_secret": "",
        "redirect_uri": "https://app.example.com/",
        "claim_map": "",
        "allow_http": False,
    }
    kwargs.update(overrides)
    return OidcSettings.parse(**kwargs)


def test_parse_google_settings():
    settings = _parse(client_secret="secret", claim_map="hd:hd")

    assert settings.issuer == "https://accounts.google.com"
    assert settings.client_secret == "secret"
    assert settings.claim_map == {"hd": "hd"}
    assert settings.discovery_url() == "https://accounts.google.com/.well-known/openid-configuration"


def test_parse_public_client_settings():
    settings = _parse(issuer="https://idp.example.com/oauth2/default", client_secret="  ")

    assert settings.client_secret is None
    assert settings.claim_map == {}
    assert settings.discovery_url() == "https://idp.example.com/oauth2/default/.well-known/openid-configuration"


def test_parse_keeps_trailing_slash_on_issuer_but_not_in_discovery_url():
    """authentik issuers end with a slash; iss must match it exactly while discovery strips it (Discovery 1.0 4.1)."""
    settings = _parse(issuer="https://idp.example.com/application/o/evidential/")

    assert settings.issuer == "https://idp.example.com/application/o/evidential/"
    assert (
        settings.discovery_url() == "https://idp.example.com/application/o/evidential/.well-known/openid-configuration"
    )


def test_parse_trims_whitespace():
    settings = _parse(
        issuer=" https://idp.example.com ", client_id=" client ", redirect_uri=" https://app.example.com/ "
    )

    assert settings.issuer == "https://idp.example.com"
    assert settings.client_id == "client"
    assert settings.redirect_uri == "https://app.example.com/"


def test_parse_allows_redirect_uri_with_query_and_port():
    settings = _parse(redirect_uri="https://app.example.com:8443/login?next=%2Fexperiments")

    assert settings.redirect_uri == "https://app.example.com:8443/login?next=%2Fexperiments"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"issuer": ""}, "XNGIN_OIDC_ISSUER environment variable is not set.", id="missing-issuer"),
        pytest.param(
            {"client_id": " "}, "XNGIN_OIDC_CLIENT_ID environment variable is not set.", id="missing-client-id"
        ),
        pytest.param(
            {"redirect_uri": ""}, "XNGIN_OIDC_REDIRECT_URI environment variable is not set.", id="missing-redirect-uri"
        ),
        pytest.param(
            {"issuer": "http://idp.example.com"},
            "XNGIN_OIDC_ISSUER must be an https:// URL (got 'http://idp.example.com').",
            id="http-issuer",
        ),
        pytest.param(
            {"issuer": "accounts.google.com"},
            "XNGIN_OIDC_ISSUER must be an https:// URL (got 'accounts.google.com').",
            id="schemeless-issuer",
        ),
        pytest.param(
            {"issuer": "javascript:alert(1)"},
            "XNGIN_OIDC_ISSUER must be an https:// URL (got 'javascript:alert(1)').",
            id="javascript-issuer",
        ),
        pytest.param(
            {"issuer": "https:///realms/dev"},
            "XNGIN_OIDC_ISSUER must include a hostname (got 'https:///realms/dev').",
            id="issuer-without-host",
        ),
        pytest.param(
            {"issuer": "https://user:secret@idp.example.com"},
            "XNGIN_OIDC_ISSUER must not include credentials (got 'https://user:secret@idp.example.com').",
            id="issuer-with-credentials",
        ),
        pytest.param(
            {"issuer": "https://idp.example.com?tenant=1"},
            "XNGIN_OIDC_ISSUER must not include a query (got 'https://idp.example.com?tenant=1').",
            id="issuer-with-query",
        ),
        pytest.param(
            {"issuer": "https://idp.example.com#fragment"},
            "XNGIN_OIDC_ISSUER must not include a fragment (got 'https://idp.example.com#fragment').",
            id="issuer-with-fragment",
        ),
        pytest.param(
            {"issuer": "https://idp.example.com:port"},
            "XNGIN_OIDC_ISSUER has an invalid port (got 'https://idp.example.com:port').",
            id="issuer-with-invalid-port",
        ),
        pytest.param(
            {"issuer": "https://idp.example.com/realms\ndev"},
            "XNGIN_OIDC_ISSUER must not contain whitespace or control characters "
            "(got 'https://idp.example.com/realms\\ndev').",
            id="issuer-with-control-character",
        ),
        pytest.param(
            {"issuer": "https://idp.example .com"},
            "XNGIN_OIDC_ISSUER must not contain whitespace or control characters (got 'https://idp.example .com').",
            id="issuer-with-embedded-space",
        ),
        pytest.param(
            {"redirect_uri": "http://localhost:3000/"},
            "XNGIN_OIDC_REDIRECT_URI must be an https:// URL (got 'http://localhost:3000/').",
            id="http-redirect-outside-development",
        ),
        pytest.param(
            {"redirect_uri": "/login"},
            "XNGIN_OIDC_REDIRECT_URI must be an https:// URL (got '/login').",
            id="relative-redirect",
        ),
        pytest.param(
            {"redirect_uri": "https://app.example.com/#/login"},
            "XNGIN_OIDC_REDIRECT_URI must not include a fragment (got 'https://app.example.com/#/login').",
            id="redirect-with-fragment",
        ),
        pytest.param(
            {"redirect_uri": "https://user@app.example.com/"},
            "XNGIN_OIDC_REDIRECT_URI must not include credentials (got 'https://user@app.example.com/').",
            id="redirect-with-credentials",
        ),
        pytest.param(
            {"claim_map": "hd"}, "XNGIN_OIDC_CLAIM_MAP entry 'hd' is not in claim:field form.", id="claim-map-no-colon"
        ),
        pytest.param(
            {"claim_map": ":hd"},
            "XNGIN_OIDC_CLAIM_MAP entry ':hd' is not in claim:field form.",
            id="claim-map-no-claim",
        ),
        pytest.param(
            {"claim_map": "hd:"},
            "XNGIN_OIDC_CLAIM_MAP entry 'hd:' is not in claim:field form.",
            id="claim-map-no-field",
        ),
        pytest.param(
            {"claim_map": "hd:email"},
            "XNGIN_OIDC_CLAIM_MAP target 'email' is not a Principal auxiliary field. Allowed: hd.",
            id="claim-map-core-field",
        ),
        pytest.param(
            {"claim_map": "hd:hd,org_slug:hd"},
            "XNGIN_OIDC_CLAIM_MAP maps more than one claim to 'hd'.",
            id="claim-map-duplicate-target",
        ),
    ],
)
def test_parse_rejects_invalid_settings(overrides, message):
    with pytest.raises(OidcMisconfiguredError) as exc:
        _parse(**overrides)

    assert str(exc.value) == message


def test_parse_allows_http_urls_in_development():
    settings = _parse(issuer="http://localhost:8080/realms/dev", redirect_uri="http://localhost:3000/", allow_http=True)

    assert settings.issuer == "http://localhost:8080/realms/dev"
    assert settings.redirect_uri == "http://localhost:3000/"


def test_parse_claim_map_tolerates_whitespace_and_empty_entries():
    assert parse_claim_map(" hd : hd , ,") == {"hd": "hd"}
    assert parse_claim_map("") == {}


def test_from_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flags, "OIDC_ISSUER", "https://idp.example.com")
    monkeypatch.setattr(flags, "OIDC_CLIENT_ID", "public-client")
    monkeypatch.setattr(flags, "OIDC_CLIENT_SECRET", "")
    monkeypatch.setattr(flags, "OIDC_REDIRECT_URI", "https://app.example.com/")
    monkeypatch.setattr(flags, "OIDC_CLAIM_MAP", "")

    settings = OidcSettings.from_flags()

    assert settings == OidcSettings(
        issuer="https://idp.example.com",
        client_id="public-client",
        redirect_uri="https://app.example.com/",
        claim_map={},
        client_secret=None,
    )


def test_from_flags_reports_missing_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flags, "OIDC_ISSUER", "")

    with pytest.raises(OidcMisconfiguredError, match=r"XNGIN_OIDC_ISSUER environment variable is not set\."):
        OidcSettings.from_flags()
