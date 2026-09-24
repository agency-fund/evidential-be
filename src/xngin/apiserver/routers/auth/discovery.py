import datetime
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

import httpx2
import jwt
from fastapi import HTTPException, Request, status
from loguru import logger

from xngin.apiserver import flags
from xngin.apiserver.routers.auth.oidc_settings import OidcSettings, check_absolute_https_url

DISCOVERY_CACHE_LIFETIME = datetime.timedelta(minutes=60)

LOGIN_UNAVAILABLE_DETAIL = "Login is not configured on this server."


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class OidcProviderError(Exception):
    """Base class for controlled failures while communicating with an OpenID Provider."""


class OidcDiscoveryError(OidcProviderError):
    pass


class OidcProviderTimeoutError(OidcProviderError):
    pass


@dataclass(frozen=True, slots=True)
class _Endpoints:
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str


@dataclass(frozen=True, slots=True)
class _DiscoverySnapshot:
    endpoints: _Endpoints
    signing_keys: tuple[dict, ...]
    refreshed_at: datetime.datetime


@dataclass(slots=True)
class _RefreshAttempts:
    discovery: datetime.datetime
    jwks: datetime.datetime


class OidcDiscovery:
    """Loads and caches one provider's discovery document and RS256 signing keys.

    Instances are safe to share between requests. Endpoint and signing-key methods load and refresh
    provider data as needed, so callers do not manage discovery state themselves. If the discovery documents or JWKS
    keys fail to fetch, previous successful fetches will be re-used.
    """

    def __init__(
        self,
        settings: OidcSettings,
        client: httpx2.Client | None = None,
        *,
        clock: Callable[[], datetime.datetime] = _utc_now,
    ):
        self.settings = settings
        self._client = client or httpx2.Client(
            transport=httpx2.HTTPTransport(retries=2),
            timeout=15.0,
        )
        self._owns_client = client is None
        self._clock = clock
        self._lock = threading.Lock()
        try:
            self._snapshot = self._fetch_discovery()
        except OidcProviderError:
            self.close()
            raise
        self._refresh_attempts = _RefreshAttempts(
            discovery=self._snapshot.refreshed_at,
            jwks=self._snapshot.refreshed_at,
        )

    def authorization_endpoint(self) -> str:
        self._refresh_discovery_if_expired()
        return self._snapshot.endpoints.authorization_endpoint

    def token_endpoint(self) -> str:
        self._refresh_discovery_if_expired()
        return self._snapshot.endpoints.token_endpoint

    def _refresh_discovery_if_expired(self) -> None:
        if self._snapshot_is_current() or not self._should_retry_fetch(self._refresh_attempts.discovery):
            return
        with self._lock:
            if self._snapshot_is_current() or not self._should_retry_fetch(self._refresh_attempts.discovery):
                return
            self._refresh_attempts.discovery = self._clock()
            try:
                self._snapshot = self._fetch_discovery()
            except OidcProviderError as exc:
                logger.warning(f"OpenID discovery refresh failed; retaining cached discovery document: {exc}")
                return
            self._refresh_attempts.discovery = self._snapshot.refreshed_at
            self._refresh_attempts.jwks = self._snapshot.refreshed_at

    def get_signing_key(self, *, kid: object, algorithm: object) -> dict | None:
        """Returns the requested RS256 key, refreshing JWKS once when an unknown key may have rotated in."""
        self._refresh_discovery_if_expired()
        snapshot = self._snapshot
        if algorithm != "RS256" or not isinstance(kid, str) or not kid:
            return None
        if key := self._find_signing_key(snapshot, kid):
            return key

        with self._lock:
            if key := self._find_signing_key(self._snapshot, kid):
                return key

            if not self._should_retry_fetch(self._refresh_attempts.jwks):
                return None

            logger.info(f"Refreshing OpenID signing keys from {self._snapshot.endpoints.jwks_uri}")
            self._refresh_attempts.jwks = self._clock()
            signing_keys = self._fetch_signing_keys(self._snapshot.endpoints.jwks_uri)
            refreshed_snapshot = replace(self._snapshot, signing_keys=signing_keys)
            self._snapshot = refreshed_snapshot
            return self._find_signing_key(refreshed_snapshot, kid)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _snapshot_is_current(self) -> bool:
        return self._snapshot.refreshed_at >= self._clock() - DISCOVERY_CACHE_LIFETIME

    def _should_retry_fetch(self, attempted_at: datetime.datetime) -> bool:
        return attempted_at <= self._clock() - datetime.timedelta(minutes=1)

    @staticmethod
    def _find_signing_key(snapshot: _DiscoverySnapshot, kid: str) -> dict | None:
        return next((key for key in snapshot.signing_keys if key.get("kid") == kid), None)

    def _fetch_object(self, url: str) -> dict:
        try:
            response = self._client.get(url)
        except httpx2.TimeoutException as exc:
            raise OidcProviderTimeoutError(f"Fetching {url} timed out.") from exc
        except httpx2.RequestError as exc:
            raise OidcDiscoveryError(f"Fetching {url} failed: {type(exc).__name__}.") from exc
        if response.status_code != 200:
            raise OidcDiscoveryError(f"Fetching {url} failed with an unexpected status code: {response.status_code}")
        try:
            parsed = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OidcDiscoveryError(f"{url} returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise OidcDiscoveryError(f"{url} returned a non-dictionary response")
        return parsed

    def _fetch_signing_keys(self, jwks_uri: str) -> tuple[dict, ...]:
        jwks = self._fetch_object(jwks_uri)
        keys = jwks.get("keys")
        if not isinstance(keys, list):
            raise OidcDiscoveryError("JWKS response does not contain a list of keys.")
        usable = [key for key in keys if self._is_usable_signing_key(key)]
        if not usable:
            raise OidcDiscoveryError("JWKS response does not contain any usable RS256 public signing keys.")
        if len(usable) < len(keys):
            logger.warning(f"Ignoring {len(keys) - len(usable)} JWKS entries that are not usable RS256 signing keys.")
        return tuple(usable)

    def _fetch_discovery(self) -> _DiscoverySnapshot:
        discovery_url = self.settings.discovery_url()
        logger.info(f"Fetching OpenID configuration from {discovery_url}")
        config = self._fetch_object(discovery_url)
        endpoints = self._validate_config(config)
        signing_keys = self._fetch_signing_keys(endpoints.jwks_uri)
        return _DiscoverySnapshot(
            endpoints=endpoints,
            signing_keys=signing_keys,
            refreshed_at=self._clock(),
        )

    def _validate_config(self, config: dict) -> _Endpoints:
        if config.get("issuer") != self.settings.issuer:
            raise OidcDiscoveryError(
                f"Discovery document issuer '{config.get('issuer')}' does not match "
                f"{flags.ENV_XNGIN_OIDC_ISSUER} '{self.settings.issuer}'."
            )

        endpoints_may_use_http = self.settings.issuer.startswith("http://")

        def endpoint(key: str) -> str:
            value = config.get(key)
            if not isinstance(value, str) or not value:
                raise OidcDiscoveryError(f"Discovery document is missing {key}.")
            if reason := check_absolute_https_url(value, allow_http=endpoints_may_use_http):
                raise OidcDiscoveryError(f"Discovery document's {key} {reason} (got {value!r}).")
            return value

        endpoints = _Endpoints(
            authorization_endpoint=endpoint("authorization_endpoint"),
            token_endpoint=endpoint("token_endpoint"),
            jwks_uri=endpoint("jwks_uri"),
        )

        self._require_supported(config, "response_types_supported", "code", required=True)
        self._require_supported(config, "id_token_signing_alg_values_supported", "RS256", required=True)
        self._require_supported(config, "code_challenge_methods_supported", "S256", required=False)
        # When the client_secret is set, Evidential expects the provider support client_secret_post
        # (https://www.rfc-editor.org/info/rfc7591/#section-2). Google is unusual in this requirement.
        if self.settings.client_secret is not None:
            self._require_supported(
                config, "token_endpoint_auth_methods_supported", "client_secret_post", required=False
            )
        return endpoints

    @staticmethod
    def _require_supported(config: dict, field_name: str, value: str, *, required: bool) -> None:
        supported = config.get(field_name)
        if supported is None:
            if required:
                raise OidcDiscoveryError(f"Discovery document is missing {field_name}.")
            return
        if not isinstance(supported, list) or value not in supported:
            raise OidcDiscoveryError(f"Discovery document's {field_name} does not include '{value}'.")

    @staticmethod
    def _is_usable_signing_key(key: object) -> bool:
        if not isinstance(key, dict):
            return False
        if key.get("kty") != "RSA" or key.get("use", "sig") != "sig" or key.get("alg", "RS256") != "RS256":
            return False
        if not isinstance(key.get("kid"), str) or not key["kid"] or "d" in key:
            return False
        try:
            jwt.PyJWK(key, algorithm="RS256")
        except jwt.PyJWTError, ValueError:
            return False
        return True


def get_oidc_discovery(
    request: Request,
) -> OidcDiscovery:
    """Provides the application-owned discovery client for the configured identity provider."""
    discovery = getattr(request.app.state, "oidc_discovery", None)
    if not isinstance(discovery, OidcDiscovery):
        # The lifespan skips creating the discovery client in airplane mode and when testing tokens are enabled.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=LOGIN_UNAVAILABLE_DETAIL)
    return discovery
