"""A development-only OpenID Provider that signs developers in as a preconfigured user with one click.

Enable it with XNGIN_DEVIDP=true and set XNGIN_OIDC_ISSUER to the public base URL followed by /devidp. The users are
defined in devidp_config.yaml, where each user's claims override those in the template; the file is re-read on every
request, so edits take effect without a restart. Nothing here is secret: the authorization code is a plaintext
description of the pending login, and a plaintext cookie remembers the user chosen last. The signing key is cached in
.xngin_devidp_signing_key so that restarts keep issued tokens verifiable. The server refuses to start with this
enabled outside a development environment.
"""

import asyncio
import datetime as dt
import hashlib
import html
import json
import time
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import parse_qs, urlencode

import jwt
import yaml
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, Cookie, Depends, FastAPI, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jwt.utils import base64url_decode, base64url_encode
from loguru import logger
from pydantic import BaseModel, JsonValue, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.dependencies import xngin_sync_db_session
from xngin.apiserver.sqla import tables
from xngin.cli.common import write_file_atomically

PREFIX = "/devidp"
CONFIG_PATH = Path(__file__).with_name("devidp_config.yaml")
SIGNING_KEY_PATH = Path(".xngin_devidp_signing_key")
REMEMBERED_USER_COOKIE = "devidp_user"
AUTHORIZATION_CODE_LIFETIME = dt.timedelta(minutes=1)
ID_TOKEN_LIFETIME = dt.timedelta(hours=1)


class DevIdpNotAllowedError(Exception):
    pass


type Claims = dict[str, JsonValue]


class DevIdpConfig(BaseModel):
    template: Claims
    users: dict[str, Claims]

    def claims_for(self, user_id: str) -> Claims:
        return self.template | self.users[user_id]

    @model_validator(mode="after")
    def every_user_has_string_sub_and_email(self) -> Self:
        for user_id in self.users:
            claims = self.claims_for(user_id)
            if missing := [name for name in ("sub", "email") if not isinstance(claims.get(name), str)]:
                raise ValueError(f"devidp user {user_id!r} lacks string claims: {', '.join(missing)}")
        return self


class AuthorizationRequest(BaseModel):
    """An authorization code request with PKCE (RFC 6749 section 4.1.1, RFC 7636 section 4.3)."""

    response_type: Literal["code"]
    client_id: str
    redirect_uri: str
    state: str
    nonce: str
    code_challenge: str
    code_challenge_method: Literal["S256"]
    scope: str = ""

    def mismatch_with_configured_client(self) -> str | None:
        if self.client_id != flags.OIDC_CLIENT_ID:
            return f"client_id {self.client_id!r} is not {flags.ENV_XNGIN_OIDC_CLIENT_ID} ({flags.OIDC_CLIENT_ID!r})."
        if self.redirect_uri != flags.OIDC_REDIRECT_URI:
            return (
                f"redirect_uri {self.redirect_uri!r} is not {flags.ENV_XNGIN_OIDC_REDIRECT_URI} "
                f"({flags.OIDC_REDIRECT_URI!r})."
            )
        return None

    def as_query(self, **extra: str) -> str:
        return urlencode(self.model_dump() | extra)

    def redirect_back_with(self, **response_params: str) -> RedirectResponse:
        separator = "&" if "?" in self.redirect_uri else "?"
        query = urlencode(response_params | {"state": self.state})
        return RedirectResponse(f"{self.redirect_uri}{separator}{query}", status_code=303)


class UserChoice(AuthorizationRequest):
    user: str


class AuthorizationCode(BaseModel):
    user: str
    nonce: str
    code_challenge: str
    client_id: str
    redirect_uri: str
    expires_at: dt.datetime

    def encode(self) -> str:
        return base64url_encode(self.model_dump_json().encode()).decode()

    @classmethod
    def decode(cls, code: str) -> Self:
        return cls.model_validate_json(base64url_decode(code))

    def is_expired(self) -> bool:
        return self.expires_at < dt.datetime.now(dt.UTC)


def _read_config() -> DevIdpConfig:
    return DevIdpConfig.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))


def _load_or_create_signing_key(path: Path) -> rsa.RSAPrivateKey:
    if cached_key := _read_cached_signing_key(path):
        return cached_key
    generation_started = time.perf_counter()
    new_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    logger.info(f"Generated a devidp signing key in {time.perf_counter() - generation_started:.3f}s; caching in {path}")
    pem = new_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    write_file_atomically(path, pem.decode(), private=True)
    return new_key


def _read_cached_signing_key(path: Path) -> rsa.RSAPrivateKey | None:
    try:
        cached_key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except FileNotFoundError:
        return None
    except (ValueError, TypeError, UnsupportedAlgorithm) as unreadable:
        logger.warning(f"Replacing the unreadable devidp signing key in {path}: {unreadable}")
        return None
    if not isinstance(cached_key, rsa.RSAPrivateKey):
        logger.warning(f"Replacing the devidp signing key in {path}: it is not an RSA key")
        return None
    return cached_key


@cache
def _signing_key() -> rsa.RSAPrivateKey:
    return _load_or_create_signing_key(SIGNING_KEY_PATH)


@cache
def _public_jwk() -> dict[str, Any]:
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(_signing_key().public_key(), as_dict=True)
    rfc7638_thumbprint_input = json.dumps({name: jwk[name] for name in ("e", "kty", "n")}, separators=(",", ":"))
    kid = base64url_encode(hashlib.sha256(rfc7638_thumbprint_input.encode()).digest()).decode()
    return {**jwk, "kid": kid, "use": "sig", "alg": "RS256"}


def _issuer() -> str:
    return f"{flags.XNGIN_PUBLIC_API_BASE_URL}{PREFIX}"


def _registered_users_by_email(session: Session, emails: list[str]) -> dict[str, tables.User]:
    lowercase_emails = [email.lower() for email in emails]
    users = session.scalars(select(tables.User).where(func.lower(tables.User.email).in_(lowercase_emails)))
    return {user.email.lower(): user for user in users}


def _pkce_verifier_matches(code_verifier: str, code_challenge: str) -> bool:
    return base64url_encode(hashlib.sha256(code_verifier.encode()).digest()).decode() == code_challenge


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _read_config()
    await asyncio.to_thread(_public_jwk)
    yield


router = APIRouter(prefix=PREFIX, lifespan=_lifespan)


@router.get("/.well-known/openid-configuration")
async def openid_configuration() -> dict[str, Any]:
    return {
        "issuer": _issuer(),
        "authorization_endpoint": f"{_issuer()}/authorize",
        "token_endpoint": f"{_issuer()}/token",
        "jwks_uri": f"{_issuer()}/jwks",
        "response_types_supported": ["code"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
    }


@router.get("/jwks")
async def jwks() -> dict[str, Any]:
    return {"keys": [_public_jwk()]}


@router.get("/authorize", response_class=HTMLResponse)
def authorize(
    authorization: Annotated[AuthorizationRequest, Query()],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    remembered_user: Annotated[str | None, Cookie(alias=REMEMBERED_USER_COOKIE)] = None,
) -> HTMLResponse:
    if mismatch := authorization.mismatch_with_configured_client():
        return HTMLResponse(_page("Invalid authorization request", f"<p>{html.escape(mismatch)}</p>"), 400)
    users = _read_config()
    claims_by_user = {user_id: users.claims_for(user_id) for user_id in users.users}
    registered_users = _registered_users_by_email(session, [str(claims["email"]) for claims in claims_by_user.values()])
    user_links = "".join(
        _user_link(
            user_id,
            claims,
            authorization,
            is_remembered=user_id == remembered_user,
            registered_user=registered_users.get(str(claims["email"]).lower()),
        )
        for user_id, claims in claims_by_user.items()
    )
    deny_link = f'<a class="deny" href="{PREFIX}/deny?{html.escape(authorization.as_query())}">Deny</a>'
    return HTMLResponse(_page("Sign in as…", user_links + deny_link))


@router.get("/choose")
async def choose(choice: Annotated[UserChoice, Query()]) -> Response:
    if mismatch := choice.mismatch_with_configured_client():
        return HTMLResponse(_page("Invalid authorization request", f"<p>{html.escape(mismatch)}</p>"), 400)
    if choice.user not in _read_config().users:
        return HTMLResponse(_page("Unknown user", f"<p>{html.escape(choice.user)}</p>"), 400)
    code = AuthorizationCode(
        user=choice.user,
        nonce=choice.nonce,
        code_challenge=choice.code_challenge,
        client_id=choice.client_id,
        redirect_uri=choice.redirect_uri,
        expires_at=dt.datetime.now(dt.UTC) + AUTHORIZATION_CODE_LIFETIME,
    )
    response = choice.redirect_back_with(code=code.encode())
    response.set_cookie(REMEMBERED_USER_COOKIE, choice.user, path=PREFIX, httponly=True, samesite="lax")
    return response


@router.get("/deny")
async def deny(authorization: Annotated[AuthorizationRequest, Query()]) -> Response:
    if mismatch := authorization.mismatch_with_configured_client():
        return HTMLResponse(_page("Invalid authorization request", f"<p>{html.escape(mismatch)}</p>"), 400)
    return authorization.redirect_back_with(error="access_denied")


@router.post("/token")
async def token(request: Request) -> JSONResponse:
    form = {name: values[0] for name, values in parse_qs((await request.body()).decode()).items()}
    if form.get("grant_type") != "authorization_code":
        return _oauth_error("unsupported_grant_type", "Only authorization_code is supported.")
    try:
        code = AuthorizationCode.decode(form.get("code", ""))
    except ValueError:
        return _oauth_error("invalid_grant", "The code is malformed.")
    if code.is_expired():
        return _oauth_error("invalid_grant", "The code has expired.")
    if form.get("client_id") != code.client_id or form.get("redirect_uri") != code.redirect_uri:
        return _oauth_error("invalid_grant", "client_id or redirect_uri differ from the authorization request.")
    if not _pkce_verifier_matches(form.get("code_verifier", ""), code.code_challenge):
        return _oauth_error("invalid_grant", "code_verifier does not match code_challenge.")
    users = _read_config()
    if code.user not in users.users:
        return _oauth_error("invalid_grant", f"devidp user {code.user!r} no longer exists.")
    issued_at = dt.datetime.now(dt.UTC)
    claims = users.claims_for(code.user) | {
        "iss": _issuer(),
        "aud": code.client_id,
        "iat": issued_at,
        "exp": issued_at + ID_TOKEN_LIFETIME,
        "nonce": code.nonce,
    }
    id_token = jwt.encode(claims, _signing_key(), algorithm="RS256", headers={"kid": _public_jwk()["kid"]})
    return JSONResponse(
        {
            "id_token": id_token,
            "access_token": "devidp-access-token-unused",
            "token_type": "Bearer",
            "expires_in": int(ID_TOKEN_LIFETIME.total_seconds()),
        },
        headers={"Cache-Control": "no-store"},
    )


def _oauth_error(error: str, description: str) -> JSONResponse:
    return JSONResponse({"error": error, "error_description": description}, status_code=400)


def _registration_pills(registered_user: tables.User | None, claims: Claims) -> str:
    if registered_user is None:
        return ""
    privileged_pill = _pill("privileged", "privileged") if registered_user.is_privileged else ""
    if registered_user.iss is None:
        return _pill("invited", "registered") + privileged_pill
    if (registered_user.iss, registered_user.sub) == (_issuer(), claims["sub"]):
        return _pill("registered", "registered") + privileged_pill
    bound_identity = f"iss={registered_user.iss} sub={registered_user.sub}"
    return _pill("registered to a different identity", "conflict", hover_text=bound_identity) + privileged_pill


def _pill(label: str, kind: str, *, hover_text: str = "") -> str:
    return f'<span class="pill {kind}" title="{html.escape(hover_text)}">{html.escape(label)}</span>'


def _user_link(
    user_id: str,
    claims: Claims,
    authorization: AuthorizationRequest,
    *,
    is_remembered: bool,
    registered_user: tables.User | None,
) -> str:
    href = f"{PREFIX}/choose?{authorization.as_query(user=user_id)}"
    pills = (_pill("last used", "last-used") if is_remembered else "") + _registration_pills(registered_user, claims)
    return (
        f'<a class="user{" remembered" if is_remembered else ""}" href="{html.escape(href)}">'
        f"<strong>{html.escape(str(claims.get('name', user_id)))}</strong> {html.escape(str(claims['email']))}"
        f"{pills}<pre>{html.escape(json.dumps(claims, indent=2))}</pre></a>"
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>devidp: {html.escape(title)}</title><style>
body {{ font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; color: #1c2024; }}
a {{ display: block; padding: 0.75rem 1rem; margin: 0.5rem 0; border: 1px solid #cdced6; border-radius: 0.5rem;
     color: inherit; text-decoration: none; }}
a:hover {{ background: #f0f0f3; }}
a.remembered {{ border: 2px solid #3e63dd; }}
a.deny {{ color: #ce2c31; text-align: center; }}
.pill {{ display: inline-block; margin-left: 0.5rem; padding: 0.05rem 0.5rem; border-radius: 999px;
         font-size: 0.75rem; vertical-align: 0.1rem; }}
.pill.last-used {{ background: #e6edfe; color: #3a5bc7; }}
.pill.registered {{ background: #e6f6eb; color: #218358; }}
.pill.conflict {{ background: #feebec; color: #ce2c31; }}
.pill.privileged {{ background: #f3e7fc; color: #8145b5; }}
pre {{ margin: 0.5rem 0 0; font-size: 0.8rem; color: #60646c; }}
</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>"""
