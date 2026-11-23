import datetime as dt
import json
import secrets
import time
from typing import Annotated

import chafernet
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.functions import count

from xngin.apiserver import flags
from xngin.apiserver.dependencies import xngin_sync_db_session
from xngin.apiserver.request_context_middleware import add_request_fields
from xngin.apiserver.routers.admin.admin_common import find_user_by_email
from xngin.apiserver.routers.auth.discovery import LOGIN_UNAVAILABLE_DETAIL, get_oidc_discovery
from xngin.apiserver.routers.auth.oidc_settings import get_oidc_settings
from xngin.apiserver.routers.auth.principal import Principal
from xngin.apiserver.routers.auth.signup_policy import email_domain, load_signup_policy, signup_allowed
from xngin.apiserver.routers.auth.token_cryptor import TokenCryptor
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.bootstrap import create_entities_for_first_time_user

# The length of time that a session token is considered valid.
SESSION_TOKEN_LIFETIME = dt.timedelta(hours=12).seconds
SESSION_TOKEN_LOCAL_KEYSET_FILE = ".xngin_session_token_keyset"  # noqa: S105
SESSION_TOKEN_PREFIX = "xa_"  # noqa: S105
SESSION_TOKEN_AAD_VERSION = "xngin-session-v1"  # noqa: S105

# Set TESTING_TOKENS_ENABLED to allow statically defined bearer tokens to skip the JWT validation.
AIRPLANE_TOKEN = "airplane-mode-token"  # noqa: S105
PRIVILEGED_EMAIL = "testing-privileged@example.com"
PRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
TESTING_TOKENS_ENABLED = False
UNPRIVILEGED_EMAIL = "testing-unprivileged@example.com"
UNPRIVILEGED_TOKEN_FOR_TESTING = secrets.token_urlsafe(32)
UNPRIVILEGED_EMAIL_2 = "testing-unprivileged-2@example.com"
UNPRIVILEGED_TOKEN_FOR_TESTING_2 = secrets.token_urlsafe(32)
TESTING_TOKENS: dict[str, Principal] = {
    AIRPLANE_TOKEN: Principal(
        email="testing@example.com",
        hd="example.com",
        iat=int(time.time()),
        iss="airplane",
        sub="airplane",
    ),
    UNPRIVILEGED_TOKEN_FOR_TESTING: Principal(
        email=UNPRIVILEGED_EMAIL,
        hd="example.com",
        iat=int(time.time()),
        iss="testing",
        sub="testing",
    ),
    UNPRIVILEGED_TOKEN_FOR_TESTING_2: Principal(
        email=UNPRIVILEGED_EMAIL_2,
        hd="example.com",
        iat=int(time.time()),
        iss="testing",
        sub="testing",
    ),
    PRIVILEGED_TOKEN_FOR_TESTING: Principal(
        email=PRIVILEGED_EMAIL,
        hd="example.com",
        iat=int(time.time()),
        iss="testing",
        sub="testing",
    ),
}


def session_token_aad() -> bytes:
    """Returns the associated data that binds session tokens to the identity provider configuration that issued them.

    A token fails to decrypt after the issuer or client ID changes, so sessions from a previously configured identity
    provider end immediately. The raw flag values are used so that this works when OIDC is not configured (e.g. in
    airplane mode).
    """
    fields = [
        SESSION_TOKEN_AAD_VERSION,
        flags.OIDC_ISSUER.strip(),
        flags.OIDC_CLIENT_ID.strip(),
    ]
    return json.dumps(fields, separators=(",", ":")).encode()


class SessionTokenCryptor:
    """Codec for encrypted serializations of Principals."""

    def __init__(self):
        self._token_cryptor = TokenCryptor(
            ttl=SESSION_TOKEN_LIFETIME,
            keyset_env_var=flags.ENV_SESSION_TOKEN_KEYSET,
            local_keyset_filename=SESSION_TOKEN_LOCAL_KEYSET_FILE,
            prefix=SESSION_TOKEN_PREFIX,
        )
        self._aad = session_token_aad()

    def encode(self, principal: Principal) -> str:
        payload = json.dumps(principal.model_dump(), separators=(",", ":")).encode()
        return self._token_cryptor.encrypt(payload, self._aad)

    def decode(self, token: str) -> Principal:
        decrypted = self._token_cryptor.decrypt(token, self._aad)
        return Principal.model_validate_json(decrypted)


class CompatHTTPBearer(HTTPBearer):
    def __init__(self):
        super().__init__(description="Session token obtained from the auth_callback operation.", auto_error=True)

    def make_not_authenticated_error(self) -> HTTPException:
        # TODO: Migrate this to 401 so that we can remove this custom class.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Authorization header is required")


def require_valid_session_token(
    authorization: Annotated[
        HTTPAuthorizationCredentials,
        Depends(CompatHTTPBearer()),
    ],
    session_cryptor: Annotated[SessionTokenCryptor, Depends()],
) -> Principal:
    """Dependency for decoding the session token and retrieving a Principal.

    Returns:
        Principal containing the validated claims from the ID token.
    """
    token = authorization.credentials
    del authorization
    if principal := get_special_principal(token):
        return principal
    try:
        return session_cryptor.decode(token)
    except chafernet.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token invalid or expired.",
        ) from err


def require_user_from_token(
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    principal: Annotated[Principal, Depends(require_valid_session_token)],
) -> tables.User:
    """Dependency for fetching the User record matching the authenticated user's email.

    Users are never created here by the sign-up policy; that happens only at login (see sign_in), so a deleted user's
    session token cannot recreate them.

    Adds the user's ID to the request's log context as user_id.
    """
    user = _lookup_or_create(session, principal, allow_signup=False)
    if user:
        add_request_fields(user_id=user.id)
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Expired session, or user not found.",
    )


def _authenticate_existing(session: Session, user: tables.User, principal: Principal) -> tables.User | None:
    """Returns user if principal may authenticate as them, binding the principal's identity to invited users."""
    if principal.user_id is not None and principal.user_id != user.id:
        return None
    if user.last_logout.timestamp() > principal.iat:
        return None
    if user.iss == principal.iss and user.sub == principal.sub:
        return user
    # Users that are invited but who have not yet logged in will have (iss, sub) == (None, None), meaning that they
    # can authenticate with any IDP. After logging in the first time, they will only ever be able to log in with
    # that IDP.
    if user.iss is None:
        user.iss = principal.iss
        user.sub = principal.sub
        session.commit()
        return user
    return None


def sign_in(session: Session, principal: Principal) -> tables.User | None:
    """Looks up the user signing in, creating them if the sign-up policy permits.

    Called when the identity provider has just authenticated the principal. Returns None when the principal may not
    authenticate.
    """
    return _lookup_or_create(session, principal, allow_signup=True)


def _lookup_or_create(session: Session, principal: Principal, *, allow_signup: bool) -> tables.User | None:
    """Lookup or create a user based on email, iss, sub, and iat.

    When no user matches, there are three cases in which a user is created:
    1. First use of an installation: there are no users in the database yet. The user is privileged and gets an
       organization (and developer samples, depending on flags).
    2. Airplane mode: the server runs with AIRPLANE_MODE set and the request presents the airplane-mode token, as tests
       run in airplane mode do. As in case 1.
    3. Signup: allow_signup is true and the server's signup policy (see signup_policy.py) permits the email address.
       The user is unprivileged and gets only a users row; no organization or datasource is created.

    Returns None when the principal may not authenticate.
    """
    if user := find_user_by_email(session, principal.email):
        return _authenticate_existing(session, user, principal)

    user_count = session.scalar(select(count(tables.User.id)))
    if user_count == 0 or (flags.AIRPLANE_MODE and principal.iss == "airplane"):
        user = tables.User(email=principal.email, iss=principal.iss, sub=principal.sub, is_privileged=True)
        user = create_entities_for_first_time_user(
            session,
            user,
            flags.XNGIN_DEVDWH_DSN,
            bigquery_dsn=flags.XNGIN_DEVDWH_GCP_BIGQUERY_DSN,
            redshift_dsn=flags.XNGIN_DEVDWH_AWS_REDSHIFT_DSN,
        )
        session.commit()
        return user

    if not allow_signup:
        return None
    policy = load_signup_policy(session)
    if not signup_allowed(policy, principal.email):
        return None

    new_id = session.execute(
        pg_insert(tables.User)
        .values(email=principal.email, iss=principal.iss, sub=principal.sub, is_privileged=False)
        .on_conflict_do_nothing()
        .returning(tables.User.id)
    ).scalar_one_or_none()
    session.commit()
    if new_id is None:
        user = find_user_by_email(session, principal.email)
        return _authenticate_existing(session, user, principal) if user else None

    logger.info(f"Created user {new_id} by signup: domain={email_domain(principal.email)!r} signup_mode={policy.mode}")
    return session.get_one(tables.User, new_id)


def enable_testing_tokens():
    """Configures the authentication system to enable tokens used in unit tests."""
    global TESTING_TOKENS_ENABLED
    TESTING_TOKENS_ENABLED = True


def get_special_principal(token: str) -> Principal | None:
    """In testing or airplane mode, specific tokens map to specific principals."""
    if TESTING_TOKENS_ENABLED and token in TESTING_TOKENS:
        return TESTING_TOKENS[token]
    if flags.AIRPLANE_MODE and token == AIRPLANE_TOKEN:
        return TESTING_TOKENS[AIRPLANE_TOKEN]
    return None


def disable(app):
    """Disables interaction with internet-dependent authentication resources."""

    def unavailable():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=LOGIN_UNAVAILABLE_DETAIL)

    # Neither the environment nor the identity provider is consulted; login endpoints report that login is unavailable.
    app.dependency_overrides[get_oidc_settings] = unavailable
    app.dependency_overrides[get_oidc_discovery] = unavailable


def setup(app):
    """Configures FastAPI dependencies for OIDC."""

    # If we are not in airplane mode, there is no setup to do.
    if not flags.AIRPLANE_MODE:
        return

    logger.warning("AIRPLANE_MODE is set.")

    disable(app)
