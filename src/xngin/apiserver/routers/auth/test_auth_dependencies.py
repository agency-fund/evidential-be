import contextlib
import os
import time
from datetime import datetime, timedelta

import fastapi
import pytest
from chafernet import Keyset
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xngin.apiserver import database, flags
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.routers.admin.admin_common import DEFAULT_NO_DWH_SOURCE_NAME, default_organization_name
from xngin.apiserver.routers.admin.test_admin_api import find_ds_with_name
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.apiserver.routers.auth.auth_dependencies import (
    AIRPLANE_TOKEN,
    PRIVILEGED_TOKEN_FOR_TESTING,
    SESSION_TOKEN_LIFETIME,
    SESSION_TOKEN_PREFIX,
    TESTING_TOKENS,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
    SessionTokenCryptor,
    require_user_from_token,
    require_valid_session_token,
    sign_in,
)
from xngin.apiserver.routers.auth.principal import Principal
from xngin.apiserver.routers.auth.token_cryptor import TokenCryptor, TokenCryptorMisconfiguredError
from xngin.apiserver.settings import NoDwh
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.bootstrap import (
    ALT_TESTING_DWH_DATASOURCE_NAME,
    PGSTRESS_DATASOURCE_NAME,
    TESTING_DWH_DATASOURCE_NAME,
)
from xngin.apiserver.testing.helpers import delete_seeded_users
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_TABLE_NAME


@contextlib.contextmanager
def temporary_env_var(name: str, value: str):
    """Temporarily set environment variable for the duration of the context."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is not None:
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


@contextlib.contextmanager
def temporary_unset_env_var(name: str):
    previous = os.environ.get(name)
    os.environ.pop(name, None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ[name] = previous


def test_require_valid_session_token_missing_prefix():
    cryp = SessionTokenCryptor()
    with pytest.raises(HTTPException, match="token invalid") as exc:
        require_valid_session_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="abc"), cryp)
    assert exc.value.status_code == 401


def test_require_valid_session_token_misconfigured():
    cryp = SessionTokenCryptor()
    with pytest.raises(TokenCryptorMisconfiguredError):
        require_valid_session_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="xa_abc"), cryp)


def test_require_valid_session_token():
    with temporary_env_var(flags.ENV_SESSION_TOKEN_KEYSET, Keyset.generate().to_base64()):
        cryp = SessionTokenCryptor()
        expected = Principal(email="test@example.com", hd="", iat=0, iss="", sub="")
        valid_token = cryp.encode(expected)
        actual = require_valid_session_token(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=valid_token,
            ),
            cryp,
        )
        assert actual == expected


@pytest.mark.parametrize("variant", ["x", ".", " ", "==", "  ", "===", ";", "\n"])
def test_require_valid_session_token_invalid(variant):
    with temporary_env_var(flags.ENV_SESSION_TOKEN_KEYSET, Keyset.generate().to_base64()):
        cryp = SessionTokenCryptor()
        expected = Principal(email="test@example.com", hd="", iat=0, iss="", sub="")
        with pytest.raises(HTTPException, match="token invalid") as exc:
            require_valid_session_token(
                HTTPAuthorizationCredentials(
                    scheme="Bearer",
                    credentials=cryp.encode(expected) + variant,
                ),
                cryp,
            )
        assert exc.value.status_code == 401
        with pytest.raises(HTTPException, match="token invalid") as exc:
            require_valid_session_token(
                HTTPAuthorizationCredentials(
                    scheme="Bearer",
                    credentials=variant + cryp.encode(expected),
                ),
                cryp,
            )
        assert exc.value.status_code == 401


@pytest.fixture(name="session_token_env")
def fixture_session_token_env(monkeypatch: pytest.MonkeyPatch):
    """Configures a session token keyset and the identity provider settings used by session_token_aad()."""
    monkeypatch.setenv(flags.ENV_SESSION_TOKEN_KEYSET, Keyset.generate().to_base64())
    monkeypatch.setattr(flags, "OIDC_ISSUER", "https://idp.example.com")
    monkeypatch.setattr(flags, "OIDC_CLIENT_ID", "client-id")


SESSION_PRINCIPAL = Principal(email="test@example.com", hd="", iat=0, iss="https://idp.example.com", sub="sub")


def test_session_token_round_trips(session_token_env):
    token = SessionTokenCryptor().encode(SESSION_PRINCIPAL)

    assert SessionTokenCryptor().decode(token) == SESSION_PRINCIPAL


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        pytest.param("OIDC_ISSUER", "https://other-idp.example.com", id="issuer"),
        pytest.param("OIDC_CLIENT_ID", "other-client-id", id="client-id"),
    ],
)
def test_session_token_is_rejected_after_idp_configuration_changes(
    session_token_env, monkeypatch: pytest.MonkeyPatch, setting, value
):
    token = SessionTokenCryptor().encode(SESSION_PRINCIPAL)
    monkeypatch.setattr(flags, setting, value)

    with pytest.raises(HTTPException, match="token invalid") as exc:
        require_valid_session_token(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=token), SessionTokenCryptor()
        )
    assert exc.value.status_code == 401


def test_session_token_without_aad_is_rejected(session_token_env):
    """Tokens encrypted without session_token_aad() are rejected."""
    legacy_cryptor = TokenCryptor(
        ttl=SESSION_TOKEN_LIFETIME,
        keyset_env_var=flags.ENV_SESSION_TOKEN_KEYSET,
        local_keyset_filename="file-does-not-exist",
        prefix=SESSION_TOKEN_PREFIX,
    )
    legacy_token = legacy_cryptor.encrypt(SESSION_PRINCIPAL.model_dump_json().encode(), b"")

    with pytest.raises(HTTPException, match="token invalid") as exc:
        require_valid_session_token(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=legacy_token), SessionTokenCryptor()
        )
    assert exc.value.status_code == 401


def test_token_cryptor_uses_custom_prefix_and_env_var():
    env_var_name = "XNGIN_TEST_TOKEN_KEYSET"
    with temporary_env_var(env_var_name, Keyset.generate().to_base64()):
        cryptor = TokenCryptor(
            ttl=60,
            keyset_env_var=env_var_name,
            local_keyset_filename="file-does-not-exist",
            prefix="xp_",
        )
        token = cryptor.encrypt(b"payload", b"")
        assert token.startswith("xp_")
        assert cryptor.decrypt(token, b"") == b"payload"


def test_token_cryptor_reads_configured_local_keyset_file(tmp_path):
    keyset_file = tmp_path / "my-local-keyset"
    keyset_file.write_text(Keyset.generate().to_base64())
    env_var_name = "XNGIN_TEST_TOKEN_KEYSET"
    with temporary_env_var(env_var_name, "local"):
        cryptor = TokenCryptor(
            ttl=60,
            keyset_env_var=env_var_name,
            local_keyset_filename=str(keyset_file),
            prefix="xp_",
        )
        token = cryptor.encrypt(b"payload", b"")
        assert cryptor.decrypt(token, b"") == b"payload"


def test_user_from_token_invite(xngin_session: Session):
    """
    Tests that invited users are updated with the IDP details on their first login, and
    that they are then bound to that IDP afterwards.
    """
    delete_seeded_users(xngin_session)

    xngin_session.add(tables.User(email="u1@example.com", iss="iss", sub="sub"))
    # Invited users have (iss, sub) = (None, None).
    xngin_session.add(tables.User(email="inv@example.com", iss=None, sub=None))
    xngin_session.commit()

    invited_user_principal = Principal(
        email="inv@example.com", iss="invited", sub="invited", hd="", iat=int(time.time())
    )

    second_user = require_user_from_token(xngin_session, invited_user_principal)
    assert second_user.iss == "invited"
    assert second_user.sub == "invited"

    second_user_again = require_user_from_token(xngin_session, invited_user_principal)
    assert second_user_again.iss == "invited"
    assert second_user_again.sub == "invited"

    different_idp_iss = invited_user_principal.model_copy(update={"iss": "otheridp"})
    with pytest.raises(HTTPException, match="user not found") as exc:
        require_user_from_token(xngin_session, different_idp_iss)
    assert exc.value.status_code == 401

    different_idp_sub = invited_user_principal.model_copy(update={"sub": "othersub"})
    with pytest.raises(HTTPException, match="user not found") as exc:
        require_user_from_token(xngin_session, different_idp_sub)
    assert exc.value.status_code == 401


def test_user_from_token_when_users_exist(xngin_session: Session):
    unpriv = require_user_from_token(xngin_session, TESTING_TOKENS[UNPRIVILEGED_TOKEN_FOR_TESTING])
    assert not unpriv.is_privileged
    priv = require_user_from_token(xngin_session, TESTING_TOKENS[PRIVILEGED_TOKEN_FOR_TESTING])
    assert priv.is_privileged

    with pytest.raises(HTTPException, match="user not found") as exc:
        require_user_from_token(
            xngin_session,
            Principal(email="usernotfound@example.com", iss="", sub="", hd="", iat=int(time.time())),
        )
    assert exc.value.status_code == 401


def test_user_from_token_initial_setup(xngin_session: Session):
    # emulate first time developer experience by deleting the seeded users
    delete_seeded_users(xngin_session)

    first_user = require_user_from_token(
        xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="", iat=int(time.time()))
    )
    assert first_user.is_privileged
    xngin_session.refresh(first_user, ["organizations"])
    assert len(first_user.organizations) == 1
    assert first_user.organizations[0].name == "firstuser's Organization"

    with pytest.raises(HTTPException, match="user not found") as exc:
        require_user_from_token(
            xngin_session,
            Principal(email="seconduser@example.com", iss="", sub="", hd="", iat=int(time.time())),
        )
    assert exc.value.status_code == 401


def test_user_from_token_expired(xngin_session: Session):
    # emulate first time developer experience by deleting the seeded users
    delete_seeded_users(xngin_session)

    user = require_user_from_token(
        xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="", iat=0)
    )

    now = datetime.now()
    user.last_logout = now + timedelta(days=365)
    xngin_session.commit()

    with pytest.raises(fastapi.HTTPException, match="Expired session") as exc:
        require_user_from_token(
            xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="", iat=int(now.timestamp()))
        )
    assert exc.value.status_code == 401


def test_initial_user_setup_matches_testing_dwh(xngin_session: Session):
    delete_seeded_users(xngin_session)

    first_user = require_user_from_token(
        xngin_session, Principal(email="initial@example.com", iss="", sub="", hd="", iat=int(time.time()))
    )
    xngin_session.commit()

    # Validate directly from the db that our default org was created with datasources.
    xngin_session.refresh(first_user, ["organizations"])
    assert len(first_user.organizations) == 1
    organization = first_user.organizations[0]
    assert organization.name == "initial's Organization"
    assert organization.name == default_organization_name("initial@example.com")
    datasources: list[tables.Datasource] = organization.datasources
    assert len(datasources) == 4, [d.name for d in datasources]

    # Validate that we added the testing dwh datasource.
    ds = find_ds_with_name(datasources, TESTING_DWH_DATASOURCE_NAME)
    ds_config = ds.get_config()
    with DwhSession.open(ds_config.dwh) as dwh:
        sa_table = dwh.inspect_table(TESTING_DWH_TABLE_NAME)
    assert "id" in sa_table.columns

    _ = find_ds_with_name(datasources, ALT_TESTING_DWH_DATASOURCE_NAME)
    _ = find_ds_with_name(datasources, PGSTRESS_DATASOURCE_NAME)

    # Autogenerated NoDwh source should also exist.
    ds = find_ds_with_name(datasources, DEFAULT_NO_DWH_SOURCE_NAME)
    assert isinstance(ds.get_config().dwh, NoDwh)

    experiment_count = xngin_session.scalar(
        select(func.count())
        .select_from(tables.Experiment)
        .where(tables.Experiment.datasource_id.in_([datasource.id for datasource in datasources]))
    )
    assert experiment_count == 8

    # The bandit samples carry draws so the autofail and DWH pull jobs have work on their first run.
    experiments_with_draws = set(
        xngin_session.scalars(
            select(tables.Experiment.name)
            .join(tables.Draw, tables.Draw.experiment_id == tables.Experiment.id)
            .where(tables.Experiment.datasource_id.in_([datasource.id for datasource in datasources]))
            .distinct()
        )
    )
    assert experiments_with_draws == {"MAB", "MAB - wide DWH"}

    snapshot_count_rows = (
        xngin_session.execute(
            select(tables.Snapshot.experiment_id, func.count())
            .group_by(tables.Snapshot.experiment_id)
            .where(
                tables.Snapshot.experiment_id.in_(
                    select(tables.Experiment.id).where(
                        tables.Experiment.datasource_id.in_([datasource.id for datasource in datasources])
                    )
                )
            )
        )
    ).all()
    snapshot_counts: dict[str, int] = {experiment_id: count for experiment_id, count in snapshot_count_rows}
    assert len(snapshot_counts) == 2
    assert set(snapshot_counts.values()) == {10}


def _set_signup_policy(session: Session, mode: tables.SignupMode, domains: list[str] | None = None) -> None:
    """Stores the sign-up mode and domain settings directly, as the admin API would."""
    settings = session.get(tables.ServerSettings, tables.SERVER_SETTINGS_ID)
    if settings is None:
        settings = tables.ServerSettings()
        session.add(settings)
    settings.signup_mode = mode
    session.execute(delete(tables.DomainSettings))
    session.add_all(tables.DomainSettings(domain=domain) for domain in domains or [])
    session.commit()


def _principal(email: str, *, iss: str = "https://idp.example.com", sub: str = "sub-1", iat: int | None = None):
    return Principal(email=email, iss=iss, sub=sub, hd="", iat=int(time.time()) if iat is None else iat)


def _user_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(tables.User)) or 0


def _users_with_email_ci(session: Session, email: str) -> list[tables.User]:
    return list(session.scalars(select(tables.User).where(func.lower(tables.User.email) == email.lower())))


def _sign_up(session: Session, principal: Principal) -> tables.User:
    user = sign_in(session, principal)
    assert user is not None
    return user


def _assert_rejected(session: Session, principal: Principal):
    assert sign_in(session, principal) is None
    with pytest.raises(HTTPException, match="user not found") as exc:
        require_user_from_token(session, principal)
    assert exc.value.status_code == 401


def test_signup_without_settings_row_rejects_unknown_user(xngin_session: Session):
    assert xngin_session.get(tables.ServerSettings, tables.SERVER_SETTINGS_ID) is None
    users_before = _user_count(xngin_session)

    _assert_rejected(xngin_session, _principal("stranger@example.org"))

    assert _user_count(xngin_session) == users_before
    assert xngin_session.scalar(select(func.count()).select_from(tables.Organization)) == 0
    assert xngin_session.get(tables.ServerSettings, tables.SERVER_SETTINGS_ID) is None


@pytest.mark.parametrize("mode", ["existing", "existing_and_domains", "anyone"])
def test_invited_user_signs_in_under_every_signup_mode(xngin_session: Session, mode):
    _set_signup_policy(xngin_session, mode, ["example.org"])
    invited = tables.User(email="invitee@example.com")
    xngin_session.add(invited)
    xngin_session.commit()

    user = require_user_from_token(xngin_session, _principal("invitee@example.com", sub="invitee-sub"))

    assert user.id == invited.id
    assert (user.iss, user.sub) == ("https://idp.example.com", "invitee-sub")


def test_requests_do_not_sign_up_users(xngin_session: Session):
    _set_signup_policy(xngin_session, "anyone")
    users_before = _user_count(xngin_session)

    with pytest.raises(HTTPException, match="user not found"):
        require_user_from_token(xngin_session, _principal("newcomer@example.net"))

    assert _user_count(xngin_session) == users_before


def test_signup_existing_rejects_unknown_user(xngin_session: Session):
    _set_signup_policy(xngin_session, "existing", ["example.org"])
    users_before = _user_count(xngin_session)

    _assert_rejected(xngin_session, _principal("stranger@example.org"))

    assert _user_count(xngin_session) == users_before


@pytest.mark.parametrize("devdwh_dsn", ["", flags.XNGIN_DEVDWH_DSN or "postgresql+psycopg://u:p@localhost/dwh"])
def test_signup_anyone_creates_bare_unprivileged_user(
    xngin_session: Session, monkeypatch: pytest.MonkeyPatch, records, devdwh_dsn
):
    monkeypatch.setattr(flags, "XNGIN_DEVDWH_DSN", devdwh_dsn)
    _set_signup_policy(xngin_session, "anyone")
    users_before = _user_count(xngin_session)

    user = _sign_up(xngin_session, _principal("newcomer@example.net", sub="newcomer-sub"))

    assert user.email == "newcomer@example.net"
    assert user.is_privileged is False
    assert user.iss == "https://idp.example.com"
    assert user.sub == "newcomer-sub"
    assert _user_count(xngin_session) == users_before + 1
    xngin_session.refresh(user, ["organizations"])
    assert user.organizations == []
    assert xngin_session.scalar(select(func.count()).select_from(tables.Organization)) == 0
    assert xngin_session.scalar(select(func.count()).select_from(tables.Datasource)) == 0

    created = [r for r in records if r["message"].startswith("Created user")]
    assert len(created) == 1
    assert user.id in created[0]["message"]
    assert "domain='example.net'" in created[0]["message"]
    assert "signup_mode=anyone" in created[0]["message"]


def test_signup_anyone_returns_same_user_on_later_requests(xngin_session: Session):
    _set_signup_policy(xngin_session, "anyone")
    principal = _principal("newcomer@example.net")

    first = _sign_up(xngin_session, principal)
    second = require_user_from_token(xngin_session, principal)

    assert first.id == second.id
    assert len(_users_with_email_ci(xngin_session, "newcomer@example.net")) == 1


def test_signup_existing_and_domains_match_creates_user(xngin_session: Session):
    _set_signup_policy(xngin_session, "existing_and_domains", ["example.org", "example.net"])

    user = _sign_up(xngin_session, _principal("Member@EXAMPLE.org"))

    assert user.is_privileged is False
    # The stored email is exactly as presented by the identity provider.
    assert user.email == "Member@EXAMPLE.org"


@pytest.mark.parametrize(
    "email",
    [
        "member@sub.example.org",
        "member@example.org.example.net",
        "member@example.org.",
        "member@example.org@example.com",
    ],
)
def test_signup_existing_and_domains_miss_rejects_user(xngin_session: Session, email):
    _set_signup_policy(xngin_session, "existing_and_domains", ["example.org"])
    users_before = _user_count(xngin_session)

    _assert_rejected(xngin_session, _principal(email))

    assert _user_count(xngin_session) == users_before


def test_signup_existing_and_domains_empty_list_rejects_user(xngin_session: Session):
    _set_signup_policy(xngin_session, "existing_and_domains", [])

    _assert_rejected(xngin_session, _principal("member@example.org"))


def test_signup_existing_and_domains_ignores_domains_that_do_not_allow_signup(xngin_session: Session):
    _set_signup_policy(xngin_session, "existing_and_domains", [])
    xngin_session.add(tables.DomainSettings(domain="example.org", allow_signup=False))
    xngin_session.commit()

    _assert_rejected(xngin_session, _principal("member@example.org"))


@pytest.mark.parametrize(
    ("mode", "domains"),
    [
        pytest.param("existing", ["example.org"], id="switch-to-existing"),
        pytest.param("existing_and_domains", ["example.net"], id="domain-removed"),
    ],
)
def test_signed_up_user_survives_policy_changes(xngin_session: Session, mode, domains):
    _set_signup_policy(xngin_session, "existing_and_domains", ["example.org"])
    principal = _principal("member@example.org")
    created = _sign_up(xngin_session, principal)

    _set_signup_policy(xngin_session, mode, domains)

    assert require_user_from_token(xngin_session, principal).id == created.id
    # Others from the removed domain are no longer admitted.
    _assert_rejected(xngin_session, _principal("another@example.org"))


def test_signed_up_user_honours_last_logout(xngin_session: Session):
    _set_signup_policy(xngin_session, "anyone")
    user = _sign_up(xngin_session, _principal("newcomer@example.net", iat=0))

    now = datetime.now()
    user.last_logout = now + timedelta(days=365)
    xngin_session.commit()

    with pytest.raises(HTTPException, match="Expired session") as exc:
        require_user_from_token(xngin_session, _principal("newcomer@example.net", iat=int(now.timestamp())))
    assert exc.value.status_code == 401
    assert len(_users_with_email_ci(xngin_session, "newcomer@example.net")) == 1


def test_invited_user_matches_case_insensitively(xngin_session: Session):
    """An invited user is bound on first sign-in even when the IDP presents their email in a different case."""
    xngin_session.add(tables.User(email="alice@example.com"))
    xngin_session.commit()
    users_before = _user_count(xngin_session)

    user = require_user_from_token(xngin_session, _principal("Alice@example.com", sub="alice-sub"))

    assert user.email == "alice@example.com"
    assert user.iss == "https://idp.example.com"
    assert user.sub == "alice-sub"
    assert _user_count(xngin_session) == users_before
    assert [u.email for u in _users_with_email_ci(xngin_session, "alice@example.com")] == ["alice@example.com"]


def test_emails_are_unique_case_insensitively(xngin_session: Session):
    xngin_session.add(tables.User(email="alice@example.com"))
    xngin_session.commit()
    xngin_session.add(tables.User(email="Alice@example.com"))
    with pytest.raises(IntegrityError, match="ix_users_email_lower"):
        xngin_session.commit()
    xngin_session.rollback()


def test_signup_race_loser_returns_winning_user(xngin_session: Session, monkeypatch: pytest.MonkeyPatch):
    """When a concurrent request creates the user between the lookup and the insert, the existing row is used."""
    _set_signup_policy(xngin_session, "anyone")
    email = "racer@example.net"
    original_find = auth_dependencies.find_user_by_email
    calls = 0

    def find_after_concurrent_insert(session: Session, email_arg: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            # The winning request inserts and commits the user after our lookup found nothing.
            with database.get_session() as other_session:
                # The winner's email differs in case, as it would if the IDP changed how it presents the address.
                other_session.add(tables.User(email=email.upper(), iss="https://idp.example.com", sub="sub-1"))
                other_session.commit()
            return None
        return original_find(session, email_arg)

    monkeypatch.setattr(auth_dependencies, "find_user_by_email", find_after_concurrent_insert)

    user = _sign_up(xngin_session, _principal(email))

    assert calls == 2
    assert user.email == email.upper()
    assert len(_users_with_email_ci(xngin_session, email)) == 1


def test_first_user_bootstrap_ignores_signup_policy(xngin_session: Session):
    delete_seeded_users(xngin_session)
    _set_signup_policy(xngin_session, "existing")

    first_user = require_user_from_token(xngin_session, _principal("founder@example.org"))

    assert first_user.is_privileged
    xngin_session.refresh(first_user, ["organizations"])
    assert [o.name for o in first_user.organizations] == ["founder's Organization"]


def test_airplane_mode_creates_privileged_user_regardless_of_signup_policy(
    xngin_session: Session, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(flags, "AIRPLANE_MODE", True)
    monkeypatch.setattr(flags, "XNGIN_DEVDWH_DSN", "")
    _set_signup_policy(xngin_session, "existing")

    user = require_user_from_token(xngin_session, TESTING_TOKENS[AIRPLANE_TOKEN])

    assert user.is_privileged
    xngin_session.refresh(user, ["organizations"])
    assert [o.name for o in user.organizations] == ["testing's Organization"]
