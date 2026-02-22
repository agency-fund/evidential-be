import contextlib
import os
import time
from datetime import datetime, timedelta

import fastapi
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver import flags
from xngin.apiserver.conftest import delete_seeded_users
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.routers.admin.admin_common import DEFAULT_NO_DWH_SOURCE_NAME
from xngin.apiserver.routers.admin.test_admin import find_ds_with_name
from xngin.apiserver.routers.auth.auth_dependencies import (
    PRIVILEGED_TOKEN_FOR_TESTING,
    TESTING_TOKENS,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
    require_user_from_token,
    require_valid_session_token,
    session_token_crypter_dependency,
)
from xngin.apiserver.routers.auth.principal import Principal
from xngin.apiserver.routers.auth.token_crypter import TokenCrypter, TokenCrypterMisconfiguredError
from xngin.apiserver.routers.common_enums import DataType
from xngin.apiserver.settings import NoDwh, ParticipantsDef
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.bootstrap import (
    ALT_TESTING_DWH_DATASOURCE_NAME,
    DEFAULT_ORGANIZATION_NAME,
    TESTING_DWH_DATASOURCE_NAME,
)
from xngin.xsecrets.nacl_provider import NaclProviderKeyset


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


async def test_require_valid_session_token_missing_prefix():
    cryp = session_token_crypter_dependency(ttl=60)
    with pytest.raises(HTTPException, match="token invalid") as exc:
        await require_valid_session_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="abc"), cryp)
    assert exc.value.status_code == 401


async def test_require_valid_session_token_misconfigured():
    cryp = session_token_crypter_dependency(ttl=60)
    with pytest.raises(TokenCrypterMisconfiguredError):
        await require_valid_session_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="xa_abc"), cryp)


async def test_require_valid_session_token():
    with temporary_env_var(flags.ENV_SESSION_TOKEN_KEYSET, NaclProviderKeyset.create().serialize_base64()):
        cryp = session_token_crypter_dependency(ttl=60)
        expected = Principal(email="test@example.com", hd="", iat=0, iss="", sub="")
        valid_token = cryp.encrypt(expected.model_dump_json().encode())
        actual = await require_valid_session_token(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=valid_token,
            ),
            cryp,
        )
        assert actual == expected


@pytest.mark.parametrize("variant", ["x", ".", " ", "==", "  ", "===", ";", "\n"])
async def test_require_valid_session_token_invalid(variant):
    with temporary_env_var(flags.ENV_SESSION_TOKEN_KEYSET, NaclProviderKeyset.create().serialize_base64()):
        cryp = session_token_crypter_dependency(ttl=60)
        expected = Principal(email="test@example.com", hd="", iat=0, iss="", sub="")
        with pytest.raises(HTTPException, match="token invalid") as exc:
            await require_valid_session_token(
                HTTPAuthorizationCredentials(
                    scheme="Bearer",
                    credentials=cryp.encrypt(expected.model_dump_json().encode()) + variant,
                ),
                cryp,
            )
        assert exc.value.status_code == 401
        with pytest.raises(HTTPException, match="token invalid") as exc:
            await require_valid_session_token(
                HTTPAuthorizationCredentials(
                    scheme="Bearer",
                    credentials=variant + cryp.encrypt(expected.model_dump_json().encode()),
                ),
                cryp,
            )
        assert exc.value.status_code == 401


def test_token_crypter_uses_custom_prefix_and_env_var():
    env_var_name = "XNGIN_TEST_TOKEN_KEYSET"
    with temporary_env_var(env_var_name, NaclProviderKeyset.create().serialize_base64()):
        crypter = TokenCrypter(
            ttl=60,
            keyset_env_var=env_var_name,
            local_keyset_filename=".xngin_test_token_keyset",
            prefix="xp_",
        )
        token = crypter.encrypt(b"payload")
        assert token.startswith("xp_")
        assert crypter.decrypt(token) == b"payload"


def test_token_crypter_reads_configured_local_keyset_file(tmp_path):
    keyset_file = tmp_path / "my-local-keyset"
    keyset_file.write_text(NaclProviderKeyset.create().serialize_base64())
    env_var_name = "XNGIN_TEST_TOKEN_KEYSET"
    with temporary_env_var(env_var_name, "local"):
        crypter = TokenCrypter(
            ttl=60,
            keyset_env_var=env_var_name,
            local_keyset_filename=str(keyset_file),
            prefix="xp_",
        )
        token = crypter.encrypt(b"payload")
        assert crypter.decrypt(token) == b"payload"


def test_token_crypter_can_fallback_to_noop_provider_when_keyset_missing():
    env_var_name = "XNGIN_TEST_TOKEN_KEYSET"
    with temporary_unset_env_var(env_var_name):
        crypter = TokenCrypter(
            ttl=60,
            keyset_env_var=env_var_name,
            local_keyset_filename=".xngin_test_token_keyset",
            prefix="xp_",
            allow_noop_fallback=True,
        )
        token = crypter.encrypt("payload")
        assert crypter.decrypt(token) == b"payload"


async def test_user_from_token_invite(xngin_session: AsyncSession, ppost):
    """
    Tests that invited users are updated with the IDP details on their first login, and
    that they are then bound to that IDP afterwards.
    """
    await delete_seeded_users(xngin_session)

    xngin_session.add(tables.User(email="u1@example.com", iss="iss", sub="sub"))
    # Invited users have (iss, sub) = (None, None).
    xngin_session.add(tables.User(email="inv@example.com", iss=None, sub=None))
    await xngin_session.commit()

    invited_user_principal = Principal(
        email="inv@example.com", iss="invited", sub="invited", hd="", iat=int(time.time())
    )

    second_user = await require_user_from_token(xngin_session, invited_user_principal)
    assert second_user.iss == "invited"
    assert second_user.sub == "invited"

    second_user_again = await require_user_from_token(xngin_session, invited_user_principal)
    assert second_user_again.iss == "invited"
    assert second_user_again.sub == "invited"

    different_idp_iss = invited_user_principal.model_copy(update={"iss": "otheridp"})
    with pytest.raises(HTTPException, match="user not found") as exc:
        await require_user_from_token(xngin_session, different_idp_iss)
    assert exc.value.status_code == 401

    different_idp_sub = invited_user_principal.model_copy(update={"sub": "othersub"})
    with pytest.raises(HTTPException, match="user not found") as exc:
        await require_user_from_token(xngin_session, different_idp_sub)
    assert exc.value.status_code == 401


async def test_user_from_token_when_users_exist(xngin_session: AsyncSession):
    unpriv = await require_user_from_token(xngin_session, TESTING_TOKENS[UNPRIVILEGED_TOKEN_FOR_TESTING])
    assert not unpriv.is_privileged
    priv = await require_user_from_token(xngin_session, TESTING_TOKENS[PRIVILEGED_TOKEN_FOR_TESTING])
    assert priv.is_privileged

    with pytest.raises(HTTPException, match="user not found") as exc:
        await require_user_from_token(
            xngin_session,
            Principal(email="usernotfound@example.com", iss="", sub="", hd="", iat=int(time.time())),
        )
    assert exc.value.status_code == 401


async def test_user_from_token_initial_setup(xngin_session: AsyncSession):
    # emulate first time developer experience by deleting the seeded users
    await delete_seeded_users(xngin_session)

    first_user = await require_user_from_token(
        xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="", iat=int(time.time()))
    )
    assert first_user.is_privileged
    await xngin_session.refresh(first_user, ["organizations"])
    assert len(first_user.organizations) == 1

    with pytest.raises(HTTPException, match="user not found") as exc:
        await require_user_from_token(
            xngin_session,
            Principal(email="seconduser@example.com", iss="", sub="", hd="", iat=int(time.time())),
        )
    assert exc.value.status_code == 401


async def test_user_from_token_expired(xngin_session: AsyncSession):
    # emulate first time developer experience by deleting the seeded users
    await delete_seeded_users(xngin_session)

    user = await require_user_from_token(
        xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="", iat=0)
    )

    now = datetime.now()
    user.last_logout = now + timedelta(days=365)
    await xngin_session.commit()

    with pytest.raises(fastapi.HTTPException, match="Expired session") as exc:
        await require_user_from_token(
            xngin_session, Principal(email="firstuser@example.com", iss="", sub="", hd="", iat=int(now.timestamp()))
        )
    assert exc.value.status_code == 401


async def test_initial_user_setup_matches_testing_dwh(xngin_session: AsyncSession):
    await delete_seeded_users(xngin_session)

    first_user = await require_user_from_token(
        xngin_session, Principal(email="initial@example.com", iss="", sub="", hd="", iat=int(time.time()))
    )
    await xngin_session.commit()

    # Validate directly from the db that our default org was created with datasources.
    await xngin_session.refresh(first_user, ["organizations"])
    assert len(first_user.organizations) == 1
    organization = first_user.organizations[0]
    assert organization.name == DEFAULT_ORGANIZATION_NAME
    datasources: list[tables.Datasource] = await organization.awaitable_attrs.datasources
    assert len(datasources) == 3, [d.name for d in datasources]

    # Validate that we added the testing dwh datasource.
    ds = find_ds_with_name(datasources, TESTING_DWH_DATASOURCE_NAME)
    ds_config = ds.get_config()
    pt_def = ds_config.participants[0]
    # Assert it's a "schema" type, not the old "sheets" type.
    assert isinstance(pt_def, ParticipantsDef)
    # Check auto-generated ParticipantsDef is aligned with the test dwh.
    async with DwhSession(ds_config.dwh) as dwh:
        sa_table = await dwh.inspect_table(pt_def.table_name)
    col_names = {c.name for c in sa_table.columns}
    field_names = {f.field_name for f in pt_def.fields}
    assert col_names == field_names
    for field in pt_def.fields:
        col = sa_table.columns[field.field_name]
        assert DataType.match(col.type) == field.data_type

    _ = find_ds_with_name(datasources, ALT_TESTING_DWH_DATASOURCE_NAME)

    # Autogenerated NoDwh source should also exist.
    ds = find_ds_with_name(datasources, DEFAULT_NO_DWH_SOURCE_NAME)
    assert isinstance(ds.get_config().dwh, NoDwh)
