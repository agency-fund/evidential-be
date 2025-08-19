import hashlib
import secrets
import string

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.sqla import tables

API_KEY_PREFIX = "xat"
HASH_PURPOSE = b"xnginapikey1"


class ApiKeyError(Exception):
    pass


class ApiKeyRequiredError(ApiKeyError):
    pass


class InvalidApiKeyError(ApiKeyError):
    pass


KEY_ALPHABET = [*string.ascii_lowercase, *string.ascii_uppercase, *string.digits]


def hash_key_or_raise(key: str | bytes | None) -> str:
    """Transforms the plain API key to the hashed (stored) value."""
    if not key:
        raise ApiKeyRequiredError()
    if isinstance(key, str):
        if not key.startswith(API_KEY_PREFIX):
            raise InvalidApiKeyError()
        key = key.encode()
    elif isinstance(key, bytes):
        if not key.startswith(API_KEY_PREFIX.encode()):
            raise InvalidApiKeyError()
    else:
        raise InvalidApiKeyError()
    return hashlib.blake2b(key, person=HASH_PURPOSE).hexdigest()


def make_key() -> tuple[str, str]:
    """Generates an API key.

    The general format is ``"xat_" || id || random``.

    The xat_ prefix is to make the value easily recognizable as an xngin API token. The {id} value is used to identify
    this specific key; it is not considered secret, may be presented in UIs, and is useful when we need to identify
    specific API keys without worrying about revealing the secret portion.
    """
    id_ = "".join([secrets.choice(KEY_ALPHABET) for _ in range(6)])
    rnd = "".join([secrets.choice(KEY_ALPHABET) for _ in range(32)])
    key = f"{API_KEY_PREFIX}_{id_}_{rnd}"
    return id_, key


async def require_valid_api_key(session: AsyncSession, api_key: str | None, datasource_id: str):
    """Queries the database for a matching API key with privileges on the config referenced by config_id."""
    key_hash = hash_key_or_raise(api_key)
    stmt = (
        select(tables.ApiKey.id)
        .join(tables.Datasource)
        .where(tables.ApiKey.datasource_id == datasource_id)
        .where(tables.ApiKey.key == key_hash)
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()

    if not row:
        raise InvalidApiKeyError()
