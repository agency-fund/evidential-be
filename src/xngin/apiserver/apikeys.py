import hashlib
import secrets
import string

from sqlalchemy import select
from sqlalchemy.orm import Session

from xngin.apiserver.models.tables import ApiKey, Datasource
import logging

logger = logging.getLogger(__name__)

API_KEY_PREFIX = "xat"
HASH_PURPOSE = b"xnginapikey1"


class ApiKeyError(Exception):
    pass


class ApiKeyRequiredError(ApiKeyError):
    pass


class InvalidApiKeyError(ApiKeyError):
    pass


KEY_ALPHABET = [*string.ascii_lowercase, *string.ascii_uppercase, *string.digits]


def hash_key(key: str | bytes):
    """Transforms the plain API key to the hashed (stored) value."""
    if isinstance(key, str):
        key = key.encode()
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


def require_valid_api_key(session: Session, api_key: str | None, datasource_id: str):
    """Queries the database for a matching API key with privileges on the config referenced by config_id."""
    if not api_key:
        raise ApiKeyRequiredError()
    if not api_key.startswith(API_KEY_PREFIX):
        raise InvalidApiKeyError()
    key_hash = hash_key(api_key)
    stmt = (
        select(ApiKey.id)
        .join(Datasource)
        .where(ApiKey.datasource_id == datasource_id)
        .where(ApiKey.key == key_hash)
    )
    result = session.execute(stmt)
    row = result.scalar_one_or_none()

    if not row:
        raise InvalidApiKeyError()
