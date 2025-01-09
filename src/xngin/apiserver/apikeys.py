import hashlib
import secrets
import string

from sqlalchemy import select
from sqlalchemy.orm import Session

from xngin.apiserver.models.tables import (
    ApiKey as ApiKeyDB,
    ApiKeyDatasource as ApiKeyDatasourceDB,
)
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
    if isinstance(key, str):
        key = key.encode()
    return hashlib.blake2b(key, person=HASH_PURPOSE).hexdigest()


def make_key() -> tuple[str, str]:
    id_ = "".join([secrets.choice(KEY_ALPHABET) for _ in range(6)])
    rnd = "".join([secrets.choice(KEY_ALPHABET) for _ in range(32)])
    key = f"{API_KEY_PREFIX}_{id_}_{rnd}"
    return id_, key


def require_valid_api_key(session: Session, api_key: str | None, config_id: str):
    if not api_key:
        raise ApiKeyRequiredError()
    key_hash = hash_key(api_key)
    stmt = (
        select(ApiKeyDB)
        .join(ApiKeyDatasourceDB)
        .where(ApiKeyDB.key == key_hash)
        .where(ApiKeyDatasourceDB.datasource_id == config_id)
    )
    result = session.execute(stmt)
    db_key = result.scalar_one_or_none()

    if not db_key:
        raise InvalidApiKeyError()
