"""Utilities for the management of the local testing database.

This corresponds to the "testing" config specified in xngin.testing.settings.json.
"""

import hashlib
from pathlib import Path

from sqlalchemy.orm import Session

from xngin.apiserver.models import tables
from xngin.apiserver.settings import Dsn, RemoteDatabaseConfig
from xngin.apiserver.testing.testing_dwh_def import TESTING_PARTICIPANT_DEF

TESTING_DWH_RAW_DATA = Path(__file__).parent.parent / "testdata/testing_dwh.csv.zst"


def compact_hash(path: Path):
    """Computes a hash of the input CSV so that we can determine whether to re-create the test warehouse."""
    h = hashlib.blake2b(digest_size=2)
    with open(path, "rb") as source:
        h.update(source.read())
    with open(__file__, "rb") as source:
        h.update(source.read())
    return int.from_bytes(h.digest())


def create_user_and_first_datasource(
    session: Session, *, email: str, dsn: str | None, privileged: bool
):
    """Creates a User with an organization, a datasource, and a participant type.

    Assumes dsn refers to a testing_dwh instance.
    """
    user = tables.User(email=email, is_privileged=privileged)
    session.add(user)
    organization = tables.Organization(name="My Organization")
    session.add(organization)
    organization.users.append(user)
    if dsn:
        config = RemoteDatabaseConfig(
            participants=[TESTING_PARTICIPANT_DEF], type="remote", dwh=Dsn.from_url(dsn)
        )
        datasource = tables.Datasource(
            name="Local DWH", organization=organization
        ).set_config(config)
        session.add(datasource)
    return user
