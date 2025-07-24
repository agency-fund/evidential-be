"""Utilities for the management of the local testing database.

This corresponds to the "testing" config specified in xngin.testing.settings.json.
"""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.session import Session

from xngin.apiserver.models import tables
from xngin.apiserver.settings import Dsn, RemoteDatabaseConfig
from xngin.apiserver.testing.testing_dwh_def import TESTING_PARTICIPANT_DEF

TESTING_DWH_RAW_DATA = Path(__file__).parent.parent / "testdata/testing_dwh.csv.zst"


def create_user_and_first_datasource(
    session: Session | AsyncSession, *, email: str, dsn: str | None, privileged: bool
) -> tables.User:
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
            id=tables.datasource_id_factory(),
            name="Local DWH",
            organization=organization,
        ).set_config(config)
        session.add(datasource)
    return user
