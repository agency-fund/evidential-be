"""Utilities for boostrapping entities in our app db."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.session import Session

from xngin.apiserver.models import tables
from xngin.apiserver.settings import Dsn, NoDwh, RemoteDatabaseConfig
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF

DEFAULT_ORGANIZATION_NAME = "My Organization"
DEFAULT_DWH_SOURCE_NAME = "Local DWH"
DEFAULT_NO_DWH_SOURCE_NAME = "Default no-DWH Source"


def add_nodwh_datasource_to_org(session, organization):
    """Adds a NoDWH datasource to the given organization."""
    nodwh_config = RemoteDatabaseConfig(participants=[], type="remote", dwh=NoDwh())
    nodwh_datasource = tables.Datasource(
        name=DEFAULT_NO_DWH_SOURCE_NAME, organization=organization
    ).set_config(nodwh_config)
    session.add(nodwh_datasource)


def create_user_and_first_datasource(
    session: Session | AsyncSession, *, email: str, dsn: str | None, privileged: bool
) -> tables.User:
    """Creates a User with an organization, a datasource, and a participant type.

    Assumes dsn refers to a testing_dwh instance.
    """
    user = tables.User(email=email, is_privileged=privileged)
    session.add(user)
    organization = tables.Organization(name=DEFAULT_ORGANIZATION_NAME)
    session.add(organization)
    organization.users.append(user)

    # create a datasource from input
    if dsn:
        config = RemoteDatabaseConfig(
            participants=[TESTING_DWH_PARTICIPANT_DEF],
            type="remote",
            dwh=Dsn.from_url(dsn),
        )
        datasource = tables.Datasource(
            id=tables.datasource_id_factory(),
            name=DEFAULT_DWH_SOURCE_NAME,
            organization=organization,
        ).set_config(config)
        session.add(datasource)

    add_nodwh_datasource_to_org(session, organization)

    return user
