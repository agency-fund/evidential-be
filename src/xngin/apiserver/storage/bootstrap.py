"""Utilities for boostrapping entities in our app db."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.session import Session

from xngin.apiserver.settings import Dsn, NoDwh, RemoteDatabaseConfig
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF

DEFAULT_ORGANIZATION_NAME = "My Organization"
DEFAULT_DWH_SOURCE_NAME = "Local DWH"
DEFAULT_NO_DWH_SOURCE_NAME = "API Only"


def add_nodwh_datasource_to_org(session, organization):
    """Adds a NoDWH datasource to the given organization."""
    nodwh_config = RemoteDatabaseConfig(participants=[], type="remote", dwh=NoDwh())
    nodwh_datasource = tables.Datasource(name=DEFAULT_NO_DWH_SOURCE_NAME, organization=organization).set_config(
        nodwh_config
    )
    session.add(nodwh_datasource)


def _create_first_datasource(session: Session | AsyncSession, organization: tables.Organization, dsn: str | None):
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


def setup_user_and_first_datasource(session: Session | AsyncSession, user: tables.User, dsn: str | None) -> tables.User:
    """Adds models to User such that they can have a good first time experience with the application.

    Users will have an organization and a NoDWH datasource created for them. If dsn is provided, we create that
    datasource and a participant type for them (assuming DSN refers to a testing_dwh instance).
    """
    session.add(user)
    organization = tables.Organization(name=DEFAULT_ORGANIZATION_NAME)
    session.add(organization)
    organization.users.append(user)
    _create_first_datasource(session, organization, dsn)
    return user
