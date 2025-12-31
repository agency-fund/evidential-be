from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.settings import NoDwh, RemoteDatabaseConfig
from xngin.apiserver.sqla import tables

DEFAULT_NO_DWH_SOURCE_NAME = "API Only"


async def create_datasource_impl(
    session: AsyncSession, org: tables.Organization, name: str, config: RemoteDatabaseConfig
) -> tables.Datasource:
    datasource = tables.Datasource(id=tables.datasource_id_factory(), name=name, organization=org).set_config(config)
    session.add(datasource)
    return datasource


async def create_organization_impl(session: AsyncSession, user: tables.User, name: str) -> tables.Organization:
    organization = tables.Organization(name=name)
    session.add(organization)
    organization.users.append(user)  # Add the creating user to the organization

    nodwh_config = RemoteDatabaseConfig(participants=[], type="remote", dwh=NoDwh())
    nodwh_datasource = tables.Datasource(name=DEFAULT_NO_DWH_SOURCE_NAME, organization=organization).set_config(
        nodwh_config
    )
    session.add(nodwh_datasource)
    return organization
