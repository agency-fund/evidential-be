from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.models import tables
from xngin.apiserver.routers.admin.admin_api_types import CreateDatasourceResponse
from xngin.apiserver.settings import (
    Dwh,
    RemoteDatabaseConfig,
)


async def create_dummy_datasource(
    session: AsyncSession,
    organization_id: str,
) -> tables.Datasource:
    """Creates a User with an organization and a datasource for testing purposes."""
    dwh = Dwh.validate({
        "driver": "postgresql+psycopg2",
        "host": "dummy",
        "port": 5432,
        "user": "dummy",
        "password": "dummy",
        "dbname": "dummy",
        "sslmode": "disable",
        "search_path": None,
    })
    config = RemoteDatabaseConfig(participants=[], type="remote", dwh=dwh)
    dummy_datasource = tables.Datasource(
        name="Dummy Datasource",
        organization_id=organization_id,
    ).set_config(config)
    session.add(dummy_datasource)
    await session.commit()

    return CreateDatasourceResponse(id=dummy_datasource.id)
