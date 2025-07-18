import sqlalchemy

from xngin.apiserver.models import tables


def is_user_authorized_on_datasource(
    user: tables.User, datasource_id: str
) -> sqlalchemy.Select:
    """Create a query that checks if a user is authorized to manage a datasource."""
    return (
        sqlalchemy.select(tables.Datasource)
        .join(tables.Organization)
        .join(tables.UserOrganization)
        .where(
            tables.UserOrganization.user_id == user.id,
            tables.Datasource.id == datasource_id,
        )
    )
