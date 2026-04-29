import sqlalchemy
from sqlalchemy import select

from xngin.apiserver.sqla import tables


def is_user_authorized_on_datasource(user: tables.User, datasource_id: str) -> sqlalchemy.Select:
    """Create a query that checks if a user is authorized to manage a datasource."""
    return (
        select(tables.Datasource.id)
        .join(tables.Organization)
        .join(tables.UserOrganization)
        .where(
            tables.UserOrganization.user_id == user.id,
            tables.Datasource.id == datasource_id,
        )
    )


def is_user_authorized_on_organization(user: tables.User, organization_id: str) -> sqlalchemy.Select:
    """Create a query that checks if a user is authorized to manage an organization."""
    return (
        select(tables.Organization.id)
        .join(tables.UserOrganization)
        .where(
            tables.UserOrganization.user_id == user.id,
            tables.Organization.id == organization_id,
        )
    )
