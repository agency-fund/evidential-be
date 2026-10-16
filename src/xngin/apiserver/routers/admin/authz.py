import sqlalchemy
from sqlalchemy import func, or_, select, true

from xngin.apiserver import flags
from xngin.apiserver.sqla import tables


def can_access_google_sheets(user: tables.User) -> bool:
    """Google Sheets requires both deployment enablement and a privileged user."""
    return flags.GOOGLE_SHEETS_ENABLED and user.is_privileged


def event_is_visible(user: tables.User) -> sqlalchemy.ColumnElement[bool]:
    """Hide events that identify a Sheets datasource or experiment, including webhook payloads."""
    if can_access_google_sheets(user):
        return true()
    sheets_datasources = select(tables.Datasource.id).where(
        tables.Datasource.config["dwh"]["driver"].astext == "google_sheets"
    )
    sheets_experiments = select(tables.Experiment.id).where(
        tables.Experiment.datasource_id.in_(sheets_datasources)
    )
    return ~or_(
        func.coalesce(tables.Event.data["datasource_id"].astext.in_(sheets_datasources), False),
        func.coalesce(tables.Event.data["experiment_id"].astext.in_(sheets_experiments), False),
        func.coalesce(tables.Event.data["request"]["body"]["datasource_id"].astext.in_(sheets_datasources), False),
        func.coalesce(tables.Event.data["request"]["body"]["experiment_id"].astext.in_(sheets_experiments), False),
    )


def is_user_authorized_on_datasource(user: tables.User, datasource_id: str) -> sqlalchemy.Select:
    """Create a query that checks if a user is authorized to manage a datasource."""
    stmt = (
        select(tables.Datasource.id)
        .join(tables.Organization)
        .join(tables.UserOrganization)
        .where(
            tables.UserOrganization.user_id == user.id,
            tables.Datasource.id == datasource_id,
        )
    )
    if not can_access_google_sheets(user):
        stmt = stmt.where(tables.Datasource.config["dwh"]["driver"].astext.is_distinct_from("google_sheets"))
    return stmt


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
