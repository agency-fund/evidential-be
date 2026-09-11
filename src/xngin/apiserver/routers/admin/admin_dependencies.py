"""FastAPI dependencies that resolve admin API path parameters to the resources they name.

Each dependency loads the resource its route addresses and raises 404 if the caller may not see it (authorization)
or if the resource does not exist.

There are two forms of dependencies: For resources that are always loaded with the same SQLAlchemy preloads, we use
a plain function so that the implementations are concise. For customizable preloads, we use a callable class so that
they can be customized via static construction. A `_with_x` suffix names an added relationship that any route may want;
a `_for_x` suffix names one specific to a handler.

test_handler_routes.py will ensure that the parameters required by all the dependencies of a route match. This avoids
a handler defining a path parameter that is not used by a dependency, and vice versa.

FastAPI dependency injection caches values by the callable's reference. This ensures that two dependencies that depend
on the same dependency will only resolve the shared dependency once. Note that the customized class-based dependencies
may still trigger multiple queries because the dependency instance is different.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import QueryableAttribute, Session, selectinload

from xngin.apiserver.dependencies import xngin_db_session, xngin_sync_db_session
from xngin.apiserver.routers.auth.auth_dependencies import require_user_from_token
from xngin.apiserver.routers.preloads import (
    EXPERIMENT_FIELDS_WITH_FILTERS,
    PreloadChain,
    build_preload_options,
)
from xngin.apiserver.sqla import tables


def raise_unless_privileged(user: tables.User) -> None:
    """Raises 403 if the given user is not privileged."""
    if not user.is_privileged:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only privileged users can perform this action.",
        )


def privileged_caller(
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> tables.User:
    """Dependency: returns the caller's User row, or raises 403 if not privileged."""
    raise_unless_privileged(user)
    return user


async def privileged_target_user(
    user_id: Annotated[str, Path(description="The ID of the user to act on.")],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    _caller: Annotated[tables.User, Depends(privileged_caller)],
) -> tables.User:
    """Resolves the user a route names, for privileged callers only.

    Requires {user_id} in the route path.
    """
    target = await session.get(tables.User, user_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return target


async def load_organization_or_raise(
    session: AsyncSession,
    user: tables.User,
    organization_id: str,
    /,
    *,
    preload: list[QueryableAttribute] | None = None,
) -> tables.Organization:
    """Reads the requested organization from the database. Raises 404 if disallowed or not found.

    Privileged users may access any organization; non-privileged users must be a member.
    """
    stmt = select(tables.Organization).where(tables.Organization.id == organization_id)
    if not user.is_privileged:
        stmt = stmt.join(tables.UserOrganization).where(tables.UserOrganization.user_id == user.id)
    if preload:
        stmt = stmt.options(*[selectinload(f) for f in preload])
    result = await session.execute(stmt)
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return org


async def _load_datasource_or_raise(
    session: AsyncSession,
    user: tables.User,
    datasource_id: str,
    /,
    *,
    organization_id: str | None = None,
    preload: list[QueryableAttribute] | None = None,
) -> tables.Datasource:
    """Reads the requested datasource from the database.

    Pass organization_id to constrain the lookup to a route that also names the organization.

    Raises 404 if disallowed or not found.
    """
    stmt = (
        select(tables.Datasource)
        .join(tables.Organization)
        .join(tables.UserOrganization)
        .where(
            tables.UserOrganization.user_id == user.id,
            tables.Datasource.id == datasource_id,
        )
    )
    if organization_id:
        stmt = stmt.where(tables.Organization.id == organization_id)
    if preload:
        stmt = stmt.options(*[selectinload(f) for f in preload])
    result = await session.execute(stmt)
    ds = result.scalar_one_or_none()
    if ds is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Datasource not found.")
    return ds


async def _load_experiment_or_raise(
    session: AsyncSession,
    ds: tables.Datasource,
    experiment_id: str,
    *,
    preload: list[QueryableAttribute] | None = None,
    nested_preload: list[PreloadChain] | None = None,
) -> tables.Experiment:
    """Reads the requested experiment (related to the given datasource) from the database.

    The .arms attribute will be eagerly loaded due to its frequent use and small size.

    Raises 404 if not found.
    """
    stmt = (
        select(tables.Experiment)
        .options(selectinload(tables.Experiment.arms))
        .where(tables.Experiment.datasource_id == ds.id)
        .where(tables.Experiment.id == experiment_id)
    )

    options = build_preload_options(preload, nested_preload)
    if options:
        stmt = stmt.options(*options)
    result = await session.execute(stmt)
    exp = result.scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")
    return exp


class _Organization:
    """Resolves the organization a route names.

    Requires {organization_id} in the route path.
    """

    def __init__(self, *, preload: list[QueryableAttribute] | None = None) -> None:
        self.preload = preload

    async def __call__(
        self,
        organization_id: Annotated[str, Path()],
        session: Annotated[AsyncSession, Depends(xngin_db_session)],
        user: Annotated[tables.User, Depends(require_user_from_token)],
    ) -> tables.Organization:
        return await load_organization_or_raise(session, user, organization_id, preload=self.preload)


organization = _Organization()
organization_with_members = _Organization(preload=[tables.Organization.users])


class _Datasource:
    """Resolves the datasource a route names.

    Requires {datasource_id} in the route path.
    """

    def __init__(self, *, preload: list[QueryableAttribute] | None = None) -> None:
        self.preload = preload

    async def __call__(
        self,
        datasource_id: Annotated[str, Path()],
        session: Annotated[AsyncSession, Depends(xngin_db_session)],
        user: Annotated[tables.User, Depends(require_user_from_token)],
    ) -> tables.Datasource:
        return await _load_datasource_or_raise(session, user, datasource_id, preload=self.preload)


datasource = _Datasource()
datasource_with_organization = _Datasource(preload=[tables.Datasource.organization])
datasource_with_api_keys = _Datasource(preload=[tables.Datasource.api_keys, tables.Datasource.organization])


class _DatasourceSync:
    """Synchronously resolves the datasource a route names.

    Requires {datasource_id} in the route path.
    """

    def __init__(self, *, preload: list[QueryableAttribute] | None = None) -> None:
        self.preload = preload

    def __call__(
        self,
        datasource_id: Annotated[str, Path()],
        session: Annotated[Session, Depends(xngin_sync_db_session)],
        user: Annotated[tables.User, Depends(require_user_from_token)],
    ) -> tables.Datasource:
        stmt = (
            select(tables.Datasource)
            .join(tables.Organization)
            .join(tables.UserOrganization)
            .where(
                tables.UserOrganization.user_id == user.id,
                tables.Datasource.id == datasource_id,
            )
        )
        if self.preload:
            stmt = stmt.options(*(selectinload(field) for field in self.preload))
        ds = session.execute(stmt).scalar_one_or_none()
        if ds is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Datasource not found.")
        return ds


datasource_sync = _DatasourceSync()
datasource_with_api_keys_sync = _DatasourceSync(preload=[tables.Datasource.api_keys, tables.Datasource.organization])


def experiment_sync(
    experiment_id: Annotated[str, Path()],
    ds: Annotated[tables.Datasource, Depends(datasource_sync)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> tables.Experiment:
    """Synchronously resolves the experiment a route names.

    Requires {datasource_id} and {experiment_id} in the route path.
    """
    stmt = (
        select(tables.Experiment)
        .options(selectinload(tables.Experiment.arms))
        .where(tables.Experiment.datasource_id == ds.id)
        .where(tables.Experiment.id == experiment_id)
    )
    exp = session.execute(stmt).scalar_one_or_none()
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")
    return exp


async def _org_datasource(
    organization_id: Annotated[str, Path()],
    datasource_id: Annotated[str, Path()],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
    user: Annotated[tables.User, Depends(require_user_from_token)],
) -> tables.Datasource:
    """Resolves the datasource a route names, within the organization it names.

    Requires {organization_id} and {datasource_id} in the route path.

    Routes that name the organization in their path must constrain the datasource lookup to it, so that a
    datasource belonging to another of the caller's organizations cannot be reached under the wrong path.
    """
    return await _load_datasource_or_raise(session, user, datasource_id, organization_id=organization_id)


class _Experiment:
    """Resolves the experiment a route names.

    Requires {datasource_id} and {experiment_id} in the route path.

    The owning datasource comes from the `datasource` dependency, so a route that declares both this and
    `datasource` pays for one datasource lookup rather than two.
    """

    def __init__(
        self,
        *,
        preload: list[QueryableAttribute] | None = None,
        nested_preload: list[PreloadChain] | None = None,
    ) -> None:
        self.preload = preload
        self.nested_preload = nested_preload

    async def __call__(
        self,
        experiment_id: Annotated[str, Path()],
        ds: Annotated[tables.Datasource, Depends(datasource)],
        session: Annotated[AsyncSession, Depends(xngin_db_session)],
    ) -> tables.Experiment:
        return await _load_experiment_or_raise(
            session,
            ds,
            experiment_id,
            preload=self.preload,
            nested_preload=self.nested_preload,
        )


experiment = _Experiment()
experiment_with_contexts = _Experiment(preload=[tables.Experiment.contexts])
experiment_for_analysis = _Experiment(
    preload=[tables.Experiment.contexts],
    nested_preload=[EXPERIMENT_FIELDS_WITH_FILTERS],
)
experiment_for_csv_export = _Experiment(nested_preload=[EXPERIMENT_FIELDS_WITH_FILTERS])
experiment_for_ui = _Experiment(
    preload=[tables.Experiment.webhooks, tables.Experiment.contexts],
    nested_preload=[EXPERIMENT_FIELDS_WITH_FILTERS],
)
# Note the contrast with the variants above: this loads the experiment's own filters, not the filters
# hanging off each of its fields.
experiment_for_sample_calls = _Experiment(preload=[tables.Experiment.experiment_filters])


async def org_experiment(
    experiment_id: Annotated[str, Path()],
    ds: Annotated[tables.Datasource, Depends(_org_datasource)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> tables.Experiment:
    """Resolves the experiment a route names, within the organization it names.

    Requires {organization_id}, {datasource_id}, and {experiment_id} in the route path.
    """
    return await _load_experiment_or_raise(session, ds, experiment_id)


async def webhook(
    webhook_id: Annotated[str, Path()],
    org: Annotated[tables.Organization, Depends(organization)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> tables.Webhook:
    """Resolves the webhook a route names.

    Requires {organization_id} and {webhook_id} in the route path.
    """
    wh = await session.scalar(
        select(tables.Webhook).where(
            tables.Webhook.id == webhook_id,
            tables.Webhook.organization_id == org.id,
        )
    )
    if wh is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")
    return wh


async def event(
    event_id: Annotated[str, Path()],
    org: Annotated[tables.Organization, Depends(organization)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> tables.Event:
    """Resolves the event a route names.

    Requires {organization_id} and {event_id} in the route path.
    """
    evt = await session.scalar(
        select(tables.Event).where(
            tables.Event.id == event_id,
            tables.Event.organization_id == org.id,
        )
    )
    if evt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found.")
    return evt


async def snapshot(
    snapshot_id: Annotated[str, Path()],
    exp: Annotated[tables.Experiment, Depends(org_experiment)],
    session: Annotated[AsyncSession, Depends(xngin_db_session)],
) -> tables.Snapshot:
    """Resolves the snapshot a route names.

    Requires {organization_id}, {datasource_id}, {experiment_id}, and {snapshot_id} in the route path.
    """
    snap = await session.scalar(
        select(tables.Snapshot).where(
            tables.Snapshot.experiment_id == exp.id,
            tables.Snapshot.id == snapshot_id,
        )
    )
    if snap is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found")
    return snap


def privileged_target_user_sync(
    user_id: Annotated[str, Path(description="The ID of the user to act on.")],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
    _caller: Annotated[tables.User, Depends(privileged_caller)],
) -> tables.User:
    """Synchronously resolves the target user for a privileged caller."""
    target = session.get(tables.User, user_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return target


class _OrganizationSync:
    def __init__(self, *, preload: list[QueryableAttribute] | None = None) -> None:
        self.preload = preload

    def __call__(
        self,
        organization_id: Annotated[str, Path()],
        session: Annotated[Session, Depends(xngin_sync_db_session)],
        user: Annotated[tables.User, Depends(require_user_from_token)],
    ) -> tables.Organization:
        stmt = select(tables.Organization).where(tables.Organization.id == organization_id)
        if not user.is_privileged:
            stmt = stmt.join(tables.UserOrganization).where(tables.UserOrganization.user_id == user.id)
        if self.preload:
            stmt = stmt.options(*(selectinload(field) for field in self.preload))
        organization = session.execute(stmt).scalar_one_or_none()
        if organization is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
        return organization


organization_sync = _OrganizationSync()
organization_with_members_sync = _OrganizationSync(preload=[tables.Organization.users])


def webhook_sync(
    webhook_id: Annotated[str, Path()],
    organization: Annotated[tables.Organization, Depends(organization_sync)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> tables.Webhook:
    webhook = session.scalar(
        select(tables.Webhook).where(
            tables.Webhook.id == webhook_id,
            tables.Webhook.organization_id == organization.id,
        )
    )
    if webhook is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")
    return webhook


def event_sync(
    event_id: Annotated[str, Path()],
    organization: Annotated[tables.Organization, Depends(organization_sync)],
    session: Annotated[Session, Depends(xngin_sync_db_session)],
) -> tables.Event:
    event = session.scalar(
        select(tables.Event).where(
            tables.Event.id == event_id,
            tables.Event.organization_id == organization.id,
        )
    )
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found.")
    return event
