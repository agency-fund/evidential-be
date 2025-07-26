from collections.abc import Awaitable, Callable
from typing import TypeVar

import sqlalchemy
from fastapi import HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)


T = TypeVar("T")


async def handle_delete(
    session: AsyncSession,
    allow_missing: bool,
    is_authorized: sqlalchemy.Select,
    get_resource_or_none: sqlalchemy.Select | Callable[[AsyncSession], Awaitable[T]],
    deleter: Callable[[AsyncSession, T], Awaitable[None]] | None = None,
):
    """Generic delete request handler.

    If the user does not have permission to access the resource, regardless of whether or not it exists, a
    403 will be raised.

    If the user does have proper permission, but the requested resource does not exist, we return a 404
    unless allow_missing is set to true.

    These behaviors are consistent with Google's AIP-135 (https://google.aip.dev/135).

    :param session: SQLAlchemy session
    :param allow_missing: When True, a 204 will be returned even if the item does not exist.
    :param is_authorized: Query that returns one row if the user is authorized to perform the delete, or zero rows if they are not.
    :param get_resource_or_none: Query that returns the SQLAlchemy ORM instance of the resource being deleted.
    :param deleter: If specified, will be invoked instead of a standard database session delete operation. The method
        will be passed the return value of get_resource_or_none.
    :return:
    """
    allowed = (await session.execute(is_authorized)).scalar_one_or_none() is not None
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized for this resource.",
        )
    if isinstance(get_resource_or_none, sqlalchemy.Select):
        resource = (await session.execute(get_resource_or_none)).scalar_one_or_none()
    else:
        resource = await get_resource_or_none(session)
    if resource is None:
        if allow_missing:
            return GENERIC_SUCCESS
        raise HTTPException(404)
    if deleter:
        await deleter(session, resource)
    else:
        await session.delete(resource)
    await session.commit()
    return GENERIC_SUCCESS
