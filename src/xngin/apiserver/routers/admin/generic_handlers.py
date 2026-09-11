from collections.abc import Callable

import sqlalchemy
from fastapi import HTTPException, Response
from sqlalchemy.orm import Session
from starlette import status

GENERIC_SUCCESS = Response(status_code=status.HTTP_204_NO_CONTENT)


def handle_delete[T](
    session: Session,
    allow_missing: bool,
    is_authorized: sqlalchemy.Select,
    get_resource_or_none: sqlalchemy.Select | Callable[[Session], T | None],
    deleter: Callable[[Session, T], None] | None = None,
):
    """Generic delete request handler.

    If the user does not have permission to access the resource, regardless of whether or not it exists, a
    403 will be raised.

    If the user does have proper permission, but the requested resource does not exist, we return a 404
    unless allow_missing is set to true.

    These behaviors are consistent with Google's AIP-135 (https://google.aip.dev/135).

    :param session: SQLAlchemy session
    :param allow_missing: When True, a 204 will be returned even if the item does not exist.
    :param is_authorized: Query that proves the caller may delete the resource at the requested path.
    :param get_resource_or_none: Query that returns the SQLAlchemy ORM instance of the resource being deleted. Do not
                                 fetch by child ID alone; parent IDs from the request must constrain the lookup to
                                 prevent cross-parent deletion.
    :param deleter: If specified, will be invoked instead of a standard database session delete operation. The method
        will be passed the return value of get_resource_or_none.
    :return:
    """
    allowed = session.execute(is_authorized).scalar_one_or_none() is not None
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized for this resource.",
        )
    if isinstance(get_resource_or_none, sqlalchemy.Select):
        resource = session.execute(get_resource_or_none).scalar_one_or_none()
    else:
        resource = get_resource_or_none(session)
    if resource is None:
        if allow_missing:
            return GENERIC_SUCCESS
        raise HTTPException(404)
    if deleter:
        deleter(session, resource)
    else:
        session.delete(resource)
    return GENERIC_SUCCESS
