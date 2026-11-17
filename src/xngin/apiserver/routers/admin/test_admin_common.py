import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xngin.apiserver import database
from xngin.apiserver.routers.admin import admin_common
from xngin.apiserver.routers.admin.admin_common import find_or_create_user_by_email
from xngin.apiserver.sqla import tables


def test_find_or_create_user_by_email_returns_user_created_concurrently(
    xngin_session: Session, monkeypatch: pytest.MonkeyPatch
):
    """When a concurrent request creates a case variant between the lookup and the insert, that user is returned."""
    email = "invitee@example.net"

    def find_after_concurrent_insert(_session: Session, _email: str):
        with database.get_session() as other_session:
            other_session.add(tables.User(email=email.upper()))
            other_session.commit()

    monkeypatch.setattr(admin_common, "find_user_by_email", find_after_concurrent_insert)

    user = find_or_create_user_by_email(xngin_session, email)

    assert user.email == email.upper()
    matching = select(func.count()).select_from(tables.User).where(func.lower(tables.User.email) == email)
    assert xngin_session.scalar(matching) == 1
