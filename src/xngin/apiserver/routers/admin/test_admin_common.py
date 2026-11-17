import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xngin.apiserver import database
from xngin.apiserver.limits import MAX_LENGTH_OF_NAME_VALUE
from xngin.apiserver.routers.admin import admin_common
from xngin.apiserver.routers.admin.admin_common import default_organization_name, find_or_create_user_by_email
from xngin.apiserver.sqla import tables


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("alice@example.org", "alice's Organization"),
        ("Alice.Smith@example.org", "Alice.Smith's Organization"),
        ("a@b@example.org", "a@b's Organization"),
        ("no-at-sign", "no-at-sign's Organization"),
        ("x" * 85 + "@example.org", "x" * 85 + "'s Organization"),
        ("x" * 200 + "@example.org", "x" * 85 + "'s Organization"),
    ],
)
def test_default_organization_name(email, expected):
    name = default_organization_name(email)
    assert name == expected
    assert len(name) <= MAX_LENGTH_OF_NAME_VALUE


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
