"""Tests for the user-management endpoints in admin_api.py.

All tests interact with the system exclusively through the AdminAPIClient. No direct database access.
"""

from collections.abc import Iterable
from typing import Protocol

from xngin.apiserver.conftest import expect_status_code
from xngin.apiserver.routers.admin.admin_api_types import (
    AddMemberToOrganizationRequest,
    CreateOrganizationRequest,
    CreateUserRequest,
    PatchUserRequest,
)
from xngin.apiserver.routers.admin.admin_common import DEFAULT_NO_DWH_SOURCE_NAME
from xngin.apiserver.routers.auth.auth_dependencies import (
    PRIVILEGED_EMAIL,
    UNPRIVILEGED_EMAIL,
)
from xngin.apiserver.testing.admin_api_client import AdminAPIClient


class _HasIdAndEmail(Protocol):
    id: str
    email: str


def _user_id_for(items: Iterable[_HasIdAndEmail], email: str) -> str:
    for item in items:
        if item.email == email:
            return item.id
    raise AssertionError(f"User with email {email} not found in {[i.email for i in items]}")


def test_list_users_requires_privileged(aclient_unpriv: AdminAPIClient):
    with expect_status_code(403):
        aclient_unpriv.list_users()
    # require_privileged runs unconditionally regardless of scope; pin the scope=mine variant too so
    # nobody can "relax" only the mine path without realizing both are gated.
    with expect_status_code(403):
        aclient_unpriv.list_users(scope="mine")


def test_list_users_returns_seeded_users(aclient: AdminAPIClient):
    response = aclient.list_users().data
    emails = {u.email for u in response.items}
    assert {PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL}.issubset(emails)
    by_email = {u.email: u for u in response.items}
    assert by_email[PRIVILEGED_EMAIL].is_privileged is True
    assert by_email[UNPRIVILEGED_EMAIL].is_privileged is False
    # The PRIVILEGED user has just authenticated to make this very request, which sets iss/sub via
    # the require_user_from_token dependency. The UNPRIVILEGED user has not authenticated yet.
    assert by_email[PRIVILEGED_EMAIL].has_logged_in is True
    assert by_email[UNPRIVILEGED_EMAIL].has_logged_in is False
    # Seeded users have no organizations.
    assert by_email[PRIVILEGED_EMAIL].organizations == []


def test_list_users_returns_organizations(aclient: AdminAPIClient):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="users-test-org")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    response = aclient.list_users().data
    by_email = {u.email: u for u in response.items}
    unpriv_orgs = {o.id for o in by_email[UNPRIVILEGED_EMAIL].organizations}
    assert org_id in unpriv_orgs
    priv_orgs = {o.id for o in by_email[PRIVILEGED_EMAIL].organizations}
    assert org_id in priv_orgs


def test_list_users_email_contains(aclient: AdminAPIClient):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="filter-test")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="alice@filter-test.example")
    )
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="bob@filter-test.example")
    )

    response = aclient.list_users(email_contains="ALICE").data  # case-insensitive
    emails = {u.email for u in response.items}
    assert emails == {"alice@filter-test.example"}


def test_list_users_scope_mine_filters_to_caller_orgs(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    """scope=mine returns only users that share at least one organization with the caller."""
    foreign_org = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="foreign-org")).data.id
    aclient_unpriv.add_member_to_organization(
        organization_id=foreign_org, body=AddMemberToOrganizationRequest(email="outsider@example.com")
    )

    shared_org = aclient.create_organizations(body=CreateOrganizationRequest(name="shared-org")).data.id
    aclient.add_member_to_organization(
        organization_id=shared_org, body=AddMemberToOrganizationRequest(email="insider@example.com")
    )

    # scope=all sees both outsider and insider; scope=mine sees only insider (and the caller themselves).
    all_emails = {u.email for u in aclient.list_users(scope="all").data.items}
    assert {"outsider@example.com", "insider@example.com"}.issubset(all_emails)

    mine_emails = {u.email for u in aclient.list_users(scope="mine").data.items}
    assert "insider@example.com" in mine_emails
    assert "outsider@example.com" not in mine_emails
    assert PRIVILEGED_EMAIL in mine_emails


def test_list_users_includes_created_at(aclient: AdminAPIClient):
    items = aclient.list_users().data.items
    assert all(u.created_at is not None for u in items)


def test_list_users_email_contains_escapes_like_wildcards(aclient: AdminAPIClient):
    """Underscore and percent characters must be matched literally, not as LIKE wildcards."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="escape-test")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="under_score@example.com")
    )
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="dotted.match@example.com")
    )

    # Without escaping, `_` would be a single-character wildcard and would match dotted.match too.
    response = aclient.list_users(email_contains="under_score").data
    emails = {u.email for u in response.items}
    assert emails == {"under_score@example.com"}


def test_list_users_pagination(aclient: AdminAPIClient):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="page-test")).data.id
    for i in range(3):
        aclient.add_member_to_organization(
            organization_id=org_id, body=AddMemberToOrganizationRequest(email=f"page{i}@example.com")
        )

    seen: set[str] = set()
    page_token: str | None = None
    pages_fetched = 0
    while True:
        page = aclient.list_users(page_size=2, page_token=page_token).data
        assert len(page.items) <= 2
        seen.update(u.email for u in page.items)
        pages_fetched += 1
        if not page.next_page_token:
            break
        page_token = page.next_page_token

    # Seeded users (privileged + unprivileged) plus the three added above.
    assert {f"page{i}@example.com" for i in range(3)}.issubset(seen)
    assert {PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL}.issubset(seen)
    assert pages_fetched >= 3


def test_list_users_sorted_by_email_asc(aclient: AdminAPIClient):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="sort-test")).data.id
    for email in ("zeta@example.com", "alpha@example.com", "mu@example.com"):
        aclient.add_member_to_organization(organization_id=org_id, body=AddMemberToOrganizationRequest(email=email))

    response = aclient.list_users().data
    emails = [u.email for u in response.items]
    assert emails == sorted(emails)


def test_create_user_succeeds(aclient: AdminAPIClient):
    """A privileged user can create a new User by email; the new user has sensible defaults."""
    response = aclient.create_user(body=CreateUserRequest(email="invitee@example.com")).data
    assert response.id

    after = {u.email: u for u in aclient.list_users().data.items}
    assert "invitee@example.com" in after
    invitee = after["invitee@example.com"]
    assert invitee.id == response.id
    assert invitee.is_privileged is False
    assert invitee.has_logged_in is False
    assert invitee.organizations == []


def test_create_user_idempotent_on_duplicate_email(aclient: AdminAPIClient):
    """Calling create_user twice with the same email returns the same id and creates no extra row."""
    first = aclient.create_user(body=CreateUserRequest(email="invitee@example.com")).data
    second = aclient.create_user(body=CreateUserRequest(email="invitee@example.com")).data
    assert first.id == second.id

    # Exactly one row with that email exists.
    matching = [u for u in aclient.list_users().data.items if u.email == "invitee@example.com"]
    assert len(matching) == 1


def test_create_user_does_not_affect_existing_user(aclient: AdminAPIClient):
    """If the email already belongs to an existing user (even with orgs), create_user is a no-op."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="prior-org")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    before = next(u for u in aclient.list_users().data.items if u.email == UNPRIVILEGED_EMAIL)
    response = aclient.create_user(body=CreateUserRequest(email=UNPRIVILEGED_EMAIL)).data
    assert response.id == before.id

    after = next(u for u in aclient.list_users().data.items if u.email == UNPRIVILEGED_EMAIL)
    assert after.is_privileged == before.is_privileged
    assert {o.id for o in after.organizations} == {o.id for o in before.organizations}


def test_create_user_requires_privileged(aclient_unpriv: AdminAPIClient):
    with expect_status_code(403):
        aclient_unpriv.create_user(body=CreateUserRequest(email="rejected@example.com"))


def test_get_user_requires_privileged(aclient_unpriv: AdminAPIClient):
    with expect_status_code(403):
        aclient_unpriv.get_user(user_id="u_anything")


def test_get_user_not_found(aclient: AdminAPIClient):
    with expect_status_code(404):
        aclient.get_user(user_id="u_doesnotexist")


def test_get_user_returns_details(aclient: AdminAPIClient):
    target_id = _user_id_for(aclient.list_users().data.items, UNPRIVILEGED_EMAIL)
    response = aclient.get_user(user_id=target_id).data
    assert response.id == target_id
    assert response.email == UNPRIVILEGED_EMAIL
    assert response.is_privileged is False
    assert response.has_logged_in is False
    assert response.organizations == []


def test_get_user_returns_organizations_with_counts(aclient: AdminAPIClient):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="get-user-org")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    target_id = _user_id_for(aclient.list_users().data.items, UNPRIVILEGED_EMAIL)
    response = aclient.get_user(user_id=target_id).data
    org_ids = {o.id for o in response.organizations}
    assert org_id in org_ids
    matching = next(o for o in response.organizations if o.id == org_id)
    # The organization has at least one user (the unprivileged user we just added). The privileged
    # user was implicitly added when they created the org.
    assert matching.user_count is not None and matching.user_count >= 1
    assert matching.experiment_count == 0
    assert matching.created_at is not None
    # joined_at is populated in the user-scoped context (and is >= the org's created_at since the
    # user was added immediately after the org was created).
    assert matching.joined_at is not None
    assert matching.joined_at >= matching.created_at


def test_list_organizations_does_not_populate_joined_at(aclient: AdminAPIClient):
    """joined_at is only meaningful in the user-scoped GET endpoint; list_organizations returns null."""
    aclient.create_organizations(body=CreateOrganizationRequest(name="joined-at-unset"))
    items = aclient.list_organizations().data.items
    assert items
    assert all(o.joined_at is None for o in items)


def test_patch_user_grant_privilege(aclient: AdminAPIClient):
    target_id = _user_id_for(aclient.list_users().data.items, UNPRIVILEGED_EMAIL)
    aclient.patch_user(user_id=target_id, body=PatchUserRequest(is_privileged=True))

    after = {u.email: u for u in aclient.list_users().data.items}
    assert after[UNPRIVILEGED_EMAIL].is_privileged is True

    aclient.patch_user(user_id=target_id, body=PatchUserRequest(is_privileged=False))


def test_patch_user_revoke_privilege_when_others_exist(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    users = aclient.list_users().data.items
    other_id = _user_id_for(users, UNPRIVILEGED_EMAIL)
    self_id = _user_id_for(users, PRIVILEGED_EMAIL)

    # Grant privilege to the other user so we can demote ourselves.
    aclient.patch_user(user_id=other_id, body=PatchUserRequest(is_privileged=True))
    aclient.patch_user(user_id=self_id, body=PatchUserRequest(is_privileged=False))

    # PRIVILEGED_EMAIL is now unprivileged; verify via the now-privileged aclient_unpriv.
    after = {u.email: u for u in aclient_unpriv.list_users().data.items}
    assert after[PRIVILEGED_EMAIL].is_privileged is False
    assert after[UNPRIVILEGED_EMAIL].is_privileged is True

    # Restore the seeded state. Only UNPRIVILEGED_EMAIL is privileged now, so cleanup must run via
    # aclient_unpriv (which is now privileged) to re-promote PRIVILEGED_EMAIL first.
    aclient_unpriv.patch_user(user_id=self_id, body=PatchUserRequest(is_privileged=True))
    aclient.patch_user(user_id=other_id, body=PatchUserRequest(is_privileged=False))


def test_patch_user_revoke_privilege_last_privileged_rejected(aclient: AdminAPIClient):
    self_id = _user_id_for(aclient.list_users().data.items, PRIVILEGED_EMAIL)
    with expect_status_code(403, text="privileged"):
        aclient.patch_user(user_id=self_id, body=PatchUserRequest(is_privileged=False))


def test_patch_user_not_found(aclient: AdminAPIClient):
    with expect_status_code(404):
        aclient.patch_user(user_id="u_doesnotexist", body=PatchUserRequest(is_privileged=True))


def test_patch_user_no_change_when_payload_empty(aclient: AdminAPIClient):
    target_id = _user_id_for(aclient.list_users().data.items, UNPRIVILEGED_EMAIL)
    aclient.patch_user(user_id=target_id, body=PatchUserRequest())
    after = {u.email: u for u in aclient.list_users().data.items}
    assert after[UNPRIVILEGED_EMAIL].is_privileged is False


def test_patch_user_requires_privileged(aclient_unpriv: AdminAPIClient):
    with expect_status_code(403):
        aclient_unpriv.patch_user(user_id="u_anything", body=PatchUserRequest(is_privileged=True))


def test_delete_user_self_rejected(aclient: AdminAPIClient):
    self_id = _user_id_for(aclient.list_users().data.items, PRIVILEGED_EMAIL)
    with expect_status_code(403, text="yourself"):
        aclient.delete_user(user_id=self_id)


def test_delete_user_ok(aclient: AdminAPIClient):
    target_id = _user_id_for(aclient.list_users().data.items, UNPRIVILEGED_EMAIL)
    aclient.delete_user(user_id=target_id)

    emails = {u.email for u in aclient.list_users().data.items}
    assert UNPRIVILEGED_EMAIL not in emails


def test_delete_user_cascades_org_membership(aclient: AdminAPIClient):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="cascade-test")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    target_id = _user_id_for(aclient.list_users().data.items, UNPRIVILEGED_EMAIL)
    aclient.delete_user(user_id=target_id)

    org = aclient.get_organization(organization_id=org_id).data
    assert UNPRIVILEGED_EMAIL not in {u.email for u in org.users}


def test_delete_user_not_found(aclient: AdminAPIClient):
    with expect_status_code(404):
        aclient.delete_user(user_id="u_doesnotexist")


def test_delete_user_requires_privileged(aclient_unpriv: AdminAPIClient):
    with expect_status_code(403):
        aclient_unpriv.delete_user(user_id="u_anything")


def test_list_organizations_privileged_with_scope_all_sees_orgs_they_are_not_a_member_of(
    aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient
):
    # The unprivileged user creates an org of their own. The original privileged user is not a
    # member of that org. Default scope (`mine`) hides it; `scope=all` shows it.
    new_org_id = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="other-users-org")).data.id

    # Default scope is `mine` — the original priv user is not a member, so org is absent.
    default_org_ids = {o.id for o in aclient.list_organizations().data.items}
    assert new_org_id not in default_org_ids

    # Explicit scope=all reveals it.
    all_org_ids = {o.id for o in aclient.list_organizations(scope="all").data.items}
    assert new_org_id in all_org_ids

    # And the privileged caller can now GET that org's details too (priv-bypass on get_organization_or_raise).
    org_detail = aclient.get_organization(organization_id=new_org_id).data
    assert org_detail.name == "other-users-org"
    assert PRIVILEGED_EMAIL not in {u.email for u in org_detail.users}


def test_list_organizations_scope_all_requires_privilege(aclient_unpriv: AdminAPIClient):
    with expect_status_code(403):
        aclient_unpriv.list_organizations(scope="all")


def test_get_organization_unprivileged_404s_on_non_member_org(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    """Non-privileged callers still get 404 on orgs they are not members of."""
    foreign_org = aclient.create_organizations(body=CreateOrganizationRequest(name="foreign-non-priv-test")).data.id
    with expect_status_code(404):
        aclient_unpriv.get_organization(organization_id=foreign_org)


def test_list_organizations_unprivileged_sees_only_own(aclient_unpriv: AdminAPIClient, aclient: AdminAPIClient):
    aclient.create_organizations(body=CreateOrganizationRequest(name="not-shared"))

    items = aclient_unpriv.list_organizations().data.items
    names = {o.name for o in items}
    assert "not-shared" not in names


def test_list_organizations_name_contains(aclient: AdminAPIClient):
    aclient.create_organizations(body=CreateOrganizationRequest(name="alpha team"))
    aclient.create_organizations(body=CreateOrganizationRequest(name="bravo team"))
    aclient.create_organizations(body=CreateOrganizationRequest(name="alphabet"))

    response = aclient.list_organizations(name_contains="ALPHA").data
    names = {o.name for o in response.items}
    assert names == {"alpha team", "alphabet"}


def test_list_organizations_name_contains_escapes_like_wildcards(aclient: AdminAPIClient):
    aclient.create_organizations(body=CreateOrganizationRequest(name="under_score"))
    aclient.create_organizations(body=CreateOrganizationRequest(name="dotted.match"))

    # `_` is a LIKE single-character wildcard. Escape must make it match literally.
    response = aclient.list_organizations(name_contains="under_score").data
    names = {o.name for o in response.items}
    assert names == {"under_score"}


def test_list_organizations_pagination(aclient: AdminAPIClient):
    for n in ("aaa", "bbb", "ccc"):
        aclient.create_organizations(body=CreateOrganizationRequest(name=n))

    seen: set[str] = set()
    page_token: str | None = None
    pages_fetched = 0
    while True:
        page = aclient.list_organizations(page_size=2, page_token=page_token).data
        assert len(page.items) <= 2
        seen.update(o.name for o in page.items)
        pages_fetched += 1
        if not page.next_page_token:
            break
        page_token = page.next_page_token

    assert {"aaa", "bbb", "ccc"}.issubset(seen)
    assert pages_fetched >= 2


def test_list_organizations_includes_created_at_and_counts(aclient: AdminAPIClient):
    """list_organizations with include_stats returns aggregated user_count and experiment_count."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="counts-test")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="extra1@example.com")
    )
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="extra2@example.com")
    )

    found = next(o for o in aclient.list_organizations(include_stats=True).data.items if o.id == org_id)
    assert found.created_at is not None
    # Creator (PRIVILEGED_EMAIL) is auto-added; plus the two extras = 3.
    assert found.user_count == 3
    assert found.experiment_count == 0


def test_list_organizations_omits_counts_by_default(aclient: AdminAPIClient):
    """Without include_stats, user_count and experiment_count are null."""
    aclient.create_organizations(body=CreateOrganizationRequest(name="no-stats-test"))

    items = aclient.list_organizations().data.items
    assert items
    assert all(o.user_count is None for o in items)
    assert all(o.experiment_count is None for o in items)


def test_list_organizations_sorted_by_name_asc(aclient: AdminAPIClient):
    for n in ("zeta", "alpha", "mu"):
        aclient.create_organizations(body=CreateOrganizationRequest(name=n))

    response = aclient.list_organizations().data
    names = [o.name for o in response.items]
    assert names == sorted(names)


def test_remove_member_self_returns_403(aclient: AdminAPIClient):
    """Removing yourself from an organization returns 403 with an explanatory message."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="self-remove-test")).data.id
    self_id = _user_id_for(aclient.list_users().data.items, PRIVILEGED_EMAIL)
    with expect_status_code(403, text="cannot remove yourself"):
        aclient.remove_member_from_organization(organization_id=org_id, user_id=self_id)


def test_remove_member_privileged_bypasses_membership(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    # The unprivileged user creates an org of their own that the original privileged user is not a
    # member of.
    org_id = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="bypass-test")).data.id
    aclient_unpriv.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="invitee@example.com")
    )

    # Verify the privileged user is NOT a member of bypass-test, then remove invitee anyway.
    by_email = {u.email: u for u in aclient.list_users().data.items}
    priv_org_ids = {o.id for o in by_email[PRIVILEGED_EMAIL].organizations}
    assert org_id not in priv_org_ids, "privileged user must not be a member of bypass-test"
    invitee_id = by_email["invitee@example.com"].id

    aclient.remove_member_from_organization(organization_id=org_id, user_id=invitee_id)

    # Verify membership-only semantics: the invitee's User record must still exist; only the org
    # membership row should be gone.
    after = {u.email: u for u in aclient.list_users().data.items}
    assert "invitee@example.com" in after, "removing membership must not delete the user record"
    assert org_id not in {o.id for o in after["invitee@example.com"].organizations}


def test_get_organization_users_include_is_privileged(aclient: AdminAPIClient):
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="priv-flag-test")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    org = aclient.get_organization(organization_id=org_id).data
    by_email = {u.email: u for u in org.users}
    assert by_email[PRIVILEGED_EMAIL].is_privileged is True
    assert by_email[UNPRIVILEGED_EMAIL].is_privileged is False


def test_create_organization_unprivileged_succeeds(aclient_unpriv: AdminAPIClient):
    """An unprivileged user can create an organization and read it back as the sole member."""
    org_name = "unpriv-created"
    create_response = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name=org_name)).data
    assert create_response.id

    org = aclient_unpriv.get_organization(organization_id=create_response.id).data
    assert org.id == create_response.id
    assert org.name == org_name
    assert [u.email for u in org.users] == [UNPRIVILEGED_EMAIL]

    assert len(org.datasources) == 1
    nodwh = org.datasources[0]
    assert nodwh.type == "remote"
    assert nodwh.driver == "none"
    assert nodwh.name == DEFAULT_NO_DWH_SOURCE_NAME
    assert nodwh.organization_id == create_response.id
    assert nodwh.organization_name == org_name


def test_unprivileged_user_can_create_multiple_organizations(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    """An unprivileged user can create multiple orgs; the privileged user doesn't see them via scope=mine."""
    first_id = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="unpriv-first")).data.id
    second_id = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="unpriv-second")).data.id

    visible_ids = {o.id for o in aclient_unpriv.list_organizations().data.items}
    assert {first_id, second_id}.issubset(visible_ids)

    priv_visible_ids = {o.id for o in aclient.list_organizations().data.items}
    assert first_id not in priv_visible_ids
    assert second_id not in priv_visible_ids


def test_create_organization_unprivileged_does_not_grant_privileges_elsewhere(aclient_unpriv: AdminAPIClient):
    """Creating an org must not grant access to other privileged-only endpoints."""
    aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="still-not-an-admin"))

    with expect_status_code(403):
        aclient_unpriv.list_users()
    with expect_status_code(403):
        aclient_unpriv.patch_user(user_id="u_anything", body=PatchUserRequest(is_privileged=True))
    with expect_status_code(403):
        aclient_unpriv.delete_user(user_id="u_anything")
    with expect_status_code(403):
        aclient_unpriv.list_organizations(scope="all")


def test_unprivileged_users_orgs_isolated_from_each_other(
    aclient_unpriv: AdminAPIClient, aclient_unpriv_2: AdminAPIClient
):
    """Orgs created by one unprivileged user are not visible to another unprivileged user."""
    a_org_id = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="a-only")).data.id
    b_org_id = aclient_unpriv_2.create_organizations(body=CreateOrganizationRequest(name="b-only")).data.id

    a_visible_ids = {o.id for o in aclient_unpriv.list_organizations().data.items}
    assert a_org_id in a_visible_ids
    assert b_org_id not in a_visible_ids

    b_visible_ids = {o.id for o in aclient_unpriv_2.list_organizations().data.items}
    assert b_org_id in b_visible_ids
    assert a_org_id not in b_visible_ids

    with expect_status_code(404):
        aclient_unpriv.get_organization(organization_id=b_org_id)
    with expect_status_code(404):
        aclient_unpriv_2.get_organization(organization_id=a_org_id)


def test_creator_removes_member_user_record_persists_and_booted_user_sees_no_orgs(
    aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient
):
    """Creator removes a member; the booted user's account remains usable and they see zero orgs."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="boot-test")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    unpriv_id = _user_id_for(aclient.get_organization(organization_id=org_id).data.users, UNPRIVILEGED_EMAIL)
    aclient.remove_member_from_organization(organization_id=org_id, user_id=unpriv_id)

    identity = aclient_unpriv.caller_identity().data
    assert identity.email == UNPRIVILEGED_EMAIL

    booted_visible_ids = {o.id for o in aclient_unpriv.list_organizations().data.items}
    assert org_id not in booted_visible_ids


def test_member_can_remove_another_member(aclient: AdminAPIClient, aclient_unpriv: AdminAPIClient):
    """All org members have equal rights: a non-creator member can remove other members (including the creator)."""
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="equal-rights")).data.id
    aclient.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email=UNPRIVILEGED_EMAIL)
    )

    priv_user_id = _user_id_for(aclient_unpriv.get_organization(organization_id=org_id).data.users, PRIVILEGED_EMAIL)
    aclient_unpriv.remove_member_from_organization(organization_id=org_id, user_id=priv_user_id)

    priv_visible_ids = {o.id for o in aclient.list_organizations().data.items}
    assert org_id not in priv_visible_ids

    identity = aclient.caller_identity().data
    assert identity.email == PRIVILEGED_EMAIL


def test_unprivileged_non_member_cannot_remove_members(
    aclient_unpriv: AdminAPIClient, aclient_unpriv_2: AdminAPIClient
):
    """A user who is not a member of an org cannot remove anyone from it."""
    org_id = aclient_unpriv.create_organizations(body=CreateOrganizationRequest(name="closed-club")).data.id
    aclient_unpriv.add_member_to_organization(
        organization_id=org_id, body=AddMemberToOrganizationRequest(email="victim@example.com")
    )

    victim_id = _user_id_for(aclient_unpriv.get_organization(organization_id=org_id).data.users, "victim@example.com")

    # The generic delete handler returns 403 when the caller is not authorized for the resource.
    with expect_status_code(403):
        aclient_unpriv_2.remove_member_from_organization(organization_id=org_id, user_id=victim_id)

    # Confirm the victim is still a member after the failed removal attempt.
    members_after = {u.email for u in aclient_unpriv.get_organization(organization_id=org_id).data.users}
    assert "victim@example.com" in members_after
