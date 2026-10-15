"""Tests for the Project Wellbeing demo seed.

The full build runs against the dev warehouse's project_wellbeing_dwh table, which `task bootstrap-dwh-database`
loads. Tests that need it are skipped when the table is missing so local runs without it still pass.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.routers.auth.auth_dependencies import PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL
from xngin.apiserver.settings import Dsn
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage import demo_seed

# A well-formed DSN for a warehouse the tests make unreachable by patching DwhSession.open (see _make_unreachable).
UNREACHABLE_DWH_DSN = "postgresql+psycopg://postgres:postgres@127.0.0.1:5433/dwh?sslmode=disable"

EXPECTED_EXPERIMENTS = {
    "Iron Pilot: SMS reminders",
    "Glasses for reading",
    "Iron Support: peer champions",
    "Health Report redesign",
    "Handwashing reminders",
    "Dental kit design",
    "Nutrition tips: who responds to what",
}


@pytest.fixture(name="demo_dwh_dsn")
def fixture_demo_dwh_dsn() -> str:
    """Returns the dev warehouse DSN, skipping the test unless the demo table is loaded there."""
    if not flags.XNGIN_DEVDWH_DSN or not demo_seed._demo_table_loaded(Dsn.from_url(flags.XNGIN_DEVDWH_DSN)):
        pytest.skip(f"{demo_seed.TABLE} is not loaded; run `task bootstrap-dwh-database`.")
    return flags.XNGIN_DEVDWH_DSN


def _make_unreachable(mocker) -> None:
    mocker.patch.object(demo_seed.DwhSession, "open", side_effect=ConnectionError("connection refused"))


def _demo_orgs(session: Session) -> list[tables.Organization]:
    session.expire_all()
    return list(
        session.scalars(select(tables.Organization).where(tables.Organization.name == demo_seed.DEMO_ORGANIZATION_NAME))
    )


def test_seed_for_emails_builds_demo_once(xngin_session: Session, demo_dwh_dsn: str):
    """Builds the whole demo for a listed user, skips them on a re-run, and reports unknown emails."""
    # Mixed case checks that allowlist matching ignores case.
    results = demo_seed.seed_for_emails(xngin_session, demo_dwh_dsn, [PRIVILEGED_EMAIL.upper(), "nobody@example.com"])
    assert results == {PRIVILEGED_EMAIL.upper(): "created", "nobody@example.com": "no such user"}

    (org,) = _demo_orgs(xngin_session)
    assert [u.email for u in org.users] == [PRIVILEGED_EMAIL]
    warehouse = next(ds for ds in org.datasources if ds.name == demo_seed.DEMO_WAREHOUSE_DATASOURCE_NAME)
    assert len(warehouse.api_keys) == 1
    experiments = [e for ds in org.datasources for e in ds.experiments]
    assert {e.name for e in experiments} == EXPECTED_EXPERIMENTS

    # Re-running is a no-op for users who already have the demo.
    assert demo_seed.seed_for_emails(xngin_session, demo_dwh_dsn, [PRIVILEGED_EMAIL]) == {PRIVILEGED_EMAIL: "exists"}
    assert len(_demo_orgs(xngin_session)) == 1


def test_seed_for_emails_rolls_back_a_failed_build(xngin_session: Session, demo_dwh_dsn: str, mocker):
    """A failure partway through leaves no demo behind for that user and does not stop the other users."""
    mocker.patch.object(demo_seed, "_create_glasses", side_effect=RuntimeError("boom"))

    results = demo_seed.seed_for_emails(xngin_session, demo_dwh_dsn, [PRIVILEGED_EMAIL, "nobody@example.com"])

    assert results == {PRIVILEGED_EMAIL: "failed", "nobody@example.com": "no such user"}
    assert _demo_orgs(xngin_session) == []


def test_seed_for_emails_fails_when_table_missing(xngin_session: Session, demo_dwh_dsn: str, mocker):
    mocker.patch.object(demo_seed, "TABLE", "no_such_table")

    results = demo_seed.seed_for_emails(xngin_session, demo_dwh_dsn, [PRIVILEGED_EMAIL])

    assert results == {PRIVILEGED_EMAIL: "failed"}
    assert _demo_orgs(xngin_session) == []


def test_seed_for_emails_fails_when_warehouse_unreachable(xngin_session: Session, mocker):
    _make_unreachable(mocker)
    results = demo_seed.seed_for_emails(xngin_session, UNREACHABLE_DWH_DSN, [PRIVILEGED_EMAIL, UNPRIVILEGED_EMAIL])

    assert results == {PRIVILEGED_EMAIL: "failed", UNPRIVILEGED_EMAIL: "failed"}
    assert _demo_orgs(xngin_session) == []


def test_maybe_create_project_wellbeing_skips_when_disabled(xngin_session: Session, mocker):
    mocker.patch.object(flags, "XNGIN_CREATE_DEMO_ORG", False)
    create = mocker.patch.object(demo_seed, "_create_demo_isolated")
    user = xngin_session.scalars(select(tables.User).where(tables.User.email == PRIVILEGED_EMAIL)).one()

    demo_seed.maybe_create_project_wellbeing(xngin_session, user, UNREACHABLE_DWH_DSN)

    create.assert_not_called()


@pytest.mark.parametrize("dwh_dsn", [None, "", UNREACHABLE_DWH_DSN])
def test_maybe_create_project_wellbeing_skips_without_warehouse(xngin_session: Session, mocker, dwh_dsn):
    """No configured warehouse, or one that cannot be reached, skips the demo without raising."""
    _make_unreachable(mocker)
    mocker.patch.object(flags, "XNGIN_CREATE_DEMO_ORG", True)
    create = mocker.patch.object(demo_seed, "_create_demo_isolated")
    user = xngin_session.scalars(select(tables.User).where(tables.User.email == PRIVILEGED_EMAIL)).one()

    demo_seed.maybe_create_project_wellbeing(xngin_session, user, dwh_dsn)

    create.assert_not_called()


def test_maybe_create_project_wellbeing_creates_when_table_loaded(xngin_session: Session, demo_dwh_dsn: str, mocker):
    """With the flag on and the table loaded, first-time setup hands off to the isolated builder."""
    mocker.patch.object(flags, "XNGIN_CREATE_DEMO_ORG", True)
    create = mocker.patch.object(demo_seed, "_create_demo_isolated", return_value=True)
    user = xngin_session.scalars(select(tables.User).where(tables.User.email == PRIVILEGED_EMAIL)).one()

    demo_seed.maybe_create_project_wellbeing(xngin_session, user, demo_dwh_dsn)

    create.assert_called_once()
    assert create.call_args.args[1] is user
