import threading
import time

import pytest
import sqlalchemy
from sqlalchemy import text

from xngin.apiserver.dwh.dwh_session import CannotFindTableError, DwhSession
from xngin.apiserver.exceptions_common import DwhTimeoutError
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_TABLE_NAME


@pytest.fixture(name="dwh_config")
def fixture_dwh_config(testing_datasource):
    return testing_datasource.ds.get_config().dwh


def test_run_hands_fn_the_warehouse_session(dwh_config):
    with DwhSession.open(dwh_config) as dwh:
        assert dwh.run(lambda session: session.execute(text("SELECT 1")).scalar_one()) == 1


def test_all_warehouse_calls_run_on_one_helper_thread(dwh_config):
    """The block reuses one warehouse Session, which is only safe if every call shares one thread."""
    with DwhSession.open(dwh_config) as dwh:
        idents = {dwh.run(lambda _session: threading.get_ident()) for _ in range(3)}
    assert len(idents) == 1
    assert idents != {threading.get_ident()}


def test_the_deadline_covers_the_whole_block(dwh_config, mocker):
    release = threading.Event()

    def stall(*_args, **_kwargs):
        release.wait()

    mocker.patch.object(DwhSession, "_inspect_table_blocking", stall)
    try:
        with (
            pytest.raises(DwhTimeoutError, match=r"did not finish inspecting a table within 0\.05s"),
            DwhSession.open(dwh_config, timeout=0.05) as dwh,
        ):
            dwh.inspect_table("dwh")
    finally:
        release.set()


def test_a_timed_out_block_does_not_close_the_session_on_the_callers_thread(dwh_config, mocker):
    """Closing a Session while the abandoned call is still using it is the hazard this class removes."""
    release = threading.Event()
    closed_on = threading.Event()
    closed_by: list[int] = []
    real_exit = DwhSession._close_blocking

    def record_exit(self):
        closed_by.append(threading.get_ident())
        real_exit(self)
        closed_on.set()

    def stall(*_args, **_kwargs):
        release.wait()

    mocker.patch.object(DwhSession, "_inspect_table_blocking", stall)
    mocker.patch.object(DwhSession, "_close_blocking", record_exit)
    try:
        with pytest.raises(DwhTimeoutError), DwhSession.open(dwh_config, timeout=0.05) as dwh:
            dwh.inspect_table("dwh")
        # The block has exited, but the close is queued behind the call we gave up on.
        assert not closed_on.is_set()
    finally:
        release.set()
    assert closed_on.wait(timeout=10)
    assert closed_by != [threading.get_ident()]


def test_a_clean_block_does_not_wait_for_the_warehouse_session_to_close(dwh_config, mocker):
    release = threading.Event()
    close_started = threading.Event()
    closed = threading.Event()
    real_exit = DwhSession._close_blocking

    def record_exit(self):
        close_started.set()
        release.wait()
        real_exit(self)
        closed.set()

    mocker.patch.object(DwhSession, "_close_blocking", record_exit)
    started_at = time.monotonic()
    try:
        with DwhSession.open(dwh_config) as dwh:
            dwh.inspect_table("dwh")
        assert time.monotonic() - started_at < 1.0
        assert close_started.wait(timeout=10)
        assert not closed.is_set()
    finally:
        release.set()
    assert closed.wait(timeout=10)


def test_connecting_is_bounded(dwh_config, mocker):
    """_create_engine resolves DNS synchronously, so entering has to be under the deadline too."""
    release = threading.Event()
    closed = threading.Event()
    real_connect = DwhSession._connect_blocking
    real_close = DwhSession._close_blocking

    def stall(self):
        release.wait()
        real_connect(self)

    def record_close(self):
        real_close(self)
        closed.set()

    mocker.patch.object(DwhSession, "_connect_blocking", stall)
    mocker.patch.object(DwhSession, "_close_blocking", record_close)
    try:
        with (
            pytest.raises(DwhTimeoutError, match=r"did not finish connecting within 0\.05s"),
            DwhSession.open(dwh_config, timeout=0.05),
        ):
            pass
        assert not closed.is_set()
    finally:
        release.set()
    assert closed.wait(timeout=10)


def test_a_later_block_works_after_one_times_out(dwh_config, mocker):
    """A timed-out block must not leave anything behind that poisons the next one."""
    release = threading.Event()

    def stall(*_args, **_kwargs):
        release.wait()

    patched = mocker.patch.object(DwhSession, "_inspect_table_blocking", stall)
    try:
        with pytest.raises(DwhTimeoutError), DwhSession.open(dwh_config, timeout=0.05) as dwh:
            dwh.inspect_table("dwh")
    finally:
        release.set()
    mocker.stop(patched)
    with DwhSession.open(dwh_config) as dwh:
        assert dwh.inspect_table("dwh") is not None


def test_inspecting_a_missing_table_names_only_that_table(dwh_config, mocker):
    """The error must not list the warehouse's other tables.

    Building that list reflects every table's schema, which puts customer
    table names into logs and Sentry titles.
    """
    reflect = mocker.patch.object(sqlalchemy.MetaData, "reflect")
    with DwhSession.open(dwh_config) as dwh, pytest.raises(CannotFindTableError) as excinfo:
        dwh.inspect_table("no_such_table")
    assert (
        str(excinfo.value)
        == "The table 'no_such_table' does not exist. Check the table name, and the tables in the database, "
        "and try again."
    )

    assert TESTING_DWH_TABLE_NAME not in str(excinfo.value)
    reflect.assert_not_called()
