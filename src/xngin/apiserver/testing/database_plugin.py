"""pytest plugin giving each test run and pytest-xdist worker its own Postgres databases; see pg_helpers.

It is registered with -p in pyproject.toml rather than living in a conftest so that it also runs in the pytest-xdist
controller, which reports the databases that workers kept after failures once all tests have finished.
"""

import sys

import psycopg
import pytest

from xngin.apiserver.testing.pg_helpers import finish_test_databases_pg, with_database_suffix

_KEPT_KEY = "kept_test_databases"
_kept = pytest.StashKey[list[str]]()


def pytest_configure(config: pytest.Config):
    """Points the application at a database unique to this test run and worker."""
    # Imported here rather than at module level because flags reads environment variables, such as SENTRY_DSN, as it is
    # imported, and -p plugins are imported before pytest-env sets them.
    from xngin.apiserver import flags  # noqa: PLC0415

    config.stash[_kept] = []
    if flags.DATABASE_URL:
        flags.DATABASE_URL = with_database_suffix(flags.DATABASE_URL)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    """Drops this process's test databases, unless a test failed or the run was cut short."""
    keep = bool(session.testsfailed) or exitstatus not in {pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED}
    try:
        kept = finish_test_databases_pg(keep=keep)
    except psycopg.Error as exc:
        # Don't turn the run into an INTERNALERROR; a later run drops what is left.
        print(f"\nCould not drop test databases: {exc}", file=sys.stderr)
        return
    if keep:
        session.config.stash[_kept].extend(kept)
        workeroutput = getattr(session.config, "workeroutput", None)
        if workeroutput is not None:
            workeroutput[_KEPT_KEY] = kept


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """Collects the databases that a pytest-xdist worker kept."""
    node.config.stash[_kept].extend(getattr(node, "workeroutput", {}).get(_KEPT_KEY, []))


def pytest_terminal_summary(terminalreporter, config: pytest.Config):
    kept: list[str] = config.stash[_kept]
    if kept:
        terminalreporter.section("kept test databases")
        terminalreporter.write_line("Databases used by the failing tests were kept for inspection:")
        for name in sorted(kept):
            terminalreporter.write_line(f"  {name}")
