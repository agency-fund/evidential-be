import os
import re
from string.templatelib import Template
from types import SimpleNamespace

import pytest
import sqlalchemy
from sqlalchemy import Column, Index, Integer, MetaData, String, Table

from xngin.apiserver import flags
from xngin.apiserver.testing import database_plugin, pg_helpers
from xngin.apiserver.testing.pg_helpers import TEMPLATE_PREFIX


def test_with_database_suffix_in_xdist_worker(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")

    url = pg_helpers.with_database_suffix("postgresql+psycopg://u:secret@localhost:5499/test_xngin?sslmode=disable")

    parsed = sqlalchemy.make_url(url)
    assert parsed.database == f"test_xngin_{pg_helpers._CHECKOUT_ID}_p{os.getppid()}_gw3"
    assert parsed.password == "secret"
    assert parsed.query == {"sslmode": "disable"}


def test_with_database_suffix_without_xdist(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    url = pg_helpers.with_database_suffix("postgresql+psycopg://u@localhost/test_xngin")

    assert sqlalchemy.make_url(url).database == f"test_xngin_{pg_helpers._CHECKOUT_ID}_p{os.getpid()}_main"


def _metadata(*, extra_column: bool = False) -> MetaData:
    metadata = MetaData()
    columns: list[Column] = [Column("id", Integer, primary_key=True), Column("a", String), Column("b", String)]
    if extra_column:
        columns.append(Column("c", String))
    table = Table("t", metadata, *columns)
    Index("ix_t_a", table.c.a)
    Index("ix_t_b", table.c.b)
    return metadata


def test_schema_digest_is_stable_and_tracks_schema_changes():
    assert pg_helpers._schema_digest(_metadata()) == pg_helpers._schema_digest(_metadata())
    assert pg_helpers._schema_digest(_metadata()) != pg_helpers._schema_digest(_metadata(extra_column=True))


class _FakeConnection:
    """Records the databases that _drop_leftovers() drops."""

    def __init__(self, newest_first: list[str]):
        self._rows = [(name,) for name in newest_first]
        self.dropped: list[str] = []

    def execute(self, query):
        if isinstance(query, Template) and query.strings[0].startswith("DROP DATABASE"):
            self.dropped.append(query.values[0])
        return SimpleNamespace(fetchall=lambda: self._rows)


def test_drop_leftovers_keeps_live_runs_and_newest_templates():
    checkout = pg_helpers._CHECKOUT_ID
    dead_pid = 99_999_999  # Larger than any pid_max.
    current = f"{TEMPLATE_PREFIX}current"
    others = [f"{TEMPLATE_PREFIX}other{i}" for i in range(6)]
    conn = _FakeConnection([
        others[0],
        current,
        *others[1:],
        f"test_xngin_{checkout}_p{dead_pid}_gw0",
        f"test_queries_{checkout}_p{dead_pid}_main",
        f"test_xngin_{checkout}_p{os.getpid()}_gw1",
        f"test_xngin_0000aaaa_p{dead_pid}_gw0",  # Another checkout's: its pids can't be checked here.
        "dwh",
    ])

    pg_helpers._drop_leftovers(conn, current)  # type: ignore[arg-type]

    assert conn.dropped == [
        *others[pg_helpers._OTHER_TEMPLATES_KEPT :],
        f"test_xngin_{checkout}_p{dead_pid}_gw0",
        f"test_queries_{checkout}_p{dead_pid}_main",
    ]


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        ("sqlite:///xngin.db", "sqlite:///xngin.db"),
        ("not a url", "not a url"),
        ("", ""),
        ("postgresql+psycopg://u@localhost/test_xngin", re.compile(r".*/test_xngin_[0-9a-f]{8}_p\d+_(main|gw\d+)$")),
    ],
)
def test_plugin_suffixes_only_postgres_urls(monkeypatch, database_url, expected):
    monkeypatch.setattr(flags, "DATABASE_URL", database_url)

    database_plugin.pytest_configure(SimpleNamespace(stash={}))  # type: ignore[arg-type]

    if isinstance(expected, re.Pattern):
        assert expected.match(flags.DATABASE_URL)
    else:
        assert expected == flags.DATABASE_URL
