from datetime import UTC, datetime
from string.templatelib import Template
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, Index, Integer, MetaData, String, Table

from xngin.apiserver import flags
from xngin.apiserver.testing import database_plugin, pg_helpers
from xngin.apiserver.testing.pg_helpers import TEMPLATE_PREFIX

_SUFFIX = pg_helpers._DATABASE_SUFFIX


@pytest.mark.parametrize(
    "create",
    [
        pg_helpers.create_empty_test_database_pg,
        lambda url: pg_helpers.create_test_database_pg(url, MetaData()),
    ],
    ids=["empty", "from_template"],
)
def test_create_refuses_unsuffixed_names(create):
    with pytest.raises(ValueError, match="Refusing to recreate 'xngin'"):
        create("postgresql+psycopg://u@localhost/xngin")


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


def test_drop_leftovers_drops_day_old_databases_and_all_but_newest_templates():
    current = f"{TEMPLATE_PREFIX}current"
    others = [f"{TEMPLATE_PREFIX}other{i}" for i in range(6)]
    conn = _FakeConnection([
        others[0],
        current,
        *others[1:],
        "test_xngin_20261022t120000_11111111_gw0",  # Started 25 hours ago.
        "test_queries_20261022t120000_11111111_main",
        "test_xngin_20261022t140000_22222222_gw1",  # Started 23 hours ago.
        "dwh",
    ])

    pg_helpers._drop_leftovers(conn, current, datetime(2026, 10, 23, 13, 0, tzinfo=UTC))  # type: ignore[arg-type]

    assert conn.dropped == [
        *others[pg_helpers._OTHER_TEMPLATES_KEPT :],
        "test_xngin_20261022t120000_11111111_gw0",
        "test_queries_20261022t120000_11111111_main",
    ]


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        ("", ""),
        (
            "postgresql+psycopg://u:secret@localhost:5499/test_xngin?sslmode=disable",
            f"postgresql+psycopg://u:secret@localhost:5499/test_xngin{_SUFFIX}?sslmode=disable",
        ),
        ("postgres://u@localhost/xngin", f"postgres://u@localhost/xngin{_SUFFIX}"),
    ],
    ids=["unset", "postgresql", "postgres"],
)
def test_plugin_suffixes_database_url(monkeypatch, database_url, expected):
    monkeypatch.setattr(flags, "DATABASE_URL", database_url)

    database_plugin.pytest_configure(SimpleNamespace(stash={}))  # type: ignore[arg-type]

    assert expected == flags.DATABASE_URL
