import base64
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select

from xngin.apiserver.pagination import (
    MAX_PAGE_TOKEN_AGE_SECONDS,
    InvalidPageTokenError,
    SortField,
    build_next_page_token,
    clamp_page_size,
    decode_datetime,
    decode_page_token,
    encode_datetime,
    encode_page_token,
    paginate,
)


def test_page_token_round_trip():
    token = encode_page_token(["sn_abc123", 42])
    cursor = decode_page_token(token)
    assert cursor.created_at > 0
    assert cursor.keys == ["sn_abc123", 42]


def test_page_token_round_trip_with_microseconds():
    ts = datetime(2025, 6, 15, 10, 30, 0, 123456, tzinfo=UTC)
    token = encode_page_token([encode_datetime(ts), "ev_xyz"])
    cursor = decode_page_token(token)
    assert decode_datetime(cursor.keys[0]) == ts
    assert cursor.keys[1] == "ev_xyz"


def test_page_token_uses_compact_aliases():
    token = encode_page_token(["v1", "v2"])
    padded = token + "=" * (-len(token) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded))
    assert payload["k"] == ["v1", "v2"]
    assert isinstance(payload["c"], int)


def test_decode_expired_token():
    now = int(datetime.now(UTC).timestamp())
    payload = {"c": now - MAX_PAGE_TOKEN_AGE_SECONDS - 1, "k": ["v1"]}
    token = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    with pytest.raises(InvalidPageTokenError, match="Invalid page token"):
        decode_page_token(token)


def test_decode_invalid_token():
    with pytest.raises(InvalidPageTokenError, match="Invalid page token"):
        decode_page_token("not-a-valid-token!!!")


def test_decode_empty_json():
    token = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    with pytest.raises(InvalidPageTokenError, match="Invalid page token"):
        decode_page_token(token)


def test_clamp_page_size_none():
    assert clamp_page_size(None) == 20


def test_clamp_page_size_zero():
    assert clamp_page_size(0) == 20


def test_clamp_page_size_within_range():
    assert clamp_page_size(50) == 50


def test_clamp_page_size_exceeds_max():
    assert clamp_page_size(200) == 100


def test_clamp_page_size_negative():
    assert clamp_page_size(-5) == 1


def test_clamp_page_size_custom_defaults():
    assert clamp_page_size(None, default=10, maximum=50) == 10
    assert clamp_page_size(100, default=10, maximum=50) == 50


def test_paginate_supports_multi_field_cursor_desc():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    events = Table(
        "events",
        metadata,
        Column("id", String, primary_key=True),
        Column("score", Integer, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            events.insert(),
            [
                {"id": "e", "score": 5},
                {"id": "d", "score": 5},
                {"id": "c", "score": 5},
                {"id": "z", "score": 4},
                {"id": "a", "score": 3},
            ],
        )
        token = encode_page_token([5, "d"])
        query = paginate(
            select(events.c.score, events.c.id),
            sort_fields=[
                SortField.numeric(column=events.c.score, attr="score", direction="desc"),
                SortField(column=events.c.id, attr="id", direction="desc"),
            ],
            page_token=token,
            page_size=10,
        )
        rows = conn.execute(query).all()
    assert rows == [(5, "c"), (4, "z"), (3, "a")]


def test_paginate_applies_skip_after_cursor():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    events = Table(
        "events",
        metadata,
        Column("id", String, primary_key=True),
        Column("score", Integer, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            events.insert(),
            [
                {"id": "e", "score": 5},
                {"id": "d", "score": 5},
                {"id": "c", "score": 5},
                {"id": "z", "score": 4},
                {"id": "a", "score": 3},
            ],
        )
        token = encode_page_token([5, "d"])
        query = paginate(
            select(events.c.score, events.c.id),
            sort_fields=[
                SortField.numeric(column=events.c.score, attr="score", direction="desc"),
                SortField(column=events.c.id, attr="id", direction="desc"),
            ],
            page_token=token,
            page_size=10,
            skip=1,
        )
        rows = conn.execute(query).all()
    assert rows == [(4, "z"), (3, "a")]


def test_paginate_rejects_cursor_with_wrong_field_count():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    events = Table(
        "events",
        metadata,
        Column("id", String, primary_key=True),
        Column("score", Integer, nullable=False),
    )
    metadata.create_all(engine)
    token = encode_page_token([5])
    with pytest.raises(InvalidPageTokenError, match="Invalid page token"):
        paginate(
            select(events.c.score, events.c.id),
            sort_fields=[
                SortField.numeric(column=events.c.score, attr="score", direction="desc"),
                SortField(column=events.c.id, attr="id", direction="desc"),
            ],
            page_token=token,
            page_size=10,
        )


def test_sortfield_bool_serializes_as_zero_or_one():
    field = SortField.bool(Column("is_active"), attr="is_active", direction="desc")
    assert field.encode(True) == 1
    assert field.encode(False) == 0
    assert field.decode(1) is True
    assert field.decode(0) is False
    with pytest.raises(InvalidPageTokenError, match="Invalid page token"):
        field.decode(2)


def test_build_next_page_token_no_more_pages():
    rows = [
        SimpleNamespace(created_at=datetime(2025, 1, 2, tzinfo=UTC), id="b"),
        SimpleNamespace(created_at=datetime(2025, 1, 1, tzinfo=UTC), id="a"),
    ]
    sort_fields = [
        SortField.timestamp(column=Column("created_at"), attr="created_at", direction="desc"),
        SortField(column=Column("id"), attr="id", direction="desc"),
    ]
    trimmed, token = build_next_page_token(rows, page_size=5, sort_fields=sort_fields)
    assert trimmed == rows
    assert token == ""


def test_build_next_page_token_has_more_pages():
    rows = [
        SimpleNamespace(created_at=datetime(2025, 1, 3, tzinfo=UTC), id="c"),
        SimpleNamespace(created_at=datetime(2025, 1, 2, tzinfo=UTC), id="b"),
        SimpleNamespace(created_at=datetime(2025, 1, 1, tzinfo=UTC), id="a"),
    ]
    sort_fields = [
        SortField.timestamp(column=Column("created_at"), attr="created_at", direction="desc"),
        SortField(column=Column("id"), attr="id", direction="desc"),
    ]
    trimmed, token = build_next_page_token(rows, page_size=2, sort_fields=sort_fields)
    assert len(trimmed) == 2
    assert trimmed[0].id == "c"
    assert trimmed[1].id == "b"
    assert token != ""
    cursor = decode_page_token(token)
    assert decode_datetime(cursor.keys[0]) == datetime(2025, 1, 2, tzinfo=UTC)
    assert cursor.keys[1] == "b"


def test_build_next_page_token_exact_page_size():
    rows = [
        SimpleNamespace(created_at=datetime(2025, 1, 2, tzinfo=UTC), id="b"),
        SimpleNamespace(created_at=datetime(2025, 1, 1, tzinfo=UTC), id="a"),
    ]
    sort_fields = [
        SortField.timestamp(column=Column("created_at"), attr="created_at", direction="desc"),
        SortField(column=Column("id"), attr="id", direction="desc"),
    ]
    trimmed, token = build_next_page_token(rows, page_size=2, sort_fields=sort_fields)
    assert len(trimmed) == 2
    assert token == ""
