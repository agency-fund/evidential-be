"""Cursor-based pagination utilities following Google AIP-158."""

import base64
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Select, and_, or_
from sqlalchemy.sql.elements import SQLColumnExpression

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_PAGE_TOKEN_AGE_SECONDS = 3 * 24 * 60 * 60


class InvalidPageTokenError(Exception):
    """Raised when a page token is missing required fields or has invalid values."""

    def __init__(self, message: str = "Invalid page token"):
        super().__init__(message)


class PageCursor(BaseModel):
    """The data encoded within an opaque page token."""

    model_config = ConfigDict(serialize_by_alias=True, extra="forbid", populate_by_name=True)
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
        validation_alias="c",
        serialization_alias="c",
    )
    keys: list[Any] = Field(validation_alias="k", serialization_alias="k", min_length=1)


def _encode_datetime(value: datetime) -> str:
    """Encode a datetime value for cursor serialization."""
    return value.isoformat()


def _decode_datetime(value: Any) -> datetime:
    """Decode a datetime value from cursor serialization."""
    try:
        return datetime.fromisoformat(value)
    except Exception as exc:
        raise InvalidPageTokenError() from exc


def _encode_numeric(value: object) -> int | float:
    """Encode an int/float value for cursor serialization."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidPageTokenError()
    return value


def _decode_numeric(value: object) -> int | float:
    """Decode an int/float value from cursor serialization."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidPageTokenError()
    return value


def _encode_bool(value: object) -> int:
    """Encode a bool value for cursor serialization as 0 or 1."""
    if not isinstance(value, bool):
        raise InvalidPageTokenError()
    return 1 if value else 0


def _decode_bool(value: object) -> bool:
    """Decode a bool value from cursor serialization (expects integer 0 or 1)."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise InvalidPageTokenError()
    return value == 1


@dataclass(frozen=True)
class PaginationQuery:
    """Describes the pagination request parameters."""

    page_size: int
    page_token: str | None
    skip: int = 0


def pagination_query_params(
    page_size: Annotated[
        int,
        Query(
            description="Maximum number of items to return per page.",
            ge=1,
            le=MAX_PAGE_SIZE,
        ),
    ] = DEFAULT_PAGE_SIZE,
    page_token: Annotated[
        str | None,
        Query(description="Token from a previous response to fetch the next page."),
    ] = None,
    skip: Annotated[
        int,
        Query(
            description="Number of items to skip after page_token (or from the start when page_token is omitted).",
            ge=0,
        ),
    ] = 0,
) -> PaginationQuery:
    """Dependency describing pagination request parameters."""
    return PaginationQuery(page_size=page_size, page_token=page_token or None, skip=skip)


@dataclass(frozen=True)
class SortField:
    """Describes one ordered field used for seek-pagination and token generation.

    `column` is the SQLAlchemy expression used to build ORDER BY and cursor WHERE
    predicates.

    `attr` is the attribute name read from each returned row object when building
    the next page token. In other words, for each row in `build_next_page_token`,
    the value is read via `getattr(row, attr)`. For ORM queries this should match
    the mapped model attribute name (for example, `created_at` or `id`).

    Use the static factories when possible:
    - `SortField.timestamp(...)` for datetime values
    - `SortField.numeric(...)` for int/float values
    - `SortField.bool(...)` for bool values (serialized as 0/1)
    """

    column: SQLColumnExpression[Any]
    attr: str
    direction: Literal["asc", "desc"] = "desc"
    encode: Callable[[Any], Any] = lambda value: value
    decode: Callable[[Any], Any] = lambda value: value

    @staticmethod
    def timestamp(
        column: SQLColumnExpression[Any],
        attr: str,
        direction: Literal["asc", "desc"] = "desc",
    ) -> "SortField":
        """Create a SortField for datetime values."""
        return SortField(
            column=column,
            attr=attr,
            direction=direction,
            encode=_encode_datetime,
            decode=_decode_datetime,
        )

    @staticmethod
    def numeric(
        column: SQLColumnExpression[Any],
        attr: str,
        direction: Literal["asc", "desc"] = "desc",
    ) -> "SortField":
        """Create a SortField for int/float values."""
        return SortField(
            column=column,
            attr=attr,
            direction=direction,
            encode=_encode_numeric,
            decode=_decode_numeric,
        )

    @staticmethod
    def bool(
        column: SQLColumnExpression[Any],
        attr: str,
        direction: Literal["asc", "desc"] = "desc",
    ) -> "SortField":
        """Create a SortField for bool values serialized as 0/1."""
        return SortField(
            column=column,
            attr=attr,
            direction=direction,
            encode=_encode_bool,
            decode=_decode_bool,
        )


def _encode_page_token(values: Sequence[Any]) -> str:
    """Encode a cursor position as an opaque, URL-safe page token."""
    cursor = PageCursor(keys=list(values))
    return base64.urlsafe_b64encode(cursor.model_dump_json().encode()).decode()


def _decode_page_token(token: str) -> PageCursor:
    """Decode a page token into a PageCursor.

    Raises InvalidPageTokenError if the token is malformed.
    """
    try:
        raw = base64.urlsafe_b64decode(token)
        cursor = PageCursor.model_validate_json(raw)
    except Exception as exc:
        raise InvalidPageTokenError() from exc
    # TODO: Use chafernet to encrypt/decrypt page token and enforce timestamps.
    now = int(datetime.now(UTC).timestamp())
    if now - cursor.created_at > MAX_PAGE_TOKEN_AGE_SECONDS:
        raise InvalidPageTokenError()
    return cursor


def paginate(
    query: Select[Any],
    ordering: Sequence[SortField],
    pagination: PaginationQuery,
) -> Select[Any]:
    """Apply cursor-based WHERE, ORDER BY, OFFSET, and LIMIT.

    ordering must define a deterministic order, usually with a unique tie-breaker
    (e.g. id) as the last field.
    Fetches page_size + 1 rows so the caller can detect a next page.

    Raises InvalidPageTokenError if page_token is present but invalid.
    """
    if not ordering:
        raise ValueError("ordering must not be empty")
    if pagination.skip < 0:
        raise ValueError("skip must be non-negative")

    order_by = [field.column.desc() if field.direction == "desc" else field.column.asc() for field in ordering]
    query = query.order_by(None).order_by(*order_by)

    if pagination.page_token:
        cursor = _decode_page_token(pagination.page_token)
        if len(cursor.keys) != len(ordering):
            raise InvalidPageTokenError()
        try:
            cursor_values = [field.decode(value) for field, value in zip(ordering, cursor.keys, strict=True)]
        except InvalidPageTokenError:
            raise
        except Exception as exc:
            raise InvalidPageTokenError() from exc
        # Expand lexicographic cursor comparison into OR-of-prefix predicates. This allows mixed-direction orderings
        # (e.g. score DESC, id ASC).
        disjuncts = []
        for idx, field in enumerate(ordering):
            prefix = [ordering[prefix_idx].column == cursor_values[prefix_idx] for prefix_idx in range(idx)]
            op = field.column < cursor_values[idx] if field.direction == "desc" else field.column > cursor_values[idx]
            disjuncts.append(and_(*prefix, op))
        query = query.where(or_(*disjuncts))

    if pagination.skip:
        query = query.offset(pagination.skip)
    return query.limit(pagination.page_size + 1)


def build_next_page_token(
    rows: list[Any],
    page_size: int,
    ordering: Sequence[SortField],
) -> tuple[list[Any], str]:
    """Given rows (possibly page_size+1 long), return (trimmed_rows, next_page_token).

    If len(rows) > page_size, trims to page_size and builds a token from the
    last included row. Otherwise returns "" indicating no more pages.
    """
    if len(rows) > page_size:
        rows = rows[:page_size]
        last = rows[-1]
        values = [field.encode(getattr(last, field.attr)) for field in ordering]
        token = _encode_page_token(values)
        return rows, token
    return rows, ""
