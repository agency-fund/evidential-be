from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from loguru import logger
from psycopg.rows import TupleRow

from xngin.ops import performance

if TYPE_CHECKING:
    from string.templatelib import Template

    from psycopg import AsyncConnection
    from sqlalchemy.ext.asyncio import AsyncSession


@asynccontextmanager
async def with_driver_connection(session: AsyncSession) -> AsyncIterator[AsyncConnection]:
    async_conn = await session.connection()
    raw_conn = await async_conn.get_raw_connection()
    driver_conn = raw_conn.driver_connection
    if driver_conn is None:
        raise RuntimeError("failed getting driver connection")
    yield driver_conn


async def select_as_csv(
    session: AsyncSession,
    select_sql: Template,
    buffer_size_bytes: int,
    newline_framed: bool = False,
    include_header: bool = False,
) -> AsyncIterator[bytes]:
    """Yield CSV bytes for a SELECT statement executed via COPY TO STDOUT.

    This method can be used to efficiently read bulk data from Postgres without SQLAlchemy overhead.

    Args:
        session: SQLAlchemy async session.
        select_sql: SELECT statement to wrap in a COPY ... TO STDOUT query.
        buffer_size_bytes: Approximate threshold at which buffered bytes are yielded.
        newline_framed: If True, only yield through the last newline to preserve whole CSV rows.
        include_header: If True, request a CSV header row from Postgres.

    Yields:
        CSV byte chunks from the COPY stream. The final yielded chunk may be smaller than
        buffer_size_bytes.
    """
    header_sql = t", HEADER TRUE" if include_header else t""  # type: ignore[misc]
    copy_query = t"COPY ({select_sql:q}) TO STDOUT WITH (FORMAT CSV{header_sql:q})"  # type: ignore[misc]
    yield_count = 0
    with performance.timing() as timings:
        async with with_driver_connection(session) as driver_conn:
            buffer = bytearray()
            async with driver_conn.cursor() as cursor, cursor.copy(copy_query) as copy:
                # Each chunk is approximately one row; e.g. reading two ID fields might be ~60 bytes.
                async for chunk in copy:
                    buffer.extend(chunk)
                    if len(buffer) < buffer_size_bytes:
                        continue
                    if newline_framed:
                        last_newline = buffer.rfind(b"\n")
                        if last_newline >= 0:
                            yield_count += 1
                            yield bytes(buffer[: last_newline + 1])
                            del buffer[: last_newline + 1]
                    else:
                        yield_count += 1
                        yield bytes(buffer)
                        buffer.clear()
            if buffer:
                yield_count += 1
                yield bytes(buffer)
    logger.info("select_as_csv streamed {} chunks in {}s", yield_count, timings.elapsed)


async def stream(session: AsyncSession, select_query: Template, size: int) -> AsyncGenerator[TupleRow]:
    """Streams the results of a query, asking libpq to buffer up to size rows at a time."""
    async with with_driver_connection(session) as driver_conn, driver_conn.cursor() as cursor:
        async for row in cursor.stream(select_query, size=size):
            yield row
