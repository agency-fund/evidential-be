from collections.abc import Callable

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.dwh.inspection_types import ParticipantsSchema
from xngin.apiserver.models import tables
from xngin.apiserver.settings import SheetRef


class GSheetCache:
    """Implements a simple read-through cache for Google Sheets data backed by a database."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(
        self,
        key: SheetRef,
        fetcher: Callable[[], ParticipantsSchema],
        refresh=False,
    ):
        if self.session is None:
            return fetcher()

        cache_key = f"{key.url}!{key.worksheet}"
        entry = None
        if not refresh:
            entry = await self.session.get(tables.CacheTable, cache_key)
        if not entry:
            result = fetcher()
            entry = tables.CacheTable(key=cache_key, value=result.model_dump_json())
            try:
                self.session.add(entry)
                await self.session.commit()
            except IntegrityError:
                await self.session.rollback()
                await self.session.execute(
                    update(tables.CacheTable)
                    .where(tables.CacheTable.key == entry.key)
                    .values(value=entry.value)
                )
                await self.session.commit()
        return ParticipantsSchema.model_validate_json(entry.value)
