from collections.abc import Callable

from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from xngin.apiserver.models.tables import CacheTable
from xngin.apiserver.settings import SheetRef
from xngin.schema.sheet_types import ParticipantSchema


class GSheetCache:
    """Implements a simple read-through cache for Google Sheets data backed by a SQLite database."""

    def __init__(self, session: Session):
        self.session = session

    def get(
        self,
        key: SheetRef,
        fetcher: Callable[[], ParticipantSchema],
        refresh=False,
    ):
        cache_key = f"{key.url}!{key.worksheet}"
        entry = None
        if not refresh:
            entry = self.session.get(CacheTable, cache_key)
        if not entry:
            result = fetcher()
            entry = CacheTable(key=cache_key, value=result.model_dump_json())
            try:
                self.session.add(entry)
                self.session.commit()
            # This fallback should support sqlite and postgresql
            except IntegrityError:
                self.session.rollback()
                self.session.execute(
                    update(CacheTable)
                    .where(CacheTable.key == entry.key)
                    .values(value=entry.value)
                )
                self.session.commit()
        return ParticipantSchema.model_validate_json(entry.value)
