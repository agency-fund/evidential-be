from collections.abc import Callable

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from xngin.apiserver.models.tables import CacheTable
from xngin.apiserver.settings import SheetRef
from xngin.schema.schema_types import ParticipantsSchema


class GSheetCache:
    """Implements a simple read-through cache for Google Sheets data backed by a database."""

    def __init__(self, session: Session):
        self.session = session

    def get(
        self,
        key: SheetRef,
        fetcher: Callable[[], ParticipantsSchema],
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
            except IntegrityError:
                self.session.rollback()
                self.session.execute(
                    update(CacheTable)
                    .where(CacheTable.key == entry.key)
                    .values(value=entry.value)
                )
                self.session.commit()
        return ParticipantsSchema.model_validate_json(entry.value)
