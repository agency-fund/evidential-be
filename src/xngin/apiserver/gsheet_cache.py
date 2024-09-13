from typing import Callable

from sqlalchemy.orm import Session

from xngin.apiserver.database import Cache
from xngin.apiserver.settings import SheetRef
from xngin.sheets.config_sheet import ConfigWorksheet


class GSheetCache:
    """Implements a simple read-through cache for Google Sheets data backed by a SQLite database."""

    def __init__(self, session: Session):
        self.session = session

    def get(
        self,
        key: SheetRef,
        fetcher: Callable[[], ConfigWorksheet],
        refresh=False,
    ):
        cache_key = f"{key.url}!{key.worksheet}"
        entry = None
        if not refresh:
            entry = self.session.get(Cache, cache_key)
        if not entry:
            result = fetcher()
            entry = Cache(key=cache_key, value=result.model_dump_json())
            self.session.add(entry)
            self.session.commit()
        return ConfigWorksheet.model_validate_json(entry.value)
