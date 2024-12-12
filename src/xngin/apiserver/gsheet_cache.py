from collections.abc import Callable

from sqlalchemy.orm import Session

from xngin.apiserver.database import Cache
from xngin.apiserver.settings import SheetRef
from xngin.sheets.config_sheet import ConfigWorksheet


class GSheetCache:
    """Implements a simple read-through cache for Google Sheets data backed by a SQLite database."""

    def __init__(self, session: Session, auto_handle_conflicts: bool = False):
        """Set auto_handle_conflicts=True if the Session's dialect can auto-resolve conflicts on insert."""
        self.session = session
        self.auto_handle_conflicts = auto_handle_conflicts

    def get(
        self,
        key: SheetRef,
        fetcher: Callable[[], ConfigWorksheet],
        refresh=False,
    ):
        cache_key = f"{key.url}!{key.worksheet}"
        entry = None
        if not refresh or not self.auto_handle_conflicts:
            entry = self.session.get(Cache, cache_key)
        if not entry:
            result = fetcher()
            entry = Cache(key=cache_key, value=result.model_dump_json())
            self.session.add(entry)
            self.session.commit()
        elif refresh:  # then do an UPDATE of the entry
            result = fetcher()
            entry.value = result.model_dump_json()
            self.session.commit()
        return ConfigWorksheet.model_validate_json(entry.value)
