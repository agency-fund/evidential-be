from datetime import datetime


def _dates_equal(db_date: datetime, other_date: datetime):
    """Compare dates with or without timezone info, honoring the db_date's timezone."""
    if db_date.tzinfo is None:
        return db_date == other_date.replace(tzinfo=None)
    return db_date == other_date


def assert_dates_equal(db_date: datetime, other_date: datetime):
    """Asserts that the db_date is equal to other_date, honoring the db_date's timezone."""
    assert _dates_equal(db_date, other_date), f"Date {db_date} is not equal to {other_date}"
