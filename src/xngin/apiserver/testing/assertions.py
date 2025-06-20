from datetime import datetime
from typing import Any

from deepdiff import DeepDiff


def assert_same(
    actual: Any, expected: Any, deepdiff_kwargs=None, extra: str | None = None
):
    """Compare two values in a float-tolerant way using DeepDiff.

    If extra= is set, it will be displayed when the assertions fail.
    """
    # Setting math_epsilon to a non-None value delegates numeric comparisons to
    # https://docs.python.org/3/library/math.html#math.isclose.
    if deepdiff_kwargs is None:
        deepdiff_kwargs = dict()
    diff = DeepDiff(expected, actual, math_epsilon=0, **deepdiff_kwargs)
    assert not diff, (
        f"Objects differ:\n{diff.pretty()}" + f"\n{extra}\n" if extra else ""
    )


def _dates_equal(db_date: datetime, other_date: datetime):
    """Compare dates with or without timezone info, honoring the db_date's timezone."""
    if db_date.tzinfo is None:
        return db_date == other_date.replace(tzinfo=None)
    return db_date == other_date


def assert_dates_equal(db_date: datetime, other_date: datetime):
    """Asserts that the db_date is equal to other_date, honoring the db_date's timezone."""
    assert _dates_equal(db_date, other_date), (
        f"Date {db_date} is not equal to {other_date}"
    )
