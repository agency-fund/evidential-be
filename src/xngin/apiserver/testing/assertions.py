from typing import Any

from deepdiff import DeepDiff


def assert_same(actual: Any, expected: Any, extra: str | None = None):
    """Compare two values in a float-tolerant way using DeepDiff.

    If extra= is set, it will be displayed when the assertions fail.
    """
    # Setting math_epsilon to a non-None value delegates numeric comparisons to
    # https://docs.python.org/3/library/math.html#math.isclose.
    diff = DeepDiff(actual, expected, math_epsilon=0)
    assert not diff, (
        f"Objects differ:\n{diff.pretty()}" + f"\n{extra}\n" if extra else ""
    )
