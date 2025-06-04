import enum
from typing import Self


class ExperimentState(enum.StrEnum):
    """
    Experiment lifecycle states.

    note: [starting state], [[terminal state]]
    [DESIGNING]->[ASSIGNED]->{[[ABANDONED]], COMMITTED}->[[ABORTED]]
    """

    DESIGNING = "designing"  # TODO: https://github.com/agency-fund/xngin/issues/352
    ASSIGNED = "assigned"  # TODO: rename to "REVIEWING"
    ABANDONED = "abandoned"
    COMMITTED = "committed"
    # TODO: Consider adding two more states:
    # Add an ACTIVE state that is only derived in a View when the state is COMMITTED and the query
    # time is between experiment start and end.
    # Add a COMPLETE state that is only derived in a View when the state is COMMITTED and query time
    # is after experiment end.
    ABORTED = "aborted"


class StopAssignmentReason(enum.StrEnum):
    """The reason assignments were stopped."""

    @classmethod
    def from_str(cls, value: str | None) -> Self | None:
        """Create StopAssignmentReason from string. Returns None if value is None."""
        return None if value is None else cls(value)

    END_DATE = "end_date"  # end date reached
    MANUAL = "manual"  # manually stopped by user
    TARGET_N = "target_n"  # target total number of participants across all arms reached
