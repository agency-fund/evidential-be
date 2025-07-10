import enum
from typing import Self

import sqlalchemy.sql
from loguru import logger
from sqlalchemy.dialects.postgresql.json import JSON, JSONB


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

    PREASSIGNED = (
        "preassigned"  # preassigned experiments do not allow additional assignments
    )
    END_DATE = "end_date"  # end date reached
    MANUAL = "manual"  # manually stopped by user
    TARGET_N = "target_n"  # target total number of participants across all arms reached


class DwhDataType(enum.StrEnum):
    """Defines the supported data types for fields in the data source."""

    BOOLEAN = "boolean"
    CHARACTER_VARYING = "character varying"
    UUID = "uuid"
    DATE = "date"
    INTEGER = "integer"
    DOUBLE_PRECISION = "double precision"
    NUMERIC = "numeric"
    TIMESTAMP_WITHOUT_TIMEZONE = "timestamp without time zone"
    TIMESTAMP_WITH_TIMEZONE = "timestamp with time zone"
    BIGINT = "bigint"
    JSONB = "jsonb (unsupported)"
    JSON = "json (unsupported)"
    UNKNOWN = "unsupported"
    # NOTE: If adding types, the frontend (e.g. data-type-badge.tsx when viewing participant type
    # details) should also be updated to badge appropriately in the UI.

    @classmethod
    def match(cls, value):
        """Maps a Python or SQLAlchemy type or value's type to the corresponding DataType.

        Value may be a Python type or a SQLAlchemy type.
        """
        if value in DwhDataType:
            return DwhDataType[value]
        if value is str:
            return DwhDataType.CHARACTER_VARYING
        if isinstance(value, sqlalchemy.sql.sqltypes.UUID):
            return DwhDataType.UUID
        if isinstance(value, sqlalchemy.sql.sqltypes.String):
            return DwhDataType.CHARACTER_VARYING
        if isinstance(value, sqlalchemy.sql.sqltypes.Boolean):
            return DwhDataType.BOOLEAN
        if isinstance(value, sqlalchemy.sql.sqltypes.BigInteger):
            return DwhDataType.BIGINT
        if isinstance(value, sqlalchemy.sql.sqltypes.Integer):
            return DwhDataType.INTEGER
        if isinstance(value, sqlalchemy.sql.sqltypes.Double):
            return DwhDataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.Float):
            return DwhDataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.Numeric):
            return DwhDataType.NUMERIC
        if isinstance(value, sqlalchemy.sql.sqltypes.Date):
            return DwhDataType.DATE
        if isinstance(value, sqlalchemy.sql.sqltypes.DateTime) and value.timezone:
            return DwhDataType.TIMESTAMP_WITH_TIMEZONE
        if isinstance(value, sqlalchemy.sql.sqltypes.DateTime) and not value.timezone:
            return DwhDataType.TIMESTAMP_WITHOUT_TIMEZONE
        if isinstance(value, JSONB):
            return DwhDataType.JSONB
        if isinstance(value, JSON):
            return DwhDataType.JSON
        if value is int:
            return DwhDataType.INTEGER
        if value is float:
            return DwhDataType.DOUBLE_PRECISION
        logger.warning("Unmatched type: {}", type(value))
        return DwhDataType.UNKNOWN

    @classmethod
    def supported_participant_id_types(cls) -> list["DwhDataType"]:
        """Returns the list of data types that are supported as participant IDs."""
        return [
            DwhDataType.INTEGER,
            DwhDataType.BIGINT,
            DwhDataType.UUID,
            DwhDataType.CHARACTER_VARYING,
        ]

    @classmethod
    def is_supported_type(cls, data_type: Self):
        """Returns True if the type is supported as a strata, filter, and/or metric."""
        return data_type not in {
            DwhDataType.JSONB,
            DwhDataType.JSON,
            DwhDataType.UNKNOWN,
        }

    def is_supported(self):
        """Returns True if the type is supported as a strata, filter, and/or metric."""
        return DwhDataType.is_supported_type(self)

    def filter_class(self, field_name):
        """Classifies a DataType into a filter class."""
        match self:
            # TODO: is this customer specific?
            case _ if field_name.lower().endswith("_id"):
                return FilterClass.DISCRETE
            case DwhDataType.BOOLEAN | DwhDataType.CHARACTER_VARYING | DwhDataType.UUID:
                return FilterClass.DISCRETE
            case (
                DwhDataType.DATE
                | DwhDataType.TIMESTAMP_WITHOUT_TIMEZONE
                | DwhDataType.TIMESTAMP_WITH_TIMEZONE
                | DwhDataType.INTEGER
                | DwhDataType.DOUBLE_PRECISION
                | DwhDataType.NUMERIC
                | DwhDataType.BIGINT
            ):
                return FilterClass.NUMERIC
            case _:
                return FilterClass.UNKNOWN


class FilterClass(enum.StrEnum):
    """Internal helper for grouping our supported data types by what filter relations they can use."""

    DISCRETE = "discrete"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"

    def valid_relations(self):
        """Gets the valid relation operators for this data type class."""
        match self:
            case FilterClass.DISCRETE:
                return [Relation.INCLUDES, Relation.EXCLUDES]
            case FilterClass.NUMERIC:
                return [
                    Relation.BETWEEN,
                    Relation.EXCLUDES,
                    Relation.INCLUDES,
                ]
        raise ValueError(f"{self} has no valid defined relations.")


class Relation(enum.StrEnum):
    """Defines operators for filtering values.

    INCLUDES matches when the value matches any of the provided values, including null if explicitly
    specified. For CSV fields (i.e. experiment_ids), any value in the CSV that matches the provided
    values will match, but nulls are unsupported. This is equivalent to NOT(EXCLUDES(values)).

    EXCLUDES matches when the value does not match any of the provided values, including null if
    explicitly specified. If null is not explicitly excluded, we include nulls in the result.  CSV
    fields (i.e. experiment_ids), the match will fail if any of the provided values are present
    in the value, but nulls are unsupported.

    BETWEEN matches when the value is between the two provided values (inclusive). Not allowed for CSV fields.
    """

    INCLUDES = "includes"
    EXCLUDES = "excludes"
    BETWEEN = "between"
