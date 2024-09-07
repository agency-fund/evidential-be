import enum
from datetime import datetime
from typing import List, Dict, Any, Literal

from pydantic import BaseModel


class DataType(enum.StrEnum):
    BOOLEAN = "boolean"
    CHARACTER_VARYING = "character varying"
    DATE = "date"
    INTEGER = "integer"
    DOUBLE_PRECISION = "double precision"
    NUMERIC = "numeric"
    TIMESTAMP_WITHOUT_TIMEZONE = "timestamp without time zone"
    BIGINT = "bigint"

    @classmethod
    def match(cls, value):
        """Attempt to infer the appropriate DataType for a type."""
        if value in DataType:
            return DataType[value]
        # Respect Python builtin types.
        if value is str:
            return DataType.CHARACTER_VARYING
        if value is int:
            return DataType.INTEGER
        if value is float:
            return DataType.DOUBLE_PRECISION
        raise ValueError(f"Unmatched type: {value}")


class DataTypeClass(enum.StrEnum):
    DISCRETE = "discrete"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"


class Relation(enum.StrEnum):
    INCLUDES = "includes"
    EXCLUDES = "excludes"
    BETWEEN = "between"


class AudienceSpec(BaseModel):
    """Audience specification."""

    type: str
    filters: List[Dict[str, Any]]


class DesignSpec(BaseModel):
    """Design specification."""

    experiment_id: str
    experiment_name: str
    description: str
    arms: List[Dict[str, str]]
    start_date: datetime
    end_date: datetime
    strata_cols: List[str]
    power: float
    alpha: float
    fstat_thresh: float
    metrics: List[Dict[str, Any]]


class UnimplementedResponse(BaseModel):
    todo: Literal["TODO"]


class GetStrataResponseElement(BaseModel):
    table_name: str
    column_name: str
    data_type: DataType
    strata_group: str  # TODO: rename to column_group?
    id: str
