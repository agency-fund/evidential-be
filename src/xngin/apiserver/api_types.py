import enum
from datetime import datetime
from typing import List, Dict, Any, Literal
import sqlalchemy.sql.sqltypes
from pydantic import BaseModel, Field


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
        """Attempt to infer the appropriate DataType for a value.

        Value may be a Python type or a SQLAlchemy type.
        """
        if value in DataType:
            return DataType[value]
        if value is str:
            return DataType.CHARACTER_VARYING
        if value is int:
            return DataType.INTEGER
        if value is float:
            return DataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.String):
            return DataType.CHARACTER_VARYING
        if isinstance(value, sqlalchemy.sql.sqltypes.Integer):
            return DataType.INTEGER
        if isinstance(value, sqlalchemy.sql.sqltypes.Float):
            return DataType.DOUBLE_PRECISION
        if isinstance(value, sqlalchemy.sql.sqltypes.Boolean):
            return DataType.BOOLEAN
        raise ValueError(f"Unmatched type: {value}.")

    def filter_class(self, column_name):
        """Classifies a DataType into a filter class."""
        match self:
            case DataType.BOOLEAN | DataType.CHARACTER_VARYING:
                return DataTypeClass.DISCRETE
            case (
                DataType.DATE
                | DataType.INTEGER
                | DataType.DOUBLE_PRECISION
                | DataType.NUMERIC
                | DataType.TIMESTAMP_WITHOUT_TIMEZONE
                | DataType.BIGINT
            ):
                return DataTypeClass.NUMERIC
            case _ if column_name.endswith("_id"):
                return DataTypeClass.DISCRETE
            case _:
                return DataTypeClass.UNKNOWN


class DataTypeClass(enum.StrEnum):
    DISCRETE = "discrete"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"

    def valid_relations(self):
        match self:
            case DataTypeClass.DISCRETE:
                return [Relation.INCLUDES, Relation.EXCLUDES]
            case DataTypeClass.NUMERIC:
                return [Relation.BETWEEN]
        raise ValueError(f"{self} has no valid defined relations..")


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
    data_type: DataType
    column_name: str
    description: str
    strata_group: str


class GetFiltersResponseElement(BaseModel):
    data_type: DataType
    description: str
    distinct_values: List[str]
    filter_name: str
    relations: List[Relation] = Field(..., min_length=1)
