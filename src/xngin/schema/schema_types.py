from collections import Counter
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from xngin.apiserver.api_types import DataType


class SchemaBaseModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class FieldDescriptor(SchemaBaseModel):
    field_name: Annotated[
        str, Field(description="Name of the field in the data source")
    ]
    data_type: Annotated[DataType, Field(description="The data type of this field")]
    description: Annotated[
        str, Field(description="Human-readable description of the field")
    ]
    is_unique_id: Annotated[
        bool, Field(description="Whether this field uniquely identifies records")
    ]
    is_strata: Annotated[
        bool, Field(description="Whether this field should be used for stratification")
    ]
    is_filter: Annotated[
        bool, Field(description="Whether this field can be used as a filter")
    ]
    is_metric: Annotated[
        bool, Field(description="Whether this field can be used as a metric")
    ]
    extra: Annotated[
        dict[str, str] | None, Field(description="Additional field metadata")
    ] = None

    @field_validator("description", mode="before")
    @classmethod
    def to_string_loose(cls, value) -> str:
        if not isinstance(value, str):
            return str(value)
        return value

    @field_validator("data_type", mode="before")
    @classmethod
    def to_data_type(cls, value) -> DataType:
        return DataType(value.lower())

    @field_validator(
        "is_unique_id", "is_strata", "is_filter", "is_metric", mode="before"
    )
    @classmethod
    def to_boolean(cls, value):
        truthy = {"true", "t", "yes", "y", "1"}
        falsy = {"false", "f", "no", "n", "0", ""}
        normalized = str(value).lower().strip()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        raise ValueError(f"Value '{value}' cannot be converted to a boolean.")


class ParticipantsSchema(SchemaBaseModel):
    """Represents a single worksheet describing metadata about a type of Participant."""

    table_name: Annotated[
        str, Field(description="Name of the table in the data warehouse")
    ]
    fields: Annotated[
        list[FieldDescriptor],
        Field(description="List of fields available in this table"),
    ]

    def get_unique_id_field(self):
        """Gets the name of the unique ID field."""
        return next((i.field_name for i in self.fields if i.is_unique_id), None)

    @model_validator(mode="after")
    def check_one_unique_id(self) -> "ParticipantsSchema":
        uniques = [r.field_name for r in self.fields if r.is_unique_id]
        if len(uniques) == 0:
            raise ValueError("There are no columns marked as unique ID.")
        if len(uniques) > 1:
            raise ValueError(
                f"There are {len(uniques)} columns marked as the unique ID, but there should "
                f"only be one: {', '.join(sorted(uniques))}"
            )
        return self

    @model_validator(mode="after")
    def check_unique_fields(self) -> "ParticipantsSchema":
        counted = Counter([".".join(row.field_name) for row in self.fields])
        duplicates = [item for item, count in counted.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate 'field_name' values found: {', '.join(duplicates)}."
            )
        return self

    @model_validator(mode="after")
    def check_non_empty_rows(self) -> "ParticipantsSchema":
        if len(self.fields) == 0:
            raise ValueError(
                f"{self.__class__} must contain at least one FieldDescriptor."
            )
        return self
