from collections import Counter

from pydantic import BaseModel, field_validator, model_validator

from xngin.apiserver.api_types import DataType


class FieldDescriptor(BaseModel):
    field_name: str
    data_type: DataType
    description: str
    is_unique_id: bool
    is_strata: bool
    is_filter: bool
    is_metric: bool
    extra: dict[str, str] | None = None

    model_config = {
        "strict": True,
        "extra": "forbid",
    }

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


class ParticipantSchema(BaseModel):
    """Represents a single worksheet describing metadata about a type of Participant."""

    table_name: str
    fields: list[FieldDescriptor]

    model_config = {
        "strict": True,
        "extra": "forbid",
    }

    def get_unique_id_field(self):
        """Gets the name of the unique ID field."""
        return next((i.field_name for i in self.fields if i.is_unique_id), None)

    @model_validator(mode="after")
    def check_one_unique_id(self) -> "ParticipantSchema":
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
    def check_unique_fields(self) -> "ParticipantSchema":
        counted = Counter([".".join(row.field_name) for row in self.fields])
        duplicates = [item for item, count in counted.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate 'field_name' values found: {', '.join(duplicates)}."
            )
        return self

    @model_validator(mode="after")
    def check_non_empty_rows(self) -> "ParticipantSchema":
        if len(self.fields) == 0:
            raise ValueError(
                f"{self.__class__} must contain at least one FieldDescriptor."
            )
        return self
