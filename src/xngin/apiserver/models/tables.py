import json
import secrets
from datetime import datetime
from typing import Self

from pydantic import TypeAdapter
from sqlalchemy import ForeignKey, String, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship

from xngin.apiserver.routers.admin_api_types import (
    InspectDatasourceTableResponse,
    InspectParticipantTypesResponse,
)
from xngin.apiserver.settings import DatasourceConfig

# JSONBetter is JSON for most databases but JSONB for Postgres.
JSONBetter = JSON().with_variant(JSONB(), "postgresql")

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def unique_id_factory(prefix):
    def generate():
        return prefix + "_" + "".join([secrets.choice(ALPHABET) for _ in range(16)])

    return generate


class Base(DeclarativeBase):
    pass


class CacheTable(Base):
    """Stores cached values."""

    __tablename__ = "cache"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str]


class ApiKey(Base):
    """Stores API keys. Each API key grants access to a single datasource."""

    __tablename__ = "apikeys"

    id: Mapped[str] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(unique=True)
    datasource_id: Mapped[str] = mapped_column(
        ForeignKey("datasources.id", ondelete="CASCADE")
    )
    datasource: Mapped["Datasource"] = relationship(back_populates="api_keys")


class Organization(Base):
    """Represents an organization that has users and can own datasources."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=unique_id_factory("o")
    )
    name: Mapped[str] = mapped_column(String(255))

    # Relationships
    users: Mapped[list["User"]] = relationship(
        secondary="user_organizations", back_populates="organizations"
    )
    datasources: Mapped[list["Datasource"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class User(Base):
    """Represents a user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=unique_id_factory("u")
    )
    email: Mapped[str] = mapped_column(String(255), unique=True)
    # TODO: properly handle federated auth
    iss: Mapped[str | None] = mapped_column(String(255), default=None)
    sub: Mapped[str | None] = mapped_column(String(255), default=None)

    # Relationships
    organizations: Mapped[list["Organization"]] = relationship(
        secondary="user_organizations", back_populates="users"
    )


class UserOrganization(Base):
    """Maps a User to an Organization."""

    __tablename__ = "user_organizations"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )

    organization: Mapped["Organization"] = relationship(viewonly=True)
    user: Mapped["User"] = relationship(viewonly=True)


class Datasource(Base):
    """Stores a DatasourceConfig and maps it to an Organization."""

    __tablename__ = "datasources"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=unique_id_factory("ds")
    )
    name: Mapped[str] = mapped_column(String(255))
    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    config: Mapped[dict] = mapped_column(
        JSON, comment="JSON serialized form of DatasourceConfig"
    )

    table_list: Mapped[list[str] | None] = mapped_column(
        type_=JSONBetter, comment="List of table names available in this datasource"
    )
    table_list_updated: Mapped[datetime | None] = mapped_column(
        comment="Timestamp of the last update to `inspected_tables`"
    )

    organization: Mapped["Organization"] = relationship(back_populates="datasources")
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="datasource", cascade="all, delete-orphan"
    )

    def get_config(self) -> DatasourceConfig:
        """Deserializes the config field into a DatasourceConfig."""
        return TypeAdapter(DatasourceConfig).validate_python(self.config)

    def set_config(self, value: DatasourceConfig) -> Self:
        """Sets the config field to the serialized DatasourceConfig.

        Raises ValidationError if the config is invalid.
        """
        # Dump the model to JSON because this is how we can serialize the SecretStr values.
        as_json = value.model_dump_json()

        # Validate that we are persisting a valid DatasourceConfig because Pydantic only validates on model creation.
        # This will raise if there is an error.
        TypeAdapter(DatasourceConfig).validate_json(as_json)

        self.config = json.loads(as_json)
        return self

    def set_table_list(self, tables: list[str] | None) -> Self:
        if tables is None:
            self.table_list = None
            self.table_list_updated = None
        else:
            self.table_list = tables
            self.table_list_updated = datetime.now()
        return self

    def get_table_list(self) -> list[str] | None:
        return self.table_list

    def clear_table_list(self) -> Self:
        return self.set_table_list(None)


class DatasourceTablesInspected(Base):
    """Stores details of the most recent listing of tables in a datasource."""

    __tablename__ = "datasource_tables_inspected"

    datasource_id: Mapped[str] = mapped_column(
        ForeignKey("datasources.id", ondelete="CASCADE"), primary_key=True
    )
    table_name: Mapped[str] = mapped_column(primary_key=True)

    response: Mapped[dict | None] = mapped_column(
        type_=JSONBetter, comment="Serialized InspectDatasourceTablesResponse."
    )
    response_last_updated: Mapped[datetime | None] = mapped_column(
        comment="Timestamp of the last update to `response`"
    )

    def get_response(self):
        return InspectDatasourceTableResponse.model_validate(self.response)

    def set_response(self, value: InspectDatasourceTableResponse) -> Self:
        self.response = value.model_dump()
        self.response_last_updated = datetime.now()
        return self


class ParticipantTypesInspected(Base):
    """Stores details of the most recent participant type inspection (including exemplar values)."""

    __tablename__ = "participant_types_inspected"

    datasource_id: Mapped[str] = mapped_column(
        ForeignKey("datasources.id", ondelete="CASCADE"), primary_key=True
    )
    participant_type: Mapped[str] = mapped_column(primary_key=True)

    response: Mapped[dict | None] = mapped_column(
        type_=JSONBetter, comment="Serialized InspectParticipantTypesResponse."
    )
    response_last_updated: Mapped[datetime | None] = mapped_column(
        comment="Timestamp of the last update to `response`"
    )

    def get_response(self):
        return InspectParticipantTypesResponse.model_validate(self.response)

    def set_response(self, value: InspectParticipantTypesResponse) -> Self:
        self.response = value.model_dump()
        self.response_last_updated = datetime.now()
        return self

    def clear_response(self):
        self.response = None
        self.response_last_updated = None
