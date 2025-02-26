import json
import secrets
from datetime import datetime
import enum
from typing import ClassVar, Self
import uuid

from pydantic import TypeAdapter
import sqlalchemy
from sqlalchemy import ForeignKey, String, JSON
from sqlalchemy.types import TypeEngine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship

from xngin.apiserver.routers.admin_api_types import (
    InspectDatasourceTableResponse,
    InspectParticipantTypesResponse,
)
from xngin.apiserver.settings import DatasourceConfig

# JSONBetter is JSON for most databases but JSONB for Postgres.
JSONBetter = JSON().with_variant(postgresql.JSONB(), "postgresql")

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def unique_id_factory(prefix):
    def generate():
        return prefix + "_" + "".join([secrets.choice(ALPHABET) for _ in range(16)])

    return generate


class Base(DeclarativeBase):
    # See https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html#customizing-the-type-map
    type_annotation_map: ClassVar[dict[type, TypeEngine]] = {
        # For pg specifically, use the binary form
        sqlalchemy.JSON: JSONBetter,
        datetime: sqlalchemy.TIMESTAMP(timezone=True),
        # uuid.UUID: sqlalchemy.Uuid().with_variant(sqlalchemy.Uuid(as_uuid=False), "sqlite"),
    }

    def to_dict(self):
        """Quick and dirty dump to dict for debugging."""
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }


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
        sqlalchemy.JSON, comment="JSON serialized form of DatasourceConfig"
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
        """Sets the config field to the serialized DatasourceConfig."""
        # Round-trip via JSON to serialize SecretStr values correctly.
        self.config = json.loads(value.model_dump_json())
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


class ExperimentState(enum.StrEnum):
    """
    Experiment lifecycle states.

    note: [starting state], [[terminal state]]
    [DESIGNING]->ASSIGNED->{[[ABANDONED]], COMMITTED}->[[ABORTED]]
    """

    DESIGNING = "designing"
    ASSIGNED = "assigned"
    ABANDONED = "abandoned"
    COMMITTED = "committed"
    # TODO: Consider adding two more states:
    # Add an ACTIVE state that is only derived in a View when the state is COMMITTED and the query
    # time is between experiment start and end.
    # Add a COMPLETE state that is only derived in a View when the state is COMMITTED and query time
    # is after experiment end.
    ABORTED = "aborted"


class ArmAssignment(Base):
    """Stores experiment treatment assignments."""

    __tablename__ = "arm_assignments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        sqlalchemy.Uuid(), ForeignKey("experiments.id", ondelete="CASCADE")
    )
    participant_type: Mapped[str] = mapped_column(String(255))
    participant_id: Mapped[str] = mapped_column(String(255))
    arm_id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.Uuid())
    strata: Mapped[sqlalchemy.JSON]

    experiment: Mapped["Experiment"] = relationship(back_populates="arm_assignments")

    # A participant should only be assigned to one arm ever per experiment.
    __table_args__ = (
        sqlalchemy.UniqueConstraint(
            "experiment_id", "participant_id", name="uniq_participant"
        ),
    )


class Experiment(Base):
    """Stores experiment metadata."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(sqlalchemy.Uuid(), primary_key=True)
    datasource_id: Mapped[str] = mapped_column(
        String(255)
    )  # TODO: setup a proper relation on Datasource
    state: Mapped[ExperimentState]
    # We presume updates to descriptions/names/times won't happen frequently.
    # TODO: set up a GIN index if using postgres. Or, denormalize start_date/end_date/description/other editable fields. Build index on fields needed for pagination and filtering.
    design_spec: Mapped[sqlalchemy.JSON]
    audience_spec: Mapped[sqlalchemy.JSON]
    # TODO: store power analysis json
    assign_summary: Mapped[sqlalchemy.JSON]
    created_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(), onupdate=sqlalchemy.sql.func.now()
    )

    arm_assignments: Mapped[list["ArmAssignment"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
