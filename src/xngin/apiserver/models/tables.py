import json
import secrets
from datetime import UTC, datetime
from typing import ClassVar, Self
from uuid import UUID

import sqlalchemy
from pydantic import TypeAdapter
from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine
from xngin.apiserver.api_types import (
    Arm,
    AudienceSpec,
    BalanceCheck,
    DesignSpec,
    PowerResponse,
)
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.routers.admin_api_types import (
    InspectDatasourceTableResponse,
    InspectParticipantTypesResponse,
)
from xngin.apiserver.routers.experiments_api_types import AssignSummary
from xngin.apiserver.settings import DatasourceConfig

# JSONBetter is JSON for most databases but JSONB for Postgres.
JSONBetter = sqlalchemy.JSON().with_variant(postgresql.JSONB(), "postgresql")

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
    events: Mapped[list["Event"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class Event(Base):
    """Represents events that occur in an organization."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(primary_key=True, default=unique_id_factory("evt"))
    created_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    type: Mapped[str] = mapped_column(
        comment="The type of event. E.g. `experiment.created`"
    )
    data: Mapped[dict] = mapped_column(
        type_=JSONBetter,
        comment="The event payload. This will always be a JSON object with a `type` field.",
    )

    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    organization: Mapped["Organization"] = relationship(back_populates="events")
    tasks: Mapped[list["Task"]] = relationship(back_populates="event", cascade="all, delete-orphan")

    __table_args__ = (Index("event_stream", "organization_id", created_at),)


class Task(Base):
    """Represents a task in the task queue."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(primary_key=True, default=unique_id_factory("task"))
    created_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(),
        onupdate=sqlalchemy.sql.func.now(),
    )
    task_type: Mapped[str] = mapped_column(
        comment="The type of task. E.g. `event.created`"
    )
    embargo_until: Mapped[datetime | None] = mapped_column(
        comment="If set, the task will not be processed until after this time."
    )
    retry_count: Mapped[int] = mapped_column(
        server_default="0",
        comment="Number of times this task has been retried."
    )
    payload: Mapped[dict | None] = mapped_column(
        type_=JSONBetter,
        comment="The task payload. This will be a JSON object with task-specific data.",
    )
    event_id: Mapped[str | None] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"),
        comment="Optional reference to an event that triggered this task."
    )

    event: Mapped["Event | None"] = relationship(back_populates="tasks")

    __table_args__ = (
        Index("idx_tasks_embargo", "embargo_until"),
        Index("idx_tasks_type", "task_type"),
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
        type_=JSONBetter, comment="JSON serialized form of DatasourceConfig"
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
    experiments: Mapped[list["Experiment"]] = relationship(
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
            self.table_list_updated = datetime.now(UTC)
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
        self.response_last_updated = datetime.now(UTC)
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
        # This value may contain Python datetime objects. The default JSON serializer doesn't serialize them
        # but the Pydantic serializer turns them into ISO8601 strings. This could be better.
        self.response = json.loads(value.model_dump_json())
        self.response_last_updated = datetime.now(UTC)
        return self

    def clear_response(self):
        self.response = None
        self.response_last_updated = None


class ArmAssignment(Base):
    """Stores experiment treatment assignments."""

    __tablename__ = "arm_assignments"

    experiment_id: Mapped[UUID] = mapped_column(
        sqlalchemy.Uuid(as_uuid=False),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        primary_key=True,
    )
    participant_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    participant_type: Mapped[str] = mapped_column(String(255))
    arm_id: Mapped[UUID] = mapped_column(sqlalchemy.Uuid(as_uuid=False))
    strata: Mapped[sqlalchemy.JSON] = mapped_column(
        comment="JSON serialized form of a list of Strata objects (from Assignment.strata)."
    )

    experiment: Mapped["Experiment"] = relationship(back_populates="arm_assignments")

    def strata_names(self):
        return [s["field_name"] for s in self.strata]

    def strata_values(self):
        return [s["strata_value"] for s in self.strata]


class Experiment(Base):
    """Stores experiment metadata."""

    __tablename__ = "experiments"

    id: Mapped[UUID] = mapped_column(sqlalchemy.Uuid(as_uuid=False), primary_key=True)
    datasource_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("datasources.id", ondelete="CASCADE")
    )
    state: Mapped[ExperimentState]
    start_date: Mapped[datetime] = mapped_column(
        comment="Target start date of the experiment. Denormalized from design_spec."
    )
    end_date: Mapped[datetime] = mapped_column(
        comment="Target end date of the experiment. Denormalized from design_spec."
    )
    # We presume updates to descriptions/names/times won't happen frequently.
    design_spec: Mapped[sqlalchemy.JSON] = mapped_column(
        comment="JSON serialized form of DesignSpec."
    )
    audience_spec: Mapped[sqlalchemy.JSON] = mapped_column(
        comment="JSON serialized form of AudienceSpec."
    )
    power_analyses: Mapped[sqlalchemy.JSON | None] = mapped_column(
        comment="JSON serialized form of a PowerResponse. Not required since some experiments may not have data to run power analyses."
    )
    assign_summary: Mapped[sqlalchemy.JSON] = mapped_column(
        comment="JSON serialized form of AssignSummary."
    )
    created_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(), onupdate=sqlalchemy.sql.func.now()
    )

    arm_assignments: Mapped[list["ArmAssignment"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )

    datasource: Mapped["Datasource"] = relationship(back_populates="experiments")

    def get_arms(self) -> list[Arm]:
        ds = self.get_design_spec()
        return ds.arms

    def get_arm_ids(self) -> list[UUID]:
        return [arm.arm_id for arm in self.get_arms()]

    def get_arm_names(self) -> list[str]:
        return [arm.arm_name for arm in self.get_arms()]

    def get_design_spec(self) -> DesignSpec:
        return TypeAdapter(DesignSpec).validate_python(self.design_spec)

    def get_audience_spec(self) -> AudienceSpec:
        return TypeAdapter(AudienceSpec).validate_python(self.audience_spec)

    def get_power_analyses(self) -> PowerResponse | None:
        if self.power_analyses is None:
            return None
        return TypeAdapter(PowerResponse).validate_python(self.power_analyses)

    def get_assign_summary(self) -> AssignSummary:
        return TypeAdapter(AssignSummary).validate_python(self.assign_summary)

    def get_balance_check(self) -> BalanceCheck:
        return TypeAdapter(BalanceCheck).validate_python(
            self.assign_summary["balance_check"]
        )
