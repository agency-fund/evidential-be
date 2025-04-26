import json
import secrets
from datetime import UTC, datetime
from typing import ClassVar, Self
import uuid

import sqlalchemy
from pydantic import TypeAdapter
from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine
from xngin.apiserver.stateless_api_types import (
    Arm,
    ArmSize,
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
from xngin.events import EventDataTypes

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
    arms: Mapped[list["ArmTable"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    users: Mapped[list["User"]] = relationship(
        secondary="user_organizations", back_populates="organizations"
    )
    datasources: Mapped[list["Datasource"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    events: Mapped[list["Event"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    webhooks: Mapped[list["Webhook"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class Webhook(Base):
    """Represents an API webhook.

    The bodies of the outbound webhooks are defined by types in src.xngin.apiserver.webhooks.
    """

    __tablename__ = "webhooks"

    id: Mapped[str] = mapped_column(primary_key=True, default=unique_id_factory("wh"))
    # The type of webhook; e.g. experiment.created. These are user-visible arbitrary strings.
    type: Mapped[str] = mapped_column()
    # The URL to post the event to. The payload body depends on the type of webhook.
    url: Mapped[str] = mapped_column()
    # The token that will be sent in the Authorization header.
    auth_token: Mapped[str | None] = mapped_column()

    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    organization: Mapped["Organization"] = relationship(back_populates="webhooks")


class Event(Base):
    """Represents events that occur in an organization.

    All .data values should correspond to a Pydantic type defined in the xngin.events module.
    """

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(primary_key=True, default=unique_id_factory("evt"))
    created_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    # The type of event. E.g. `experiment.created`
    type: Mapped[str] = mapped_column()
    # The event payload. This will always be a JSON object with a `type` field.
    data: Mapped[dict] = mapped_column(
        type_=JSONBetter,
    )

    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    organization: Mapped["Organization"] = relationship(back_populates="events")

    def set_data(self, data: EventDataTypes):
        as_json = data.model_dump_json()
        TypeAdapter(EventDataTypes).validate_json(as_json)
        self.data = json.loads(as_json)
        return self

    def get_data(self) -> EventDataTypes | None:
        if self.data is None:
            return None
        return TypeAdapter(EventDataTypes).validate_python(self.data)

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
    # The type of task. E.g. `experiment.created`
    task_type: Mapped[str] = mapped_column()
    # Status of the task: 'pending', 'running', 'success', or 'dead'.
    status: Mapped[str] = mapped_column(server_default="pending")
    # Time until which the task should not be processed. Defaults to created_at.
    embargo_until: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    # Number of times this task has been retried.
    retry_count: Mapped[int] = mapped_column(server_default="0")
    # The task payload. This will be a JSON object with task-specific data.
    payload: Mapped[dict | None] = mapped_column(type_=JSONBetter)
    # An optional informative message about the state of this task.
    message: Mapped[str | None] = mapped_column()

    __table_args__ = (Index("idx_tasks_embargo", "embargo_until"),)


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

    # True when this user is considered to be privileged.
    is_privileged: Mapped[bool] = mapped_column(server_default=sqlalchemy.sql.false())

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
    # JSON serialized form of DatasourceConfig
    config: Mapped[dict] = mapped_column(type_=JSONBetter)

    # List of table names available in this datasource
    table_list: Mapped[list[str] | None] = mapped_column(type_=JSONBetter)
    # Timestamp of the last update to `inspected_tables`
    table_list_updated: Mapped[datetime | None] = mapped_column()

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

    # Serialized InspectDatasourceTablesResponse.
    response: Mapped[dict | None] = mapped_column(type_=JSONBetter)
    # Timestamp of the last update to `response`
    response_last_updated: Mapped[datetime | None] = mapped_column()

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

    # Serialized InspectParticipantTypesResponse.
    response: Mapped[dict | None] = mapped_column(type_=JSONBetter)
    # Timestamp of the last update to `response`
    response_last_updated: Mapped[datetime | None] = mapped_column()

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

    experiment_id: Mapped[str] = mapped_column(
        String(length=36),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        primary_key=True,
    )
    participant_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    participant_type: Mapped[str] = mapped_column(String(255))
    arm_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("arms.id", ondelete="CASCADE")
    )
    # JSON serialized form of a list of Strata objects (from Assignment.strata).
    strata: Mapped[list[dict[str, str]]] = mapped_column(type_=JSONBetter)

    experiment: Mapped["Experiment"] = relationship(back_populates="arm_assignments")
    arm: Mapped["ArmTable"] = relationship(back_populates="arm_assignments")

    def strata_names(self) -> list[str]:
        """Returns the names of the strata fields."""
        return [s["field_name"] for s in self.strata]

    def strata_values(self) -> list[str]:
        """Returns the values of the strata fields as strings."""
        return [s["strata_value"] for s in self.strata]


class Experiment(Base):
    """Stores experiment metadata."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    datasource_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("datasources.id", ondelete="CASCADE")
    )
    experiment_type: Mapped[str] = mapped_column(
        comment="Should be one of the ExperimentType literals."
    )
    participant_type: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    # Describe your experiment and hypothesis here.
    description: Mapped[str] = mapped_column(String(2000))
    state: Mapped[ExperimentState]
    # Target start date of the experiment. Denormalized from design_spec.
    start_date: Mapped[datetime] = mapped_column()
    # Target end date of the experiment. Denormalized from design_spec.
    end_date: Mapped[datetime] = mapped_column()

    # JSON serialized form of DesignSpec.
    design_spec: Mapped[dict] = mapped_column(type_=JSONBetter)
    # JSON serialized form of AudienceSpec.
    audience_spec: Mapped[dict] = mapped_column(type_=JSONBetter)
    # JSON serialized form of a PowerResponse. Not required since some experiments may not have data to run power analyses.
    power_analyses: Mapped[dict | None] = mapped_column(type_=JSONBetter)
    # JSON serialized form of AssignSummary.
    assign_summary: Mapped[dict] = mapped_column(type_=JSONBetter)
    created_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(), onupdate=sqlalchemy.sql.func.now()
    )

    arm_assignments: Mapped[list["ArmAssignment"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    arms: Mapped[list["ArmTable"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    datasource: Mapped["Datasource"] = relationship(back_populates="experiments")

    def get_arms(self) -> list[Arm]:
        ds = self.get_design_spec()
        return ds.arms

    def get_arm_ids(self) -> list[str]:
        return [str(arm.arm_id) for arm in self.get_arms()]

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
        """Constructs an AssignSummary from the experiment's arms and arm_assignments."""
        balance_check = self.get_balance_check()
        arm_sizes = [
            ArmSize(
                arm=Arm(arm_id=uuid.UUID(arm.id), arm_name=arm.name),
                size=len(arm.arm_assignments),
            )
            for arm in self.arms
        ]
        return AssignSummary(
            balance_check=balance_check,
            arm_sizes=arm_sizes,
            sample_size=sum(arm_size.size for arm_size in arm_sizes),
        )

    def get_balance_check(self) -> BalanceCheck | None:
        if self.assign_summary is not None:
            return TypeAdapter(BalanceCheck).validate_python(
                self.assign_summary["balance_check"]
            )
        return None


class ArmTable(Base):
    """Representation of arms of an experiment."""

    __tablename__ = "arms"
    # TODO: Ensure arm names are unique within an organization
    #       Do this as part of Issue #278; will need to backfill in such a way that old arms are
    #       made unique e.g. suffixing with a part of the experiment id.
    # __table_args__ = (
    #     sqlalchemy.UniqueConstraint("name", "organization_id", name="uix_arm_name_org"),
    # )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(String(2000))
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="CASCADE")
    )
    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(), onupdate=sqlalchemy.sql.func.now()
    )

    organization: Mapped["Organization"] = relationship(back_populates="arms")
    experiment: Mapped["Experiment"] = relationship(back_populates="arms")
    arm_assignments: Mapped[list["ArmAssignment"]] = relationship(
        back_populates="arm", cascade="all, delete-orphan"
    )
