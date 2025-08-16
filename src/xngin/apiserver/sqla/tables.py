"""Defines our app db tables and models using the SQLAlchemy ORM."""

import json
import secrets
from datetime import UTC, datetime
from typing import ClassVar, Self

import sqlalchemy
from pydantic import TypeAdapter
from sqlalchemy import Float, ForeignKey, Index, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine

from xngin.apiserver.settings import DatasourceConfig, EncryptedDsn
from xngin.events import EventDataTypes

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def unique_id_factory(prefix: str):
    def generate() -> str:
        return prefix + "_" + "".join([secrets.choice(ALPHABET) for _ in range(16)])

    return generate


arm_id_factory = unique_id_factory("arm")
datasource_id_factory = unique_id_factory("ds")
event_id_factory = unique_id_factory("evt")
experiment_id_factory = unique_id_factory("exp")
organization_id_factory = unique_id_factory("o")
task_id_factory = unique_id_factory("task")
user_id_factory = unique_id_factory("u")
webhook_id_factory = unique_id_factory("wh")


class Base(AsyncAttrs, DeclarativeBase):
    # See https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html#customizing-the-type-map
    type_annotation_map: ClassVar[dict[type, TypeEngine]] = {
        datetime: sqlalchemy.TIMESTAMP(timezone=True),
    }

    def to_dict(self):
        """Quick and dirty dump to dict for debugging."""
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


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
    datasource_id: Mapped[str] = mapped_column(ForeignKey("datasources.id", ondelete="CASCADE"))
    datasource: Mapped["Datasource"] = relationship(back_populates="api_keys")


class Organization(Base):
    """Represents an organization that has users and can own datasources."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(primary_key=True, default=organization_id_factory)
    name: Mapped[str] = mapped_column(String(255))

    arms: Mapped[list["Arm"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    users: Mapped[list["User"]] = relationship(secondary="user_organizations", back_populates="organizations")
    datasources: Mapped[list["Datasource"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    webhooks: Mapped[list["Webhook"]] = relationship(back_populates="organization", cascade="all, delete-orphan")


class Webhook(Base):
    """Represents an API webhook.

    The bodies of the outbound webhooks are defined by types in src.xngin.apiserver.webhooks.
    """

    __tablename__ = "webhooks"

    id: Mapped[str] = mapped_column(primary_key=True, default=webhook_id_factory)
    # User-friendly name for the webhook
    name: Mapped[str] = mapped_column(server_default="")
    # The type of webhook; e.g. experiment.created. These are user-visible arbitrary strings.
    type: Mapped[str] = mapped_column()
    # The URL to post the event to. The payload body depends on the type of webhook.
    url: Mapped[str] = mapped_column()
    # The token that will be sent in the Webhook-Token header.
    auth_token: Mapped[str | None] = mapped_column()

    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    organization: Mapped["Organization"] = relationship(back_populates="webhooks")
    experiments: Mapped[list["Experiment"]] = relationship(secondary="experiment_webhooks", back_populates="webhooks")


class Event(Base):
    """Represents events that occur in an organization.

    All .data values should correspond to a Pydantic type defined in the xngin.events module.
    """

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(primary_key=True, default=event_id_factory)
    created_at: Mapped[datetime] = mapped_column(server_default=sqlalchemy.sql.func.now())
    # The type of event. E.g. `experiment.created`
    type: Mapped[str] = mapped_column()
    # The event payload. This will always be a JSON object with a `type` field.
    data: Mapped[dict] = mapped_column(postgresql.JSONB)

    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
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

    id: Mapped[str] = mapped_column(primary_key=True, default=task_id_factory)
    created_at: Mapped[datetime] = mapped_column(server_default=sqlalchemy.sql.func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(),
        onupdate=sqlalchemy.sql.func.now(),
    )
    # The type of task. E.g. `experiment.created`
    task_type: Mapped[str] = mapped_column()
    # Status of the task: 'pending', 'running', 'success', or 'dead'.
    status: Mapped[str] = mapped_column(server_default="pending")
    # Time until which the task should not be processed. Defaults to created_at.
    embargo_until: Mapped[datetime] = mapped_column(server_default=sqlalchemy.sql.func.now())
    # Number of times this task has been retried.
    retry_count: Mapped[int] = mapped_column(server_default="0")
    # The task payload. This will be a JSON object with task-specific data.
    payload: Mapped[dict | None] = mapped_column(postgresql.JSONB)
    # An optional informative message about the state of this task.
    message: Mapped[str | None] = mapped_column()

    __table_args__ = (Index("idx_tasks_embargo", "embargo_until"),)


class User(Base):
    """Represents a user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(primary_key=True, default=user_id_factory)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    # TODO: properly handle federated auth
    iss: Mapped[str | None] = mapped_column(String(255))
    sub: Mapped[str | None] = mapped_column(String(255))

    # True when this user is considered to be privileged.
    is_privileged: Mapped[bool] = mapped_column(server_default=sqlalchemy.sql.false())

    organizations: Mapped[list["Organization"]] = relationship(secondary="user_organizations", back_populates="users")


class UserOrganization(Base):
    """Maps a User to an Organization."""

    __tablename__ = "user_organizations"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True)

    organization: Mapped["Organization"] = relationship(viewonly=True)
    user: Mapped["User"] = relationship(viewonly=True)


class ExperimentWebhook(Base):
    """Maps an Experiment to a Webhook for many-to-many relationship."""

    __tablename__ = "experiment_webhooks"

    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), primary_key=True)
    webhook_id: Mapped[str] = mapped_column(ForeignKey("webhooks.id", ondelete="CASCADE"), primary_key=True)

    experiment: Mapped["Experiment"] = relationship(viewonly=True)
    webhook: Mapped["Webhook"] = relationship(viewonly=True)


class Datasource(Base):
    """Stores a DatasourceConfig and maps it to an Organization.

    When creating a Datasource entity, take care to manually set the id column value before calling .set_config(). This
    is important because we need the primary key before we encrypt the datasource config.
    """

    __tablename__ = "datasources"

    id: Mapped[str] = mapped_column(primary_key=True, default=datasource_id_factory)
    name: Mapped[str] = mapped_column(String(255))
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    # JSON serialized form of DatasourceConfig
    config: Mapped[dict] = mapped_column(postgresql.JSONB)

    # List of table names available in this datasource
    table_list: Mapped[list[str] | None] = mapped_column(postgresql.JSONB)
    # Timestamp of the last update to `inspected_tables`
    table_list_updated: Mapped[datetime | None] = mapped_column()

    organization: Mapped["Organization"] = relationship(back_populates="datasources")
    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="datasource", cascade="all, delete-orphan")
    experiments: Mapped[list["Experiment"]] = relationship(back_populates="datasource", cascade="all, delete-orphan")

    def get_config(self) -> DatasourceConfig:
        """Deserializes the config field into a DatasourceConfig."""
        config: DatasourceConfig = TypeAdapter(DatasourceConfig).validate_python(self.config)
        if isinstance(config.dwh, EncryptedDsn):
            config = config.model_copy(update={"dwh": config.dwh.decrypt(self.id)})
        return config

    def set_config(self, config: DatasourceConfig) -> Self:
        """Sets the config field to the serialized DatasourceConfig.

        Raises ValidationError if the config is invalid.
        """
        if isinstance(config.dwh, EncryptedDsn):
            config = config.model_copy(update={"dwh": config.dwh.encrypt(self.id)})

        # Round-trip the new model through validation so that we can validate it before committing it to the database.
        # This will raise if there is an error.
        TypeAdapter(DatasourceConfig).validate_python(config.model_dump())
        self.config = config.model_dump()
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

    datasource_id: Mapped[str] = mapped_column(ForeignKey("datasources.id", ondelete="CASCADE"), primary_key=True)
    table_name: Mapped[str] = mapped_column(primary_key=True)

    # Serialized InspectDatasourceTablesResponse.
    response: Mapped[dict | None] = mapped_column(postgresql.JSONB)
    # Timestamp of the last update to `response`
    response_last_updated: Mapped[datetime | None] = mapped_column()


class ParticipantTypesInspected(Base):
    """Stores details of the most recent participant type inspection (including exemplar values)."""

    __tablename__ = "participant_types_inspected"

    datasource_id: Mapped[str] = mapped_column(ForeignKey("datasources.id", ondelete="CASCADE"), primary_key=True)
    participant_type: Mapped[str] = mapped_column(primary_key=True)

    # Serialized InspectParticipantTypesResponse.
    response: Mapped[dict | None] = mapped_column(postgresql.JSONB)
    # Timestamp of the last update to `response`
    response_last_updated: Mapped[datetime | None] = mapped_column()

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
    arm_id: Mapped[str] = mapped_column(String(36), ForeignKey("arms.id", ondelete="CASCADE"))
    # JSON serialized form of a list of Strata objects (from Assignment.strata).
    strata: Mapped[list[dict[str, str]]] = mapped_column(postgresql.JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=sqlalchemy.sql.func.now())

    experiment: Mapped["Experiment"] = relationship(back_populates="arm_assignments")
    arm: Mapped["Arm"] = relationship(back_populates="arm_assignments")

    def strata_names(self) -> list[str]:
        """Returns the names of the strata fields."""
        return [s["field_name"] for s in self.strata]

    def strata_values(self) -> list[str]:
        """Returns the values of the strata fields as strings."""
        return [s["strata_value"] for s in self.strata]


class Experiment(Base):
    """Stores experiment metadata.

    Use the ExperimentStorageConverter to set/get the different JSONB columns with the appropriate
    storage models, as well as derive other API types from the Experiment db record.
    """

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    datasource_id: Mapped[str] = mapped_column(String(255), ForeignKey("datasources.id", ondelete="CASCADE"))

    experiment_type: Mapped[str] = mapped_column()
    participant_type: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    # Describe your experiment and hypothesis here.
    description: Mapped[str] = mapped_column(String(2000))
    # The experiment state should be one of xngin.apiserver.routers.common_enums.ExperimentState.
    state: Mapped[str]
    # Target start date of the experiment. Denormalized from design_spec.
    start_date: Mapped[datetime] = mapped_column()
    # Target end date of the experiment. Denormalized from design_spec.
    end_date: Mapped[datetime] = mapped_column()
    # The timestamp when experiment assignment was stopped. New participants cannot be assigned.
    stopped_assignments_at: Mapped[datetime | None] = mapped_column()
    # The reason assignments were stopped. See xngin.apiserver.routers.common_enums.StopAssignmentReason.
    stopped_assignments_reason: Mapped[str | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=sqlalchemy.sql.func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(), onupdate=sqlalchemy.sql.func.now()
    )
    n_trials: Mapped[int] = mapped_column(server_default="0")

    # Bandit config params
    prior_type: Mapped[str | None] = mapped_column()
    reward_type: Mapped[str | None] = mapped_column()

    # Frequentist config params
    # JSON serialized form of an experiment's specified dwh fields used for strata/metrics/filters.
    design_spec_fields: Mapped[dict | None] = mapped_column(postgresql.JSONB)
    # JSON serialized form of a PowerResponse. Not required since some experiments may not have data to run
    # power analyses.
    power_analyses: Mapped[dict | None] = mapped_column(postgresql.JSONB)
    # JSON serialized form of a BalanceCheck. May be null if the experiment type doesn't support
    # balance checks.
    balance_check: Mapped[dict | None] = mapped_column(postgresql.JSONB)

    # Frequentist experiment types i.e. online and preassigned
    power: Mapped[float | None] = mapped_column()
    alpha: Mapped[float | None] = mapped_column()
    fstat_thresh: Mapped[float | None] = mapped_column()

    arm_assignments: Mapped[list["ArmAssignment"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", lazy="raise"
    )
    arms: Mapped[list["Arm"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")
    datasource: Mapped["Datasource"] = relationship(back_populates="experiments")
    webhooks: Mapped[list["Webhook"]] = relationship(secondary="experiment_webhooks", back_populates="experiments")
    draws: Mapped[list["Draw"]] = relationship(
        "Draw",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )
    contexts: Mapped[list["Context"]] = relationship(
        "Context",
        back_populates="experiment",
        cascade="all, delete-orphan",
        primaryjoin="and_(Experiment.id==Context.experiment_id," + "Experiment.experiment_type=='cmab')",
    )


class Arm(Base):
    """Representation of arms of an experiment."""

    __tablename__ = "arms"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(String(2000))
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(server_default=sqlalchemy.sql.func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=sqlalchemy.sql.func.now(), onupdate=sqlalchemy.sql.func.now()
    )

    # Prior variables
    mu_init: Mapped[float | None] = mapped_column()
    sigma_init: Mapped[float | None] = mapped_column()
    mu: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    covariance: Mapped[list[list[float]] | None] = mapped_column(ARRAY(Float))

    alpha_init: Mapped[float | None] = mapped_column()
    beta_init: Mapped[float | None] = mapped_column()
    alpha: Mapped[float | None] = mapped_column()
    beta: Mapped[float | None] = mapped_column()

    organization: Mapped["Organization"] = relationship(back_populates="arms")
    experiment: Mapped["Experiment"] = relationship(back_populates="arms")
    arm_assignments: Mapped[list["ArmAssignment"]] = relationship(back_populates="arm", cascade="all, delete-orphan")
    draws: Mapped[list["Draw"]] = relationship(
        "Draw",
        back_populates="arm",
        cascade="all, delete-orphan",
    )


class Draw(Base):
    """
    Base model for draws.
    """

    __tablename__ = "draws"

    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), primary_key=True)
    participant_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    created_at: Mapped[datetime] = mapped_column(server_default=sqlalchemy.sql.func.now())
    observed_at: Mapped[datetime | None] = mapped_column()
    observation_type: Mapped[str | None] = mapped_column()

    # Observation data
    participant_type: Mapped[str] = mapped_column(String(255))
    arm_id: Mapped[str] = mapped_column(ForeignKey("arms.id", ondelete="CASCADE"))
    outcome: Mapped[float | None] = mapped_column()
    context_val: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    current_mu: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    current_covariance: Mapped[list[list[float]] | None] = mapped_column(ARRAY(Float))
    current_alpha: Mapped[float | None] = mapped_column()
    current_beta: Mapped[float | None] = mapped_column()

    arm: Mapped[Arm] = relationship("Arm", back_populates="draws", lazy="joined")
    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="draws", lazy="joined")


class Context(Base):
    """
    ORM for managing context for an experiment
    """

    __tablename__ = "context"

    id: Mapped[str] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(String(2000))
    value_type: Mapped[str] = mapped_column()

    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="contexts")
