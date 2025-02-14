import json
import secrets

from pydantic import TypeAdapter
from sqlalchemy import ForeignKey, String, JSON
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship

from xngin.apiserver.settings import DatasourceConfig

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def newid():
    return "".join([secrets.choice(ALPHABET) for _ in range(16)])


class Base(DeclarativeBase):
    pass


class CacheTable(Base):
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
    """Represents an organization that can own datasources."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=newid)
    name: Mapped[str] = mapped_column(String(255))

    # Relationships
    users: Mapped[list["User"]] = relationship(
        secondary="user_organizations", back_populates="organizations"
    )
    datasources: Mapped[list["Datasource"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class User(Base):
    """Represents a user in the system."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=newid)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    # TODO: properly handle federated auth
    iss: Mapped[str | None] = mapped_column(String(255), default=None)
    sub: Mapped[str | None] = mapped_column(String(255), default=None)

    # Relationships
    organizations: Mapped[list["Organization"]] = relationship(
        secondary="user_organizations", back_populates="users"
    )


class UserOrganization(Base):
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
    """Represents a data source in the system.

    This contains a serialized settings.Datasource value.
    """

    __tablename__ = "datasources"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=newid)
    name: Mapped[str] = mapped_column(String(255))
    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )
    config: Mapped[dict] = mapped_column(
        JSON, comment="JSON serialized form of DatasourceConfig"
    )

    organization: Mapped["Organization"] = relationship(back_populates="datasources")
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="datasource", cascade="all, delete-orphan"
    )

    def get_config(self) -> DatasourceConfig:
        """Parses the config field and returns the Datasource.config."""
        return TypeAdapter(DatasourceConfig).validate_python(self.config)

    def set_config(self, value: DatasourceConfig):
        """Sets the config field to the serialized form of the given Datasource.config."""
        # Round-trip via JSON to serialize SecretStr values correctly.
        self.config = json.loads(value.model_dump_json())
