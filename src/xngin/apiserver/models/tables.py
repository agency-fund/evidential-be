import json
import secrets

from pydantic import TypeAdapter
from sqlalchemy import ForeignKey, String, JSON
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship

from xngin.apiserver.settings import DatasourceConfig

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

    organization: Mapped["Organization"] = relationship(back_populates="datasources")
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="datasource", cascade="all, delete-orphan"
    )

    def get_config(self) -> DatasourceConfig:
        """Deserializes the config field into a DatasourceConfig."""
        return TypeAdapter(DatasourceConfig).validate_python(self.config)

    def set_config(self, value: DatasourceConfig):
        """Sets the config field to the serialized DatasourceConfig."""
        # Round-trip via JSON to serialize SecretStr values correctly.
        self.config = json.loads(value.model_dump_json())
