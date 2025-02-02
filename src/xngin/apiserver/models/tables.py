from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship


class Base(DeclarativeBase):
    pass


class CacheTable(Base):
    __tablename__ = "cache"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str]


class ApiKeyTable(Base):
    """Stores API keys.

    API keys have a 1:M relationship with datasources.
    """

    __tablename__ = "apikeys"

    id: Mapped[str] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(unique=True)
    datasources: Mapped[list["ApiKeyDatasourceTable"]] = relationship(
        back_populates="apikey", cascade="all, delete-orphan"
    )


class ApiKeyDatasourceTable(Base):
    """Stores the list of datasources that an API key has privileges on."""

    __tablename__ = "apikey_datasources"

    apikey_id: Mapped[str] = mapped_column(
        ForeignKey("apikeys.id", ondelete="CASCADE"), primary_key=True
    )
    datasource_id: Mapped[str] = mapped_column(primary_key=True)
    apikey: Mapped[ApiKeyTable] = relationship(back_populates="datasources")


class Organization(Base):
    """Represents an organization that can own datasources."""

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    iss: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    sub: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)

    # Relationships
    organizations: Mapped[list["Organization"]] = relationship(
        secondary="user_organizations", back_populates="users"
    )


class UserOrganization(Base):
    """Association table for the many-to-many relationship between users and organizations."""

    __tablename__ = "user_organizations"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id"), primary_key=True
    )


class Datasource(Base):
    """Represents a data source in the system."""

    __tablename__ = "datasources"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE")
    )

    # Relationships
    organization: Mapped["Organization"] = relationship(back_populates="datasources")
