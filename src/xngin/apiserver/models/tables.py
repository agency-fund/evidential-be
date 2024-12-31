from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship


class Base(DeclarativeBase):
    pass


class Cache(Base):
    __tablename__ = "cache"

    key = mapped_column(String, primary_key=True)
    value = mapped_column(String)


class ApiKey(Base):
    """Stores API keys.

    API keys have a 1:M relationship with datasources.
    """

    __tablename__ = "apikeys"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)

    key: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    datasources: Mapped[list["ApiKeyDatasource"]] = relationship(
        "ApiKeyDatasource", back_populates="apikey", cascade="all, delete-orphan"
    )


class ApiKeyDatasource(Base):
    """Stores the list of datasources that an API key has privileges on."""

    __tablename__ = "apikey_datasources"

    apikey_id: Mapped[str] = mapped_column(
        ForeignKey("apikeys.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )

    datasource_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)

    apikey: Mapped[ApiKey] = relationship(ApiKey, back_populates="datasources")
