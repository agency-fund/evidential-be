import os
from pathlib import Path

from sqlalchemy import create_engine, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship
from sqlalchemy.orm import sessionmaker

DEFAULT_POSTGRES_DIALECT = "postgresql+psycopg"

# TODO: replace with something that looks upwards until it finds pyproject.toml.
DEFAULT_SQLITE_DB = Path(__file__).parent.parent.parent.parent / "xngin.db"


def get_server_database_url():
    """Gets a SQLAlchemy-compatible URL string from the environment."""
    # Hosting providers may set hosted database URL as DATABASE_URL.
    if database_url := os.environ.get("DATABASE_URL"):
        return generic_url_to_sa_url(database_url)
    if xngin_db := os.environ.get("XNGIN_DB"):
        return xngin_db
    return f"sqlite:///{DEFAULT_SQLITE_DB}"


def generic_url_to_sa_url(database_url):
    """Converts postgres:// to a SQLAlchemy-compatible value that includes a dialect."""
    if database_url.startswith("postgres://"):
        database_url = (
            DEFAULT_POSTGRES_DIALECT + "://" + database_url[len("postgres://") :]
        )
    return database_url


SQLALCHEMY_DATABASE_URL = get_server_database_url()


def get_connect_args():
    default = {}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite:"):
        default.update({"check_same_thread": False})
    return default


engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=get_connect_args())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Cache(Base):
    __tablename__ = "cache"

    # TODO: handle non-sqlite SQLALCHEMY_DATABASE_URLs
    key = mapped_column(String, primary_key=True)
    value = mapped_column(String)


class ApiKey(Base):
    __tablename__ = "apikeys"

    id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)

    key: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    datasources: Mapped[list["ApiKeyDatasource"]] = relationship(
        back_populates="apikey", cascade="all, delete-orphan"
    )


class ApiKeyDatasource(Base):
    __tablename__ = "apikey_datasources"

    apikey_id: Mapped[str] = mapped_column(
        ForeignKey("apikeys.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )

    datasource_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)

    apikey: Mapped[ApiKey] = relationship(back_populates="datasources")


def setup():
    Base.metadata.create_all(bind=engine)
