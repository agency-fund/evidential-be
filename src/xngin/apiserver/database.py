import os
from pathlib import Path

from sqlalchemy import create_engine, String
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.orm import sessionmaker

# TODO: replace with something that looks upwards until it finds pyproject.toml.
DEFAULT_SQLITE_DB = Path(__file__).parent.parent.parent.parent / "xngin.db"
SQLALCHEMY_DATABASE_URL = os.environ.get("XNGIN_DB", f"sqlite:///{DEFAULT_SQLITE_DB}")


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


def setup():
    Base.metadata.create_all(bind=engine)
