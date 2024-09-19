import os
from pathlib import Path

from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# TODO: replace with something that looks upwards until it finds pyproject.toml.
DEFAULT_DB = Path(__file__).parent.parent.parent.parent / "xngin.db"
SQLALCHEMY_DATABASE_URL = os.environ.get("XNGIN_DB", f"sqlite:///{DEFAULT_DB}")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Cache(Base):
    __tablename__ = "cache"

    key = Column(String, primary_key=True)
    value = Column(String)


def setup():
    Base.metadata.create_all(bind=engine)
