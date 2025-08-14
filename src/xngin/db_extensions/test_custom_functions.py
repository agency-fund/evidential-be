from types import MappingProxyType

import pytest
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.functions import func

from xngin.db_extensions import custom_functions

# Suppress for tests that create an engine with "bigquery" dialect
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Your application has authenticated using end user credentials from Google Cloud SDK.*:UserWarning"
)


class Base(DeclarativeBase):
    pass


class SampleTable(Base):
    __tablename__ = "test_table"

    id1: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    id2: Mapped[str] = mapped_column(primary_key=True)
    string_col: Mapped[str]
    float_col: Mapped[float]


class SampleTableNoPK(Base):
    """
    Mapped class that simulates a table without any primary key.

    See:
    - https://docs.sqlalchemy.org/en/20/faq/ormconfiguration.html#how-do-i-map-a-table-that-has-no-primary-key
    - https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html#orm-imperative-table-configuration
    """

    __table__ = Table(
        "nopk_table",
        MetaData(),
        Column("string_col", String),
        Column("int_col", Integer),
    )
    __mapper_args__ = MappingProxyType({"primary_key": [__table__.c.int_col]})


# Marking these tests that use the bigquery dialect as integration-only since
# it will try to auth against google using app default credentials.
@pytest.mark.integration
def test_random_compilation_on_different_engine_dialects():
    expected_results = {
        "postgresql": "random()",
        "bigquery": "rand()",
    }

    for dialect, expected in expected_results.items():
        engine = create_engine(f"{dialect}://")
        query = select(custom_functions.Random(sa_table=SampleTable.__table__))
        sql_text = str(query.compile(engine))
        assert f"SELECT {expected}" in sql_text, f"Engine {engine.dialect}"


@pytest.mark.integration
def test_stddev_pop_compilation_on_different_engine_dialects():
    expected_results = {
        "postgresql": "SELECT STDDEV_POP(test_table.float_col) AS stddev_pop_1 FROM test_table",
        "bigquery": "SELECT stddev_pop(`test_table`.`float_col`) AS `stddev_pop_1` FROM `test_table`",
    }

    for dialect, expected in expected_results.items():
        engine = create_engine(f"{dialect}://")
        query = select(func.stddev_pop(SampleTable.float_col)).select_from(SampleTable)
        sql_text = str(query.compile(engine)).replace("\n", "")
        assert expected == sql_text, f"Engine {engine.dialect}"


def test_deterministic_random():
    """Test that deterministic random orders by primary key."""
    # We still allow use of sqlite:// here because we aren't actually testing anything SQLite related.
    engine = create_engine("sqlite://")

    # deterministic random enabled
    custom_functions.USE_DETERMINISTIC_RANDOM = True
    query = select(SampleTable).order_by(custom_functions.Random(sa_table=SampleTable.__table__))
    sql_text = str(query.compile(engine))
    assert "ORDER BY test_table.id1, test_table.id2" in sql_text

    # deterministic random enabled and no primary key
    custom_functions.USE_DETERMINISTIC_RANDOM = True
    query_nopk = select(SampleTableNoPK).order_by(custom_functions.Random(sa_table=SampleTableNoPK.__table__))
    sql_text = str(query_nopk.compile(engine))
    assert "ORDER BY nopk_table.int_col, nopk_table.string_col" in sql_text

    # normal case
    custom_functions.USE_DETERMINISTIC_RANDOM = False
    query = select(SampleTable).order_by(custom_functions.Random(sa_table=SampleTable.__table__))
    sql_text = str(query.compile(engine))
    assert "ORDER BY random()" in sql_text
