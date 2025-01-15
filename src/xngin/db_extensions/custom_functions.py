from typing import Any

import numpy as np
from sqlalchemy import inspect

from sqlalchemy.ext import compiler
from sqlalchemy.sql._typing import _ColumnExpressionOrLiteralArgument
from sqlalchemy.sql.functions import func, FunctionElement

stddev_pop = func.stddev_pop

# Set this to True to override the our_random() behavior to return a deterministic value instead.
USE_DETERMINISTIC_RANDOM = False


class Random(FunctionElement):
    """Returns a RANDOM() call compatible with the databases we use.

    When USE_DETERMINISTIC_RANDOM is True, the RANDOM is replaced by the primary key of the table passed
    via the sa_table argument.

    Also see: conftest.use_deterministic_random
    """

    name = "random"
    inherit_cache = False

    def __init__(self, *clauses: _ColumnExpressionOrLiteralArgument[Any], sa_table):
        super().__init__(*clauses)
        self.sa_table = sa_table


def deterministic_random(element):
    if element.sa_table is None:
        raise ValueError(
            "our_random requires sa_table= to be an inspectable table-like entity."
        )
    meta = inspect(element.sa_table)
    if len(meta.primary_key) == 1:
        return ", ".join(str(c) for c in meta.primary_key.columns)
    return ", ".join(str(c) for c in meta.columns)


@compiler.compiles(Random)
def our_random(element, _compiler, **_kw):
    """Generates RANDOM()."""
    if USE_DETERMINISTIC_RANDOM:
        return deterministic_random(element)

    return "RANDOM()"


@compiler.compiles(Random, "bigquery")
def _bq_random(element, _compiler, **_kw):
    """Generates BigQuery-compatible RAND()."""
    if USE_DETERMINISTIC_RANDOM:
        return deterministic_random(element)
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/mathematical_functions#rand
    return "RAND()"


@compiler.compiles(stddev_pop, "mysql")
@compiler.compiles(stddev_pop, "postgresql")
def _std_default(element, compiler, **_kw):
    return f"STDDEV_POP({compiler.process(element.clauses)})"


@compiler.compiles(stddev_pop, "mssql")
def _std_mssql(element, compiler, **_kw):
    return f"STDEVP({compiler.process(element.clauses)})"


class NumpyStddev:
    """SQLite extension function to compute the standard deviation (population) using numpy.

    This only needs to be registered on SQLite connections.

    Register with:
    ```
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def register_sqlite_functions(dbapi_connection, _):
            NumpyStddev.register(dbapi_connection)
    ```
    """

    SQL_FUNCTION_NAME = "stddev_pop"

    def __init__(self):
        self.values: list[float] = []

    def step(self, value):
        if value is not None:
            self.values.append(float(value))

    def finalize(self):
        return float(np.std(self.values, ddof=0)) if self.values else None

    @classmethod
    def register(cls, dbapi_connection):
        dbapi_connection.create_aggregate(cls.SQL_FUNCTION_NAME, 1, NumpyStddev)
