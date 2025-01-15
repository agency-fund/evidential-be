import numpy as np
from sqlalchemy import inspect, ColumnCollection

from sqlalchemy.ext import compiler
from sqlalchemy.sql.functions import func

stddev_pop = func.stddev_pop

# Set this to True to override the our_random() behavior to return a deterministic value instead.
USE_DETERMINISTIC_RANDOM = False


def our_random(sa_table=None):
    """Returns a RANDOM() call.

    When USE_DETERMINISTIC_RANDOM is True, the RANDOM is replaced by the primary key of the table passed
    via the sa_table argument.

    Also see: conftest.use_deterministic_random
    """
    if USE_DETERMINISTIC_RANDOM:
        if sa_table is None:
            raise ValueError(
                "our_random requires sa_table= to be an inspectable table-like entity."
            )
        # Find a suitable key (or keys) to order by.
        meta = inspect(sa_table)
        if len(meta.primary_key) > 0:
            return ColumnCollection(
                columns=[(c.name, c) for c in meta.columns.values() if c.primary_key]
            )
        # If we can't order by a single primary key, order by all the columns.
        return ColumnCollection(columns=list(sorted(meta.columns.items())))
    return func.random()


@compiler.compiles(our_random, "bigquery")
def _bq_random(_element, _compiler, **_kw):
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
