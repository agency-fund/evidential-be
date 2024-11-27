import numpy as np
from sqlalchemy import inspect

from sqlalchemy.ext import compiler
from sqlalchemy.sql.functions import func

stddev_pop = func.stddev_pop

# Hack: Allow the conftest setup function to let us use a deterministic random function for sorting.
TESTING = False


def our_random(sa_table=None):
    """Returns a RANDOM() call.

    When in a unit test, this returns the value returned by expr_for_tests().
    """
    if TESTING:
        return inspect(sa_table).primary_key[0]
    return func.random()


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
