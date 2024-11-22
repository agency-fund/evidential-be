import numpy as np


class NumpyStddev:
    """SQLite extension function to compute the standard deviation using numpy.

    Register with:
    ```
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def register_sqlite_functions(dbapi_connection, _):
            NumpyStddev.register(dbapi_connection)
    ```
    """

    SQL_FUNCTION_NAME = "stddev"

    def __init__(self):
        self.values: list[float] = []

    def step(self, value):
        if value is not None:
            self.values.append(float(value))

    def finalize(self):
        return float(np.std(self.values, ddof=1)) if self.values else None

    @classmethod
    def register(cls, dbapi_connection):
        dbapi_connection.create_aggregate(cls.SQL_FUNCTION_NAME, 1, NumpyStddev)
