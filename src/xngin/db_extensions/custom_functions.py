from typing import Any

from sqlalchemy import Numeric, inspect
from sqlalchemy.ext import compiler
from sqlalchemy.sql._typing import _ColumnExpressionOrLiteralArgument
from sqlalchemy.sql.functions import FunctionElement, GenericFunction, func

# Set this to True to override the our_random() behavior to return a deterministic value instead.
USE_DETERMINISTIC_RANDOM = False


class stddev_pop(GenericFunction):  # noqa: N801
    """Returns the population standard deviation of the expression.
    This is a wrapper around the SQL standard function STDDEV_POP().
    In some databases, this is called STDEVP().
    """

    name = "stddev_pop"
    inherit_cache = True
    type = Numeric()


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
    """Helper to implement a deterministic random."""
    if element.sa_table is None:
        raise ValueError(
            "our_random requires sa_table= to be an inspectable table-like entity."
        )
    meta = inspect(element.sa_table)
    if meta.primary_key.columns:
        columns = (c for c in meta.primary_key.columns)
    else:
        columns = (c for c in meta.columns)
    return ", ".join(str(c) for c in sorted(columns, key=lambda c: c.name))


@compiler.compiles(Random)
def _default_random(element, _compiler, **_kw):
    """Generates RANDOM()."""
    if USE_DETERMINISTIC_RANDOM:
        return deterministic_random(element)

    return _compiler.process(func.random())


@compiler.compiles(Random, "bigquery")
def _bq_random(element, _compiler, **_kw):
    """Generates BigQuery-compatible RAND()."""
    if USE_DETERMINISTIC_RANDOM:
        return deterministic_random(element)
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/mathematical_functions#rand
    return "rand()"


@compiler.compiles(stddev_pop, "mysql")
@compiler.compiles(stddev_pop, "postgresql")
def _std_default(element, compiler, **_kw):
    return f"STDDEV_POP({compiler.process(element.clauses)})"


@compiler.compiles(stddev_pop, "mssql")
def _std_mssql(element, compiler, **_kw):
    return f"STDEVP({compiler.process(element.clauses)})"
