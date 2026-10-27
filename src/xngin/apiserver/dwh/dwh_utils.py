import unicodedata

from sqlalchemy import URL, Engine

REDSHIFT_HOSTNAME_SUFFIXES = ("redshift.amazonaws.com", "redshift-serverless.amazonaws.com")

# BigQuery table names: letters (L), marks (M), numbers (N), connectors (Pc), dashes (Pd), and spaces (Zs), up to
# 1,024 UTF-8 bytes. https://cloud.google.com/bigquery/docs/tables#table_naming
_BQ_TABLE_NAME_CATEGORIES = frozenset({"Pc", "Pd", "Zs"})
_BQ_TABLE_NAME_MAX_BYTES = 1024


def is_redshift(host_or_url: str | URL) -> bool:
    """
    Returns true iff the hostname string or URL indicates that this is connecting to Redshift.

    Will NOT work with Redshift's custom domains feature!
    """
    if isinstance(host_or_url, str):
        return host_or_url.endswith(REDSHIFT_HOSTNAME_SUFFIXES)
    return host_or_url.host is not None and host_or_url.host.endswith(REDSHIFT_HOSTNAME_SUFFIXES)


def is_bigquery(host_or_url: str | URL) -> bool:
    """Returns true iff the hostname string or URL indicates that this is connecting to BigQuery."""
    if isinstance(host_or_url, str):
        return host_or_url.startswith("bigquery")
    return host_or_url.drivername == "bigquery"


def is_postgres(driver_or_url: str | URL) -> bool:
    """Returns true iff the SQLAlchemy driver name string or URL indicates that this is connecting to Postgres."""
    if isinstance(driver_or_url, str):
        return driver_or_url.startswith("postgresql")
    return driver_or_url.drivername in {"postgresql", "postgresql+psycopg", "postgresql+psycopg2"}


def is_bigquery_table_name(name: str) -> bool:
    """Whether name is a table name BigQuery could have created, which is safe inside a backtick-quoted identifier.

    sqlalchemy-bigquery quotes identifiers with backticks but does not escape them, so a name must not contain
    anything with meaning inside backticks: the closing backtick, the backslash escape, line breaks, the dot that
    separates path components, the * that makes a wildcard table, and NUL, which BigQuery treats as the end of the
    name. None of those can appear in a real table name.
    """
    return 0 < len(name.encode()) <= _BQ_TABLE_NAME_MAX_BYTES and all(
        unicodedata.category(c)[0] in "LMN" or unicodedata.category(c) in _BQ_TABLE_NAME_CATEGORIES for c in name
    )


def extra_engine_setup(engine: Engine):
    """Do any extra configuration if needed before a connection is made."""
    # Handle Redshift incompatibilities
    if is_redshift(engine.url) and hasattr(engine.dialect, "_set_backslash_escapes"):
        engine.dialect._set_backslash_escapes = lambda _: None
