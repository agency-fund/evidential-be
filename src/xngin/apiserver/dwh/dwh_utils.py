from sqlalchemy import URL, Engine

REDSHIFT_HOSTNAME_SUFFIXES = ("redshift.amazonaws.com", "redshift-serverless.amazonaws.com")


def is_redshift(host_or_url: str | URL) -> bool:
    """Returns true iff the hostname string or URL indicates that this is connecting to Redshift."""
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


def extra_engine_setup(engine: Engine):
    """Do any extra configuration if needed before a connection is made."""
    # Handle Redshift incompatibilities
    if is_redshift(engine.url) and hasattr(engine.dialect, "_set_backslash_escapes"):
        engine.dialect._set_backslash_escapes = lambda _: None
