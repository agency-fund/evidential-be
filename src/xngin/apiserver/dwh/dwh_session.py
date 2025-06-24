"""Data warehouse session context manager for database connections."""

import google.api_core.exceptions
import sqlalchemy
from loguru import logger
from sqlalchemy import Engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.dns.safe_resolve import safe_resolve
from xngin.apiserver.settings import (
    SA_LOGGER_NAME_FOR_DWH,
    TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES,
    Dwh,
    infer_table,
    safe_url,
)


class DwhDatabaseDoesNotExistError(Exception):
    """Raised when the target database or dataset does not exist."""


class DwhSession:
    """Context manager for data warehouse database connections.

    Encapsulates database connection logic and provides convenient access
    to both SQLAlchemy Session and Engine objects.
    """

    def __init__(self, dwh_config: Dwh):
        """Initialize with data warehouse configuration.

        Args:
            dwh_config: The data warehouse configuration (Dsn or BqDsn)
        """
        self.dwh_config = dwh_config
        self._engine = None
        self._session = None

    def __enter__(self) -> "DwhSession":
        """Enter the context manager and create database connections."""
        self._engine = self._create_engine()
        self._session = Session(self._engine)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up database connections."""
        if self._session:
            self._session.close()
        if self._engine:
            self._engine.dispose()

    @property
    def session(self) -> Session:
        """Get the SQLAlchemy session."""
        if self._session is None:
            raise RuntimeError(
                "DwhSession not entered - use 'with DwhSession(...) as dwh:'"
            )
        return self._session

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            raise RuntimeError(
                "DwhSession not entered - use 'with DwhSession(...) as dwh:'"
            )
        return self._engine

    def infer_table(self, table_name: str) -> sqlalchemy.Table:
        """Infer table structure with built-in reflection support.

        Args:
            table_name: Name of the table to infer

        Returns:
            SQLAlchemy Table object with inferred schema
        """
        return infer_table(
            self.engine, table_name, self.dwh_config.supports_table_reflection()
        )

    def list_tables(self) -> list[str]:
        """Get a list of table names from the data warehouse.

        Returns:
            List of table names (strings) available in the data warehouse

        Raises:
            DwhDatabaseDoesNotExistError: When the target database/dataset does not exist
        """
        try:
            # Hack for redshift's lack of reflection support.
            if self.dwh_config.is_redshift():
                query = text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema IN (:search_path) ORDER BY table_name"
                )
                result = self.session.execute(
                    query, {"search_path": self.dwh_config.search_path or "public"}
                )
                return result.scalars().all()
            inspected = sqlalchemy.inspect(self.engine)
            return list(
                sorted(inspected.get_table_names() + inspected.get_view_names())
            )
        except OperationalError as exc:
            if self._is_postgres_database_not_found_error(exc):
                raise DwhDatabaseDoesNotExistError(str(exc)) from exc
            raise
        except google.api_core.exceptions.NotFound as exc:
            # Google returns a 404 when authentication succeeds but when the specified datasource does not exist.
            raise DwhDatabaseDoesNotExistError(str(exc)) from exc

    def _is_postgres_database_not_found_error(self, exc: OperationalError) -> bool:
        """Check if the exception indicates a Postgres database does not exist."""
        return (
            exc.args
            and isinstance(exc.args[0], str)
            and "FATAL:  database" in exc.args[0]
            and "does not exist" in exc.args[0]
        )

    def _create_engine(self) -> Engine:
        """Create a SQLAlchemy Engine for the customer database.

        This method replicates the logic from RemoteDatabaseConfig.dbengine().
        """
        url = self.dwh_config.to_sqlalchemy_url()
        connect_args: dict = {}

        if url.get_backend_name() == "postgresql":
            connect_args["connect_timeout"] = TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES
            # Replace the Postgres' client default DNS lookup with one that applies security checks first
            connect_args["hostaddr"] = safe_resolve(url.host)

        logger.info(
            f"Connecting to customer dwh: url={safe_url(url)}, "
            f"backend={url.get_backend_name()}, connect_args={connect_args}"
        )

        engine = sqlalchemy.create_engine(
            url,
            connect_args=connect_args,
            logging_name=SA_LOGGER_NAME_FOR_DWH,
            execution_options={"logging_token": "dwh"},
            echo=flags.ECHO_SQL,
            poolclass=sqlalchemy.pool.NullPool,
        )

        self._extra_engine_setup(engine)
        return engine

    def _extra_engine_setup(self, engine: Engine):
        """Do any extra configuration if needed before a connection is made.

        This method replicates the logic from RemoteDatabaseConfig._extra_engine_setup().
        """
        # Handle search_path for PostgreSQL
        if hasattr(self.dwh_config, "search_path") and self.dwh_config.search_path:

            @event.listens_for(engine, "connect", insert=True)
            def set_search_path(dbapi_connection, _connection_record):
                existing_autocommit = dbapi_connection.autocommit
                dbapi_connection.autocommit = True
                cursor = dbapi_connection.cursor()
                cursor.execute(f"SET SESSION search_path={self.dwh_config.search_path}")
                cursor.close()
                dbapi_connection.autocommit = existing_autocommit

        # Handle Redshift incompatibilities
        if self.dwh_config.is_redshift() and hasattr(
            engine.dialect, "_set_backslash_escapes"
        ):
            engine.dialect._set_backslash_escapes = lambda _: None
