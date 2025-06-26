"""Data warehouse session context manager for database connections."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import google.api_core.exceptions
import sqlalchemy
from loguru import logger
from sqlalchemy import Engine, distinct, event, text
from sqlalchemy.exc import NoSuchTableError, OperationalError
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.dns.safe_resolve import safe_resolve
from xngin.apiserver.dwh.inspection_types import FieldDescriptor
from xngin.apiserver.dwh.inspections import generate_field_descriptors
from xngin.apiserver.dwh.queries import query_for_participants
from xngin.apiserver.routers.common_api_types import (
    FilterClass,
    GetFiltersResponseDiscrete,
    GetFiltersResponseElement,
    GetFiltersResponseNumericOrDate,
)
from xngin.apiserver.settings import (
    SA_LOGGER_NAME_FOR_DWH,
    TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES,
    Dwh,
)


class DwhDatabaseDoesNotExistError(Exception):
    """Raised when the target database or dataset does not exist."""


def _safe_url(url: sqlalchemy.engine.url.URL) -> sqlalchemy.engine.url.URL:
    """Prepares a URL for presentation or capture in logs by stripping sensitive values."""
    cleaned = url.set(password="redacted")
    for qp in ("credentials_base64", "credentials_info"):
        if cleaned.query.get(qp):
            cleaned = cleaned.update_query_dict({qp: "redacted"})
    return cleaned


def _is_postgres_database_not_found_error(exc: OperationalError) -> bool:
    """Returns true when the exception indicates a Postgres database does not exist."""
    return (
        exc.args
        and isinstance(exc.args[0], str)
        and "FATAL:  database" in exc.args[0]
        and "does not exist" in exc.args[0]
    )


@dataclass
class GetParticipantsResult:
    """Result of getting participants from a data warehouse table."""

    sa_table: sqlalchemy.Table
    participants: list


@dataclass
class InferTableWithDescriptorsResult:
    """Result of inferring table structure with field descriptors."""

    sa_table: sqlalchemy.Table
    db_schema: dict[str, FieldDescriptor]
    mapper: Callable[[str, FieldDescriptor], GetFiltersResponseElement]


class CannotFindTableError(Exception):
    """Raised when we cannot find a table in the database."""

    def __init__(self, table_name, existing_tables):
        self.table_name = table_name
        self.alternatives = existing_tables
        if existing_tables:
            self.message = f"The table '{table_name}' does not exist. Known tables: {', '.join(sorted(existing_tables))}"
        else:
            self.message = f"The table '{table_name}' does not exist; the database does not contain any tables."

    def __str__(self):
        return self.message


class DwhSession:
    """Async context manager for data warehouse database connections.

    This class defines most of the interactions we have with customer data warehouses. The underlying connections to
    the DWH are using blocking SQLAlchemy drivers, and this class wraps them in threads and adapts them to async so that
    we can call them without blocking the request thread.
    """

    def __init__(self, dwh_config: Dwh):
        """Initialize with data warehouse configuration.

        Args:
            dwh_config: The data warehouse configuration (Dsn or BqDsn)
        """
        self.dwh_config = dwh_config
        self._engine: Engine | None = None
        self._session: Session | None = None

    def _enter_blocking(self):
        self._engine = self._create_engine()
        self._session = Session(self._engine)

    async def __aenter__(self) -> "DwhSession":
        """Enter the context manager and create database connections."""
        await asyncio.get_event_loop().run_in_executor(None, self._enter_blocking)
        return self

    def _exit_blocking(self):
        if self._session:
            self._session.close()
        if self._engine:
            self._engine.dispose()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up database connections."""
        await asyncio.get_event_loop().run_in_executor(None, self._exit_blocking)

    @property
    def session(self) -> Session:
        """Get the synchronous SQLAlchemy session.

        The returned Session is synchronous. When the returned session is used on the API server, take care to wrap it
        in a thread so that it doesn't block FastAPI's request thread.
        """
        if self._session is None:
            raise RuntimeError(
                "DwhSession not entered - use 'async with DwhSession(...) as dwh:'"
            )
        return self._session

    def _infer_table_blocking(
        self, table_name: str, use_reflection: bool | None = None
    ) -> sqlalchemy.Table:
        if use_reflection is None:
            use_reflection = self.dwh_config.supports_table_reflection()
        metadata = sqlalchemy.MetaData()
        try:
            if use_reflection:
                return sqlalchemy.Table(
                    table_name, metadata, autoload_with=self._engine, quote=False
                )
            # This method of introspection should only be used if the db dialect doesn't support Sqlalchemy2 reflection.
            return self._infer_table_from_cursor_blocking(self._engine, table_name)
        except sqlalchemy.exc.ProgrammingError:
            logger.exception(
                "Failed to create a Table! use_reflection: {}", use_reflection
            )
            raise
        except NoSuchTableError as nste:
            metadata.reflect(self._engine)
            existing_tables = metadata.tables.keys()
            raise CannotFindTableError(table_name, existing_tables) from nste

    def _infer_table_from_cursor_blocking(
        self, engine: sqlalchemy.engine.Engine, table_name: str
    ) -> sqlalchemy.Table:
        """Creates a SQLAlchemy Table instance from cursor description metadata."""

        columns = []
        metadata = sqlalchemy.MetaData()
        try:
            with engine.begin() as connection:
                safe_table = sqlalchemy.quoted_name(table_name, quote=True)
                # Create a select statement - this is safe from SQL injection
                query = (
                    sqlalchemy.select(text("*")).select_from(text(safe_table)).limit(0)
                )
                result = connection.execute(query)
                description = result.cursor.description
                # print("CURSOR DESC: ", result.cursor.description)
                for col in description:
                    # Unpack cursor.description tuple
                    (
                        name,
                        type_code,
                        _,  # display_size,
                        internal_size,
                        precision,
                        scale,
                        null_ok,
                    ) = col

                    # Map Redshift type codes to SQLAlchemy types. Not comprehensive.
                    # https://docs.sqlalchemy.org/en/20/core/types.html
                    # Comment shows both pg_type.typename / information_schema.data_type
                    sa_type: type[sqlalchemy.TypeEngine] | sqlalchemy.TypeEngine
                    match type_code:
                        case 16:  # BOOL / boolean
                            sa_type = sqlalchemy.Boolean
                        case 20:  # INT8 / bigint
                            sa_type = sqlalchemy.BigInteger
                        case 23:  # INT4 / integer
                            sa_type = sqlalchemy.Integer
                        case 701:  # FLOAT8 / double precision
                            sa_type = sqlalchemy.Double
                        case 1043:  # VARCHAR / character varying
                            sa_type = sqlalchemy.String(internal_size)
                        case 1082:  # DATE / date
                            sa_type = sqlalchemy.Date
                        case 1114:  # TIMESTAMP / timestamp without time zone
                            sa_type = sqlalchemy.DateTime
                        case 1700:  # NUMERIC / numeric
                            sa_type = sqlalchemy.Numeric(precision, scale)
                        case _:  # type_code == 25
                            # Default to Text for unknown types
                            sa_type = sqlalchemy.Text

                    columns.append(
                        sqlalchemy.Column(
                            name,
                            sa_type,
                            nullable=null_ok if null_ok is not None else True,
                        )
                    )
                return sqlalchemy.Table(table_name, metadata, *columns, quote=False)
        except NoSuchTableError as nste:
            metadata.reflect(engine)
            existing_tables = metadata.tables.keys()
            raise CannotFindTableError(table_name, existing_tables) from nste

    async def infer_table(
        self, table_name: str, use_reflection: bool | None = None
    ) -> sqlalchemy.Table:
        """Infer table structure with built-in reflection support.

        Args:
            table_name: Name of the table to infer
            use_reflection: Whether to use SQLAlchemy reflection. If None, uses config default.

        Returns:
            SQLAlchemy Table object with inferred schema
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._infer_table_blocking, table_name, use_reflection
        )

    def _infer_table_with_descriptors_blocking(
        self, table_name: str, unique_id_field: str, use_reflection: bool | None = None
    ) -> InferTableWithDescriptorsResult:
        sa_table = self._infer_table_blocking(table_name, use_reflection)
        db_schema = generate_field_descriptors(sa_table, unique_id_field)
        mapper = self.create_filter_meta_mapper(db_schema, sa_table)
        return InferTableWithDescriptorsResult(
            sa_table=sa_table, db_schema=db_schema, mapper=mapper
        )

    async def infer_table_with_descriptors(
        self, table_name: str, unique_id_field: str, use_reflection: bool | None = None
    ) -> InferTableWithDescriptorsResult:
        """Convenience method combining table inference and field descriptor generation.

        Args:
            table_name: Name of the table to inspect
            unique_id_field: The column name to use as a participant's unique identifier
            use_reflection: If not None, overrides the configuration's default behavior.

        Returns:
            InferTableWithDescriptorsResult containing both the SQLAlchemy Table and field descriptors
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._infer_table_with_descriptors_blocking,
            table_name,
            unique_id_field,
            use_reflection,
        )

    def _get_participants_blocking(
        self, table_name: str, filters, n: int, use_reflection: bool | None = None
    ) -> GetParticipantsResult:
        sa_table = self._infer_table_blocking(table_name, use_reflection)
        participants = query_for_participants(self.session, sa_table, filters, n)
        return GetParticipantsResult(sa_table=sa_table, participants=participants)

    async def get_participants(
        self, table_name: str, filters, n: int, use_reflection: bool | None = None
    ) -> GetParticipantsResult:
        """Get participants by combining table inference and querying.

        Args:
            table_name: Name of the table to query
            filters: Filter conditions to apply
            n: Number of participants to retrieve
            use_reflection: Whether to use SQLAlchemy reflection. If None, uses config default.

        Returns:
            GetParticipantsResult containing both the SQLAlchemy table and participant query results
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._get_participants_blocking,
            table_name,
            filters,
            n,
            use_reflection,
        )

    def _list_tables_blocking(self) -> list[str]:
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
            inspected = sqlalchemy.inspect(self._engine)
            return list(
                sorted(inspected.get_table_names() + inspected.get_view_names())
            )
        except OperationalError as exc:
            if _is_postgres_database_not_found_error(exc):
                raise DwhDatabaseDoesNotExistError(str(exc)) from exc
            raise
        except google.api_core.exceptions.NotFound as exc:
            # Google returns a 404 when authentication succeeds but when the specified datasource does not exist.
            raise DwhDatabaseDoesNotExistError(str(exc)) from exc

    async def list_tables(self) -> list[str]:
        """Get a list of table names from the data warehouse.

        Returns:
            List of table names (strings) available in the data warehouse

        Raises:
            DwhDatabaseDoesNotExistError: When the target database/dataset does not exist
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._list_tables_blocking
        )

    def create_filter_meta_mapper(
        self, db_schema: dict[str, FieldDescriptor], sa_table
    ) -> Callable[[str, FieldDescriptor], GetFiltersResponseElement]:
        """Create a mapper function for generating filter metadata from database columns.

        # TODO: replace this with something cleaner

        Args:
            db_schema: Dictionary mapping column names to FieldDescriptor objects
            sa_table: SQLAlchemy Table object

        Returns:
            A mapper function that takes (column_name, column_descriptor) and returns GetFiltersResponseElement
        """

        # TODO: implement caching, respecting commons.refresh
        def mapper(
            col_name: str, column_descriptor: FieldDescriptor
        ) -> GetFiltersResponseElement:
            db_col = db_schema.get(col_name)
            filter_class = db_col.data_type.filter_class(col_name)

            # Collect metadata on the values in the database.
            sa_col = sa_table.columns[col_name]
            match filter_class:
                case FilterClass.DISCRETE:
                    distinct_values = [
                        str(v)
                        for v in self.session.execute(
                            sqlalchemy.select(distinct(sa_col))
                            .where(sa_col.is_not(None))
                            .limit(1000)
                            .order_by(sa_col)
                        ).scalars()
                    ]
                    return GetFiltersResponseDiscrete(
                        field_name=col_name,
                        data_type=db_col.data_type,
                        relations=filter_class.valid_relations(),
                        description=column_descriptor.description,
                        distinct_values=distinct_values,
                    )
                case FilterClass.NUMERIC:
                    min_, max_ = self.session.execute(
                        sqlalchemy.select(
                            sqlalchemy.func.min(sa_col), sqlalchemy.func.max(sa_col)
                        ).where(sa_col.is_not(None))
                    ).first()
                    return GetFiltersResponseNumericOrDate(
                        field_name=col_name,
                        data_type=db_col.data_type,
                        relations=filter_class.valid_relations(),
                        description=column_descriptor.description,
                        min=min_,
                        max=max_,
                    )
                case _:
                    raise RuntimeError("unexpected filter class")

        return mapper

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
            f"Connecting to customer dwh: url={_safe_url(url)}, "
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
