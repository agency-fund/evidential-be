"""Context manager for data warehouse connections."""

from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Concatenate, Self

import google.api_core.exceptions
import sqlalchemy
from loguru import logger
from sqlalchemy import ColumnElement, Engine, Inspector, event, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.exc import NoSuchTableError, OperationalError
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.dns.safe_resolve import safe_resolve
from xngin.apiserver.dwh import dwh_utils, query_constructors
from xngin.apiserver.dwh.inspection_types import FieldDescriptor
from xngin.apiserver.dwh.inspections import generate_field_descriptors
from xngin.apiserver.exceptions_common import DwhConnectionError, DwhDatabaseDoesNotExistError, DwhTimeoutError
from xngin.apiserver.routers.common_api_types import Filter
from xngin.apiserver.settings import SA_LOGGER_NAME_FOR_DWH, TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES, Dsn, Dwh
from xngin.ops.threads import ThreadTimeout, timeout_thread


def _is_postgres_database_not_found_error(exc: OperationalError) -> bool:
    """Returns true when the exception indicates a Postgres database does not exist."""
    return (
        len(exc.args) > 0
        and isinstance(exc.args[0], str)
        and "FATAL:  database" in exc.args[0]
        and "does not exist" in exc.args[0]
    )


def _safe_url(url: sqlalchemy.engine.url.URL) -> sqlalchemy.engine.url.URL:
    """Prepares a URL for presentation or capture in logs by stripping sensitive values."""
    cleaned = url.set(password="redacted")  # noqa: S106
    for qp in ("credentials_base64", "credentials_info"):
        if cleaned.query.get(qp):
            cleaned = cleaned.update_query_dict({qp: "redacted"})
    return cleaned


@dataclass
class GetParticipantsResult:
    """Result of getting participants from a data warehouse table."""

    sa_table: sqlalchemy.Table
    participants: list


@dataclass
class InspectTableWithDescriptorsResult:
    """Result of inspecting table structure."""

    sa_table: sqlalchemy.Table
    db_schema: dict[str, FieldDescriptor]


class CannotFindTableError(Exception):
    """Raised when we cannot find a table in the database."""

    def __init__(self, table_name, existing_tables):
        self.table_name = table_name
        self.alternatives = existing_tables
        if existing_tables:
            self.message = (
                f"The table '{table_name}' does not exist. Known tables: {', '.join(sorted(existing_tables))}"
            )
        else:
            self.message = f"The table '{table_name}' does not exist; the database does not contain any tables."

    def __str__(self):
        return self.message


class DwhSession:
    """Deadline-bounded context manager for customer data warehouse connections.

    This class defines most of the interactions we have with customer data warehouses. Their drivers
    are blocking and not all of them have async equivalents, so every interaction runs on one helper
    thread this object owns, and all of them share one deadline that starts when the block is
    entered:

        with DwhSession.open(dwh_config, timeout=30) as dwh:
            sa_table = dwh.inspect_table("participants")
            outcomes = dwh.run(get_participant_metrics, sa_table, metrics, "uid", ids)

    Each DwhSession manages a single thread, and runs all the DWH queries on that thread. This allows
    us to detect that the DWH interactions are not responding within a timeout, and raise an exception when
    the timeout is reached.

    A DWH that stops responding raises DwhTimeoutError rather than hanging the caller. The
    query itself cannot be cancelled: it keeps running on the helper thread until it finishes.

    SQLAlchemy's threading model requires that we access each Session from only one thread, so we open
    the connection and close the DWH Session on that thread. Do not pass SQLALchemy objects such as Session or
    ORM instances that are from the application database because that will violate SQLAlchemy's thread safety
    commitments.

    Each warehouse interaction is implemented as a pair: a private _x_blocking() running the query itself, and a
    public x() that dispatches it onto the helper thread.
    """

    # Set by _connect_blocking, which open() runs before handing the object to anyone.
    _engine: Engine
    _session: Session

    @classmethod
    @contextmanager
    def open(cls, dwh_config: Dwh, *, timeout: float | None = None) -> Iterator[Self]:
        """Connect to a warehouse for the duration of the block.

        Args:
            dwh_config: The data warehouse configuration (Dsn or BqDsn)
            timeout: seconds this block may spend interacting with the warehouse, in total.
                Defaults to flags.DWH_TIMEOUT_SECS, read here rather than bound as a default so
                that the deployment-wide budget stays adjustable.
        """
        seconds = flags.DWH_TIMEOUT_SECS if timeout is None else timeout
        with timeout_thread(seconds=seconds, name="dwh") as deadline:
            dwh = cls(dwh_config, deadline, seconds)
            try:
                # Bounds connecting too: _create_engine resolves DNS synchronously. Keeping it
                # inside the try ensures that a timed-out connection is closed once it finishes.
                dwh._on_worker("connecting", dwh._connect_blocking)
                yield dwh
            finally:
                dwh._close()

    def __init__(self, dwh_config: Dwh, deadline: ThreadTimeout, timeout_secs: float):
        """Not for direct use; open() owns the helper thread and the connection this needs."""
        self._dwh_config = dwh_config
        self._timeout = deadline
        self._timeout_secs = timeout_secs

    def _connect_blocking(self) -> None:
        self._engine = self._create_engine()
        self._session = Session(self._engine)

    def _close_blocking(self) -> None:
        session = getattr(self, "_session", None)
        try:
            if session is not None:
                session.close()
        finally:
            engine = getattr(self, "_engine", None)
            if engine is not None:
                engine.dispose()

    @staticmethod
    def _log_close_failure(future: Future[None]) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Failed to close the data warehouse connection; abandoning it.")

    def _close(self) -> None:
        """Queue closing the warehouse connection without waiting for it.

        After a timeout the helper is still inside the call we gave up on, so the close is queued
        behind it rather than run here. Clean exits use the same path so warehouse teardown never
        holds up the caller. Failures are logged by the completion callback, but otherwise ignored,
        because there is nothing we can do about them here anyway.
        """
        self._timeout.submit(self._close_blocking).add_done_callback(self._log_close_failure)

    def _on_worker[T, **P](self, label: str, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run one warehouse interaction on the helper thread, under this block's deadline."""
        try:
            return self._timeout.run(fn, *args, **kwargs)
        except TimeoutError as exc:
            raise DwhTimeoutError(f"The data warehouse did not finish {label} within {self._timeout_secs:g}s.") from exc

    def run[T, **P](self, fn: Callable[Concatenate[Session, P], T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call fn(warehouse_session, *args, **kwargs) under this block's deadline.

        For queries that live outside this class because they are substantial enough to deserve
        their own modules. fn runs on the helper thread and must touch nothing but the Session it is
        handed: an application-database Session, or an ORM object belonging to the caller, would end
        up in use from two threads at once.

        The Session is never returned to callers, only passed to fn, so that no one can hold it
        past the deadline or use it off the helper thread.
        """
        return self._on_worker(getattr(fn, "__name__", "a query"), fn, self._session, *args, **kwargs)

    def _inspect_table_blocking(self, table_name: str, *, use_sa_autoload: bool | None = None) -> sqlalchemy.Table:
        if use_sa_autoload is None:
            use_sa_autoload = self._dwh_config.supports_sa_autoload()
        metadata = sqlalchemy.MetaData()
        try:
            if use_sa_autoload:
                return sqlalchemy.Table(table_name, metadata, autoload_with=self._engine, quote=False)
            # This method of introspection should only be used if the db dialect doesn't support Sqlalchemy2 reflection.
            return self._inspect_table_from_cursor_blocking(self._engine, table_name)
        except sqlalchemy.exc.ProgrammingError:
            logger.exception("Failed to create a Table! use_sa_autoload: {}", use_sa_autoload)
            raise
        except NoSuchTableError as nste:
            metadata.reflect(self._engine)
            existing_tables = metadata.tables.keys()
            raise CannotFindTableError(table_name, existing_tables) from nste

    def _inspect_table_from_cursor_blocking(
        self, engine: sqlalchemy.engine.Engine, table_name: str
    ) -> sqlalchemy.Table:
        """Creates a SQLAlchemy Table instance from cursor description metadata."""

        columns = []
        metadata = sqlalchemy.MetaData()
        try:
            with engine.begin() as connection:
                query = query_constructors.create_inspect_table_from_cursor_query(table_name)
                result = connection.execute(query)
                description = result.cursor.description
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
                    sa_type: type[sqlalchemy.types.TypeEngine] | sqlalchemy.types.TypeEngine
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

    def inspect_table(self, table_name: str, use_sa_autoload: bool | None = None) -> sqlalchemy.Table:
        """Inspect table structure using a variety of backend-specific workarounds.

        The only fields guaranteed to be set on the the returned Table.columns field are
        .name, .type, and .nullable.

        Args:
            table_name: Name of the table to inspect. Only unqualified table names are supported.
            use_sa_autoload: Whether to use SQLAlchemy reflection. If None, uses config default.

        Returns:
            SQLAlchemy Table object
        """
        return self._on_worker(
            "inspecting a table",
            self._inspect_table_blocking,
            table_name,
            use_sa_autoload=use_sa_autoload,
        )

    def _inspect_table_with_descriptors_blocking(
        self, table_name: str, unique_id_field: str, use_sa_autoload: bool | None = None
    ) -> InspectTableWithDescriptorsResult:
        sa_table = self._inspect_table_blocking(table_name, use_sa_autoload=use_sa_autoload)
        db_schema = generate_field_descriptors(sa_table, unique_id_field)
        return InspectTableWithDescriptorsResult(sa_table=sa_table, db_schema=db_schema)

    def inspect_table_with_descriptors(
        self, table_name: str, unique_id_field: str, use_sa_autoload: bool | None = None
    ) -> InspectTableWithDescriptorsResult:
        """Convenience method combining table inspection and field descriptor generation.

        Args:
            table_name: Name of the table to inspect
            unique_id_field: The column name to use as a participant's unique identifier
            use_sa_autoload: If not None, overrides the configuration's default behavior.

        Returns:
            InspectTableWithDescriptorsResult containing both the SQLAlchemy Table and field descriptors
        """
        return self._on_worker(
            "inspecting a table",
            self._inspect_table_with_descriptors_blocking,
            table_name,
            unique_id_field,
            use_sa_autoload,
        )

    def _get_result_blocking(
        self,
        table_name: str,
        filters: list[Filter],
        compose_query: Callable[[sqlalchemy.Table, list[ColumnElement]], sqlalchemy.Select],
        use_sa_autoload: bool | None = None,
    ) -> GetParticipantsResult:
        """Inspects ``table_name``, runs the query built by ``compose_query``, and wraps the resulting rows.

        ``compose_query`` receives the inspected table and the SQLAlchemy-translated ``filters`` and returns
        the query to execute. Both individual-level and cluster-level sampling paths provide their
        custom composer while sharing remaining logic.
        """
        sa_table = self._inspect_table_blocking(table_name, use_sa_autoload=use_sa_autoload)
        sqla_filters = query_constructors.create_query_filters(sa_table, filters)
        participants = self._session.execute(compose_query(sa_table, sqla_filters)).all()
        return GetParticipantsResult(sa_table=sa_table, participants=list(participants))

    def _get_participants_blocking(
        self,
        table_name: str,
        select_columns: set[str],
        filters: list[Filter],
        n: int,
        use_sa_autoload: bool | None = None,
    ) -> GetParticipantsResult:
        return self._get_result_blocking(
            table_name,
            filters,
            lambda sa_table, sqla_filters: query_constructors.compose_query(sa_table, select_columns, sqla_filters, n),
            use_sa_autoload,
        )

    def _get_clusters_blocking(
        self,
        table_name: str,
        select_columns: set[str],
        filters: list[Filter],
        desired_n_clusters: int,
        cluster_key: str,
        use_sa_autoload: bool | None = None,
    ) -> GetParticipantsResult:
        return self._get_result_blocking(
            table_name,
            filters,
            lambda sa_table, sqla_filters: query_constructors.compose_cluster_query(
                sa_table,
                select_columns | {cluster_key},
                sqla_filters,
                desired_n_clusters,
                cluster_key,
            ),
            use_sa_autoload,
        )

    def get_participants(
        self,
        table_name: str,
        *,
        select_columns: set[str],
        filters: list[Filter],
        n: int,
        use_sa_autoload: bool | None = None,
    ) -> GetParticipantsResult:
        """Get participants by combining table inspection and querying.

        Caveats documented on inspect_table() apply to the returned Table.

        Args:
            table_name: Name of the table to query
            select_columns: DWH columns to return.
            filters: Filter conditions to apply
            n: Number of participants to retrieve
            use_sa_autoload: Whether to use SQLAlchemy reflection. If None, uses config default.

        Returns:
            GetParticipantsResult containing both the SQLAlchemy table and participant query results
        """
        return self._on_worker(
            "reading participants",
            self._get_participants_blocking,
            table_name,
            select_columns,
            filters,
            n,
            use_sa_autoload,
        )

    def get_clusters_of_participants(
        self,
        table_name: str,
        *,
        select_columns: set[str],
        filters: list[Filter],
        desired_n_clusters: int,
        cluster_key: str,
        use_sa_autoload: bool | None = None,
    ) -> GetParticipantsResult:
        """Get participants from a random sample of clusters.

        The random sample is taken over distinct values of ``cluster_key`` after applying ``filters``.
        Returned participants are the filtered rows belonging to the sampled clusters.

        Caveats documented on inspect_table() apply to the returned Table.

        Args:
            table_name: Name of the table to query
            select_columns: DWH columns to return. ``cluster_key`` is always included.
            filters: Filter conditions to apply before sampling clusters and fetching participants
            desired_n_clusters: Number of clusters to sample
            cluster_key: Column containing cluster identifiers
            use_sa_autoload: Whether to use SQLAlchemy reflection. If None, uses config default.

        Returns:
            GetParticipantsResult containing both the SQLAlchemy table and participant query results
        """
        return self._on_worker(
            "sampling clusters",
            self._get_clusters_blocking,
            table_name,
            select_columns,
            filters,
            desired_n_clusters,
            cluster_key,
            use_sa_autoload,
        )

    def _list_tables_blocking(self) -> list[str]:
        try:
            # Hack for redshift's lack of reflection support.
            if isinstance(self._dwh_config, Dsn) and self._dwh_config.is_redshift():
                query = text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = ANY(current_schemas(false)) "
                    "AND table_type IN ('BASE TABLE', 'VIEW') "
                    "ORDER BY table_name"
                )
                result = self._session.execute(query)
                return list(result.scalars().all())
            inspected = sqlalchemy.inspect(self._engine)

            if not isinstance(inspected, Inspector):
                raise TypeError(f"Unexpected type of inspector: {type(inspected)}")
            return list(sorted(inspected.get_table_names() + inspected.get_view_names()))

        except OperationalError as exc:
            if _is_postgres_database_not_found_error(exc):
                raise DwhDatabaseDoesNotExistError(str(exc)) from exc
            raise DwhConnectionError(exc) from exc
        except google.api_core.exceptions.NotFound as exc:
            # Google returns a 404 when authentication succeeds but when the specified datasource does not exist.
            raise DwhDatabaseDoesNotExistError(str(exc)) from exc

    def list_tables(self) -> list[str]:
        """Get a list of table names from the data warehouse.

        Returns:
            List of table names (strings) available in the data warehouse

        Raises:
            DwhDatabaseDoesNotExistError: When the target database/dataset does not exist
        """
        return self._on_worker("listing tables", self._list_tables_blocking)

    def _connectivity_check_blocking(self) -> None:
        """Runs a minimal query to validate database connectivity and credentials."""
        try:
            self._session.execute(text("SELECT 1"))
        except OperationalError as exc:
            if _is_postgres_database_not_found_error(exc):
                raise DwhDatabaseDoesNotExistError(str(exc)) from exc
            raise DwhConnectionError(exc) from exc
        except google.api_core.exceptions.NotFound as exc:
            raise DwhDatabaseDoesNotExistError(str(exc)) from exc

    def connectivity_check(self) -> None:
        """Validate that the configured warehouse is reachable and credentials are valid."""
        self._on_worker("a connectivity check", self._connectivity_check_blocking)

    def _create_engine(self) -> Engine:
        """Create a SQLAlchemy Engine for the customer database."""
        url = self._dwh_config.to_sqlalchemy_url()
        if url.host is None:
            # This should never happen, but check just in case.
            raise DwhDatabaseDoesNotExistError(f"No host found in URL: {url}")

        connect_args: dict = {}

        if dwh_utils.is_postgres(url):
            connect_args["connect_timeout"] = TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES
            # Replace the Postgres' client default DNS lookup with one that applies security checks first
            connect_args["hostaddr"] = safe_resolve(url.host)

        logger.info(
            f"Connecting to customer dwh: url={_safe_url(url)}, "
            f"backend={url.get_backend_name()}, connect_args={connect_args}"
        )
        try:
            engine = sqlalchemy.create_engine(
                url,
                connect_args=connect_args,
                logging_name=SA_LOGGER_NAME_FOR_DWH,
                execution_options={"logging_token": "dwh"},
                poolclass=sqlalchemy.pool.NullPool,
            )
        except Exception as exc:
            raise DwhConnectionError(exc) from exc

        self._extra_engine_setup(engine)
        return engine

    def _extra_engine_setup(self, engine: Engine):
        """Do any extra configuration if needed before a connection is made."""
        # Handle search_path for PostgreSQL & Redshift
        if isinstance(self._dwh_config, Dsn) and self._dwh_config.search_path:
            search_path_sql_arg = self._dwh_config.search_path

            @event.listens_for(engine, "connect", insert=True)
            def set_search_path(dbapi_connection: DBAPIConnection, _connection_record):
                existing_autocommit = dbapi_connection.autocommit
                dbapi_connection.autocommit = True
                cursor = dbapi_connection.cursor()
                try:
                    # Postgres-compatible SQL via DBAPI parameterized query to set the search path with
                    # a user-specified, possibly comma-separated string.
                    cursor.execute(
                        "SELECT set_config('search_path', %(schemas)s, false)",
                        {"schemas": search_path_sql_arg},
                    )
                finally:
                    cursor.close()
                    dbapi_connection.autocommit = existing_autocommit

        dwh_utils.extra_engine_setup(engine)
