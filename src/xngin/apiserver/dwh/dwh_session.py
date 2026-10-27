"""Context manager for data warehouse connections."""

from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Self

import google.api_core.exceptions
import sqlalchemy
from loguru import logger
from sqlalchemy import ColumnElement, Engine, Inspector, event, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.exc import NoSuchTableError, OperationalError
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.dns.safe_resolve import safe_resolve
from xngin.apiserver.dwh import dwh_utils, participant_metrics_queries, queries, query_constructors
from xngin.apiserver.dwh.analysis_types import ParticipantOutcome
from xngin.apiserver.exceptions_common import DwhConnectionError, DwhDatabaseDoesNotExistError, DwhTimeoutError
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest, Filter
from xngin.apiserver.settings import SA_LOGGER_NAME_FOR_DWH, TIMEOUT_SECS_FOR_CUSTOMER_POSTGRES, BqDsn, Dsn, Dwh
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


class CannotFindTableError(Exception):
    """Raised when we cannot find a table in the database.

    The message names only the missing table. It must never list the other tables in the warehouse:
    their names are customer data and this message ends up in logs, Sentry titles and API responses.
    """

    def __init__(self, table_name: str):
        self.table_name = table_name
        self.message = (
            f"The table '{table_name}' does not exist. Check the table name, and the tables in the database, "
            f"and try again."
        )

    def __str__(self):
        return self.message


class UnsupportedTableNameError(CannotFindTableError):
    """Raised for a BigQuery table name that BigQuery could not have created; see dwh_utils.is_bigquery_table_name.

    Such names include dotted ones, which sqlalchemy-bigquery would read as dataset.table, reaching outside the data
    source's dataset.
    """

    def __init__(self, table_name: str):
        super().__init__(table_name)
        self.message = f"The table name '{table_name}' is not a valid BigQuery table name."


def _mark_bigquery_timestamps_as_aware(table: sqlalchemy.Table) -> None:
    """Reports BigQuery TIMESTAMP columns as timezone-aware.

    A BigQuery TIMESTAMP is an instant in UTC and the driver returns timezone-aware datetimes for it, but
    sqlalchemy-bigquery reflects it as sqlalchemy.TIMESTAMP with timezone=False, indistinguishable from DATETIME (which
    really is naive). Flip the flag so the column is reported as a TIMESTAMP_WITH_TIMEZONE, matching the values the
    driver returns.
    """
    for column in table.columns:
        if type(column.type) is sqlalchemy.sql.sqltypes.TIMESTAMP and not column.type.timezone:
            column.type = sqlalchemy.TIMESTAMP(timezone=True)


class DwhSession:
    """Deadline-bounded context manager for customer data warehouse connections.

    This class defines every interaction the application has with customer data warehouses. Each one
    is a named public method. All interactions with the customer DWH should be implemented on this class.

    The warehouse drivers are blocking and not all of them have async equivalents, so every
    interaction runs on one helper thread this object owns, and all of them share one deadline that
    starts when the block is entered:

        with DwhSession.open(dwh_config, timeout=30) as dwh:
            outcomes = dwh.get_participant_metrics("participants", ...)

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
    public x() that dispatches it onto the helper thread. Larger queries live in the queries and
    participant_metrics_queries modules and are passed the Session by their _x_blocking() method.
    """

    # Set by _connect_blocking, which open() runs before handing the object to anyone.
    _engine: Engine
    _session: Session

    @classmethod
    @contextmanager
    def open(cls, dwh_config: Dwh, *, timeout: float | None = None) -> Generator[Self]:
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
        self._inspected_tables: dict[tuple[str, bool], sqlalchemy.Table] = {}

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

    def _inspect_table_blocking(self, table_name: str, *, use_sa_autoload: bool | None = None) -> sqlalchemy.Table:
        """Returns the Table for table_name, inspecting the DWH only the first time it is requested."""
        self._check_table_name(table_name)
        if use_sa_autoload is None:
            use_sa_autoload = self._dwh_config.supports_sa_autoload()
        key = (table_name, use_sa_autoload)
        sa_table = self._inspected_tables.get(key)
        if sa_table is None:
            sa_table = self._reflect_table_blocking(table_name, use_sa_autoload=use_sa_autoload)
            self._inspected_tables[key] = sa_table
        return sa_table

    def _check_table_name(self, table_name: str) -> None:
        """Refuses BigQuery table names that BigQuery could not have created, including dotted ones.

        Every method that takes a table name inspects it first, so this is the one place names are checked. On
        Postgres and Redshift, a table name is quoted as a single identifier, so a dot in it is part of the name and
        cannot name a schema: the data source's search_path alone decides where tables are found.
        """
        if isinstance(self._dwh_config, BqDsn) and not dwh_utils.is_bigquery_table_name(table_name):
            raise UnsupportedTableNameError(table_name)

    def _reflect_table_blocking(self, table_name: str, *, use_sa_autoload: bool) -> sqlalchemy.Table:
        metadata = sqlalchemy.MetaData()
        try:
            if use_sa_autoload:
                table = sqlalchemy.Table(table_name, metadata, autoload_with=self._engine)
                if dwh_utils.is_bigquery(self._engine.url):
                    _mark_bigquery_timestamps_as_aware(table)
                return table
            # This method of introspection should only be used if the db dialect doesn't support Sqlalchemy2 reflection.
            return self._inspect_table_from_cursor_blocking(self._engine, table_name)
        except sqlalchemy.exc.ProgrammingError:
            logger.exception("Failed to create a Table! use_sa_autoload: {}", use_sa_autoload)
            raise
        except NoSuchTableError as nste:
            # Do not reflect the rest of the warehouse here to suggest alternatives: that fetches every
            # table's schema and has blown the DWH deadline on datasets with a few hundred tables.
            raise CannotFindTableError(table_name) from nste

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
                        case 1184:  # TIMESTAMPTZ / timestamp with time zone
                            sa_type = sqlalchemy.DateTime(timezone=True)
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
                return sqlalchemy.Table(table_name, metadata, *columns)
        except NoSuchTableError as nste:
            raise CannotFindTableError(table_name) from nste
        except sqlalchemy.exc.ProgrammingError as pe:
            # The query names a table that does not exist: SQLSTATE 42P01, undefined_table.
            if getattr(pe.orig, "pgcode", None) == "42P01" or getattr(pe.orig, "sqlstate", None) == "42P01":
                raise CannotFindTableError(table_name) from pe
            raise

    def inspect_table(self, table_name: str, use_sa_autoload: bool | None = None) -> sqlalchemy.Table:
        """Inspect table structure using a variety of backend-specific workarounds.

        The only fields guaranteed to be set on the the returned Table.columns field are
        .name, .type, and .nullable. Inspecting the same table again on this DwhSession returns the
        same Table without contacting the DWH.

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

    def _get_participant_metrics_blocking(
        self,
        table_name: str,
        metric_names: list[str],
        unique_id_field: str,
        participant_ids: list[str],
        use_sa_autoload: bool | None,
    ) -> list[ParticipantOutcome]:
        sa_table = self._inspect_table_blocking(table_name, use_sa_autoload=use_sa_autoload)
        return participant_metrics_queries.get_participant_metrics(
            self._session, sa_table, metric_names, unique_id_field, participant_ids
        )

    def get_participant_metrics(
        self,
        table_name: str,
        *,
        metric_names: list[str],
        unique_id_field: str,
        participant_ids: list[str],
        use_sa_autoload: bool | None = None,
    ) -> list[ParticipantOutcome]:
        """Read the values of metric columns for specific participants.

        Args:
            table_name: Name of the table to read. Only unqualified table names are supported.
            metric_names: The metric columns to read.
            unique_id_field: The column holding participant identifiers.
            participant_ids: The participants to read.
            use_sa_autoload: Whether to use SQLAlchemy reflection. If None, uses config default.

        Returns:
            One ParticipantOutcome per participant found in the table, with a MetricValue per metric.
            Participants absent from the table are omitted.

        Raises:
            CannotFindTableError: If the table does not exist.
            LateValidationError: If a metric or the unique id column is missing from the table.
        """
        return self._on_worker(
            "reading participant metrics",
            self._get_participant_metrics_blocking,
            table_name,
            metric_names,
            unique_id_field,
            participant_ids,
            use_sa_autoload,
        )

    def _get_raw_metric_stats_blocking(
        self, sa_table: sqlalchemy.Table, metrics: list[DesignSpecMetricRequest], filters: list[Filter]
    ) -> RowMapping:
        return queries.get_raw_metric_stats(self._session, sa_table, metrics, filters)

    def get_raw_metric_stats(
        self, sa_table: sqlalchemy.Table, *, metrics: list[DesignSpecMetricRequest], filters: list[Filter]
    ) -> RowMapping:
        """Read aggregate statistics of metric columns over the rows matching filters.

        Args:
            sa_table: The table to read, as returned by inspect_table().
            metrics: The metric columns to summarize. Only field_name is used.
            filters: Row filters applied before aggregating.

        Returns:
            One row with ``rows__count`` (rows matching the filters) and, per metric, ``{name}__mean``,
            ``{name}__stddev`` and ``{name}__count`` over its non-null values. power_adapters.build_metric_stats
            turns the row into DesignSpecMetric objects.
        """
        return self._on_worker(
            "reading metric statistics",
            self._get_raw_metric_stats_blocking,
            sa_table,
            metrics,
            filters,
        )

    def _get_cluster_sufficient_stats_blocking(
        self,
        sa_table: sqlalchemy.Table,
        cluster_column_name: str,
        outcome_column_names: Sequence[str],
        filters: list[Filter],
        outcome_shifts: Mapping[str, float] | None,
    ) -> Sequence[RowMapping]:
        return queries.get_cluster_sufficient_stats(
            self._session, sa_table, cluster_column_name, outcome_column_names, filters, outcome_shifts
        )

    def get_cluster_sufficient_stats(
        self,
        sa_table: sqlalchemy.Table,
        *,
        cluster_column_name: str,
        outcome_column_names: Sequence[str],
        filters: list[Filter],
        outcome_shifts: Mapping[str, float] | None = None,
    ) -> Sequence[RowMapping]:
        """Read per-cluster sufficient statistics of outcome columns, for cluster power calculations.

        Args:
            sa_table: The table to read, as returned by inspect_table().
            cluster_column_name: The column identifying each row's cluster. Rows with a null cluster are skipped.
            outcome_column_names: The outcome columns to summarize.
            filters: Row filters applied before grouping.
            outcome_shifts: Constant to subtract from each outcome before summing, typically its approximate
                mean, for numerical stability. The ICC is unaffected.

        Returns:
            One row per cluster with ``rows__count`` (every row in the cluster, including rows whose outcomes
            are null) and, per outcome, ``{name}__count``, ``{name}__sum`` and ``{name}__sumsq`` over its
            non-null values. power_adapters.calculate_cluster_stats turns the rows into ICC and cluster-size
            statistics.

        Raises:
            LateValidationError: If a named column is missing, or no clusters match.
        """
        return self._on_worker(
            "reading cluster statistics",
            self._get_cluster_sufficient_stats_blocking,
            sa_table,
            cluster_column_name,
            outcome_column_names,
            filters,
            outcome_shifts,
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
                # Bind parameters are customer data. With hide_parameters set, SQLAlchemy prints
                # "[SQL parameters hidden due to hide_parameters=True]" instead of the values in exception text
                # and in its own logging, which keeps them out of Railway logs and Sentry. Developers keep full
                # parameters locally.
                hide_parameters=not flags.is_dev_environment(),
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
                applied = False
                try:
                    # Postgres-compatible SQL via DBAPI parameterized query to set the search path with
                    # a user-specified, possibly comma-separated string.
                    # May raise an error if the search path syntax is invalid.
                    cursor.execute(
                        "SELECT set_config('search_path', %(schemas)s, false)",
                        {"schemas": search_path_sql_arg},
                    )
                    applied = True
                finally:
                    cursor.close()
                    if applied:
                        dbapi_connection.autocommit = existing_autocommit
                    else:
                        # A connect listener that raises runs inside pool record construction before
                        # SQLAlchemy has a record it can close, so we close the DBAPI connection here.
                        dbapi_connection.close()

        dwh_utils.extra_engine_setup(engine)
