import json
import logging
import sys
import traceback
import typing
from collections.abc import Callable, Mapping

from xngin.apiserver.flags import LogFormat

if typing.TYPE_CHECKING:
    from loguru import Message as loguru_Message
    from loguru import Record as loguru_Record


import orjson
from loguru import logger

from xngin.apiserver import flags


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Describe the caller using the LogRecord rather than by walking the stack to the first frame outside the
        # logging module: Sentry's logging integration wraps Logger.callHandlers, so that frame belongs to Sentry, and
        # every record would appear to come from sentry_sdk.integrations.logging. The logger name ("granian.access",
        # "sqlalchemy.engine.Engine.xngin_app") is also more useful than the module name a stack walk would find.
        def describe_caller(loguru_record: loguru_Record) -> None:
            loguru_record["name"] = record.name
            loguru_record["module"] = record.module
            loguru_record["function"] = record.funcName
            loguru_record["line"] = record.lineno

        message, fields = record.getMessage(), dict[str, typing.Any]()
        if record.name == GRANIAN_ACCESS_LOGGER and isinstance(record.args, Mapping):
            message, fields = _describe_granian_access(record.args)

        logger.patch(describe_caller).bind(**fields).opt(exception=record.exc_info).log(level, message)


GRANIAN_ACCESS_LOGGER = "granian.access"


def _describe_granian_access(request: Mapping[str, typing.Any]) -> tuple[str, dict[str, typing.Any]]:
    """Describes a request from Granian's access log as a message and the fields to log alongside it.

    Granian calls its access logger with a format string and a dict of request values. We ignore the format string and
    describe the request ourselves: structured logs get each value as a top-level field, so Railway can filter on them,
    and other formats get a single line of key=value pairs.

    Granian doesn't escape the values, and the path is percent-decoded, so it can contain quotes, spaces, newlines, and
    terminal escape sequences. Fields are safe because JSON escapes them; text is quoted with _logfmt_value.
    """
    path = request["path"]
    query = request["query_string"]
    duration_ms = round(request["dt_ms"], 3)
    if flags.LOG_FORMAT == LogFormat.STRUCTURED_RAILWAY:
        fields = {
            "http_method": request["method"],
            "http_path": path,
            "http_query": query,
            "http_status": request["status"],
            "http_duration_ms": duration_ms,
            "http_protocol": request["protocol"],
            "client_addr": request["addr"],
        }
        return f"{request['method']} {_logfmt_value(path)} {request['status']}", fields
    pairs = {
        "dur": duration_ms,
        "ip": request["addr"],
        "method": request["method"],
        "path": path,
        "q": query,
        "code": request["status"],
    }
    return " ".join(f"{key}={_logfmt_value(value)}" for key, value in pairs.items() if value != ""), {}


_LOGFMT_ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _logfmt_value(value: object) -> str:
    """Formats a value for a key=value log line the way logfmt does.

    Values made only of printable characters other than space, '"', '=', and backslash are written as is. Others are
    quoted, with quotes, backslashes, and non-printable characters (including Unicode format characters such as
    bidirectional overrides) backslash-escaped, so that a value can't break the line apart or alter a terminal.
    """
    text = str(value)
    if text and all(char.isprintable() and char not in ' "=\\' for char in text):
        return text
    return '"' + "".join(map(_logfmt_escape, text)) + '"'


def _logfmt_escape(char: str) -> str:
    if escape := _LOGFMT_ESCAPES.get(char):
        return escape
    if char.isprintable():
        return char
    code_point = ord(char)
    return f"\\u{code_point:04x}" if code_point <= 0xFFFF else f"\\U{code_point:08x}"


def _customize_loguru():
    # Loguru's default icon for "DEBUG" is no icon.
    logger.level("DEBUG", icon="D ")


# Distinguishes log lines from different replicas and deployments.
_RAILWAY_IDENTITY = {
    key: value
    for key, value in {
        "replica_id": flags.RAILWAY_REPLICA_ID,
        "deployment_id": flags.RAILWAY_DEPLOYMENT_ID,
    }.items()
    if value
}


def _record_to_railway_json(record: loguru_Record):
    structured: dict[str, typing.Any] = {
        # Railway only lets the Log Explorer filter on top-level fields, so contextual values (from logger.bind(),
        # logger.contextualize(), and message format arguments) go at the top level. The fields below take precedence
        # over any of them with the same name.
        **record["extra"],
        "timestamp": record["time"].isoformat(),
        "message": record["message"],
        "level": record["level"].name,
        "logger": record["name"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "process_id": record["process"].id,
        "process_name": record["process"].name,
        "thread_id": record["thread"].id,
        "thread_name": record["thread"].name,
        **_RAILWAY_IDENTITY,
    }
    if (exc := record["exception"]) is not None:
        structured.update({
            "exc_type": str(exc.type.__name__) if exc.type else None,
            "exc_msg": str(exc.value) if exc.value else None,
            "exc_tb": traceback.format_exception(exc.value, limit=15, chain=True) if exc.value else None,
        })
    # default=str and OPT_NON_STR_KEYS avoids disappearing log lines if any context variables are of an unusual type.
    try:
        return orjson.dumps(structured, default=str, option=orjson.OPT_NON_STR_KEYS).decode()
    except orjson.JSONEncodeError:
        # orjson rejects integers beyond 64 bits and strings that aren't valid UTF-8, and default= doesn't apply to
        # either. json.dumps accepts both, so fall back to it rather than lose the line.
        return json.dumps(structured, default=str)


def _stdout_railway_sink(message: loguru_Message):
    """Railway's log viewer expects a simple structured format with "message" and "level" fields."""
    serialized = _record_to_railway_json(message.record)
    print(serialized)


def _log_unraisable(unraisable: sys.UnraisableHookArgs) -> None:
    """Logs exceptions that Python cannot raise, such as those from __del__ methods and weakref callbacks."""
    message = unraisable.err_msg or "Exception ignored in"
    if unraisable.object is not None:
        message = f"{message}: {unraisable.object!r}"
    logger.opt(exception=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback)).error(message)


def setup(*, patcher: Callable[[loguru_Record], None] | None = None) -> None:
    """
    Configures logging using loguru.

    If specified, `patcher` will be applied to every record, allowing callers to enrich log records.
    """
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET, force=True)
    _customize_loguru()
    logger.configure(patcher=patcher)

    for name in logging.root.manager.loggerDict:
        existing_logger = logging.getLogger(name)
        existing_logger.handlers = []
        existing_logger.propagate = True

    _configure_third_party_levels()

    _silence_noisy_loggers()

    # Different runtime environments benefit from different logging formats. We can improve legibility for developers
    # by using a custom format when in a development environment, and generate structured logs when running in
    # production environments.
    match flags.LOG_FORMAT:
        case LogFormat.FRIENDLY:
            logger.remove()
            logger.add(
                sys.stdout,
                format="<cyan>{process.id}.{thread.name}</cyan> | <green>{time:HH:mm:ss}</green> | "
                "<level>{level.icon}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {extra} <level>{message}</level>",
            )
        case LogFormat.STRUCTURED_RAILWAY:
            logger.remove()
            logger.add(_stdout_railway_sink)
            # Replace the default unraisablehook behavior (stderr) with structured output.
            # We do not override threading.excepthook because doing so is brittle, and Sentry will already capture
            # those exceptions.
            sys.unraisablehook = _log_unraisable
        case _:
            # allow loguru default behavior
            pass


def granian_handler() -> logging.Handler:
    """Handler factory referenced by granian_log_config.json.

    This allows Granian's access logs to be emitted in the structured format, along with the context variables.
    """
    setup()
    return InterceptHandler()


def _silence_noisy_loggers():
    # Example: logger.disable("xngin.apiserver")
    pass


def _configure_third_party_levels():
    # Configure the Google SDK to log to loguru. Google's Python logging practices are poorly documented but manual
    # inspection shows that they generally put loggers in the `google` namespace. There are also some old docs:
    # https://cloud.google.com/python/docs/reference/google-cloud-developerconnect/latest
    logging.getLogger("google").setLevel(logging.INFO)  # Tip: set to WARN to log HTTP interactions with BigQuery APIs.

    # Only show warnings from SQLAlchemy pool classes. DEBUG will log on every connection open/close.
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARN)

    # sqlalchemy.orm: at INFO, this will log its internal model of the entity metadata.
    logging.getLogger("sqlalchemy.orm").setLevel(logging.WARN)

    # sqlalchemy.dialects.postgresql will log Postgres wire protocol NOTICE messages at INFO level.
    logging.getLogger("sqlalchemy.dialects.postgresql").setLevel(logging.INFO)

    # Conditionally enable the logging of SQL generated by SQLAlchemy for customer warehouses.
    # Tip: set this to DEBUG to see details on table metadata such as column name and types.
    logging.getLogger("sqlalchemy.engine.Engine.xngin_dwh").setLevel(
        logging.INFO if flags.LOG_SQL_DWH else logging.WARN
    )

    # Conditionally enable the logging of SQL generated by SQLAlchemy for the xngin app database.
    logging.getLogger("sqlalchemy.engine.Engine.xngin_app").setLevel(
        logging.INFO if flags.LOG_SQL_APP_DB else logging.WARN
    )

    logging.getLogger("httpcore").setLevel(logging.WARN)
    logging.getLogger("httpcore2").setLevel(logging.WARN)
    logging.getLogger("httpx2").setLevel(logging.WARN)
    logging.getLogger("urllib3").setLevel(logging.WARN)
    logging.getLogger("watchfiles.main").setLevel(logging.WARN)
