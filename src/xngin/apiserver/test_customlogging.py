import json
import logging
import sys
import warnings
from datetime import UTC, datetime
from uuid import UUID

import pytest
from loguru import logger

from xngin.apiserver import customlogging


@pytest.fixture
def records():
    captured = []
    handler_id = logger.add(lambda message: captured.append(message.record), level="INFO")
    yield captured
    logger.remove(handler_id)


def test_railway_json_puts_context_at_top_level(records):
    with logger.contextualize(experiment_id="exp_1"):
        logger.bind(message="clobbered?").info("hello {name}", name="world")

    (record,) = records
    structured = json.loads(customlogging._record_to_railway_json(record))

    assert structured["experiment_id"] == "exp_1"
    assert structured["name"] == "world"
    # Contextual values do not override the standard fields.
    assert structured["message"] == "hello world"
    assert "extra" not in structured


def test_railway_json_serializes_values_json_does_not_support(records):
    started_at = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    logger.bind(started_at=started_at, run_id=UUID(int=1), counts={1: "one"}).info("hello")

    (record,) = records
    structured = json.loads(customlogging._record_to_railway_json(record))

    assert structured["started_at"] == "2026-10-31T12:00:00+00:00"
    assert structured["run_id"] == "00000000-0000-0000-0000-000000000001"
    assert structured["counts"] == {"1": "one"}


def test_railway_json_falls_back_for_values_orjson_rejects(records):
    logger.bind(big=2**70, surrogate="\udc80").info("hello")

    (record,) = records
    structured = json.loads(customlogging._record_to_railway_json(record))

    assert structured["big"] == 2**70
    assert structured["surrogate"] == "\udc80"
    assert structured["message"] == "hello"


@pytest.fixture
def stdlib_logger():
    stdlib_logger = logging.getLogger("xngin.test_customlogging.stdlib")
    stdlib_logger.addHandler(customlogging.InterceptHandler())
    stdlib_logger.propagate = False
    yield stdlib_logger
    stdlib_logger.handlers.clear()
    stdlib_logger.propagate = True


def test_intercepted_records_describe_their_stdlib_origin(records, stdlib_logger, monkeypatch):
    # Wrap Logger.callHandlers the way Sentry's logging integration does.
    original_call_handlers = logging.Logger.callHandlers

    def wrapped_call_handlers(self, record):
        return original_call_handlers(self, record)

    monkeypatch.setattr(logging.Logger, "callHandlers", wrapped_call_handlers)

    stdlib_logger.warning("from stdlib")
    expected_line = sys._getframe().f_lineno - 1

    (record,) = records
    assert record["name"] == "xngin.test_customlogging.stdlib"
    assert record["function"] == "test_intercepted_records_describe_their_stdlib_origin"
    assert record["module"] == "test_customlogging"
    assert record["line"] == expected_line
    assert record["level"].name == "WARNING"


class _RaisesOnDelete:
    def __del__(self):
        raise RuntimeError("from __del__")


def test_unraisable_exceptions_are_logged(records, monkeypatch):
    monkeypatch.setattr(sys, "unraisablehook", customlogging._log_unraisable)

    _RaisesOnDelete()

    (record,) = records
    assert record["level"].name == "ERROR"
    assert "_RaisesOnDelete.__del__" in record["message"]
    assert record["exception"].type is RuntimeError


@pytest.fixture
def granian_access_logger():
    access_logger = logging.getLogger(customlogging.GRANIAN_ACCESS_LOGGER)
    original = access_logger.handlers, access_logger.propagate, access_logger.level
    access_logger.handlers = [customlogging.InterceptHandler()]
    access_logger.propagate = False
    # Other tests may raise the root logger's level, which this logger would otherwise inherit.
    access_logger.setLevel(logging.INFO)
    yield access_logger
    access_logger.handlers, access_logger.propagate, access_logger.level = original


def _log_granian_request(access_logger, **overrides):
    """Logs a request the way Granian does: a format string, with the request's values as a dict."""
    request = {
        "addr": "203.0.113.9",
        "time": "2026-10-31 12:00:00 -0700",
        "dt_ms": 12.34567,
        "status": 200,
        # A crafted path: Granian passes it percent-decoded, so it can contain quotes and newlines.
        "path": '/x" 500 0.1ms\nGET /admin',
        "query_string": "a=1&b=2",
        "method": "GET",
        "scheme": "https",
        "protocol": "HTTP/1.1",
    } | overrides
    access_logger.info('%(addr)s - "%(method)s %(path)s %(protocol)s" %(status)d', request)


def test_granian_access_logs_fields_in_structured_logs(records, granian_access_logger, monkeypatch):
    monkeypatch.setattr(customlogging.flags, "LOG_FORMAT", customlogging.LogFormat.STRUCTURED_RAILWAY)

    _log_granian_request(granian_access_logger)

    (record,) = records
    structured = json.loads(customlogging._record_to_railway_json(record))
    assert structured["message"] == 'GET "/x\\" 500 0.1ms\\nGET /admin" 200'
    assert structured["http_method"] == "GET"
    # JSON escapes the field, so it holds the path exactly as the app received it.
    assert structured["http_path"] == '/x" 500 0.1ms\nGET /admin'
    assert structured["http_query"] == "a=1&b=2"
    assert structured["http_status"] == 200
    assert structured["http_duration_ms"] == 12.346
    assert structured["http_protocol"] == "HTTP/1.1"
    assert structured["client_addr"] == "203.0.113.9"
    assert structured["logger"] == customlogging.GRANIAN_ACCESS_LOGGER


def test_granian_access_logs_key_value_pairs_in_other_formats(records, granian_access_logger, monkeypatch):
    monkeypatch.setattr(customlogging.flags, "LOG_FORMAT", customlogging.LogFormat.FRIENDLY)

    _log_granian_request(granian_access_logger, path="/v1/x")
    _log_granian_request(granian_access_logger, path="/v1/x", query_string="")

    with_query, without_query = records
    assert with_query["message"] == 'dur=12.346 ip=203.0.113.9 method=GET path=/v1/x q="a=1&b=2" code=200'
    assert "http_method" not in with_query["extra"]
    # An empty query string is omitted.
    assert without_query["message"] == "dur=12.346 ip=203.0.113.9 method=GET path=/v1/x code=200"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("/v1/x:batch@(1),2", "/v1/x:batch@(1),2", id="path-punctuation-bare"),
        pytest.param("/café", "/café", id="non-ascii-bare"),
        pytest.param("", '""', id="empty"),
        pytest.param("a b", '"a b"', id="space"),
        pytest.param("a=1&b=2", '"a=1&b=2"', id="equals"),
        pytest.param('a"b', '"a\\"b"', id="quote"),
        pytest.param("a\\b", '"a\\\\b"', id="backslash"),
        pytest.param("a\nb\rc\td", '"a\\nb\\rc\\td"', id="newline-cr-tab"),
        pytest.param("a\x1b[31mb", '"a\\u001b[31mb"', id="terminal-escape"),
        # A bidirectional override: invisible, but it reverses how the text after it is displayed.
        pytest.param("a\u202eb", '"a\\u202eb"', id="bidi-override"),
    ],
)
def test_logfmt_value(value, expected):
    assert customlogging._logfmt_value(value) == expected


@pytest.fixture
def captured_warnings():
    py_warnings = logging.getLogger("py.warnings")
    py_warnings.addHandler(customlogging.InterceptHandler())
    py_warnings.propagate = False
    # captureWarnings(True) is a no-op if logging believes it is already capturing, but pytest restores
    # warnings.showwarning around each test, so reset it first.
    logging.captureWarnings(False)
    logging.captureWarnings(True)
    yield
    logging.captureWarnings(False)
    py_warnings.handlers.clear()
    py_warnings.propagate = True


def test_python_warnings_are_logged_with_context(records, captured_warnings):
    # pytest intercepts warnings.warn(), so call the hook that logging.captureWarnings() installs directly.
    with logger.contextualize(experiment_id="exp_1"):
        warnings.showwarning(RuntimeWarning("invalid value encountered in multiply"), RuntimeWarning, "lm.py", 1992)

    (record,) = records
    structured = json.loads(customlogging._record_to_railway_json(record))
    assert structured["level"] == "WARNING"
    assert structured["logger"] == "py.warnings"
    assert structured["experiment_id"] == "exp_1"
    assert "RuntimeWarning: invalid value encountered in multiply" in structured["message"]
