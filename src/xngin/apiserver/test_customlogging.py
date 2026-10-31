import json
import logging
import sys
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
