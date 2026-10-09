import json
import logging
import sys

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
