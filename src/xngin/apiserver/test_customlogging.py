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
