import glob
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xngin.apiserver import conftest
from xngin.apiserver.main import app
from xngin.apiserver.testing.hurl import Hurl

logger = logging.getLogger(__name__)

conftest.setup(app)
client = TestClient(app)

STATIC_TESTS = tuple(glob.glob(str(Path(__file__).parent / "testdata/*.hurl")))


def trunc(s, n=4096):
    """Truncates a string at a length that is usable in unit tests."""
    if isinstance(s, bytes):
        s = str(s)
    if len(s) > n:
        s = s[:n] + "..."
    return s


@pytest.fixture
def update_api_tests_flag(pytestconfig):
    """Returns true iff the UPDATE_API_TESTS environment variable resembles a truthy value.

    TODO: replace this with something less error-prone.
    """
    return os.environ.get("UPDATE_API_TESTS", "").lower() in ("true", "1")


@pytest.mark.parametrize("script", STATIC_TESTS)
def test_api(script, update_api_tests_flag):
    with open(script) as f:
        contents = f.read()
    hurl = Hurl.from_script(contents)
    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=".hurl")  # noqa: SIM115
    # Write the actual response to a temporary file. If an exception is thrown, we optionally replace the script we just
    # executed with the new script.
    with temporary as tmpf:
        actual = hurl.model_copy()
        actual.expected_status = response.status_code
        actual.expected_response = json.dumps(response.json(), indent=2, sort_keys=True)
        tmpf.write(actual.to_script().encode("utf-8"))
    try:
        assert response.status_code == hurl.expected_status, (
            f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
        )
        assert response.json() == json.loads(hurl.expected_response), (
            f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
        )
    except AssertionError:
        if update_api_tests_flag:
            logger.info(f"Updating API test {script}.")
            shutil.copy(temporary.name, script)
        raise
