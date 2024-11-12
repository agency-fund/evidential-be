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
    with open(script, "r") as f:
        contents = f.read()
    hurl = Hurl.from_script(contents)
    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=".hurl")
    # Write the actual response to a temporary file. If an exception is thrown, we optionally replace the script we just
    # executed with the new script.
    with temporary as tmpf:
        actual = hurl.model_copy()
        actual.expected_status = response.status_code
        actual.expected_response = json.dumps(response.json(), indent=2, sort_keys=True)
        tmpf.write(actual.to_script().encode("utf-8"))
    try:
        assert (
            response.status_code == hurl.expected_status
        ), f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
        assert response.json() == json.loads(
            hurl.expected_response
        ), f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
    except AssertionError:
        if update_api_tests_flag:
            logger.info(f"Updating API test {script}.")
            shutil.copy(temporary.name, script)
        raise


def test_commit_endpoint(mocker):
    # Set up the mock response - first load our test data.
    data_file = str(Path(__file__).parent / "testdata/nonbulk/apitest.commit.hurl")
    with open(data_file, "r", encoding="utf-8") as f:
        contents = f.read()
    hurl = Hurl.from_script(contents)
    # Mock the POST using pytest-mock
    # TODO: consider using dep injection for the httpx client
    mock_response = mocker.Mock()
    mock_response.status_code = hurl.expected_status
    expected_response_json = json.loads(hurl.expected_response)
    mock_response.json.return_value = expected_response_json
    # Mock the httpx.AsyncClient.post method
    mock_post = mocker.patch("httpx.AsyncClient.post", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    # Assert that httpx.AsyncClient.post was called with the correct arguments
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    # TODO: Replace with actual URL based on settings
    assert (
        kwargs["url"]
        == "http://localhost:4001/dev/api/v1/experiment-commit/save-experiment-commit"
    )
    assert kwargs["headers"]["Authorization"] == "abc"
    assert kwargs["json"]["creator_user_id"] == "commituser"
    assert "experiment_commit_datetime" in kwargs["json"]
    assert "experiment_commit_id" in kwargs["json"]

    # Assert that the response from our API is correct
    assert response.status_code == 200
    assert response.json() == expected_response_json


def test_commit_endpoint_badconfig():
    data_file = str(Path(__file__).parent / "testdata/nonbulk/apitest.commit.hurl")
    with open(data_file, "r", encoding="utf-8") as f:
        contents = f.read()
    hurl = Hurl.from_script(contents)
    # Load our bad settings that are missing a commit action.
    hurl.headers["Config-ID"] = "customer-test-badconfig"

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    assert response.status_code == 501
