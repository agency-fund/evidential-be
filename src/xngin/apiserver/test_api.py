from datetime import datetime, timedelta
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
from xngin.apiserver.webhook_types import (
    WebhookRequestUpdate,
    WebhookRequestUpdateDescriptions,
    WebhookRequestUpdateTimestamps,
    WebhookResponse,
)

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


@pytest.fixture(name="update_api_tests_flag")
def fixture_update_api_tests_flag(pytestconfig):
    """Returns true iff the UPDATE_API_TESTS environment variable resembles a truthy value.

    TODO: replace this with something less error-prone.
    """
    return os.environ.get("UPDATE_API_TESTS", "").lower() in ("true", "1")


@pytest.mark.parametrize("script", STATIC_TESTS)
def test_api(script, update_api_tests_flag, use_deterministic_random):
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


def load_mock_response_from_hurl(mocker, file):
    """Returns a tuple with the hurl obj specified in file and a mock response."""

    # Set up the mock response - first load our test data.
    data_file = str(Path(__file__).parent / "testdata/nonbulk" / file)
    with open(data_file, encoding="utf-8") as f:
        contents = f.read()
    hurl = Hurl.from_script(contents)
    # TODO: consider using dep injection for the httpx client
    mock_response = mocker.Mock()
    mock_response.status_code = hurl.expected_status
    expected_response_as_dict = json.loads(hurl.expected_response)
    if "body" in expected_response_as_dict:
        # Extract the fake webhook response from the mock api server response,
        # then re-serialize it to use as a mock webhook response.
        mock_webhook_response_dict = expected_response_as_dict["body"]
        body_text = json.dumps(mock_webhook_response_dict)
        mock_response.text = body_text
    else:
        # We're faking something else, so just pass it through.
        mock_response.text = hurl.expected_response
    return (hurl, mock_response)


def test_commit(mocker):
    """Test /commit success case by mocking the webhook request with pytest-mock"""

    (hurl, mock_response) = load_mock_response_from_hurl(mocker, "apitest.commit.hurl")
    # Mock the httpx.AsyncClient.post method
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    assert response.status_code == 200, response.text
    # Assert that httpx.AsyncClient.post was called with the correct arguments
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "post"
    assert (
        kwargs["url"]
        == "http://localhost:4001/dev/api/v1/experiment-commit/save-experiment-commit"
    )
    assert kwargs["headers"]["Authorization"] == "abc"
    assert kwargs["json"]["creator_user_id"] == "commituser"
    assert "experiment_commit_datetime" in kwargs["json"]
    assert "experiment_commit_id" in kwargs["json"]
    # We mocked the response body as text, so need to reconstruct the serialized form
    # of the expected response to compare.
    expected_response_model = WebhookResponse(
        status_code=mock_response.status_code, body=mock_response.text
    )
    assert response.text == expected_response_model.model_dump_json()


def test_commit_when_webhook_has_non_200_status(mocker):
    (hurl, mock_response) = load_mock_response_from_hurl(mocker, "apitest.commit.hurl")
    # Override the mock status with an unaccepted code
    mock_response.status_code = 203
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    # Assert that downsream errors result in a 502 bad gateway error.
    assert response.status_code == 502, response.text
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "post"
    # And also verify that we are still passing back our WebhookResponse in the body, including the
    # upstream service's code.
    upstream_response = WebhookResponse.model_validate_json(response.text)
    assert upstream_response.status_code == 203
    assert upstream_response.body == mock_response.text


def test_commit_with_badconfig(mocker):
    """Test for error when settings are missing a commit action webhook."""

    (hurl, _) = load_mock_response_from_hurl(mocker, "apitest.commit.hurl")
    hurl.headers["Config-ID"] = "customer-test-badconfig"

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    assert response.status_code == 501


def test_update_experiment_timestamps(mocker):
    """Test /update-commit?update_type=timestamps success case"""

    (hurl, mock_response) = load_mock_response_from_hurl(
        mocker, "apitest.update-commit.timestamps.hurl"
    )
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    assert response.status_code == 200, response.text
    # Now check that the right action was used given our update_type
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "put"
    assert (
        kwargs["url"]
        == "http://localhost:4001/dev/api/v1/experiment-commit/update-timestamps-for-experiment"
    )
    # And check that we not only have the expected payload structure that the upstream server expects, but that one of
    # the values within matches our test data.
    model = WebhookRequestUpdateTimestamps.model_validate(kwargs["json"])
    assert model.start_date == datetime.fromisoformat("2024-11-15T17:15:13.576Z")


def test_update_experiment_fails_when_end_before_start(mocker):
    """Test /update-commit?update_type=timestamps with bad end_date"""

    (hurl, _) = load_mock_response_from_hurl(
        mocker, "apitest.update-commit.timestamps.hurl"
    )
    # Replace the valid body with one that has end_date < start_date
    bad_body = WebhookRequestUpdate.model_validate_json(hurl.body)
    bad_body.update_json.end_date = bad_body.update_json.start_date - timedelta(days=1)
    hurl.body = bad_body.model_dump_json()
    # Expect to fail before even making the request due to validation error.
    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )
    # Check for validation error
    assert response.status_code == 422
    response_detail = response.json()["detail"][0]
    assert response_detail["type"] == "value_error"
    assert "end_date must be after start_date" in response_detail["msg"]


def test_update_experiment_description(mocker):
    """Test /update-commit?update_type=description success case"""

    (hurl, mock_response) = load_mock_response_from_hurl(
        mocker, "apitest.update-commit.description.hurl"
    )
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    assert response.status_code == 200, response.text
    mock_request.assert_called_once()
    # Now check that the right action was used given our update_type
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "put"
    assert (
        kwargs["url"]
        == "http://localhost:4001/dev/api/v1/experiment-commit/update-experiment-commit"
    )
    model = WebhookRequestUpdateDescriptions.model_validate(kwargs["json"])
    assert model.description == "Sample new description"


def test_update_experiment_fails_with_bad_ids(mocker):
    """Test /update-commit?update_type=description fails with non-uuid ids"""

    (hurl, _) = load_mock_response_from_hurl(
        mocker, "apitest.update-commit.description.bad.hurl"
    )
    # Expect to fail before even making the request due to validation error.
    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )
    # Check for validation error
    assert response.status_code == hurl.expected_status
    response_detail = response.json()["detail"][0]
    assert response_detail["type"] == "uuid_parsing"


def test_assignment_file(mocker):
    """Test /assignment-file?experiment_id=foo1 success case"""

    (hurl, mock_response) = load_mock_response_from_hurl(
        mocker, "apitest.assignment-file.hurl"
    )
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    mock_request.assert_called_once()
    # Now check that the right action was used given our update_type
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "get"
    assert (
        kwargs["url"]
        == "http://localhost:4001/dev/api/v1/experiment-commit/get-file-name-by-experiment-id/foo1"
    )
    assert kwargs["json"] is None

    assert response.status_code == 200, response.text
    # And verify that our mocked response was packaged up properly.
    upstream_response = WebhookResponse.model_validate_json(response.text)
    assert upstream_response.status_code == 200
    assert upstream_response.body == mock_response.text
