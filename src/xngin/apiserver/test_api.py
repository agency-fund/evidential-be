import glob
import json
import logging
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from json import JSONDecodeError
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xngin.apiserver import conftest, constants, flags
from xngin.apiserver.api_types import CommitRequest
from xngin.apiserver.main import app
from xngin.apiserver.testing.assertions import assert_same
from xngin.apiserver.testing.xurl import Xurl
from xngin.apiserver.webhook_types import (
    WebhookUpdateCommitRequest,
    WebhookUpdateDescriptionRequest,
    WebhookUpdateTimestampsRequest,
    WebhookResponse,
)

logger = logging.getLogger(__name__)

conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


def mark_nondeterministic_tests(c):
    """Marks known nondeterministic tests with a mark that allows us to skip them except when requested."""
    return c


API_TESTS = [
    mark_nondeterministic_tests(c)
    for c in glob.glob(str(Path(__file__).parent / "testdata/*.xurl"))
]


def trunc(s, n=4096):
    """Truncates a string at a length that is usable in unit tests."""
    if isinstance(s, bytes):
        s = str(s)
    if len(s) > n:
        s = s[:n] + "..."
    return s


@pytest.fixture(name="update_api_tests_flag")
def fixture_update_api_tests_flag(pytestconfig):
    """Returns true iff the UPDATE_API_TESTS environment variable resembles a truthy value."""
    return flags.UPDATE_API_TESTS


@pytest.mark.parametrize("script", API_TESTS)
def test_api(script, update_api_tests_flag, use_deterministic_random):
    """Runs all the API_TESTS test scripts.

    Test scripts may omit asserting equality of actual response and expected response on specific paths. For example:

    [test_api]
    {
        "deepdiff_kwargs": {
            "exclude_paths": [
                "root['assignments']",
                "root['f_statistic']",
                "root['p_value']"
            ]
        }
    }

    Furthermore, arbitrary args can be passed to deepdiff.diff via the deepdiff_kwargs key.
    """

    def parse_trailer(trailer):
        """Extracts the value of deepdiff_kwargs from the JSON following any [test_api] header, or empty dict."""
        if not trailer:
            return {}
        matches = re.search(r"(?s)\[test_api\]\n(?P<args>.*)", trailer)
        if not matches:
            return {}
        try:
            config = json.loads(matches.group("args"))
            return config["deepdiff_kwargs"]
        except JSONDecodeError as e:
            pytest.fail(
                f"The script {script} contains an invalid JSON body in the [test_api] trailer: {e!s}"
            )

    with open(script) as f:
        contents = f.read()
    xurl = Xurl.from_script(contents)
    response = client.request(
        xurl.method, xurl.url, headers=xurl.headers, content=xurl.body
    )
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=".xurl")  # noqa: SIM115
    # Write the actual response to a temporary file. If an exception is thrown, we optionally replace the script we just
    # executed with the new script.
    with temporary as tmpf:
        actual = xurl.model_copy()
        actual.expected_status = response.status_code
        actual.expected_response = json.dumps(response.json(), indent=2, sort_keys=True)
        tmpf.write(actual.to_script().encode("utf-8"))
    try:
        assert response.status_code == xurl.expected_status, (
            f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
        )
        deepdiff_kwargs = parse_trailer(xurl.trailer)
        assert_same(
            response.json(),
            json.loads(xurl.expected_response),
            deepdiff_kwargs=deepdiff_kwargs,
            extra=f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}",
        )
    except AssertionError:
        if update_api_tests_flag:
            logger.info(f"Updating API test {script}.")
            shutil.copy(temporary.name, script)
        raise


def load_mock_response_from_xurl(mocker, file):
    """Returns a tuple with the Xurl obj specified in file and a mock response."""

    # Set up the mock response - first load our test data.
    data_file = str(Path(__file__).parent / "testdata/nonbulk" / file)
    with open(data_file, encoding="utf-8") as f:
        contents = f.read()
    xurl = Xurl.from_script(contents)
    # TODO: consider using dep injection for the httpx client
    mock_response = mocker.Mock()
    mock_response.status_code = xurl.expected_status
    expected_response_as_dict = json.loads(xurl.expected_response)
    if "body" in expected_response_as_dict:
        # Extract the fake webhook response from the mock api server response,
        # then re-serialize it to use as a mock webhook response.
        mock_webhook_response_dict = expected_response_as_dict["body"]
        body_text = json.dumps(mock_webhook_response_dict)
        mock_response.text = body_text
    else:
        # We're faking something else, so just pass it through.
        mock_response.text = xurl.expected_response
    return xurl, mock_response


def test_commit(mocker):
    """Test /commit success case by mocking the webhook request with pytest-mock"""

    (xurl, mock_response) = load_mock_response_from_xurl(mocker, "apitest.commit.xurl")
    # Mock the httpx.AsyncClient.post method
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        xurl.method, xurl.url, headers=xurl.headers, content=xurl.body
    )

    assert response.status_code == 200, response.text
    # Assert that httpx.AsyncClient.post was called with the correct arguments
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
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


def test_commit_without_power_analyses(mocker):
    """Test /commit succeeds even if it has no power analysis results."""

    (xurl, mock_response) = load_mock_response_from_xurl(mocker, "apitest.commit.xurl")
    new_body = CommitRequest.model_validate_json(xurl.body)
    new_body.power_analyses = None
    # Mock the httpx.AsyncClient.post method
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        xurl.method, xurl.url, headers=xurl.headers, content=new_body.model_dump_json()
    )

    # Assert that httpx.AsyncClient.post was called
    mock_request.assert_called_once()
    assert response.status_code == 200, response.text


def test_commit_when_webhook_has_non_200_status(mocker):
    (xurl, mock_response) = load_mock_response_from_xurl(mocker, "apitest.commit.xurl")
    # Override the mock status with an unaccepted code
    mock_response.status_code = 203
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        xurl.method, xurl.url, headers=xurl.headers, content=xurl.body
    )

    # Assert that downsream errors result in a 502 bad gateway error.
    assert response.status_code == 502, response.text
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    # And also verify that we are still passing back our WebhookResponse in the body, including the
    # upstream service's code.
    upstream_response = WebhookResponse.model_validate_json(response.text)
    assert upstream_response.status_code == 203
    assert upstream_response.body == mock_response.text


def test_commit_with_badconfig(mocker):
    """Test for error when settings are missing a commit action webhook."""

    (xurl, _) = load_mock_response_from_xurl(mocker, "apitest.commit.xurl")
    xurl.headers[constants.HEADER_CONFIG_ID] = "customer-test-badconfig"

    response = client.request(
        xurl.method, xurl.url, headers=xurl.headers, content=xurl.body
    )

    assert response.status_code == 501


def test_update_experiment_timestamps(mocker):
    """Test /update-commit?update_type=timestamps success case"""

    (xurl, mock_response) = load_mock_response_from_xurl(
        mocker, "apitest.update-commit.timestamps.xurl"
    )
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        xurl.method, xurl.url, headers=xurl.headers, content=xurl.body
    )

    assert response.status_code == 200, response.text
    # Now check that the right action was used given our update_type
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "PUT"
    assert (
        kwargs["url"]
        == "http://localhost:4001/dev/api/v1/experiment-commit/update-timestamps-for-experiment"
    )
    # And check that we not only have the expected payload structure that the upstream server expects, but that one of
    # the values within matches our test data.
    model = WebhookUpdateTimestampsRequest.model_validate(kwargs["json"])
    assert model.start_date == datetime.fromisoformat("2024-11-15T17:15:13.576Z")


def test_update_experiment_fails_when_end_before_start(mocker):
    """Test /update-commit?update_type=timestamps with bad end_date"""

    (hurl, _) = load_mock_response_from_xurl(
        mocker, "apitest.update-commit.timestamps.xurl"
    )
    # Replace the valid body with one that has end_date < start_date
    bad_body = WebhookUpdateCommitRequest.model_validate_json(hurl.body)
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

    (hurl, mock_response) = load_mock_response_from_xurl(
        mocker, "apitest.update-commit.description.xurl"
    )
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    assert response.status_code == 200, response.text
    mock_request.assert_called_once()
    # Now check that the right action was used given our update_type
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "PUT"
    assert (
        kwargs["url"]
        == "http://localhost:4001/dev/api/v1/experiment-commit/update-experiment-commit"
    )
    model = WebhookUpdateDescriptionRequest.model_validate(kwargs["json"])
    assert model.description == "Sample new description"


def test_update_experiment_fails_with_bad_ids(mocker):
    """Test /update-commit?update_type=description fails with non-uuid ids"""

    (hurl, _) = load_mock_response_from_xurl(
        mocker, "apitest.update-commit.description.bad.xurl"
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

    (hurl, mock_response) = load_mock_response_from_xurl(
        mocker, "apitest.assignment-file.xurl"
    )
    mock_request = mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )

    mock_request.assert_called_once()
    # Now check that the right action was used given our update_type
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "GET"
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
