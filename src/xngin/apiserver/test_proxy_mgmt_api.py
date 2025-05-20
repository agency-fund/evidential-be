import json
from pathlib import Path

from fastapi.testclient import TestClient
from xngin.apiserver import conftest, constants
from xngin.apiserver.routers.stateless_api_types import (
    CommitRequest,
)
from xngin.apiserver.main import app
from xngin.apiserver.routers.proxy_mgmt_api_types import (
    WebhookResponse,
)
from xngin.apiserver.testing.xurl import Xurl

conftest.setup(app)
client = TestClient(app)
client.base_url = client.base_url.join(constants.API_PREFIX_V1)


def load_mock_response_from_xurl(mocker, file):
    """Returns a tuple with the Xurl obj specified in file and a mock response."""

    # Set up the mock response - first load our test data.
    data_file = str(Path(__file__).parent / "testdata" / "nonbulk" / file)
    with open(data_file, encoding="utf-8") as f:
        contents = f.read()
    xurl = Xurl.from_script(contents)
    # TODO: consider using dep injection for the httpx client
    mock_response = mocker.Mock()
    mock_response.status_code = xurl.expected_status
    expected_response_as_dict = json.loads(xurl.expected_response or "{}")
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
