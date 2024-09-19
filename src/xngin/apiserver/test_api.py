import glob
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xngin.apiserver import conftest
from xngin.apiserver.main import app
from xngin.apiserver.testing.hurl import Hurl

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


@pytest.mark.parametrize("script", STATIC_TESTS)
def test_api(script):
    with open(script, "r") as f:
        contents = f.read()
    hurl = Hurl.from_script(contents)
    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )
    temporary = tempfile.NamedTemporaryFile(delete=False)
    with temporary as tmpf:
        tmpf.write(response.content)
    assert (
        response.status_code == hurl.expected_status
    ), f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
    assert response.json() == json.loads(
        hurl.expected_response
    ), f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
