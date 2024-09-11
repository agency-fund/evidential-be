import glob
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xngin.apiserver import conftest
from xngin.apiserver.main import app
from xngin.apiserver.testing.hurl import Hurl

conftest.setup(app)
client = TestClient(app)

STATIC_TESTS = tuple(glob.glob(str(Path(__file__).parent / "testdata/*.hurl")))


@pytest.mark.parametrize("script", STATIC_TESTS)
def test_api(script):
    with open(script, "r") as f:
        contents = f.read()
    hurl = Hurl.from_script(contents)
    response = client.request(
        hurl.method, hurl.url, headers=hurl.headers, content=hurl.body
    )
    assert response.status_code == hurl.expected_status, response.content
    assert response.json() == json.loads(hurl.expected_response)
