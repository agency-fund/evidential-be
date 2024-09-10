import glob
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from xngin.apiserver.main import app
from xngin.apiserver.dependencies import settings_dependency
from xngin.apiserver.settings import SettingsForTesting, XnginSettings
from xngin.apiserver.testing.hurl import Hurl


def get_settings_for_test() -> XnginSettings:
    filename = Path(__file__).parent / "testdata/xngin.testing.settings.json"
    with open(filename) as f:
        try:
            contents = f.read()
            return TypeAdapter(SettingsForTesting).validate_json(contents)
        except ValidationError as pyve:
            print(f"Failed to parse {filename}. Contents:\n{contents}")
            raise pyve


# https://fastapi.tiangolo.com/advanced/testing-dependencies/#use-the-appdependency_overrides-attribute
app.dependency_overrides[settings_dependency] = get_settings_for_test

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
