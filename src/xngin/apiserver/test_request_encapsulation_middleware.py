"""Unit tests for RequestEncapsulationMiddleware.

Tests use a standalone Starlette app with TestClient to validate middleware behavior.
"""

import json

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from xngin.apiserver.request_encapsulation_middleware import RequestEncapsulationMiddleware


async def echo_handler(request: Request) -> JSONResponse:
    """Test endpoint that echoes the request body back as JSON."""
    body = await request.body()
    if body:
        data = json.loads(body)
    else:
        data = None
    return JSONResponse({"received": data})


def create_test_app(unwrap_param: str = "_unwrap") -> Starlette:
    """Creates a Starlette app with RequestEncapsulationMiddleware for testing."""
    return Starlette(
        routes=[
            Route("/echo", echo_handler, methods=["GET", "POST", "PUT", "PATCH"]),
        ],
        middleware=[
            Middleware(RequestEncapsulationMiddleware, unwrap_param=unwrap_param),
        ],
    )


@pytest.fixture(scope="module")
def client():
    app = create_test_app()
    return TestClient(app)


@pytest.mark.parametrize(
    "method,query_path,input_data,expected",
    [
        (
            "POST",
            "/data/payload",
            {"data": {"payload": {"actual": "content"}}},
            {"actual": "content"},
        ),
        (
            "PUT",
            "/wrapper/inner",
            {"wrapper": {"inner": {"value": 42}}},
            {"value": 42},
        ),
        (
            "PATCH",
            "/updates",
            {"updates": {"field": "new_value"}},
            {"field": "new_value"},
        ),
    ],
    ids=["post", "put", "patch"],
)
def test_unwraps_nested_json_with_http_methods(client, method, query_path, input_data, expected):
    """Test unwrapping works with POST, PUT, and PATCH methods."""

    response = client.request(
        method,
        f"/echo?_unwrap={query_path}",
        json=input_data,
    )

    assert response.status_code == 200
    assert response.json() == {"received": expected}


@pytest.mark.parametrize(
    "query_path,input_data,expected",
    [
        ("/message", {"message": "hello world"}, "hello world"),
        ("/nullable", {"nullable": None}, None),
        ("/list", {"list": [1, 2, 3]}, [1, 2, 3]),
        ("/flag", {"flag": True}, True),
        ("/value", {"value": 0}, 0),
        ("/items/0", {"items": [{"first": "element"}, "second"]}, {"first": "element"}),
        ("", {"k": "v"}, {"k": "v"}),
        (
            "/level1/level2/level3/level4",
            {"level1": {"level2": {"level3": {"level4": {"deep": "value"}}}}},
            {"deep": "value"},
        ),
    ],
    ids=["string", "null", "array", "boolean", "zero_value", "zeroth_item", "full", "deep"],
)
def test_unwraps_primitive_values(client, query_path, input_data, expected):
    """Test unwrapping various primitive types and values."""

    response = client.post(
        f"/echo?_unwrap={query_path}",
        json=input_data,
    )

    assert response.status_code == 200
    assert response.json() == {"received": expected}


@pytest.mark.parametrize(
    "query_path,input_data,expected",
    [
        ("/text", {"text": ""}, ""),
        ("/obj", {"obj": {}}, {}),
        ("/arr", {"arr": []}, []),
    ],
    ids=["empty_string", "empty_object", "empty_array"],
)
def test_unwraps_empty_values(client, query_path, input_data, expected):
    """Test unwrapping empty strings, objects, and arrays."""

    response = client.post(
        f"/echo?_unwrap={query_path}",
        json=input_data,
    )

    assert response.status_code == 200
    assert response.json() == {"received": expected}


@pytest.mark.parametrize(
    "query_path,input_data",
    [
        ("data", {"data": "value"}),  # Invalid JSON Pointer (must start with /)
        ("/nonexistent", {"data": "value"}),  # Path doesn't exist
        ("/data/missing", {"data": {"present": "value"}}),  # Nested path doesn't exist
        ("/items/10", {"items": ["a", "b", "c"]}),  # Array index out of bounds
    ],
    ids=["invalid_pointer", "undefined_path", "undefined_nested_path", "array_out_of_bounds"],
)
def test_error_on_invalid_paths(client, query_path, input_data):
    response = client.post(
        f"/echo?_unwrap={query_path}",
        json=input_data,
    )

    assert response.status_code == 400
    assert "_unwrap is not a valid JSONPointer path" in response.json()["message"]


@pytest.mark.parametrize(
    "unwrap_param,query_param,query_value,input_data,expected",
    [
        (
            "_extract",
            "_extract",
            "/inner",
            {"inner": {"custom": "param"}},
            {"custom": "param"},
        ),
        (
            "_extract",
            "_unwrap",
            "/inner",
            {"inner": {"data": "value"}},
            {"inner": {"data": "value"}},
        ),
    ],
    ids=["custom_param_works", "default_param_ignored"],
)
def test_custom_unwrap_parameter(unwrap_param, query_param, query_value, input_data, expected):
    """Test using a custom unwrap parameter name."""
    app = create_test_app(unwrap_param=unwrap_param)
    client = TestClient(app)

    response = client.post(
        f"/echo?{query_param}={query_value}",
        json=input_data,
    )

    assert response.status_code == 200
    assert response.json() == {"received": expected}


def test_handles_empty_request_body(client):
    response = client.post(
        "/echo?_unwrap=/data",
        data=None,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == {"received": None}


@pytest.mark.parametrize(
    "query_value,input_key,expected_data",
    [
        ("/foo~1bar", "foo/bar", {"special": "key"}),
        ("/foo~0bar", "foo~bar", {"tilde": "value"}),
    ],
    ids=["slash_escape", "tilde_escape"],
)
def test_unwraps_with_special_characters_in_keys(client, query_value, input_key, expected_data):
    """Test JSON Pointer escape sequences for special characters."""

    response = client.post(
        f"/echo?_unwrap={query_value}",
        json={input_key: expected_data},
    )

    assert response.status_code == 200
    assert response.json() == {"received": expected_data}


@pytest.mark.parametrize(
    "query_value",
    ["%2Ffoo+bar", "/foo%20bar"],
    ids=["plus_escaped_space", "percent_escaped_space"],
)
def test_unwraps_with_url_encoded_spaces(client, query_value):
    """Test URL encoding of spaces in query parameters."""

    response = client.post(
        f"/echo?_unwrap={query_value}",
        json={"foo bar": {"tilde": "value"}},
    )

    assert response.status_code == 200
    assert response.json() == {"received": {"tilde": "value"}}


@pytest.mark.parametrize(
    "content_type", ["Application/JSON", "application/json; charset=utf-8"], ids=("casing", "charset")
)
def test_content_type_case_insensitive(client, content_type):
    response = client.post(
        "/echo?_unwrap=/data",
        json={"data": "value"},
        headers={"Content-Type": content_type},
    )

    assert response.status_code == 200
    assert response.json() == {"received": "value"}


def test_multiple_unwrap_parameters_uses_last(client):
    # Starlette's QueryParams.get() returns the last value when multiple exist
    response = client.post(
        "/echo?_unwrap=/data&_unwrap=/other",
        json={"data": {"first": "value"}, "other": {"second": "value"}},
    )

    assert response.status_code == 200
    # Starlette returns the last value for duplicate query parameters
    assert response.json() == {"received": {"second": "value"}}


def test_preserves_query_parameters_in_scope(client):
    response = client.post(
        "/echo?_unwrap=/data&other=param",
        json={"data": {"unwrapped": "value"}},
    )

    assert response.status_code == 200
    assert response.json() == {"received": {"unwrapped": "value"}}


def test_does_not_unwrap_without_query_parameter(client):
    original_data = {"data": {"payload": {"actual": "content"}}}
    response = client.post("/echo", json=original_data)

    assert response.status_code == 200
    assert response.json() == {"received": original_data}


def test_does_not_unwrap_get_requests(client):
    response = client.get("/echo?_unwrap=/data")

    assert response.status_code == 200
    assert response.json() == {"received": None}


def test_does_not_unwrap_non_json_content_type(client):
    # Send valid JSON but with wrong content type
    response = client.post(
        "/echo?_unwrap=/data",
        content=json.dumps({"data": "value"}).encode(),
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 200
    # Should receive the full body since it wasn't unwrapped due to content type
    assert response.json() == {"received": {"data": "value"}}
