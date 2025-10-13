"""Unit tests for RequestEncapsulationMiddleware.

Tests use a standalone Starlette app with TestClient to validate middleware behavior.
"""

import json

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


class TestRequestEncapsulationMiddleware:  # noqa: PLR0904
    """Test suite for RequestEncapsulationMiddleware."""

    def test_unwraps_nested_json_with_post(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/data/payload",
            json={"data": {"payload": {"actual": "content"}}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"actual": "content"}}

    def test_unwraps_nested_json_with_put(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.put(
            "/echo?_unwrap=/wrapper/inner",
            json={"wrapper": {"inner": {"value": 42}}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"value": 42}}

    def test_unwraps_nested_json_with_patch(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.patch(
            "/echo?_unwrap=/updates",
            json={"updates": {"field": "new_value"}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"field": "new_value"}}

    def test_does_not_unwrap_without_query_parameter(self):
        app = create_test_app()
        client = TestClient(app)

        original_data = {"data": {"payload": {"actual": "content"}}}
        response = client.post("/echo", json=original_data)

        assert response.status_code == 200
        assert response.json() == {"received": original_data}

    def test_does_not_unwrap_get_requests(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.get("/echo?_unwrap=/data")

        assert response.status_code == 200
        assert response.json() == {"received": None}

    def test_does_not_unwrap_non_json_content_type(self):
        app = create_test_app()
        client = TestClient(app)

        # Send valid JSON but with wrong content type
        response = client.post(
            "/echo?_unwrap=/data",
            content=json.dumps({"data": "value"}).encode(),
            headers={"Content-Type": "text/plain"},
        )

        assert response.status_code == 200
        # Should receive the full body since it wasn't unwrapped due to content type
        assert response.json() == {"received": {"data": "value"}}

    def test_unwraps_deeply_nested_path(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/level1/level2/level3/level4",
            json={"level1": {"level2": {"level3": {"level4": {"deep": "value"}}}}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"deep": "value"}}

    def test_unwraps_array_element(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/items/1",
            json={"items": ["first", {"target": "value"}, "third"]},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"target": "value"}}

    def test_unwraps_primitive_value(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/message",
            json={"message": "hello world"},
        )

        assert response.status_code == 200
        assert response.json() == {"received": "hello world"}

    def test_unwraps_null_value(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/nullable",
            json={"nullable": None},
        )

        assert response.status_code == 200
        assert response.json() == {"received": None}

    def test_unwraps_array_value(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/list",
            json={"list": [1, 2, 3]},
        )

        assert response.status_code == 200
        assert response.json() == {"received": [1, 2, 3]}

    def test_unwraps_boolean_value(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/flag",
            json={"flag": True},
        )

        assert response.status_code == 200
        assert response.json() == {"received": True}

    def test_error_on_invalid_json_pointer(self):
        app = create_test_app()
        client = TestClient(app)

        # Invalid JSON Pointer (must start with /)
        response = client.post(
            "/echo?_unwrap=data",
            json={"data": "value"},
        )

        assert response.status_code == 400
        assert "_unwrap is not a valid JSONPointer path" in response.json()["message"]

    def test_error_on_undefined_path(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/nonexistent",
            json={"data": "value"},
        )

        assert response.status_code == 400
        assert "_unwrap is not a valid JSONPointer path" in response.json()["message"]

    def test_error_on_undefined_nested_path(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/data/missing",
            json={"data": {"present": "value"}},
        )

        assert response.status_code == 400
        assert "_unwrap is not a valid JSONPointer path" in response.json()["message"]

    def test_custom_unwrap_parameter_name(self):
        app = create_test_app(unwrap_param="_extract")
        client = TestClient(app)

        response = client.post(
            "/echo?_extract=/inner",
            json={"inner": {"custom": "param"}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"custom": "param"}}

    def test_custom_unwrap_parameter_ignores_default_name(self):
        app = create_test_app(unwrap_param="_extract")
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/inner",
            json={"inner": {"data": "value"}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"inner": {"data": "value"}}}

    def test_unwraps_root_with_empty_string_pointer(self):
        app = create_test_app()
        client = TestClient(app)

        # Empty string is a valid JSON Pointer referring to the whole document
        response = client.post(
            "/echo?_unwrap=",
            json={"data": "value"},
        )

        assert response.status_code == 200
        # The root pointer should return the entire document
        assert response.json() == {"received": {"data": "value"}}

    def test_handles_empty_request_body(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/data",
            data=None,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        assert response.json() == {"received": None}

    def test_unwraps_with_special_characters_in_keys(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/foo~1bar",
            json={"foo/bar": {"special": "key"}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"special": "key"}}

    def test_unwraps_with_tilde_in_key(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/foo~0bar",
            json={"foo~bar": {"tilde": "value"}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"tilde": "value"}}

    def test_error_on_array_index_out_of_bounds(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/items/10",
            json={"items": ["a", "b", "c"]},
        )

        assert response.status_code == 400
        assert "_unwrap is not a valid JSONPointer path" in response.json()["message"]

    def test_unwraps_zero_index(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/items/0",
            json={"items": [{"first": "element"}, "second"]},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"first": "element"}}

    def test_unwraps_number_zero(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/value",
            json={"value": 0},
        )

        assert response.status_code == 200
        assert response.json() == {"received": 0}

    def test_unwraps_empty_string(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/text",
            json={"text": ""},
        )

        assert response.status_code == 200
        assert response.json() == {"received": ""}

    def test_unwraps_empty_object(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/obj",
            json={"obj": {}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {}}

    def test_unwraps_empty_array(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/arr",
            json={"arr": []},
        )

        assert response.status_code == 200
        assert response.json() == {"received": []}

    def test_multiple_unwrap_parameters_uses_last(self):
        app = create_test_app()
        client = TestClient(app)

        # Starlette's QueryParams.get() returns the last value when multiple exist
        response = client.post(
            "/echo?_unwrap=/data&_unwrap=/other",
            json={"data": {"first": "value"}, "other": {"second": "value"}},
        )

        assert response.status_code == 200
        # Starlette returns the last value for duplicate query parameters
        assert response.json() == {"received": {"second": "value"}}

    def test_preserves_query_parameters_in_scope(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/data&other=param",
            json={"data": {"unwrapped": "value"}},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"unwrapped": "value"}}

    def test_content_type_case_insensitive(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/data",
            json={"data": "value"},
            headers={"Content-Type": "Application/JSON"},
        )

        assert response.status_code == 200
        assert response.json() == {"received": "value"}

    def test_content_type_with_charset(self):
        app = create_test_app()
        client = TestClient(app)

        response = client.post(
            "/echo?_unwrap=/data",
            content=json.dumps({"data": {"value": "test"}}).encode(),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

        assert response.status_code == 200
        assert response.json() == {"received": {"value": "test"}}
