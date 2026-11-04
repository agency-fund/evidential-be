from fastapi import FastAPI
from fastapi.testclient import TestClient

from xngin.apiserver import exceptionhandlers
from xngin.apiserver.exceptions_common import DwhPermissionDeniedError
from xngin.apiserver.request_context_middleware import REQUEST_ID_HEADER, RequestContextMiddleware


def test_dwh_errors_answer_with_their_status_and_the_dwh_error_response_shape():
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    exceptionhandlers.setup(app)

    @app.get("/")
    def denied():
        raise DwhPermissionDeniedError("msg", hint="h", table_name="t")

    response = TestClient(app).get("/", headers={REQUEST_ID_HEADER: "req-1"})

    assert response.status_code == 424
    assert response.json() == {
        "message": "msg",
        "code": "dwh_permission_denied",
        "hint": "h",
        "hint_command": None,
        "dwh_message": None,
        "table_name": "t",
        "request_id": "req-1",
    }
