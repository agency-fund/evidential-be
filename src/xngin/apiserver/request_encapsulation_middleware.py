import json

import starlette.datastructures
from jsonpath import JSONPointer, JSONPointerError
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_SUPPORTED_METHODS = {"PATCH", "POST", "PUT"}


class RequestEncapsulationBadRequestError(Exception):
    pass


class RequestEncapsulationMiddleware:
    """ASGI middleware that unwraps nested JSON request bodies using JSON Pointer paths.

    Activates on POST, PATCH, or PUT requests with Content-Type: application/json when the
    unwrap_param query parameter is present. Extracts a value from the request
    body and replaces the body with that extracted value.

    This middleware will block requests and return a 400 in these cases: when an empty request body
    is sent, or when the JSON pointer is invalid, or when the JSON pointer doesn't refer to
    a defined value within the request body. This middleware does not explicitly handle
    JSONDecodeErrors.

    In this example, the request handler will receive {"actual": "content"}.

        POST /api/endpoint?_unwrap=/data/payload
        Content-Type: application/json
        {"data": {"payload": {"actual": "content"}}}
    """

    def __init__(self, app: ASGIApp, *, unwrap_param: str = "_unwrap"):
        """Initialize the middleware.

        Args:
            app: The ASGI application to wrap.
            unwrap_param: Query parameter name that specifies the JSON Pointer path.
        """
        self._app = app
        self._unwrap_param = unwrap_param

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        method = scope["method"]
        if method not in _SUPPORTED_METHODS:
            await self._app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        content_type = headers.get("content-type")
        if content_type is None or not content_type.lower().startswith("application/json"):
            await self._app(scope, receive, send)
            return
        params = starlette.datastructures.QueryParams(scope["query_string"])
        unwrap = params.get(self._unwrap_param, None)
        if unwrap is None:
            await self._app(scope, receive, send)
            return

        rw = _RequestRewriter(self._app, unwrap_arg=self._unwrap_param, unwrap_path=unwrap)
        await rw(scope, receive, send)


class _RequestRewriter:
    """Rewrites the request body by extracting a value at a JSON Pointer path.

    Intercepts the request body, parses it as JSON, extracts the value at the
    specified path, and replaces the body with the extracted value.
    """

    def __init__(self, app: ASGIApp, *, unwrap_arg: str, unwrap_path: str) -> None:
        """Initialize the rewriter.

        Args:
            app: The ASGI application to wrap.
            unwrap_arg: Name of the query parameter (used in error messages).
            unwrap_path: JSON Pointer path to extract from the request body.
        """
        self._app = app
        self._unwrap_arg = unwrap_arg
        self._unwrap_path = unwrap_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        self._recv = receive
        try:
            await self._app(scope, self.receive, send)
        except RequestEncapsulationBadRequestError as err:
            resp = JSONResponse({"message": str(err)}, status_code=400)
            await resp(scope, self._recv, send)

    async def receive(self) -> Message:
        message = await self._recv()
        assert message["type"] == "http.request"
        body = message["body"]
        if body == b"":
            raise RequestEncapsulationBadRequestError(f"{self._unwrap_arg} requires a non-empty request body.")
        more_body = message.get("more_body", False)
        if more_body:
            message = await self._recv()
            if message["body"] != b"":
                raise NotImplementedError("streaming requests not supported")

        parsed = json.loads(body)
        try:
            resolved = JSONPointer(self._unwrap_path).resolve(parsed)
        except JSONPointerError as err:
            raise RequestEncapsulationBadRequestError(f"{self._unwrap_arg} is not a valid JSONPointer path.") from err
        message["body"] = json.dumps(resolved).encode()
        return message
