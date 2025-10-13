import json

import jsonpath.pointer
import starlette.datastructures
from fastapi import HTTPException
from jsonpath import JSONPointer, JSONPointerError
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestEncapsulationMiddleware:
    """ASGI middleware that unwraps nested JSON request bodies using JSON Pointer paths.

    Activates on POST requests with application/json content when the unwrap_param
    query parameter is present. Extracts a nested value from the request body and
    replaces the body with that extracted value.

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
        if method != "POST":
            await self._app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if headers.get("content-type") != "application/json":
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
        self._send = send
        await self._app(scope, self.receive, send)

    async def receive(self) -> Message:
        message = await self._recv()
        assert message["type"] == "http.request"
        body = message["body"]
        more_body = message.get("more_body", False)
        if more_body:
            message = await self._recv()
            if message["body"] != b"":
                raise NotImplementedError("streaming requests not supported")

        parsed = json.loads(body)
        try:
            resolved = JSONPointer(self._unwrap_path).resolve(parsed)
        except JSONPointerError as e:
            raise HTTPException(
                status_code=400,
                detail=f"{self._unwrap_arg} does not contain a valid JSONPointer path.",
            ) from e
        if resolved is jsonpath.pointer.UNDEFINED:
            raise HTTPException(
                status_code=400,
                detail=f"{self._unwrap_arg} refers to an undefined value.",
            )
        message["body"] = json.dumps(resolved).encode()
        return message
