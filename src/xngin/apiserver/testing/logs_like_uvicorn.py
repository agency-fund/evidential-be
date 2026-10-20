from loguru import logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class LogsLikeUvicorn:
    """ASGI wrapper that logs at the two moments Uvicorn does, so a test can check what Uvicorn's lines would carry.

    "access" is logged when the application sends the response headers. Uvicorn writes its access line there: inside
    the request, after the dependencies and the handler have run.

    "after app" is logged once the request has fully unwound, whether or not the application raised. Uvicorn logs the
    traceback of an unhandled exception there, outside the application's middleware.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def logging_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                logger.info("access")
            await send(message)

        try:
            await self._app(scope, receive, logging_send)
        finally:
            if scope["type"] == "http":
                logger.info("after app")
