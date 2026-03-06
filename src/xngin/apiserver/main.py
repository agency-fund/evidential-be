import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from xngin.apiserver import (
    customlogging,
    database,
    exceptionhandlers,
    middleware,
    routes,
)
from xngin.apiserver.openapi import custom_openapi
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.ops import sentry
from xngin.xsecrets import secretservice

customlogging.setup()
sentry.setup()


class MisconfiguredError(Exception):
    pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting server: {__name__}")

    # Security: Disable the Google Cloud SDK's use of GCE metadata service by pointing it at localhost. This service
    # operates on behalf of customers who provide their own credentials. By setting these variables (and aborting if
    # GOOGLE_APPLICATION_CREDENTIALS) is set, we can reduce the chance that an implementation bug would result in
    # outbound requests using our privileges unintentionally.
    os.environ["GCE_METADATA_HOST"] = "127.0.0.1"
    os.environ["GCE_METADATA_IP"] = "127.0.0.1"
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") != "":
        raise MisconfiguredError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is set. "
            "This should not be set on a running xngin API server because it "
            "could cause the server to authenticate with the service account "
            "credentials on behalf of end-user requests. "
            "Please unset GOOGLE_APPLICATION_CREDENTIALS and try again."
        )
    else:
        async with database.setup():
            yield


app = FastAPI(
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)
exceptionhandlers.setup(app)
middleware.setup(app)
secretservice.setup()
routes.register(app)
auth_dependencies.setup(app)

app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]


def main_dev():
    """Entrypoint for instances of this service running in development environments.

    This is used instead of `fastapi dev` because it starts faster.
    """
    logger.info(f"Starting Uvicorn for development: {__name__}.main_dev.")
    uvicorn.run(
        "xngin.apiserver.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        proxy_headers=False,
        reload_dirs=["src"],
        reload_excludes=["test_*"],
        log_config={"version": 1, "disable_existing_loggers": False},
    )


def main_live():
    """Entrypoint for instances of this service running in environments where structured logging is desired.

    This is equivalent to the default `fastapi run` CLI behavior.
    """
    logger.info(f"Starting Uvicorn for live deployments: {__name__}.main_live.")
    # This replicates the `fastapi run` CLI behavior and also disables Uvicorn's default logging behavior.
    uvicorn.run(
        "xngin.apiserver.main:app",
        host="0.0.0.0",
        port=8000,
        log_config={"version": 1, "disable_existing_loggers": False},
    )
