import os
from contextlib import asynccontextmanager

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


app = FastAPI(lifespan=lifespan)
exceptionhandlers.setup(app)
middleware.setup(app)
customlogging.setup()
secretservice.setup()
routes.register(app)
auth_dependencies.setup(app)

app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]
