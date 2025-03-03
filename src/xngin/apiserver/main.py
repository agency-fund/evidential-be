import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from xngin.apiserver import database, exceptionhandlers, middleware, constants
from xngin.apiserver.flags import PUBLISH_ALL_DOCS
from xngin.apiserver.routers import (
    experiments,
    experiments_api,
    experiments_proxy_mgmt_api,
    oidc,
    admin,
    oidc_dependencies,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if sentry_dsn := os.environ.get("SENTRY_DSN"):
    import sentry_sdk

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.environ.get("ENVIRONMENT", "local"),
        traces_sample_rate=1.0,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
    )


class MisconfiguredError(Exception):
    pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
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
        yield


# TODO: pass parameters to Swagger to support OIDC/PKCE
app = FastAPI(lifespan=lifespan)
exceptionhandlers.setup(app)
middleware.setup(app)

app.include_router(
    experiments_api.router, prefix=constants.API_PREFIX_V1, tags=["Experiment Design"]
)

app.include_router(
    experiments.router, prefix=constants.API_PREFIX_V1, tags=["Experiment Management"]
)

app.include_router(
    experiments_proxy_mgmt_api.router,
    prefix=constants.API_PREFIX_V1,
    tags=["Experiment Management Webhooks"],
)


def enable_oidc_api():
    app.include_router(
        oidc.router,
        prefix=constants.API_PREFIX_V1,
        tags=["Auth"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )


def enable_admin_api():
    app.include_router(
        admin.router,
        prefix=constants.API_PREFIX_V1,
        tags=["Admin"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )


if oidc.is_enabled():
    enable_oidc_api()

if oidc.is_enabled() and admin.is_enabled():
    enable_admin_api()

oidc_dependencies.setup(app)


def custom_openapi():
    """Customizes the generated OpenAPI schema."""
    if app.openapi_schema:  # cache
        return app.openapi_schema

    # Overrides the operationId values in the OpenAPI spec to generate humane names
    # based on the method name. This avoids generating long, ugly names downstream.
    # Note: ensure all API methods have names you'd like to appear in the generated APIs.
    for route in app.routes:
        if isinstance(route, APIRoute):
            # uses the Python API method name
            route.operation_id = route.name

    openapi_schema = get_openapi(
        title="xngin: Experiments API",
        version="0.9.0",
        summary="",
        description="",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def main():
    database.setup()

    import uvicorn

    # Handy for debugging in your IDE
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("UVICORN_PORT", 8000)))


if __name__ == "__main__":
    main()
