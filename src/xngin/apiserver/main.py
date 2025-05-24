import dataclasses
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from xngin.apiserver import (
    customlogging,
    exceptionhandlers,
    middleware,
    database,
)
from xngin.apiserver.flags import PUBLISH_ALL_DOCS
from xngin.apiserver.routers import (
    admin,
    experiments,
    proxy_mgmt_api,
    oidc,
    oidc_dependencies,
    stateless_api,
    healthchecks,
)
from loguru import logger

from xngin.apiserver.settings import get_settings_for_server

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
    logger.info(f"Starting server: {__name__}")

    settings = get_settings_for_server()
    logger.info(f"trusted_ips: {settings.trusted_ips}")
    logger.info(
        f"database connection timeout (seconds): {settings.db_connect_timeout_secs}"
    )

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
        database.setup()
        yield


# TODO: pass parameters to Swagger to support OIDC/PKCE
app = FastAPI(lifespan=lifespan)
exceptionhandlers.setup(app)
middleware.setup(app)
customlogging.setup()

app.include_router(experiments.router, tags=["Experiment Integration"])

app.include_router(
    stateless_api.router,
    tags=["Stateless Experiment Design"],
)

app.include_router(
    proxy_mgmt_api.router,
    tags=["Stateless Experiment Design"],
)

app.include_router(healthchecks.router, tags=["Health Checks"], include_in_schema=False)


def enable_oidc_api():
    app.include_router(
        oidc.router,
        tags=["Auth"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )


def enable_admin_api():
    app.include_router(
        admin.router,
        tags=["Admin"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )


if oidc.is_enabled():
    enable_oidc_api()

if oidc.is_enabled() and admin.is_enabled():
    enable_admin_api()

oidc_dependencies.setup(app)


@dataclasses.dataclass
class TagDocumentation:
    visible: bool
    definition: dict[str, str]


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

    visible_tags = [
        TagDocumentation(
            visible=True,
            definition={
                "name": "Stateless Experiment Design",
                "description": " (⚠️ New clients: use Experiment Integration APIs.) Methods for designing and saving experiments in which a client manages all state persistence.",
            },
        ),
        TagDocumentation(
            visible=True,
            definition={
                "name": "Experiment Integration",
                "description": "Methods for a client to use when integrating Evidential experiments and assignments with their own serving infrastructure.",
            },
        ),
        TagDocumentation(
            visible=PUBLISH_ALL_DOCS,
            definition={"name": "Auth", "description": "Methods for handling SSO."},
        ),
        TagDocumentation(
            visible=PUBLISH_ALL_DOCS,
            definition={
                "name": "Admin",
                "description": "Methods supporting the Evidential UI.",
            },
        ),
    ]
    openapi_schema = get_openapi(
        title="xngin: Experiments API",
        version="0.9.0",
        contact={
            "name": "Agency Fund",
            "url": "https://www.agency.fund",
            "email": "evidential-support@agency.fund",
        },
        summary="",
        description="",
        tags=[
            tag.definition
            for tag in sorted(visible_tags, key=lambda t: t.definition["name"])
            if tag.visible
        ],
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
