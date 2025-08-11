import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from xngin.apiserver import (
    constants,
    customlogging,
    database,
    exceptionhandlers,
    middleware,
    routes,
)
from xngin.apiserver.openapi import custom_openapi
from xngin.apiserver.routers.auth import auth_dependencies
from xngin.xsecrets import secretservice

if sentry_dsn := os.environ.get("SENTRY_DSN"):
    import sentry_sdk
    from sentry_sdk import scrubber

    denylist = [
        *scrubber.DEFAULT_DENYLIST,
        *[
            "code",  # OIDC
            "code_verifier",  # OIDC
            "credentials_base64",  # sqlalchemy-bigquery
            "credentials_info",  # sqlalchemy-bigquery
            "database_url",  # common name of variable containing application database credentials
        ],
    ]
    pii_denylist = [
        *scrubber.DEFAULT_PII_DENYLIST,
        *[
            # Header names are exposed to Sentry in a normalized form.
            constants.HEADER_API_KEY.lower().replace("-", "_"),
            constants.HEADER_WEBHOOK_TOKEN.lower().replace("-", "_"),
            "email",
        ],
    ]

    sentry_sdk.init(
        db_query_source_threshold_ms=10,  # Capture origin of queries that exceed this time (default 100).
        dsn=sentry_dsn,
        environment=os.environ.get("ENVIRONMENT", "local"),
        traces_sample_rate=1.0,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
        send_default_pii=False,
        event_scrubber=sentry_sdk.scrubber.EventScrubber(
            denylist=denylist, pii_denylist=pii_denylist
        ),
    )


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


# TODO: pass parameters to Swagger to support OIDC/PKCE
app = FastAPI(lifespan=lifespan)
exceptionhandlers.setup(app)
middleware.setup(app)
customlogging.setup()
secretservice.setup()
routes.register(app)
auth_dependencies.setup(app)

app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]
