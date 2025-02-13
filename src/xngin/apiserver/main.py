import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from xngin.apiserver import database, exceptionhandlers, middleware, constants
from xngin.apiserver.routers import experiments_api, oidc, admin


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


# TODO: pass parameters to Swagger to support OIDC/PKCE
app = FastAPI(lifespan=lifespan)
exceptionhandlers.setup(app)
middleware.setup(app)

app.include_router(
    experiments_api.router, prefix=constants.API_PREFIX_V1, tags=["Experiment Design"]
)


def enable_oidc_api():
    app.include_router(
        oidc.router,
        prefix=constants.API_PREFIX_V1,
        tags=["Auth"],
        include_in_schema=os.environ.get("XNGIN_PUBLISH_ALL_DOCS", "").lower()
        in {"1", "true"},
    )


def enable_admin_api():
    app.include_router(
        admin.router,
        prefix=constants.API_PREFIX_V1,
        tags=["Admin"],
        include_in_schema=os.environ.get("XNGIN_PUBLISH_ALL_DOCS", "").lower()
        in {"1", "true"},
    )


if oidc.is_enabled():
    enable_oidc_api()


if oidc.is_enabled() and admin.is_enabled():
    enable_admin_api()


def custom_openapi():
    """Customizes the generated OpenAPI schema."""
    if app.openapi_schema:  # cache
        return app.openapi_schema
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
