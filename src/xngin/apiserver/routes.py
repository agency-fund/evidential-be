from typing import TYPE_CHECKING

from fastapi.responses import RedirectResponse

from xngin.apiserver import flags
from xngin.apiserver.flags import PUBLISH_ALL_DOCS
from xngin.apiserver.routers import devidp, healthchecks_api
from xngin.apiserver.routers.admin import admin_api
from xngin.apiserver.routers.admin_integrations import admin_integrations_api
from xngin.apiserver.routers.auth import auth_api
from xngin.apiserver.routers.experiments import experiments_api
from xngin.apiserver.routers.integrations import integrations_api

if TYPE_CHECKING:
    from fastapi import FastAPI

# FastAPI serves Swagger UI here in development. Elsewhere, this path redirects to the hosted API reference.
SWAGGER_UI_PATH = "/docs"
HOSTED_API_DOCS_URL = "https://docs.evidential.dev/api"


def register(app: FastAPI):
    if not flags.is_dev_environment():
        app.add_api_route(
            SWAGGER_UI_PATH,
            lambda: RedirectResponse(HOSTED_API_DOCS_URL),
            methods=["GET"],
            include_in_schema=False,
        )

    app.include_router(experiments_api.router, tags=["Experiment Integration"])

    app.include_router(healthchecks_api.router, tags=["Health Checks"], include_in_schema=False)

    app.include_router(
        auth_api.router,
        tags=["Auth"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )

    if flags.DEVIDP_ENABLED:
        if not flags.is_dev_environment():
            raise devidp.DevIdpNotAllowedError(
                f"{flags.ENV_XNGIN_DEVIDP} is set, but the development-only identity provider may only run when "
                "ENVIRONMENT is unset or dev."
            )
        app.include_router(devidp.router, include_in_schema=False)

    app.include_router(
        admin_api.router,
        tags=["Admin"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )

    app.include_router(
        admin_integrations_api.router,
        tags=["Admin: Third-Party Tools Integrations"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )

    app.include_router(
        integrations_api.router,
        tags=["Third-Party Tools Integrations"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )
