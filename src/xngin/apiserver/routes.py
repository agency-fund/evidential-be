from fastapi import FastAPI

from xngin.apiserver.flags import PUBLISH_ALL_DOCS
from xngin.apiserver.routers import healthchecks_api
from xngin.apiserver.routers.admin import admin_api
from xngin.apiserver.routers.auth import auth_api
from xngin.apiserver.routers.experiments import experiments_api


def register(app: FastAPI):
    app.include_router(experiments_api.router, tags=["Experiment Integration"])

    app.include_router(healthchecks_api.router, tags=["Health Checks"], include_in_schema=False)

    app.include_router(
        auth_api.router,
        tags=["Auth"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )

    app.include_router(
        admin_api.router,
        tags=["Admin"],
        include_in_schema=PUBLISH_ALL_DOCS,
    )
