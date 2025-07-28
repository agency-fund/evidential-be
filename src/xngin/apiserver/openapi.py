import dataclasses

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from xngin.apiserver import flags
from xngin.apiserver.flags import PUBLISH_ALL_DOCS


@dataclasses.dataclass
class TagDocumentation:
    visible: bool
    definition: dict[str, str]


def custom_openapi(app: FastAPI):
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
        title="Evidential Experiments API",
        version="0.9.0",
        contact={
            "name": "Evidential Developers",
            "url": flags.XNGIN_PRODUCT_HOMEPAGE,
            "email": flags.XNGIN_SUPPORT_EMAIL,
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
