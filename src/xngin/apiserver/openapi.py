import dataclasses
from typing import TYPE_CHECKING

from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from xngin.apiserver import constants, flags
from xngin.apiserver.flags import PUBLISH_ALL_DOCS

if TYPE_CHECKING:
    from fastapi import FastAPI


@dataclasses.dataclass
class TagDocumentation:
    visible: bool
    definition: dict[str, str]


# Rendered as the "Experiment Integration" tag description in the API docs. Per our docs conventions, multi-line text
# uses Markdown for lists only (no headings or bold), and leads with orientation before technical detail. The
# API_PREFIX and API_KEY_HEADER placeholders are substituted with the real version prefix and header name.
EXPERIMENT_INTEGRATION_DESCRIPTION = (
    """\
## Integrate experiments into your application using Evidential.

Evidential's Experiments API allows you to integrate Evidential experiments with your application or workflows using
a REST API.

Tip: This documentation is interactive. Click the "Authorize" button at the top of the page to set your data source
API key. Create an API key in [Evidential](XNGIN_PUBLIC_PROTOCOL://XNGIN_PUBLIC_HOSTNAME/).

Evidential supports many experimentation and integration strategies. The specific endpoints you will integrate with
depend on the type of experiment you are running and your integration strategy.

Preassigned A/B experiments — assignments are created when the experiment is designed, before the experiment starts:

- `GET API_PREFIX/experiments/{experiment_id}/assignments` lists all assignments.
- `GET API_PREFIX/experiments/{experiment_id}/assignments/csv` exports all assignments as CSV.
- `GET API_PREFIX/experiments/{experiment_id}/assignments/{participant_id}` looks up one participant's assignment. For
  applications that can't store assignments separately, your application can call this method at runtime.

Online A/B experiments — assignments are created as participants arrive:

- `GET API_PREFIX/experiments/{experiment_id}/assignments/{participant_id}` gets or creates a participant's assignment.
- `POST API_PREFIX/experiments/{experiment_id}/assignments/{participant_id}/assign_with_filters`
  gets or creates an assignment when the experiment uses server-side filtering.

Multi-armed Bandit (MAB) experiments — assignments are created as participants arrive, and you report each outcome:

- `GET API_PREFIX/experiments/{experiment_id}/assignments/{participant_id}` gets or creates a participant's assignment.
- `POST API_PREFIX/experiments/{experiment_id}/assignments/{participant_id}/outcome`
  records the outcome for a participant's arm.

Contextual Multi-armed Bandit (CMAB) experiments — assignments use context values you supply, and you report each
outcome:

- `POST API_PREFIX/experiments/{experiment_id}/assignments/{participant_id}/assign_cmab`
  gets or creates a participant's assignment from the context values you supply.
- `POST API_PREFIX/experiments/{experiment_id}/assignments/{participant_id}/outcome`
  records the outcome for a participant's arm.

These endpoints can be used to synchronize experiment state with your workflows:

- `GET API_PREFIX/experiments` lists experiments on the data source.
- `GET API_PREFIX/experiments/{experiment_id}` gets an experiment's design and assignment specs.

Authentication: Every method in this API requires the `API_KEY_HEADER` request header containing your data source API
key. Manage your keys under Settings, in the Data Sources section.

Error handling: Branch on the HTTP status code, not on the text of the error message. Status codes are stable, but
message text may change. 2xx codes indicate success. 4xx codes usually indicate a problem with your request. 5xx codes
indicate a server-side error and may be retried.

Retries: Retry only transient failures, meaning network errors and 5xx responses. Use exponential backoff with
jitter, and cap the number of attempts. Do not retry 4xx responses. They will fail again until you change the request
or until the experiment, data source, or assignment state changes. Assignment requests are safe to retry, because they
return the existing assignment when one was already created.
"""
    .replace("API_PREFIX", constants.API_PREFIX_V1)
    .replace("API_KEY_HEADER", constants.HEADER_API_KEY)
    .replace("XNGIN_PUBLIC_HOSTNAME", flags.XNGIN_PUBLIC_HOSTNAME)
    .replace("XNGIN_PUBLIC_PROTOCOL", flags.XNGIN_PUBLIC_PROTOCOL)
)


def custom_openapi(app: FastAPI):
    """Customizes the generated OpenAPI schema."""
    if app.openapi_schema:  # cache
        return app.openapi_schema

    # Overrides the operationId values in the OpenAPI spec to generate humane names
    # based on the Python method name. This avoids downstream code generating ugly names downstream.
    # Note: ensure all API methods have names you'd like to appear in the generated APIs.
    seen = set()
    for route in app.routes:
        if isinstance(route, APIRoute):
            if route.name in seen:
                raise RuntimeError(f"Duplicate route name: {route.name}")
            # uses the Python API method name
            route.operation_id = route.name
            seen.add(route.name)

    visible_tags = [
        TagDocumentation(
            visible=True,
            definition={
                "name": "Experiment Integration",
                "description": EXPERIMENT_INTEGRATION_DESCRIPTION,
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
        TagDocumentation(
            visible=PUBLISH_ALL_DOCS,
            definition={
                "name": "Admin: Third-Party Tools Integrations",
                "description": "Methods for configuring third-party tool integrations on Evidential UI.",
            },
        ),
        TagDocumentation(
            visible=PUBLISH_ALL_DOCS,
            definition={
                "name": "Third-Party Tools Integrations",
                "description": "Methods for specific third-party tools to integrate with Evidential.",
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
        servers=[
            {
                "url": flags.XNGIN_PUBLIC_API_BASE_URL,
                "description": flags.XNGIN_PUBLIC_API_DESCRIPTION,
            }
        ]
        if flags.XNGIN_PUBLIC_API_BASE_URL
        else None,
        tags=[tag.definition for tag in sorted(visible_tags, key=lambda t: t.definition["name"]) if tag.visible],
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema
