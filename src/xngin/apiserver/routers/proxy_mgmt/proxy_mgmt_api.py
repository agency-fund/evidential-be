from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Response
from loguru import logger
from pydantic import BaseModel

from xngin.apiserver import constants
from xngin.apiserver.dependencies import (
    datasource_config_required,
    httpx_dependency,
)
from xngin.apiserver.routers.proxy_mgmt.proxy_mgmt_api_types import (
    STANDARD_WEBHOOK_RESPONSES,
    CommitRequest,
    WebhookCommitRequest,
    WebhookResponse,
)
from xngin.apiserver.settings import (
    DatasourceConfig,
    HttpMethodTypes,
    WebhookConfig,
    WebhookUrl,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1,
)


@router.post(
    "/commit",
    summary="Commit an experiment to a remote database.",
    responses=STANDARD_WEBHOOK_RESPONSES,
)
async def commit_experiment_wh(
    response: Response,
    body: CommitRequest,
    user_id: Annotated[str, Query(...)],
    http_client: Annotated[httpx.AsyncClient, Depends(httpx_dependency)],
    config: Annotated[DatasourceConfig, Depends(datasource_config_required)],
) -> WebhookResponse:
    webhook_config = config.webhook_config
    if webhook_config is None:
        raise HTTPException(501, "Webhook not configured.")
    action = webhook_config.actions.commit
    if action is None:
        raise HTTPException(501, "Action 'commit' not configured.")

    commit_payload = WebhookCommitRequest(
        creator_user_id=user_id,
        design_spec=body.design_spec,
        power_analyses=body.power_analyses,
        experiment_assignment=body.experiment_assignment,
    )

    response.status_code, payload = await make_webhook_request(
        http_client, webhook_config, action, commit_payload
    )
    return payload


async def make_webhook_request(
    http_client: httpx.AsyncClient,
    config: WebhookConfig,
    action: WebhookUrl,
    data: BaseModel,
) -> tuple[int, WebhookResponse]:
    """Helper function to make webhook requests with common error handling.

    Returns: tuple of (status_code, WebhookResponse to use as body)
    """
    return await make_webhook_request_base(
        http_client, config, action.method, action.url, data
    )


async def make_webhook_request_base(
    http_client: httpx.AsyncClient,
    config: WebhookConfig,
    method: HttpMethodTypes,
    url: str,
    data: BaseModel | None = None,
) -> tuple[int, WebhookResponse]:
    """Like make_webhook_request() but can directly take an http method and url.

    Returns: tuple of (status_code, WebhookResponse to use as body)
    """
    headers = {}
    auth_header_value = config.common_headers.authorization
    if auth_header_value is not None:
        headers["Authorization"] = auth_header_value.get_secret_value()
    headers["Accept"] = "application/json"
    # headers["Content-Type"] is set by httpx

    try:
        # Explicitly convert to a dict via pydantic since we use custom serializers
        json_data = data.model_dump(mode="json") if data else None
        upstream_response = await http_client.request(
            method=method, url=url, headers=headers, json=json_data
        )
        webhook_response = WebhookResponse.from_httpx(upstream_response)
        status_code = 200
        # Stricter than response.raise_for_status(), we require HTTP 200:
        if upstream_response.status_code != 200:
            logger.error(
                "ERROR response %s requesting webhook: %s",
                upstream_response.status_code,
                url,
            )
            status_code = 502
    except httpx.ConnectError as e:
        logger.exception("ERROR requesting webhook (ConnectError): {}", e.request.url)
        raise HTTPException(
            status_code=502, detail=f"Error connecting to {e.request.url}: {e}"
        ) from e
    except httpx.RequestError as e:
        logger.exception("ERROR requesting webhook: {}", e.request.url)
        raise HTTPException(status_code=500, detail="server error") from e
    else:
        # Always return a WebhookResponse in the body, even on non-200 responses.
        return status_code, webhook_response
