from collections.abc import (
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass
from http import (
    HTTPMethod,
    HTTPStatus,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypedDict,
    cast,
    overload,
)
from warnings import warn

from fastapi.encoders import jsonable_encoder
from httpx import (
    USE_CLIENT_DEFAULT,
    Client,
    Response,
    Timeout,
)
from pydantic import (
    BaseModel,
    TypeAdapter,
)

from xngin.apiserver.exceptionhandlers import XHTTPValidationError
from xngin.apiserver.routers.admin.admin_api import (
    HTTPExceptionError,
    MessageError,
)
from xngin.apiserver.routers.admin.admin_api_types import (
    AddMemberToOrganizationRequest,
    AddWebhookToOrganizationRequest,
    AddWebhookToOrganizationResponse,
    CreateApiKeyResponse,
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    CreateOrganizationRequest,
    CreateOrganizationResponse,
    CreateParticipantsTypeRequest,
    CreateParticipantsTypeResponse,
    CreateSnapshotResponse,
    DeleteExperimentDataRequest,
    GetDatasourceResponse,
    GetExperimentForUiResponse,
    GetOrganizationResponse,
    GetParticipantsTypeResponse,
    GetSnapshotResponse,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    InspectParticipantTypesResponse,
    ListApiKeysResponse,
    ListDatasourcesResponse,
    ListOrganizationEventsResponse,
    ListOrganizationsResponse,
    ListParticipantsTypeResponse,
    ListSnapshotsResponse,
    ListWebhooksResponse,
    SnapshotStatus,
    UpdateArmRequest,
    UpdateDatasourceRequest,
    UpdateExperimentRequest,
    UpdateOrganizationRequest,
    UpdateOrganizationWebhookRequest,
    UpdateParticipantsTypeRequest,
    UpdateParticipantsTypeResponse,
)
from xngin.apiserver.routers.auth.auth_api_types import CallerIdentity
from xngin.apiserver.routers.common_api_types import (
    CMABContextInputRequest,
    CreateExperimentRequest,
    CreateExperimentResponse,
    ExperimentAnalysisResponse,
    GetExperimentAssignmentsResponse,
    ListExperimentsResponse,
    PowerRequest,
    PowerResponse,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


def _is_no_body_response(*, method: HTTPMethod, status: HTTPStatus) -> bool:
    """Returns true if the method or status code indicate that there is no response body."""
    return (
        method == HTTPMethod.HEAD
        or status < HTTPStatus.OK
        or status
        in {
            HTTPStatus.NO_CONTENT,
            HTTPStatus.RESET_CONTENT,
            HTTPStatus.NOT_MODIFIED,
        }
    )


class AdminAPIClientExtensions(TypedDict, total=False):
    timeout: float | tuple[float | None, float | None, float | None, float | None] | Timeout | None


@dataclass(frozen=True)
class AdminAPIClientResult[Status: HTTPStatus, Data, ModelType]:
    # `data` is the decoded response body. For streaming responses it is an
    # iterator/async iterator of decoded items rather than a fully materialized value.
    # `model` is the runtime decoder used for that body. For normal and streaming
    # responses this is a class object like `User` with static type `type[User]`; for
    # no-body responses it is `None`.
    status: Status
    data: Data
    model: ModelType
    response: Response


class AdminAPIClientValidationError(BaseModel):
    loc: Sequence[str | int]
    msg: str
    type: str


class AdminAPIClientHTTPValidationError(BaseModel):
    detail: Sequence[AdminAPIClientValidationError]


class AdminAPIClientNotDefaultStatusError(Exception):
    def __init__(
        self,
        *,
        default_status: HTTPStatus,
        result: AdminAPIClientResult[HTTPStatus, Any, object],
    ) -> None:
        super().__init__(
            f"Expected default status {default_status.value} {default_status.phrase}, "
            f"but received {result.status.value} {result.status.phrase}."
        )
        self.default_status = default_status
        self.result = result


ADMIN_API_CLIENT_NOT_REQUIRED: Any = ...


class AdminAPIClient:  # noqa: RUF100,PLR0904
    def __init__(self, client: Client) -> None:
        self.client = client

    @classmethod
    @contextmanager
    def from_app(cls, app: FastAPI, base_url: str = "http://testserver") -> Iterator[Self]:
        from fastapi.testclient import TestClient  # noqa: PLC0415

        with TestClient(app, base_url=base_url) as client:
            yield cls(client)

    @staticmethod
    def _filter_and_encode_params(
        params: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if params is None:
            return None
        return {
            param: jsonable_encoder(value)
            for param, value in params.items()
            if value is not ADMIN_API_CLIENT_NOT_REQUIRED
        } or None

    def _route_handler(
        self,
        *,
        path: str,
        method: HTTPMethod,
        default_status: HTTPStatus,
        models: Mapping[HTTPStatus, Any],
        path_params: Mapping[str, Any] | None = None,
        query_params: Mapping[str, Any] | None = None,
        header_params: Mapping[str, Any] | None = None,
        cookie_params: Mapping[str, Any] | None = None,
        body_params: Mapping[str, Any] | None = None,
        is_body_embedded: bool = False,
        is_streaming_json: bool = False,
        raise_if_not_default_status: bool = False,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any, object]:
        if not client_exts:
            client_exts = {}

        url = path
        for param, value in (self._filter_and_encode_params(path_params) or {}).items():
            value_str = f"{value:0.20f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
            url = url.replace(f"{{{param}}}", value_str)

        body = self._filter_and_encode_params(body_params)
        if body and not is_body_embedded:
            body = next(iter(body.values()))

        cookies = self._filter_and_encode_params(cookie_params)
        if cookies:
            warn(
                "Setting cookie parameters directly on an endpoint function is "
                "experimental. (This is the cause for the DeprecationWarning by httpx "
                "below.)",
                UserWarning,
                stacklevel=3,
            )

        timeout = client_exts.get("timeout")
        request_timeout = timeout or USE_CLIENT_DEFAULT
        # Scuffed isinstance() check because we don't want to import
        # starlette.testclient.Testclient for users that don't need it.
        if (
            self.client.__class__.__name__ == "TestClient"
            and self.client.__class__.__module__ == "starlette.testclient"
            and timeout
        ):
            warn(
                "Starlette's TestClient (which you probably use via "
                f"{self.__class__.__name__}.from_app()) does not support timeouts. See "
                "https://github.com/Kludex/starlette/issues/1108 for more information.",
                DeprecationWarning,
                stacklevel=3,
            )
            request_timeout = USE_CLIENT_DEFAULT

        response = self.client.request(
            method.name,
            url,
            params=self._filter_and_encode_params(query_params),
            headers=self._filter_and_encode_params(header_params),
            cookies=cookies,
            json=body,
            timeout=request_timeout,
        )
        status = HTTPStatus(response.status_code)

        if status not in models:
            model: object = Any
            result: AdminAPIClientResult[HTTPStatus, Any, object] = AdminAPIClientResult(
                status=status,
                data=response.text,
                model=model,
                response=response,
            )
        else:
            model = models[status]
            if is_streaming_json and status == default_status:

                def data_iter() -> Iterator[Any]:
                    for part in response.iter_lines():
                        yield TypeAdapter(model).validate_json(part)

                result = AdminAPIClientResult(
                    status=status,
                    data=data_iter(),
                    model=model,
                    response=response,
                )
            elif _is_no_body_response(method=method, status=status):
                result = AdminAPIClientResult(
                    status=status,
                    data=None,
                    model=model,
                    response=response,
                )
            else:
                result = AdminAPIClientResult(
                    status=status,
                    data=TypeAdapter(model).validate_json(response.text),
                    model=model,
                    response=response,
                )
        if status != default_status and raise_if_not_default_status:
            raise AdminAPIClientNotDefaultStatusError(default_status=default_status, result=result)
        return result

    @overload
    def caller_identity(
        self,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity, type[CallerIdentity]]: ...
    @overload
    def caller_identity(
        self,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity, type[CallerIdentity]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
    ): ...
    def caller_identity(
        self,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity, type[CallerIdentity]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity, type[CallerIdentity]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
            ),
            self._route_handler(
                path="/v1/m/caller-identity",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CallerIdentity,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def logout(
        self,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def logout(
        self,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
    ): ...
    def logout(
        self,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
            ),
            self._route_handler(
                path="/v1/m/logout",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_snapshot(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        snapshot_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse, type[GetSnapshotResponse]]: ...
    @overload
    def get_snapshot(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        snapshot_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse, type[GetSnapshotResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def get_snapshot(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        snapshot_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse, type[GetSnapshotResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse, type[GetSnapshotResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetSnapshotResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                    "organization_id": organization_id,
                    "snapshot_id": snapshot_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_snapshots(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        status_: list[SnapshotStatus] | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse, type[ListSnapshotsResponse]]: ...
    @overload
    def list_snapshots(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        status_: list[SnapshotStatus] | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse, type[ListSnapshotsResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def list_snapshots(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        status_: list[SnapshotStatus] | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse, type[ListSnapshotsResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse, type[ListSnapshotsResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListSnapshotsResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                    "organization_id": organization_id,
                },
                query_params={
                    "page_size": page_size,
                    "page_token": page_token,
                    "skip": skip,
                    "status": status_,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_snapshot(
        self,
        _organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        snapshot_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]: ...
    @overload
    def delete_snapshot(
        self,
        _organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        snapshot_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def delete_snapshot(
        self,
        _organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        snapshot_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: Any,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": _organization_id,
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                    "snapshot_id": snapshot_id,
                },
                query_params={
                    "allow_missing": allow_missing,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def create_snapshot(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse, type[CreateSnapshotResponse]]: ...
    @overload
    def create_snapshot(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse, type[CreateSnapshotResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def create_snapshot(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse, type[CreateSnapshotResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse, type[CreateSnapshotResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CreateSnapshotResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                    "organization_id": organization_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_organizations(
        self,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse, type[ListOrganizationsResponse]]: ...
    @overload
    def list_organizations(
        self,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse, type[ListOrganizationsResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
    ): ...
    def list_organizations(
        self,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse, type[ListOrganizationsResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse, type[ListOrganizationsResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
            ),
            self._route_handler(
                path="/v1/m/organizations",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListOrganizationsResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def create_organizations(
        self,
        body: CreateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse, type[CreateOrganizationResponse]]: ...
    @overload
    def create_organizations(
        self,
        body: CreateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse, type[CreateOrganizationResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def create_organizations(
        self,
        body: CreateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse, type[CreateOrganizationResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], CreateOrganizationResponse, type[CreateOrganizationResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CreateOrganizationResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def add_webhook_to_organization(
        self,
        body: AddWebhookToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse, type[AddWebhookToOrganizationResponse]
    ]: ...
    @overload
    def add_webhook_to_organization(
        self,
        body: AddWebhookToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse, type[AddWebhookToOrganizationResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def add_webhook_to_organization(
        self,
        body: AddWebhookToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse, type[AddWebhookToOrganizationResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse, type[AddWebhookToOrganizationResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: AddWebhookToOrganizationResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_organization_webhooks(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse, type[ListWebhooksResponse]]: ...
    @overload
    def list_organization_webhooks(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse, type[ListWebhooksResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def list_organization_webhooks(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse, type[ListWebhooksResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse, type[ListWebhooksResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListWebhooksResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_organization_webhook(
        self,
        body: UpdateOrganizationWebhookRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def update_organization_webhook(
        self,
        body: UpdateOrganizationWebhookRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def update_organization_webhook(
        self,
        body: UpdateOrganizationWebhookRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks/{webhook_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                    "webhook_id": webhook_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def regenerate_webhook_auth_token(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def regenerate_webhook_auth_token(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def regenerate_webhook_auth_token(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks/{webhook_id}/authtoken",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                    "webhook_id": webhook_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_webhook_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def delete_webhook_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def delete_webhook_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks/{webhook_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                    "webhook_id": webhook_id,
                },
                query_params={
                    "allow_missing": allow_missing,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_organization_events(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], ListOrganizationEventsResponse, type[ListOrganizationEventsResponse]
    ]: ...
    @overload
    def list_organization_events(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], ListOrganizationEventsResponse, type[ListOrganizationEventsResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def list_organization_events(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], ListOrganizationEventsResponse, type[ListOrganizationEventsResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], ListOrganizationEventsResponse, type[ListOrganizationEventsResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/events",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListOrganizationEventsResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                query_params={
                    "page_size": page_size,
                    "page_token": page_token,
                    "skip": skip,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def add_member_to_organization(
        self,
        body: AddMemberToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def add_member_to_organization(
        self,
        body: AddMemberToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def add_member_to_organization(
        self,
        body: AddMemberToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/members",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def remove_member_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        user_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def remove_member_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        user_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def remove_member_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        user_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/members/{user_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                    "user_id": user_id,
                },
                query_params={
                    "allow_missing": allow_missing,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_organization(
        self,
        body: UpdateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]: ...
    @overload
    def update_organization(
        self,
        body: UpdateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def update_organization(
        self,
        body: UpdateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: Any,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_organization(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse, type[GetOrganizationResponse]]: ...
    @overload
    def get_organization(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse, type[GetOrganizationResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def get_organization(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse, type[GetOrganizationResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse, type[GetOrganizationResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetOrganizationResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_organization_datasources(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse, type[ListDatasourcesResponse]]: ...
    @overload
    def list_organization_datasources(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse, type[ListDatasourcesResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def list_organization_datasources(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse, type[ListDatasourcesResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse, type[ListDatasourcesResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListDatasourcesResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def create_datasource(
        self,
        body: CreateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse, type[CreateDatasourceResponse]]: ...
    @overload
    def create_datasource(
        self,
        body: CreateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse, type[CreateDatasourceResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ): ...
    def create_datasource(
        self,
        body: CreateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse, type[CreateDatasourceResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse, type[CreateDatasourceResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
            ),
            self._route_handler(
                path="/v1/m/datasources",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CreateDatasourceResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: XHTTPValidationError,
                    HTTPStatus.BAD_GATEWAY: MessageError,
                    HTTPStatus.GATEWAY_TIMEOUT: MessageError,
                },
                query_params={
                    "connectivity_check": connectivity_check,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_datasource(
        self,
        body: UpdateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def update_datasource(
        self,
        body: UpdateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def update_datasource(
        self,
        body: UpdateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse, type[GetDatasourceResponse]]: ...
    @overload
    def get_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse, type[GetDatasourceResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def get_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse, type[GetDatasourceResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse, type[GetDatasourceResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetDatasourceResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def inspect_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse, type[InspectDatasourceResponse]]: ...
    @overload
    def inspect_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse, type[InspectDatasourceResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ): ...
    def inspect_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse, type[InspectDatasourceResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse, type[InspectDatasourceResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/inspect",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: InspectDatasourceResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: MessageError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: XHTTPValidationError,
                    HTTPStatus.BAD_GATEWAY: MessageError,
                    HTTPStatus.GATEWAY_TIMEOUT: MessageError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                query_params={
                    "refresh": refresh,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def inspect_table_in_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        table_name: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], InspectDatasourceTableResponse, type[InspectDatasourceTableResponse]
    ]: ...
    @overload
    def inspect_table_in_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        table_name: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], InspectDatasourceTableResponse, type[InspectDatasourceTableResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ): ...
    def inspect_table_in_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        table_name: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], InspectDatasourceTableResponse, type[InspectDatasourceTableResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], InspectDatasourceTableResponse, type[InspectDatasourceTableResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/inspect/{table_name}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: InspectDatasourceTableResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: MessageError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: XHTTPValidationError,
                    HTTPStatus.BAD_GATEWAY: MessageError,
                    HTTPStatus.GATEWAY_TIMEOUT: MessageError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "table_name": table_name,
                },
                query_params={
                    "refresh": refresh,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_datasource(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def delete_datasource(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def delete_datasource(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "organization_id": organization_id,
                },
                query_params={
                    "allow_missing": allow_missing,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], ListParticipantsTypeResponse, type[ListParticipantsTypeResponse]
    ]: ...
    @overload
    def list_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListParticipantsTypeResponse, type[ListParticipantsTypeResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def list_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListParticipantsTypeResponse, type[ListParticipantsTypeResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], ListParticipantsTypeResponse, type[ListParticipantsTypeResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListParticipantsTypeResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def create_participant_type(
        self,
        body: CreateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], CreateParticipantsTypeResponse, type[CreateParticipantsTypeResponse]
    ]: ...
    @overload
    def create_participant_type(
        self,
        body: CreateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], CreateParticipantsTypeResponse, type[CreateParticipantsTypeResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def create_participant_type(
        self,
        body: CreateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], CreateParticipantsTypeResponse, type[CreateParticipantsTypeResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], CreateParticipantsTypeResponse, type[CreateParticipantsTypeResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CreateParticipantsTypeResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def inspect_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        expensive: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], InspectParticipantTypesResponse, type[InspectParticipantTypesResponse]
    ]: ...
    @overload
    def inspect_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        expensive: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], InspectParticipantTypesResponse, type[InspectParticipantTypesResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ): ...
    def inspect_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        expensive: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], InspectParticipantTypesResponse, type[InspectParticipantTypesResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], InspectParticipantTypesResponse, type[InspectParticipantTypesResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}/inspect",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: InspectParticipantTypesResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: MessageError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: XHTTPValidationError,
                    HTTPStatus.BAD_GATEWAY: MessageError,
                    HTTPStatus.GATEWAY_TIMEOUT: MessageError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "participant_id": participant_id,
                },
                query_params={
                    "expensive": expensive,
                    "refresh": refresh,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_participant_type(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], GetParticipantsTypeResponse, type[GetParticipantsTypeResponse]
    ]: ...
    @overload
    def get_participant_type(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantsTypeResponse, type[GetParticipantsTypeResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ): ...
    def get_participant_type(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantsTypeResponse, type[GetParticipantsTypeResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], GetParticipantsTypeResponse, type[GetParticipantsTypeResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetParticipantsTypeResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: MessageError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: XHTTPValidationError,
                    HTTPStatus.BAD_GATEWAY: MessageError,
                    HTTPStatus.GATEWAY_TIMEOUT: MessageError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "participant_id": participant_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_participant_type(
        self,
        body: UpdateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse, type[UpdateParticipantsTypeResponse]
    ]: ...
    @overload
    def update_participant_type(
        self,
        body: UpdateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse, type[UpdateParticipantsTypeResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def update_participant_type(
        self,
        body: UpdateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse, type[UpdateParticipantsTypeResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse, type[UpdateParticipantsTypeResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: UpdateParticipantsTypeResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "participant_id": participant_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_participant(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def delete_participant(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def delete_participant(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "participant_id": participant_id,
                },
                query_params={
                    "allow_missing": allow_missing,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_api_keys(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse, type[ListApiKeysResponse]]: ...
    @overload
    def list_api_keys(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse, type[ListApiKeysResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def list_api_keys(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse, type[ListApiKeysResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse, type[ListApiKeysResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/apikeys",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListApiKeysResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def create_api_key(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse, type[CreateApiKeyResponse]]: ...
    @overload
    def create_api_key(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse, type[CreateApiKeyResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def create_api_key(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse, type[CreateApiKeyResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse, type[CreateApiKeyResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/apikeys",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CreateApiKeyResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_api_key(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        api_key_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def delete_api_key(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        api_key_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def delete_api_key(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        api_key_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/apikeys/{api_key_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "api_key_id": api_key_id,
                    "datasource_id": datasource_id,
                },
                query_params={
                    "allow_missing": allow_missing,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def create_experiment(
        self,
        body: CreateExperimentRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        desired_n: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        stratify_on_metrics: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse, type[CreateExperimentResponse]]: ...
    @overload
    def create_experiment(
        self,
        body: CreateExperimentRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        desired_n: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        stratify_on_metrics: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse, type[CreateExperimentResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def create_experiment(
        self,
        body: CreateExperimentRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        desired_n: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        stratify_on_metrics: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse, type[CreateExperimentResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse, type[CreateExperimentResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CreateExperimentResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                query_params={
                    "desired_n": desired_n,
                    "random_state": random_state,
                    "stratify_on_metrics": stratify_on_metrics,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def analyze_experiment(
        self,
        baseline_arm_id: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]]: ...
    @overload
    def analyze_experiment(
        self,
        baseline_arm_id: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def analyze_experiment(
        self,
        baseline_arm_id: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ExperimentAnalysisResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                query_params={
                    "baseline_arm_id": baseline_arm_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def analyze_cmab_experiment(
        self,
        body: CMABContextInputRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]]: ...
    @overload
    def analyze_cmab_experiment(
        self,
        body: CMABContextInputRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def analyze_cmab_experiment(
        self,
        body: CMABContextInputRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], ExperimentAnalysisResponse, type[ExperimentAnalysisResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze_cmab",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ExperimentAnalysisResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def commit_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def commit_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def commit_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/commit",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.CONFLICT: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def abandon_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def abandon_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def abandon_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/abandon",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.CONFLICT: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def list_organization_experiments(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]]: ...
    @overload
    def list_organization_experiments(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def list_organization_experiments(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/experiments",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListExperimentsResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "organization_id": organization_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_experiment_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse, type[GetExperimentForUiResponse]]: ...
    @overload
    def get_experiment_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse, type[GetExperimentForUiResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def get_experiment_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse, type[GetExperimentForUiResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], GetExperimentForUiResponse, type[GetExperimentForUiResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetExperimentForUiResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_experiment_assignments_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[
        Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
    ]: ...
    @overload
    def get_experiment_assignments_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def get_experiment_assignments_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[
            Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[
                    Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetExperimentAssignmentsResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_experiment_assignments_as_csv_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]: ...
    @overload
    def get_experiment_assignments_as_csv_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def get_experiment_assignments_as_csv_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/csv",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: Any,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_experiment(
        self,
        body: UpdateExperimentRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def update_experiment(
        self,
        body: UpdateExperimentRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def update_experiment(
        self,
        body: UpdateExperimentRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_experiment(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def delete_experiment(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def delete_experiment(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                query_params={
                    "allow_missing": allow_missing,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_experiment_data(
        self,
        body: DeleteExperimentDataRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def delete_experiment_data(
        self,
        body: DeleteExperimentDataRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def delete_experiment_data(
        self,
        body: DeleteExperimentDataRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/data",  # noqa: RUF100,RUF027
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_arm(
        self,
        arm_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        body: UpdateArmRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]: ...
    @overload
    def update_arm(
        self,
        arm_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        body: UpdateArmRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ): ...
    def update_arm(
        self,
        arm_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        body: UpdateArmRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            AdminAPIClientHTTPValidationError,
            type[AdminAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None, None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    AdminAPIClientHTTPValidationError,
                    type[AdminAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/arms/{arm_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: None,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "arm_id": arm_id,
                    "datasource_id": datasource_id,
                    "experiment_id": experiment_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def power_check(
        self,
        body: PowerRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse, type[PowerResponse]]: ...
    @overload
    def power_check(
        self,
        body: PowerRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse, type[PowerResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ): ...
    def power_check(
        self,
        body: PowerRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse, type[PowerResponse]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
        | AdminAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
        ]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse, type[PowerResponse]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError, type[HTTPExceptionError]]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError, type[MessageError]]
                | AdminAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError, type[XHTTPValidationError]
                ]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError, type[MessageError]]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError, type[MessageError]]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/power",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: PowerResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: MessageError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: XHTTPValidationError,
                    HTTPStatus.BAD_GATEWAY: MessageError,
                    HTTPStatus.GATEWAY_TIMEOUT: MessageError,
                },
                path_params={
                    "datasource_id": datasource_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )
