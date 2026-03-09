from collections.abc import (
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from http import (
    HTTPMethod,
    HTTPStatus,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Self,
    TypedDict,
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
    GetParticipantAssignmentResponse,
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


class AdminAPIClientResult[Status: HTTPStatus, Model](NamedTuple):
    status: Status
    data: Model
    model: type[Model]
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
        result: AdminAPIClientResult[HTTPStatus, Any],
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
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
            timeout = USE_CLIENT_DEFAULT  # Hide the warning generated by Starlette.

        response = self.client.request(
            method.name,
            url,
            params=self._filter_and_encode_params(query_params),
            headers=self._filter_and_encode_params(header_params),
            cookies=cookies,
            json=body,
            timeout=timeout or USE_CLIENT_DEFAULT,
        )
        status = HTTPStatus(response.status_code)

        if status not in models:
            model = Any
            data = response.text
        else:
            model = models[status]
            if is_streaming_json and status == default_status:

                def data_iter() -> Iterator[Any]:
                    for part in response.iter_lines():
                        yield TypeAdapter(model).validate_json(part)

                data = data_iter()
            elif _is_no_body_response(method=method, status=status):
                data = None
            else:
                data = TypeAdapter(model).validate_json(response.text)

        result = AdminAPIClientResult(
            status=status,
            data=data,
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity]: ...
    @overload
    def caller_identity(
        self,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
    ): ...
    def caller_identity(
        self,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def logout(
        self,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
    @overload
    def logout(
        self,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
    ): ...
    def logout(
        self,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Any]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def create_snapshot(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def list_organizations(
        self,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse]: ...
    @overload
    def list_organizations(
        self,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
    ): ...
    def list_organizations(
        self,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def create_organizations(
        self,
        body: CreateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse]: ...
    @overload
    def create_organizations(
        self,
        body: CreateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def create_organizations(
        self,
        body: CreateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def add_webhook_to_organization(
        self,
        body: AddWebhookToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse]: ...
    @overload
    def add_webhook_to_organization(
        self,
        body: AddWebhookToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def add_webhook_to_organization(
        self,
        body: AddWebhookToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def list_organization_webhooks(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse]: ...
    @overload
    def list_organization_webhooks(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def list_organization_webhooks(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_organization_webhook(
        self,
        body: UpdateOrganizationWebhookRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def regenerate_webhook_auth_token(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
    @overload
    def regenerate_webhook_auth_token(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def regenerate_webhook_auth_token(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_webhook_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        webhook_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationEventsResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationEventsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def add_member_to_organization(
        self,
        body: AddMemberToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
    @overload
    def add_member_to_organization(
        self,
        body: AddMemberToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def add_member_to_organization(
        self,
        body: AddMemberToOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def remove_member_from_organization(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        user_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def update_organization(
        self,
        body: UpdateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Any]: ...
    @overload
    def update_organization(
        self,
        body: UpdateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_organization(
        self,
        body: UpdateOrganizationRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def get_organization(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse]: ...
    @overload
    def get_organization(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_organization(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def list_organization_datasources(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse]: ...
    @overload
    def list_organization_datasources(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def list_organization_datasources(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def create_datasource(
        self,
        body: CreateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse]: ...
    @overload
    def create_datasource(
        self,
        body: CreateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
    ): ...
    def create_datasource(
        self,
        body: CreateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def update_datasource(
        self,
        body: UpdateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
    @overload
    def update_datasource(
        self,
        body: UpdateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_datasource(
        self,
        body: UpdateDatasourceRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def get_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse]: ...
    @overload
    def get_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def inspect_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse]: ...
    @overload
    def inspect_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
    ): ...
    def inspect_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceTableResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceTableResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
    ): ...
    def inspect_table_in_datasource(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        table_name: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_datasource(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def list_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListParticipantsTypeResponse]: ...
    @overload
    def list_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListParticipantsTypeResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def list_participant_types(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def create_participant_type(
        self,
        body: CreateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateParticipantsTypeResponse]: ...
    @overload
    def create_participant_type(
        self,
        body: CreateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateParticipantsTypeResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def create_participant_type(
        self,
        body: CreateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], InspectParticipantTypesResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], InspectParticipantTypesResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def get_participant_type(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantsTypeResponse]: ...
    @overload
    def get_participant_type(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantsTypeResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
    ): ...
    def get_participant_type(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_participant_type(
        self,
        body: UpdateParticipantsTypeRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_participant(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def list_api_keys(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse]: ...
    @overload
    def list_api_keys(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def list_api_keys(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def create_api_key(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse]: ...
    @overload
    def create_api_key(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def create_api_key(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_api_key(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        api_key_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def analyze_experiment(
        self,
        baseline_arm_id: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def analyze_cmab_experiment(
        self,
        body: CMABContextInputRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def commit_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
    @overload
    def commit_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def commit_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def abandon_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
    @overload
    def abandon_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def abandon_experiment(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def list_organization_experiments(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse]: ...
    @overload
    def list_organization_experiments(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def list_organization_experiments(
        self,
        organization_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def get_experiment_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse]: ...
    @overload
    def get_experiment_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_experiment_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def get_experiment_assignments_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse]: ...
    @overload
    def get_experiment_assignments_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_experiment_assignments_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def get_experiment_assignments_as_csv_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Any]: ...
    @overload
    def get_experiment_assignments_as_csv_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_experiment_assignments_as_csv_for_ui(
        self,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def get_experiment_assignment_for_participant(
        self,
        create_if_none: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantAssignmentResponse]: ...
    @overload
    def get_experiment_assignment_for_participant(
        self,
        create_if_none: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantAssignmentResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_experiment_assignment_for_participant(
        self,
        create_if_none: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        participant_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
            path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/{participant_id}",  # noqa: RUF100,RUF027
            method=HTTPMethod.GET,
            default_status=HTTPStatus.OK,
            models={
                HTTPStatus.OK: GetParticipantAssignmentResponse,
                HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                HTTPStatus.FORBIDDEN: HTTPExceptionError,
                HTTPStatus.NOT_FOUND: HTTPExceptionError,
                HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
            },
            path_params={
                "datasource_id": datasource_id,
                "experiment_id": experiment_id,
                "participant_id": participant_id,
            },
            query_params={
                "create_if_none": create_if_none,
                "random_state": random_state,
            },
            raise_if_not_default_status=raise_if_not_default_status,
            client_exts=client_exts,
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_experiment(
        self,
        body: UpdateExperimentRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_experiment(
        self,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_experiment_data(
        self,
        body: DeleteExperimentDataRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]: ...
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
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )

    @overload
    def power_check(
        self,
        body: PowerRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse]: ...
    @overload
    def power_check(
        self,
        body: PowerRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
        | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
    ): ...
    def power_check(
        self,
        body: PowerRequest = ADMIN_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = ADMIN_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        return self._route_handler(  # type: ignore
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
        )
