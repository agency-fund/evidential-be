from base64 import b64encode
from collections.abc import (
    Iterator,
    Mapping,
    MutableMapping,
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
    cast,
    overload,
)
from warnings import warn

from fastapi.encoders import jsonable_encoder
from fastapi.sse import ServerSentEvent
from httpx2 import (
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
    CreateUserRequest,
    CreateUserResponse,
    DeleteExperimentDataRequest,
    GetDatasourceResponse,
    GetExperimentForUiResponse,
    GetOrganizationResponse,
    GetParticipantsTypeResponse,
    GetSnapshotResponse,
    GetUserResponse,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    InspectParticipantTypesResponse,
    ListApiKeysResponse,
    ListDatasourcesResponse,
    ListOrganizationEventsResponse,
    ListOrganizationsResponse,
    ListParticipantsTypeResponse,
    ListSnapshotsResponse,
    ListUsersResponse,
    ListWebhooksResponse,
    PatchUserRequest,
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
    ListExperimentsResponse,
    PowerRequest,
    PowerResponse,
    SampleCalls,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


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


class AdminAPIClientSecurityParam(NamedTuple):
    kind: Literal[
        "http_bearer",
        "http_basic",
        "api_key_header",
        "api_key_cookie",
        "api_key_query",
    ]
    name: str
    value: str | tuple[str, str] | None


class AdminAPIClientSSE[Data](ServerSentEvent):
    data: Data | None = None


ADMIN_API_CLIENT_NOT_REQUIRED: Any = ...


class AdminAPIClient:
    def __init__(self, client: Client) -> None:
        self.client = client

    @classmethod
    @contextmanager
    def from_app(cls, app: FastAPI, base_url: str = "http://testserver") -> Iterator[Self]:
        from fastapi.testclient import TestClient

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

    @staticmethod
    def _build_file_params(
        file_params: Mapping[str, Any] | None,
    ) -> list[tuple[str, Any]] | None:
        if file_params is None:
            return None
        result: list[tuple[str, Any]] = []
        for name, value in file_params.items():
            if value is ADMIN_API_CLIENT_NOT_REQUIRED:
                continue
            values = value if isinstance(value, list) else [value]
            for v in values:
                if hasattr(v, "filename") and hasattr(v, "file"):
                    # `UploadFile`-like; duck-typed so we need not import it here.
                    result.append((name, (v.filename, v.file, v.content_type)))
                else:
                    # `bytes` / `str` / `IO[bytes]` / httpx2 `(name, content[, type])`.
                    result.append((name, v))
        return result or None

    @staticmethod
    def _build_form_params(
        form_params: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if form_params is None:
            return None
        form: dict[str, Any] = {}
        for name, value in form_params.items():
            if value is ADMIN_API_CLIENT_NOT_REQUIRED:
                continue
            encoded = jsonable_encoder(value)
            if isinstance(encoded, dict):
                # Model-as-`Form()`: flatten fields into top-level form fields
                # (only flat models round-trip; nested dicts don't url-encode).
                form.update(encoded)
            else:
                # Scalars get stringified by httpx2; lists become repeated fields.
                form[name] = encoded
        return form or None

    @staticmethod
    def _apply_security_params(
        security_params: Sequence[AdminAPIClientSecurityParam] | None,
        header_params: MutableMapping[str, Any],
        cookie_params: MutableMapping[str, Any],
        query_params: MutableMapping[str, Any],
    ) -> None:
        for kind, name, value in security_params or ():
            if value is ADMIN_API_CLIENT_NOT_REQUIRED or value is None:
                continue
            target: MutableMapping[str, Any]
            encoded: str
            if kind == "http_bearer" and isinstance(value, str):
                target, encoded = header_params, f"Bearer {value}"
            elif kind == "http_basic" and isinstance(value, tuple):
                user_pass = b64encode(f"{value[0]}:{value[1]}".encode()).decode("ascii")
                target, encoded = header_params, f"Basic {user_pass}"
            elif kind == "api_key_header" and isinstance(value, str):
                target, encoded = header_params, value
            elif kind == "api_key_cookie" and isinstance(value, str):
                target, encoded = cookie_params, value
            elif kind == "api_key_query" and isinstance(value, str):
                target, encoded = query_params, value
            else:
                raise TypeError(
                    f"Security param `{name}` of kind `{kind}` has incompatible value type `{type(value).__name__}`."
                )
            if name in target and target[name] is not ADMIN_API_CLIENT_NOT_REQUIRED:
                raise RuntimeError(
                    f"Security param `{name}` conflicts with an already-set "
                    f"{kind.split('_', 1)[0]} param of the same name."
                )
            target[name] = encoded

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
        file_params: Mapping[str, Any] | None = None,
        form_params: Mapping[str, Any] | None = None,
        security_params: Sequence[AdminAPIClientSecurityParam] | None = None,
        is_body_embedded: bool = False,
        streaming_kind: Literal["json_lines", "server_sent_events", "raw_bytes", "raw_str"] | None = None,
        raise_if_not_default_status: bool = False,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[HTTPStatus, Any]:
        if not client_exts:
            client_exts = {}

        url = path
        for param, value in (self._filter_and_encode_params(path_params) or {}).items():
            value_str = f"{value:0.20f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
            url = url.replace(f"{{{param}}}", value_str)

        headers = self._filter_and_encode_params(header_params) or {}
        cookies = self._filter_and_encode_params(cookie_params) or {}
        queries = self._filter_and_encode_params(query_params) or {}
        self._apply_security_params(security_params, headers, cookies, queries)
        if cookies:
            # Mirror httpx2's per-request-cookies DeprecationWarning ourselves
            # (we bypass `Client.request()` via `build_request` + `send`).
            warn(
                "Setting per-request cookie parameters is deprecated because cookie"
                "persistence behaviour is ambiguous. Set cookies on the client"
                "instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        timeout = client_exts.get("timeout", USE_CLIENT_DEFAULT)
        # Scuffed isinstance() check because we don't want to import
        # starlette.testclient.Testclient for users that don't need it.
        if (
            self.client.__class__.__name__ == "TestClient"
            and self.client.__class__.__module__ == "starlette.testclient"
            and timeout is not USE_CLIENT_DEFAULT
        ):
            warn(
                "Starlette's TestClient (which you probably use via "
                f"{self.__class__.__name__}.from_app()) does not support timeouts. See "
                "https://github.com/Kludex/starlette/issues/1108 for more information.",
                DeprecationWarning,
                stacklevel=3,
            )
            timeout = USE_CLIENT_DEFAULT  # Hide the warning generated by Starlette.

        files = self._build_file_params(file_params)
        form = self._build_form_params(form_params)
        if files is not None or form is not None:
            request = self.client.build_request(
                method.name,
                url,
                params=queries or None,
                headers=headers or None,
                cookies=cookies or None,
                data=form,
                files=files,
                timeout=timeout,
            )
        else:
            body = self._filter_and_encode_params(body_params)
            if body and not is_body_embedded:
                body = next(iter(body.values()))
            request = self.client.build_request(
                method.name,
                url,
                params=queries or None,
                headers=headers or None,
                cookies=cookies or None,
                json=body,
                timeout=timeout,
            )

        response = self.client.send(request, stream=streaming_kind is not None)
        status = HTTPStatus(response.status_code)

        model = models[status]
        if streaming_kind is not None and status == default_status:
            data = self._build_streaming_data(streaming_kind, response, model)
        elif streaming_kind is not None:
            # Streaming endpoint returned a non-default status (typically a JSON
            # error body). Drain it, then release the stream-mode response.
            try:
                text = "".join(response.iter_text())
            finally:
                response.close()
            data = TypeAdapter(model).validate_json(text or "null")
        else:
            # An empty body (e.g. 204 NO_CONTENT) is treated as JSON `null` so the
            # declared model still validatess.
            data = TypeAdapter(model).validate_json(response.text or "null")

        result = AdminAPIClientResult(
            status=status,
            data=data,
            model=model,
            response=response,
        )
        if status != default_status and raise_if_not_default_status:
            raise AdminAPIClientNotDefaultStatusError(default_status=default_status, result=result)
        return result

    @classmethod
    def _build_streaming_data(
        cls,
        streaming_kind: Literal["json_lines", "server_sent_events", "raw_bytes", "raw_str"],
        response: Response,
        model: Any,  # noqa: ANN401
    ) -> Iterator[Any]:
        if streaming_kind == "raw_bytes":
            return cls._close_response_after(response, response.iter_bytes())
        if streaming_kind == "raw_str":
            return cls._close_response_after(response, response.iter_text())
        if streaming_kind == "json_lines":
            return cls._close_response_after(response, cls._iter_json_lines(response, model))
        return cls._close_response_after(response, cls._iter_sse(response, model))

    @staticmethod
    def _close_response_after(response: Response, source: Iterator[Any]) -> Iterator[Any]:
        try:
            yield from source
        finally:
            response.close()

    @staticmethod
    def _iter_json_lines(
        response: Response,
        model: Any,  # noqa: ANN401
    ) -> Iterator[Any]:
        adapter = TypeAdapter(model)
        for part in response.iter_lines():
            if part:
                yield adapter.validate_json(part)

    @classmethod
    def _iter_sse(
        cls,
        response: Response,
        model: Any,  # noqa: ANN401
    ) -> Iterator[Any]:
        adapter = TypeAdapter(model)
        for fields in cls._iter_sse_event_fields(response.iter_lines()):
            if "data" in fields:
                fields = {**fields, "data": adapter.validate_json(fields["data"])}
            yield AdminAPIClientSSE[model].model_validate(fields)

    @classmethod
    def _iter_sse_event_fields(cls, lines: Iterator[str]) -> Iterator[Mapping[str, Any]]:
        fields: dict[str, Any] = {}
        data_lines: list[str] = []
        comment_lines: list[str] = []
        for line in lines:
            if line:
                cls._accumulate_sse_line(line, fields, data_lines, comment_lines)
                continue
            event = cls._finalize_sse_event(fields, data_lines, comment_lines)
            if event is not None:
                yield event
            # Spec deviation: `lastEventId` doesn't persist across events. Each
            # yielded event reflects only what was on the wire for it; events
            # without an `id:` line surface as `id=None`.
            fields, data_lines, comment_lines = {}, [], []
        event = cls._finalize_sse_event(fields, data_lines, comment_lines)
        if event is not None:
            yield event

    @staticmethod
    def _accumulate_sse_line(
        line: str,
        fields: dict[str, Any],
        data_lines: list[str],
        comment_lines: list[str],
    ) -> None:
        if line.startswith(":"):
            comment_lines.append(line[1:].removeprefix(" "))
            return
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            data_lines.append(value)
        elif field in ("event", "id"):
            fields[field] = value
        elif field == "retry" and value.isascii() and value.isdigit():
            fields[field] = int(value)

    @staticmethod
    def _finalize_sse_event(
        fields: dict[str, Any], data_lines: list[str], comment_lines: list[str]
    ) -> dict[str, Any] | None:
        if data_lines:
            fields["data"] = "\n".join(data_lines)
        if comment_lines:
            fields["comment"] = "\n".join(comment_lines)
        # Spec deviation: comment- or metadata-only events (no `data:` lines)
        # are still dispatched. The spec says to drop them, but we surface them
        # so `AdminAPIClientSSE.comment` is reachable from the client.
        return fields or None

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
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CallerIdentity]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
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
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def logout(
        self,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
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
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
            ),
            self._route_handler(
                path="/v1/m/logout",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
    def create_user(
        self,
        *,
        body: CreateUserRequest,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateUserResponse]: ...
    @overload
    def create_user(
        self,
        *,
        body: CreateUserRequest,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateUserResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def create_user(
        self,
        *,
        body: CreateUserRequest,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateUserResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateUserResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/users",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: CreateUserResponse,
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
    def list_users(
        self,
        *,
        email_contains: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        scope: Literal["all", "mine"] = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListUsersResponse]: ...
    @overload
    def list_users(
        self,
        *,
        email_contains: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        scope: Literal["all", "mine"] = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListUsersResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def list_users(
        self,
        *,
        email_contains: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        scope: Literal["all", "mine"] = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListUsersResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListUsersResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/users",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListUsersResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                query_params={
                    "email_contains": email_contains,
                    "page_size": page_size,
                    "page_token": page_token,
                    "scope": scope,
                    "skip": skip,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_user(
        self,
        *,
        user_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetUserResponse]: ...
    @overload
    def get_user(
        self,
        *,
        user_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetUserResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_user(
        self,
        *,
        user_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetUserResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetUserResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/users/{user_id}",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetUserResponse,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "user_id": user_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def patch_user(
        self,
        *,
        body: PatchUserRequest,
        user_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def patch_user(
        self,
        *,
        body: PatchUserRequest,
        user_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def patch_user(
        self,
        *,
        body: PatchUserRequest,
        user_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/users/{user_id}",
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "user_id": user_id,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def delete_user(
        self,
        *,
        user_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_user(
        self,
        *,
        user_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_user(
        self,
        *,
        user_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/users/{user_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "user_id": user_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_snapshot(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        snapshot_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse]: ...
    @overload
    def get_snapshot(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        snapshot_id: str,
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
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        snapshot_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetSnapshotResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}",
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
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        status_: list[SnapshotStatus] | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse]: ...
    @overload
    def list_snapshots(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        status_: list[SnapshotStatus] | None = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        status_: list[SnapshotStatus] | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListSnapshotsResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots",
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
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        snapshot_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_snapshot(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        snapshot_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_snapshot(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        snapshot_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots/{snapshot_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse]: ...
    @overload
    def create_snapshot(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
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
        *,
        datasource_id: str,
        experiment_id: str,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateSnapshotResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots",
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
        include_stats: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        name_contains: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        scope: Literal["mine", "all"] = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse]: ...
    @overload
    def list_organizations(
        self,
        *,
        include_stats: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        name_contains: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        scope: Literal["mine", "all"] = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def list_organizations(
        self,
        *,
        include_stats: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        name_contains: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        scope: Literal["mine", "all"] = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationsResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                query_params={
                    "include_stats": include_stats,
                    "name_contains": name_contains,
                    "page_size": page_size,
                    "page_token": page_token,
                    "scope": scope,
                    "skip": skip,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def create_organizations(
        self,
        *,
        body: CreateOrganizationRequest,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse]: ...
    @overload
    def create_organizations(
        self,
        *,
        body: CreateOrganizationRequest,
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
        *,
        body: CreateOrganizationRequest,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateOrganizationResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
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
        *,
        body: AddWebhookToOrganizationRequest,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse]: ...
    @overload
    def add_webhook_to_organization(
        self,
        *,
        body: AddWebhookToOrganizationRequest,
        organization_id: str,
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
        *,
        body: AddWebhookToOrganizationRequest,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], AddWebhookToOrganizationResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks",
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
        *,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse]: ...
    @overload
    def list_organization_webhooks(
        self,
        *,
        organization_id: str,
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
        *,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListWebhooksResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks",
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
        *,
        body: UpdateOrganizationWebhookRequest,
        organization_id: str,
        webhook_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def update_organization_webhook(
        self,
        *,
        body: UpdateOrganizationWebhookRequest,
        organization_id: str,
        webhook_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_organization_webhook(
        self,
        *,
        body: UpdateOrganizationWebhookRequest,
        organization_id: str,
        webhook_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks/{webhook_id}",
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        organization_id: str,
        webhook_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def regenerate_webhook_auth_token(
        self,
        *,
        organization_id: str,
        webhook_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def regenerate_webhook_auth_token(
        self,
        *,
        organization_id: str,
        webhook_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks/{webhook_id}/authtoken",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        organization_id: str,
        webhook_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_webhook_from_organization(
        self,
        *,
        organization_id: str,
        webhook_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_webhook_from_organization(
        self,
        *,
        organization_id: str,
        webhook_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/webhooks/{webhook_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        organization_id: str,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationEventsResponse]: ...
    @overload
    def list_organization_events(
        self,
        *,
        organization_id: str,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        organization_id: str,
        page_size: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        page_token: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        skip: int = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationEventsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListOrganizationEventsResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/events",
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
    def resend_organization_event(
        self,
        *,
        event_id: str,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def resend_organization_event(
        self,
        *,
        event_id: str,
        organization_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def resend_organization_event(
        self,
        *,
        event_id: str,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/events/{event_id}/resend",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
                    HTTPStatus.BAD_REQUEST: HTTPExceptionError,
                    HTTPStatus.UNAUTHORIZED: HTTPExceptionError,
                    HTTPStatus.FORBIDDEN: HTTPExceptionError,
                    HTTPStatus.NOT_FOUND: HTTPExceptionError,
                    HTTPStatus.UNPROCESSABLE_CONTENT: AdminAPIClientHTTPValidationError,
                },
                path_params={
                    "event_id": event_id,
                    "organization_id": organization_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def add_member_to_organization(
        self,
        *,
        body: AddMemberToOrganizationRequest,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def add_member_to_organization(
        self,
        *,
        body: AddMemberToOrganizationRequest,
        organization_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def add_member_to_organization(
        self,
        *,
        body: AddMemberToOrganizationRequest,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/members",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        organization_id: str,
        user_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def remove_member_from_organization(
        self,
        *,
        organization_id: str,
        user_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def remove_member_from_organization(
        self,
        *,
        organization_id: str,
        user_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/members/{user_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        body: UpdateOrganizationRequest,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Any]: ...
    @overload
    def update_organization(
        self,
        *,
        body: UpdateOrganizationRequest,
        organization_id: str,
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
        *,
        body: UpdateOrganizationRequest,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}",
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
        *,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse]: ...
    @overload
    def get_organization(
        self,
        *,
        organization_id: str,
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
        *,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetOrganizationResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}",
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
        *,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse]: ...
    @overload
    def list_organization_datasources(
        self,
        *,
        organization_id: str,
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
        *,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListDatasourcesResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources",
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
        *,
        body: CreateDatasourceRequest,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse]: ...
    @overload
    def create_datasource(
        self,
        *,
        body: CreateDatasourceRequest,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        body: CreateDatasourceRequest,
        connectivity_check: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
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
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateDatasourceResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
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
        *,
        body: UpdateDatasourceRequest,
        datasource_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def update_datasource(
        self,
        *,
        body: UpdateDatasourceRequest,
        datasource_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_datasource(
        self,
        *,
        body: UpdateDatasourceRequest,
        datasource_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}",
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        datasource_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse]: ...
    @overload
    def get_datasource(
        self,
        *,
        datasource_id: str,
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
        *,
        datasource_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetDatasourceResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}",
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
        *,
        datasource_id: str,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse]: ...
    @overload
    def inspect_datasource(
        self,
        *,
        datasource_id: str,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        datasource_id: str,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
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
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/inspect",
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
        *,
        datasource_id: str,
        table_name: str,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceTableResponse]: ...
    @overload
    def inspect_table_in_datasource(
        self,
        *,
        datasource_id: str,
        table_name: str,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        datasource_id: str,
        table_name: str,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
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
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], InspectDatasourceTableResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/inspect/{table_name}",
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
        *,
        datasource_id: str,
        organization_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_datasource(
        self,
        *,
        datasource_id: str,
        organization_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_datasource(
        self,
        *,
        datasource_id: str,
        organization_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/datasources/{datasource_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        datasource_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListParticipantsTypeResponse]: ...
    @overload
    def list_participant_types(
        self,
        *,
        datasource_id: str,
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
        *,
        datasource_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListParticipantsTypeResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListParticipantsTypeResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants",
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
        *,
        body: CreateParticipantsTypeRequest,
        datasource_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateParticipantsTypeResponse]: ...
    @overload
    def create_participant_type(
        self,
        *,
        body: CreateParticipantsTypeRequest,
        datasource_id: str,
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
        *,
        body: CreateParticipantsTypeRequest,
        datasource_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateParticipantsTypeResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateParticipantsTypeResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants",
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
        *,
        datasource_id: str,
        participant_id: str,
        expensive: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], InspectParticipantTypesResponse]: ...
    @overload
    def inspect_participant_types(
        self,
        *,
        datasource_id: str,
        participant_id: str,
        expensive: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        datasource_id: str,
        participant_id: str,
        expensive: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        refresh: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
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
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], InspectParticipantTypesResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}/inspect",
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
        *,
        datasource_id: str,
        participant_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantsTypeResponse]: ...
    @overload
    def get_participant_type(
        self,
        *,
        datasource_id: str,
        participant_id: str,
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
        *,
        datasource_id: str,
        participant_id: str,
        raise_if_not_default_status: bool = True,
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
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetParticipantsTypeResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}",
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
        *,
        body: UpdateParticipantsTypeRequest,
        datasource_id: str,
        participant_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse]: ...
    @overload
    def update_participant_type(
        self,
        *,
        body: UpdateParticipantsTypeRequest,
        datasource_id: str,
        participant_id: str,
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
        *,
        body: UpdateParticipantsTypeRequest,
        datasource_id: str,
        participant_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], UpdateParticipantsTypeResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}",
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
        *,
        datasource_id: str,
        participant_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_participant(
        self,
        *,
        datasource_id: str,
        participant_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_participant(
        self,
        *,
        datasource_id: str,
        participant_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/participants/{participant_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        datasource_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse]: ...
    @overload
    def list_api_keys(
        self,
        *,
        datasource_id: str,
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
        *,
        datasource_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListApiKeysResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/apikeys",
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
        *,
        datasource_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse]: ...
    @overload
    def create_api_key(
        self,
        *,
        datasource_id: str,
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
        *,
        datasource_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateApiKeyResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/apikeys",
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
        *,
        api_key_id: str,
        datasource_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_api_key(
        self,
        *,
        api_key_id: str,
        datasource_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_api_key(
        self,
        *,
        api_key_id: str,
        datasource_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/apikeys/{api_key_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        body: CreateExperimentRequest,
        datasource_id: str,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        stratify_on_metrics: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse]: ...
    @overload
    def create_experiment(
        self,
        *,
        body: CreateExperimentRequest,
        datasource_id: str,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        stratify_on_metrics: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        body: CreateExperimentRequest,
        datasource_id: str,
        random_state: int | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        stratify_on_metrics: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], CreateExperimentResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments",
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
        *,
        datasource_id: str,
        experiment_id: str,
        baseline_arm_id: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]: ...
    @overload
    def analyze_experiment(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        baseline_arm_id: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
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
        *,
        datasource_id: str,
        experiment_id: str,
        baseline_arm_id: str | None = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze",
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
        *,
        body: CMABContextInputRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]: ...
    @overload
    def analyze_cmab_experiment(
        self,
        *,
        body: CMABContextInputRequest,
        datasource_id: str,
        experiment_id: str,
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
        *,
        body: CMABContextInputRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ExperimentAnalysisResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze_cmab",
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
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def commit_experiment(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def commit_experiment(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/commit",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def abandon_experiment(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def abandon_experiment(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.CONFLICT], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/abandon",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        organization_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse]: ...
    @overload
    def list_organization_experiments(
        self,
        *,
        organization_id: str,
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
        *,
        organization_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/organizations/{organization_id}/experiments",
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
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse]: ...
    @overload
    def get_experiment_for_ui(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
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
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], GetExperimentForUiResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}",
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
    def get_experiment_sample_calls(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], SampleCalls | None]: ...
    @overload
    def get_experiment_sample_calls(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], SampleCalls | None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_experiment_sample_calls(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], SampleCalls | None]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], SampleCalls | None]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/sample-calls",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: SampleCalls | None,
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
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], Iterator[bytes]]: ...
    @overload
    def get_experiment_assignments_as_csv_for_ui(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Iterator[bytes]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def get_experiment_assignments_as_csv_for_ui(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.OK], Iterator[bytes]]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], Iterator[bytes]]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/csv",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: bytes,
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
                streaming_kind="raw_bytes",
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_experiment(
        self,
        *,
        body: UpdateExperimentRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def update_experiment(
        self,
        *,
        body: UpdateExperimentRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_experiment(
        self,
        *,
        body: UpdateExperimentRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}",
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        datasource_id: str,
        experiment_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_experiment(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_experiment(
        self,
        *,
        datasource_id: str,
        experiment_id: str,
        allow_missing: bool = ADMIN_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        body: DeleteExperimentDataRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def delete_experiment_data(
        self,
        *,
        body: DeleteExperimentDataRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def delete_experiment_data(
        self,
        *,
        body: DeleteExperimentDataRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/data",
                method=HTTPMethod.DELETE,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        arm_id: str,
        body: UpdateArmRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def update_arm(
        self,
        *,
        arm_id: str,
        body: UpdateArmRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: Literal[False],
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ): ...
    def update_arm(
        self,
        *,
        arm_id: str,
        body: UpdateArmRequest,
        datasource_id: str,
        experiment_id: str,
        raise_if_not_default_status: bool = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> (
        AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
        | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], AdminAPIClientHTTPValidationError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/arms/{arm_id}",
                method=HTTPMethod.PATCH,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
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
        *,
        body: PowerRequest,
        datasource_id: str,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: AdminAPIClientExtensions | None = None,
    ) -> AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse]: ...
    @overload
    def power_check(
        self,
        *,
        body: PowerRequest,
        datasource_id: str,
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
        *,
        body: PowerRequest,
        datasource_id: str,
        raise_if_not_default_status: bool = True,
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
    ):
        return cast(
            (
                AdminAPIClientResult[Literal[HTTPStatus.OK], PowerResponse]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.FORBIDDEN], HTTPExceptionError]
                | AdminAPIClientResult[Literal[HTTPStatus.NOT_FOUND], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.UNPROCESSABLE_CONTENT], XHTTPValidationError]
                | AdminAPIClientResult[Literal[HTTPStatus.BAD_GATEWAY], MessageError]
                | AdminAPIClientResult[Literal[HTTPStatus.GATEWAY_TIMEOUT], MessageError]
            ),
            self._route_handler(
                path="/v1/m/datasources/{datasource_id}/power",
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
