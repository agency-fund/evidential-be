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

from xngin.apiserver.routers.common_api_types import TurnConfigResponse

if TYPE_CHECKING:
    from fastapi import FastAPI


class IntegrationsAPIClientExtensions(TypedDict, total=False):
    timeout: float | tuple[float | None, float | None, float | None, float | None] | Timeout | None


class IntegrationsAPIClientResult[Status: HTTPStatus, Model](NamedTuple):
    status: Status
    data: Model
    model: type[Model]
    response: Response


class IntegrationsAPIClientValidationError(BaseModel):
    loc: Sequence[str | int]
    msg: str
    type: str


class IntegrationsAPIClientHTTPValidationError(BaseModel):
    detail: Sequence[IntegrationsAPIClientValidationError]


class IntegrationsAPIClientNotDefaultStatusError(Exception):
    def __init__(
        self,
        *,
        default_status: HTTPStatus,
        result: IntegrationsAPIClientResult[HTTPStatus, Any],
    ) -> None:
        super().__init__(
            f"Expected default status {default_status.value} {default_status.phrase}, "
            f"but received {result.status.value} {result.status.phrase}."
        )
        self.default_status = default_status
        self.result = result


class IntegrationsAPIClientSecurityParam(NamedTuple):
    kind: Literal[
        "http_bearer",
        "http_basic",
        "api_key_header",
        "api_key_cookie",
        "api_key_query",
    ]
    name: str
    value: str | tuple[str, str] | None


class IntegrationsAPIClientSSE[Data](ServerSentEvent):
    data: Data | None = None


INTEGRATIONS_API_CLIENT_NOT_REQUIRED: Any = ...


class IntegrationsAPIClient:
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
            if value is not INTEGRATIONS_API_CLIENT_NOT_REQUIRED
        } or None

    @staticmethod
    def _build_file_params(
        file_params: Mapping[str, Any] | None,
    ) -> list[tuple[str, Any]] | None:
        if file_params is None:
            return None
        result: list[tuple[str, Any]] = []
        for name, value in file_params.items():
            if value is INTEGRATIONS_API_CLIENT_NOT_REQUIRED:
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
            if value is INTEGRATIONS_API_CLIENT_NOT_REQUIRED:
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
        security_params: Sequence[IntegrationsAPIClientSecurityParam] | None,
        header_params: MutableMapping[str, Any],
        cookie_params: MutableMapping[str, Any],
        query_params: MutableMapping[str, Any],
    ) -> None:
        for kind, name, value in security_params or ():
            if value is INTEGRATIONS_API_CLIENT_NOT_REQUIRED or value is None:
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
            if name in target and target[name] is not INTEGRATIONS_API_CLIENT_NOT_REQUIRED:
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
        security_params: Sequence[IntegrationsAPIClientSecurityParam] | None = None,
        is_body_embedded: bool = False,
        streaming_kind: Literal["json_lines", "server_sent_events", "raw_bytes", "raw_str"] | None = None,
        raise_if_not_default_status: bool = False,
        client_exts: IntegrationsAPIClientExtensions | None = None,
    ) -> IntegrationsAPIClientResult[HTTPStatus, Any]:
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

        result = IntegrationsAPIClientResult(
            status=status,
            data=data,
            model=model,
            response=response,
        )
        if status != default_status and raise_if_not_default_status:
            raise IntegrationsAPIClientNotDefaultStatusError(default_status=default_status, result=result)
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
            yield IntegrationsAPIClientSSE[model].model_validate(fields)

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
        # so `IntegrationsAPIClientSSE.comment` is reachable from the client.
        return fields or None

    @overload
    def get_turn_app_config(
        self,
        *,
        experiment_id: str,
        api_key: str = INTEGRATIONS_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: IntegrationsAPIClientExtensions | None = None,
    ) -> IntegrationsAPIClientResult[Literal[HTTPStatus.OK], TurnConfigResponse]: ...
    @overload
    def get_turn_app_config(
        self,
        *,
        experiment_id: str,
        api_key: str = INTEGRATIONS_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: IntegrationsAPIClientExtensions | None = None,
    ) -> (
        IntegrationsAPIClientResult[Literal[HTTPStatus.OK], TurnConfigResponse]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict]
        | IntegrationsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], IntegrationsAPIClientHTTPValidationError
        ]
    ): ...
    def get_turn_app_config(
        self,
        *,
        experiment_id: str,
        api_key: str = INTEGRATIONS_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: IntegrationsAPIClientExtensions | None = None,
    ) -> (
        IntegrationsAPIClientResult[Literal[HTTPStatus.OK], TurnConfigResponse]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict]
        | IntegrationsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], IntegrationsAPIClientHTTPValidationError
        ]
    ):
        return cast(
            (
                IntegrationsAPIClientResult[Literal[HTTPStatus.OK], TurnConfigResponse]
                | IntegrationsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict]
                | IntegrationsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict]
                | IntegrationsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict]
                | IntegrationsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], IntegrationsAPIClientHTTPValidationError
                ]
            ),
            self._route_handler(
                path="/v1/integrations/experiments/{experiment_id}/turn-app-config",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: TurnConfigResponse,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: IntegrationsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                },
                security_params=(
                    IntegrationsAPIClientSecurityParam(kind="api_key_header", name="X-API-Key", value=api_key),
                ),
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def turn_webhook(
        self,
        *,
        webhook_id: str,
        auth_token: str | None = INTEGRATIONS_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: IntegrationsAPIClientExtensions | None = None,
    ) -> IntegrationsAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]: ...
    @overload
    def turn_webhook(
        self,
        *,
        webhook_id: str,
        auth_token: str | None = INTEGRATIONS_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: Literal[False],
        client_exts: IntegrationsAPIClientExtensions | None = None,
    ) -> (
        IntegrationsAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict]
        | IntegrationsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], IntegrationsAPIClientHTTPValidationError
        ]
    ): ...
    def turn_webhook(
        self,
        *,
        webhook_id: str,
        auth_token: str | None = INTEGRATIONS_API_CLIENT_NOT_REQUIRED,
        raise_if_not_default_status: bool = True,
        client_exts: IntegrationsAPIClientExtensions | None = None,
    ) -> (
        IntegrationsAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict]
        | IntegrationsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict]
        | IntegrationsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT], IntegrationsAPIClientHTTPValidationError
        ]
    ):
        return cast(
            (
                IntegrationsAPIClientResult[Literal[HTTPStatus.NO_CONTENT], Any]
                | IntegrationsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict]
                | IntegrationsAPIClientResult[Literal[HTTPStatus.UNAUTHORIZED], dict]
                | IntegrationsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict]
                | IntegrationsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict]
                | IntegrationsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT], IntegrationsAPIClientHTTPValidationError
                ]
            ),
            self._route_handler(
                path="/v1/integrations/turn/webhook/{webhook_id}",
                method=HTTPMethod.POST,
                default_status=HTTPStatus.NO_CONTENT,
                models={
                    HTTPStatus.NO_CONTENT: Any,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.UNAUTHORIZED: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: IntegrationsAPIClientHTTPValidationError,
                },
                path_params={
                    "webhook_id": webhook_id,
                },
                header_params={
                    "Webhook-Token": auth_token,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )
