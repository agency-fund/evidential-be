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

from xngin.apiserver.routers.common_api_types import (
    ArmBandit,
    CMABContextInputRequest,
    GetExperimentAssignmentsResponse,
    GetExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    OnlineAssignmentWithFiltersRequest,
    UpdateBanditArmOutcomeRequest,
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


class ExperimentsAPIClientExtensions(TypedDict, total=False):
    timeout: float | tuple[float | None, float | None, float | None, float | None] | Timeout | None


@dataclass(frozen=True)
class ExperimentsAPIClientResult[Status: HTTPStatus, Data, ModelType]:
    # `data` is the decoded response body. For streaming responses it is an
    # iterator/async iterator of decoded items rather than a fully materialized value.
    # `model` is the runtime decoder used for that body. For normal and streaming
    # responses this is a class object like `User` with static type `type[User]`; for
    # no-body responses it is `None`.
    status: Status
    data: Data
    model: ModelType
    response: Response


class ExperimentsAPIClientValidationError(BaseModel):
    loc: Sequence[str | int]
    msg: str
    type: str


class ExperimentsAPIClientHTTPValidationError(BaseModel):
    detail: Sequence[ExperimentsAPIClientValidationError]


class ExperimentsAPIClientNotDefaultStatusError(Exception):
    def __init__(
        self,
        *,
        default_status: HTTPStatus,
        result: ExperimentsAPIClientResult[HTTPStatus, Any, object],
    ) -> None:
        super().__init__(
            f"Expected default status {default_status.value} {default_status.phrase}, "
            f"but received {result.status.value} {result.status.phrase}."
        )
        self.default_status = default_status
        self.result = result


EXPERIMENTS_API_CLIENT_NOT_REQUIRED: Any = ...


class ExperimentsAPIClient:  # noqa: RUF100,PLR0904
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
            if value is not EXPERIMENTS_API_CLIENT_NOT_REQUIRED
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
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[HTTPStatus, Any, object]:
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
            result: ExperimentsAPIClientResult[HTTPStatus, Any, object] = ExperimentsAPIClientResult(
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

                result = ExperimentsAPIClientResult(
                    status=status,
                    data=data_iter(),
                    model=model,
                    response=response,
                )
            elif _is_no_body_response(method=method, status=status):
                result = ExperimentsAPIClientResult(
                    status=status,
                    data=None,
                    model=model,
                    response=response,
                )
            else:
                result = ExperimentsAPIClientResult(
                    status=status,
                    data=TypeAdapter(model).validate_json(response.text),
                    model=model,
                    response=response,
                )
        if status != default_status and raise_if_not_default_status:
            raise ExperimentsAPIClientNotDefaultStatusError(default_status=default_status, result=result)
        return result

    @overload
    def list_experiments(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]]: ...
    @overload
    def list_experiments(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def list_experiments(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        datasource_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[
                    Literal[HTTPStatus.OK], ListExperimentsResponse, type[ListExperimentsResponse]
                ]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments",
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ListExperimentsResponse,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                header_params={
                    "X-API-Key": api_key,
                    "Datasource-ID": datasource_id,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_experiment(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[Literal[HTTPStatus.OK], GetExperimentResponse, type[GetExperimentResponse]]: ...
    @overload
    def get_experiment(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], GetExperimentResponse, type[GetExperimentResponse]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def get_experiment(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], GetExperimentResponse, type[GetExperimentResponse]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[Literal[HTTPStatus.OK], GetExperimentResponse, type[GetExperimentResponse]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments/{experiment_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetExperimentResponse,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                },
                header_params={
                    "X-API-Key": api_key,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_experiment_assignments(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[
        Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
    ]: ...
    @overload
    def get_experiment_assignments(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def get_experiment_assignments(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[
                    Literal[HTTPStatus.OK], GetExperimentAssignmentsResponse, type[GetExperimentAssignmentsResponse]
                ]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments/{experiment_id}/assignments",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetExperimentAssignmentsResponse,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                },
                header_params={
                    "X-API-Key": api_key,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_experiment_assignments_as_csv(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]: ...
    @overload
    def get_experiment_assignments_as_csv(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def get_experiment_assignments_as_csv(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[Literal[HTTPStatus.OK], Any, type[Any]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments/{experiment_id}/assignments/csv",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: Any,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                },
                header_params={
                    "X-API-Key": api_key,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_assignment(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        max_age: int = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[
        Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
    ]: ...
    @overload
    def get_assignment(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        max_age: int = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def get_assignment(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        max_age: int = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[
                    Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
                ]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments/{experiment_id}/assignments/{participant_id}",  # noqa: RUF100,RUF027
                method=HTTPMethod.GET,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetParticipantAssignmentResponse,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                    "participant_id": participant_id,
                },
                query_params={
                    "create_if_none": create_if_none,
                    "max_age": max_age,
                },
                header_params={
                    "X-API-Key": api_key,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_assignment_filtered(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: OnlineAssignmentWithFiltersRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[
        Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
    ]: ...
    @overload
    def get_assignment_filtered(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: OnlineAssignmentWithFiltersRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def get_assignment_filtered(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: OnlineAssignmentWithFiltersRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[
                    Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
                ]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments/{experiment_id}/assignments/{participant_id}/assign_with_filters",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetParticipantAssignmentResponse,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                    "participant_id": participant_id,
                },
                query_params={
                    "create_if_none": create_if_none,
                    "random_state": random_state,
                },
                header_params={
                    "X-API-Key": api_key,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def get_assignment_cmab(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: CMABContextInputRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[
        Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
    ]: ...
    @overload
    def get_assignment_cmab(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: CMABContextInputRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def get_assignment_cmab(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: CMABContextInputRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        create_if_none: bool = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        random_state: int | None = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[
            Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
        ]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[
                    Literal[HTTPStatus.OK], GetParticipantAssignmentResponse, type[GetParticipantAssignmentResponse]
                ]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments/{experiment_id}/assignments/{participant_id}/assign_cmab",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: GetParticipantAssignmentResponse,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                    "participant_id": participant_id,
                },
                query_params={
                    "create_if_none": create_if_none,
                    "random_state": random_state,
                },
                header_params={
                    "X-API-Key": api_key,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )

    @overload
    def update_bandit_arm_with_participant_outcome(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: UpdateBanditArmOutcomeRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[True] = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> ExperimentsAPIClientResult[Literal[HTTPStatus.OK], ArmBandit, type[ArmBandit]]: ...
    @overload
    def update_bandit_arm_with_participant_outcome(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: UpdateBanditArmOutcomeRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: Literal[False],
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], ArmBandit, type[ArmBandit]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ): ...
    def update_bandit_arm_with_participant_outcome(
        self,
        api_key: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        body: UpdateBanditArmOutcomeRequest = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        experiment_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        participant_id: str = EXPERIMENTS_API_CLIENT_NOT_REQUIRED,
        *,
        raise_if_not_default_status: bool = True,
        client_exts: ExperimentsAPIClientExtensions | None = None,
    ) -> (
        ExperimentsAPIClientResult[Literal[HTTPStatus.OK], ArmBandit, type[ArmBandit]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
        | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
        | ExperimentsAPIClientResult[
            Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
            ExperimentsAPIClientHTTPValidationError,
            type[ExperimentsAPIClientHTTPValidationError],
        ]
    ):
        return cast(
            (
                ExperimentsAPIClientResult[Literal[HTTPStatus.OK], ArmBandit, type[ArmBandit]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.BAD_REQUEST], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.FORBIDDEN], dict, type[dict]]
                | ExperimentsAPIClientResult[Literal[HTTPStatus.NOT_FOUND], dict, type[dict]]
                | ExperimentsAPIClientResult[
                    Literal[HTTPStatus.UNPROCESSABLE_CONTENT],
                    ExperimentsAPIClientHTTPValidationError,
                    type[ExperimentsAPIClientHTTPValidationError],
                ]
            ),
            self._route_handler(
                path="/v1/experiments/{experiment_id}/assignments/{participant_id}/outcome",  # noqa: RUF100,RUF027
                method=HTTPMethod.POST,
                default_status=HTTPStatus.OK,
                models={
                    HTTPStatus.OK: ArmBandit,
                    HTTPStatus.BAD_REQUEST: dict,
                    HTTPStatus.FORBIDDEN: dict,
                    HTTPStatus.NOT_FOUND: dict,
                    HTTPStatus.UNPROCESSABLE_CONTENT: ExperimentsAPIClientHTTPValidationError,
                },
                path_params={
                    "experiment_id": experiment_id,
                    "participant_id": participant_id,
                },
                header_params={
                    "X-API-Key": api_key,
                },
                body_params={
                    "body": body,
                },
                raise_if_not_default_status=raise_if_not_default_status,
                client_exts=client_exts,
            ),
        )
