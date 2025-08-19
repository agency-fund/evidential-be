import psycopg.errors
import sqlalchemy
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from xngin.apiserver.apikeys import ApiKeyError
from xngin.apiserver.dependencies import CannotFindDatasourceError
from xngin.apiserver.dwh.dwh_session import CannotFindTableError
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.admin.admin_api_converters import (
    CredentialsUnavailableError,
)
from xngin.apiserver.routers.experiments.experiments_common import (
    ExperimentsAssignmentError,
)
from xngin.apiserver.settings import CannotFindParticipantsError
from xngin.stats.stats_errors import StatsError


def setup(app):
    """Registers exception handlers to the FastAPI app.

    The general goal of these exception handlers should be to return stable API responses (including meaningful HTTP
    status codes) to exceptions we recognize, and ideally not reveal too much about internal implementation details.
    """

    @app.exception_handler(CannotFindDatasourceError)
    async def exception_handler_cannotfinddatasourceerror(_request: Request, exc: CannotFindDatasourceError):
        return JSONResponse(status_code=404, content={"message": str(exc)})

    @app.exception_handler(CannotFindTableError)
    async def exception_handler_cannotfindthetableerror(_request: Request, exc: CannotFindTableError):
        return JSONResponse(status_code=404, content={"message": exc.message})

    @app.exception_handler(CannotFindParticipantsError)
    async def exception_handler_cannotfindtheparticipanterror(_request: Request, exc: CannotFindParticipantsError):
        return JSONResponse(status_code=404, content={"message": exc.message})

    @app.exception_handler(ExperimentsAssignmentError)
    async def exception_handler_experimentserror(_request: Request, exc: StatsError):
        return JSONResponse(status_code=422, content={"message": str(exc)})

    @app.exception_handler(StatsError)
    async def exception_handler_statsmodelerror(_request: Request, exc: StatsError):
        return JSONResponse(status_code=422, content={"message": str(exc)})

    @app.exception_handler(sqlalchemy.exc.OperationalError)
    async def exception_handler_sqlalchemy_opex(_request: Request, exc: sqlalchemy.exc.OperationalError):
        status = 500
        cause = getattr(exc, "orig", None) or exc.__cause__
        if isinstance(cause, psycopg.errors.ConnectionTimeout):
            status = 504
        # Return a minimal error message
        return JSONResponse(status_code=status, content={"message": str(cause) or str(exc)})

    @app.exception_handler(ApiKeyError)
    async def exception_handler_apikeys(_request: Request, _exc: ApiKeyError):
        return JSONResponse(status_code=403, content={"message": "API key missing or invalid."})

    @app.exception_handler(LateValidationError)
    async def exception_handler_latevalidation(_request: Request, exc: LateValidationError):
        return JSONResponse(status_code=422, content={"message": str(exc)})

    @app.exception_handler(ValidationError)
    async def exception_handler_pydantic_validationerror(_request: Request, exc: ValidationError):
        # This resembles FastAPI's request_validation_exception_handler but handles Pydantic ValidationErrors raised
        # by the implementation of the handlers.
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(CredentialsUnavailableError)
    async def exception_handler_credentialsunavailable(_request: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"message": str(exc)})
