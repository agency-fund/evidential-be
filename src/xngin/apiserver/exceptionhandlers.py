import psycopg.errors
import sqlalchemy
from starlette.requests import Request
from starlette.responses import JSONResponse

from xngin.apiserver.settings import (
    CannotFindTableError,
    CannotFindParticipantsError,
)


def setup(app):
    """Registers exception handlers to the FastAPI app.

    The general goal of these exception handlers should be to return stable API responses (including meaningful HTTP
    status codes) to exceptions we recognize, and ideally not reveal too much about internal implementation details.
    """

    @app.exception_handler(CannotFindTableError)
    async def exception_handler_cannotfindthetableexception(
        _request: Request, exc: CannotFindTableError
    ):
        return JSONResponse(status_code=404, content={"message": exc.message})

    @app.exception_handler(CannotFindParticipantsError)
    async def exception_handler_cannotfindtheparticipantexception(
        _request: Request, exc: CannotFindParticipantsError
    ):
        return JSONResponse(status_code=404, content={"message": exc.message})

    @app.exception_handler(sqlalchemy.exc.OperationalError)
    async def exception_handler_sqlalchemy_opex(
        _request: Request, exc: sqlalchemy.exc.OperationalError
    ):
        status = 500
        cause = getattr(exc, "orig", None) or exc.__cause__
        if isinstance(cause, psycopg.errors.ConnectionTimeout):
            status = 504
        # Return a minimal error message
        return JSONResponse(
            status_code=status, content={"message": str(cause) or str(exc)}
        )
