import psycopg.errors
import sqlalchemy
from starlette.requests import Request
from starlette.responses import JSONResponse

from xngin.apiserver.settings import (
    CannotFindTableException,
    CannotFindParticipantsException,
)


def setup(app):
    @app.exception_handler(CannotFindTableException)
    async def exception_handler_cannotfindthetableexception(
        _request: Request, exc: CannotFindTableException
    ):
        return JSONResponse(status_code=404, content={"message": exc.message})

    @app.exception_handler(CannotFindParticipantsException)
    async def exception_handler_cannotfindtheparticipantexception(
        _request: Request, exc: CannotFindParticipantsException
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
            status_code=status, content={"message": str(type(cause)) or str(exc)}
        )
