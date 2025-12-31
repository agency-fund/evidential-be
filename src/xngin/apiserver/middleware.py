from fastapi.middleware.cors import CORSMiddleware

from xngin.apiserver.request_encapsulation_middleware import RequestEncapsulationMiddleware


def setup(app):
    """Registers middleware with the FastAPI app."""
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=False,
        allow_headers=["*"],
        allow_methods=["*"],
        allow_origins=["*"],
        max_age=7200,  # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Max-Age
    )
    app.add_middleware(RequestEncapsulationMiddleware, path_prefix="/v1/experiments")
