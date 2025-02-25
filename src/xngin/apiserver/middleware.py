from fastapi.middleware.cors import CORSMiddleware


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
