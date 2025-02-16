from fastapi.middleware.cors import CORSMiddleware


def setup(app):
    """Registers middleware with the FastAPI app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
