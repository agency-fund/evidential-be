from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from granian.utils.proxies import wrap_asgi_with_proxy_headers

from xngin.apiserver import flags
from xngin.apiserver.request_context_middleware import RequestContextMiddleware
from xngin.apiserver.request_encapsulation_middleware import RequestEncapsulationMiddleware


def setup(app):
    """Registers middleware with the FastAPI app."""
    # Middleware added here runs in reverse registration order on incoming requests, and responses unwind in
    # registration order.
    app.add_middleware(
        GZipMiddleware,
        compresslevel=3,
        minimum_size=16384,
    )
    app.add_middleware(RequestEncapsulationMiddleware, path_prefix="/v1/experiments")
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=False,
        allow_headers=["*"],
        allow_methods=["*"],
        allow_origins=flags.CORS_ALLOWED_ORIGINS,
        max_age=7200,  # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Max-Age
    )
    # RequestContextMiddleware sets context variables used by logging, so we want it to be registered later in the list
    # so that it is invoked earlier in the request lifecycle.
    app.add_middleware(RequestContextMiddleware)
    if flags.is_using_reverse_proxy():
        # The reverse proxy reports the client's address in X-Forwarded-For and the original scheme in
        # X-Forwarded-Proto. Trusting them puts the client's address in access logs.
        app.add_middleware(wrap_asgi_with_proxy_headers, trusted_hosts=flags.FORWARDED_ALLOW_IPS)
