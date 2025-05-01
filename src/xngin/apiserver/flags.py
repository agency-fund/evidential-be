import os

from xngin.apiserver import constants


def truthy_env(env_var: str):
    """Return True if the environment variable is "true" or "1", or False otherwise."""
    return os.environ.get(env_var, "").lower() in {"true", "1"}


# OIDC-related flags
AIRPLANE_MODE = truthy_env("AIRPLANE_MODE")
ENV_GOOGLE_OIDC_CLIENT_ID = "GOOGLE_OIDC_CLIENT_ID"
CLIENT_ID = os.environ.get(ENV_GOOGLE_OIDC_CLIENT_ID)
ENV_GOOGLE_OIDC_CLIENT_SECRET = "GOOGLE_OIDC_CLIENT_SECRET"
CLIENT_SECRET = os.environ.get(ENV_GOOGLE_OIDC_CLIENT_SECRET)
DEFAULT_REDIRECT_URI = f"http://localhost:8000{constants.API_PREFIX_V1}/a/oidc"
OIDC_REDIRECT_URI = os.environ.get(
    "GOOGLE_OIDC_REDIRECT_URI", DEFAULT_REDIRECT_URI
)  # used for testing UI only

ALLOW_CONNECTING_TO_PRIVATE_IPS = truthy_env("ALLOW_CONNECTING_TO_PRIVATE_IPS")
ECHO_SQL = truthy_env("ECHO_SQL")
ECHO_SQL_APP_DB = truthy_env("ECHO_SQL_APP_DB")
ENABLE_ADMIN = truthy_env("ENABLE_ADMIN")
ENABLE_OIDC = truthy_env("ENABLE_OIDC")
PUBLISH_ALL_DOCS = truthy_env("XNGIN_PUBLISH_ALL_DOCS")
UPDATE_API_TESTS = truthy_env("UPDATE_API_TESTS")

XNGIN_DEVDWH_DSN = os.environ.get("XNGIN_DEVDWH_DSN", "")

# XNGIN_PUBLIC_PROTOCOL defines the protocol clients should use on our public URL. This should always be "https",
# except in dev environments.
XNGIN_PUBLIC_PROTOCOL = os.environ.get("XNGIN_PUBLIC_PROTOCOL", "https")

# XNGIN_PUBLIC_HOSTNAME defines the base hostname (and optional port) we use when constructing URLs to send to
# external systems (such as via outbound webhooks).
XNGIN_PUBLIC_HOSTNAME = os.environ.get(
    "XNGIN_PUBLIC_HOSTNAME", "main.dev.agencyfund.org"
)

LOG_SQL = truthy_env("LOG_SQL")
LOG_SQL_APP_DB = truthy_env("LOG_SQL_APP_DB")

DEBUG_LOGGING = truthy_env("DEBUG_LOGGING")
FRIENDLY_DEV_LOGGING = truthy_env("FRIENDLY_DEV_LOGGING")
