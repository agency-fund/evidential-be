import os


def truthy_env(env_var: str):
    """Return True if the environment variable is "true" or "1", or False otherwise."""
    return os.environ.get(env_var, "").lower() in {"true", "1"}


ALLOW_CONNECTING_TO_PRIVATE_IPS = truthy_env("ALLOW_CONNECTING_TO_PRIVATE_IPS")
ECHO_SQL = truthy_env("ECHO_SQL")
ECHO_SQL_APP_DB = truthy_env("ECHO_SQL_APP_DB")
ENABLE_ADMIN = truthy_env("ENABLE_ADMIN")
ENABLE_OIDC = truthy_env("ENABLE_OIDC")
PUBLISH_ALL_DOCS = truthy_env("XNGIN_PUBLISH_ALL_DOCS")
UPDATE_API_TESTS = truthy_env("UPDATE_API_TESTS")

XNGIN_DEVDWH_DSN = os.environ.get("XNGIN_DEVDWH_DSN", "")

LOG_SQL = truthy_env("LOG_SQL")
LOG_SQL_APP_DB = truthy_env("LOG_SQL_APP_DB")

DEBUG_LOGGING = truthy_env("DEBUG_LOGGING")
FRIENDLY_DEV_LOGGING = truthy_env("FRIENDLY_DEV_LOGGING")
