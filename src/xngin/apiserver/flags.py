import os


def truthy_env(env_var: str):
    """Return True if the environment variable is "true" or "1", or False otherwise."""
    return os.environ.get(env_var, "").lower() in {"true", "1"}


ECHO_SQL = truthy_env("ECHO_SQL")
ENABLE_ADMIN = truthy_env("ENABLE_ADMIN")
ENABLE_OIDC = truthy_env("ENABLE_OIDC")
PUBLISH_ALL_DOCS = truthy_env("XNGIN_PUBLISH_ALL_DOCS")
SAFE_RESOLVE_TOLERANT = truthy_env("SAFE_RESOLVE_TOLERANT")
UPDATE_API_TESTS = truthy_env("UPDATE_API_TESTS")
