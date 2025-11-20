"""Flags describes values that are read from the environment."""

import enum
import os

from xngin.apiserver import constants


def is_dev_environment():
    return os.environ.get("ENVIRONMENT", "") in {"dev", ""}


def is_railway() -> bool:
    return os.environ.get("RAILWAY_SERVICE_NAME", "") != ""


def truthy_env(env_var: str):
    """Return True if the environment variable is "true" or "1", or False otherwise."""
    return os.environ.get(env_var, "").lower() in {"true", "1"}


# Flags configuring OIDC
AIRPLANE_MODE = truthy_env("AIRPLANE_MODE")
ENV_GOOGLE_OIDC_CLIENT_ID = "GOOGLE_OIDC_CLIENT_ID"
CLIENT_ID = os.environ.get(ENV_GOOGLE_OIDC_CLIENT_ID)
ENV_GOOGLE_OIDC_CLIENT_SECRET = "GOOGLE_OIDC_CLIENT_SECRET"
CLIENT_SECRET = os.environ.get(ENV_GOOGLE_OIDC_CLIENT_SECRET)
DEFAULT_REDIRECT_URI = f"http://localhost:8000{constants.API_PREFIX_V1}/a/oidc"
OIDC_REDIRECT_URI = os.environ.get("GOOGLE_OIDC_REDIRECT_URI", DEFAULT_REDIRECT_URI)  # used for testing UI only

# XNGIN_SESSION_TOKEN_KEYSET contains a keyset for encrypting session tokens. This is generated using the
# `xngin-cli create-nacl-keyset` command. If set to "local", we will read from a local file (see:
# session_token_crypter).
ENV_SESSION_TOKEN_KEYSET = "XNGIN_SESSION_TOKEN_KEYSET"

ALLOW_CONNECTING_TO_PRIVATE_IPS = truthy_env("ALLOW_CONNECTING_TO_PRIVATE_IPS")
DISABLE_SAFEDNS_CHECK = truthy_env("DISABLE_SAFEDNS_CHECK")
PUBLISH_ALL_DOCS = truthy_env("XNGIN_PUBLISH_ALL_DOCS")
UPDATE_API_TESTS = truthy_env("UPDATE_API_TESTS")

XNGIN_DEVDWH_DSN = os.environ.get("XNGIN_DEVDWH_DSN", "")

# Hosting providers may set hosted database URL as DATABASE_URL, so we use the same.
DATABASE_URL = os.environ.get("DATABASE_URL")


# XNGIN_PUBLIC_PROTOCOL defines the protocol clients should use on our public URL. This should always be "https",
# except in dev environments.
XNGIN_PUBLIC_PROTOCOL = os.environ.get("XNGIN_PUBLIC_PROTOCOL", "https")

# XNGIN_PUBLIC_HOSTNAME defines the base hostname (and optional port) we use when constructing URLs to send to
# external systems (such as via outbound webhooks or in generated API examples).
XNGIN_PUBLIC_HOSTNAME = os.environ.get("XNGIN_PUBLIC_HOSTNAME", "example.com")

# XNGIN_PRODUCT_HOMEPAGE defines the homepage of the product.
XNGIN_PRODUCT_HOMEPAGE = os.environ.get("XNGIN_PRODUCT_HOMEPAGE", "https://example.com")

# XNGIN_SUPPORT_EMAIL defines the email address that end-users can message for support.
XNGIN_SUPPORT_EMAIL = os.environ.get("XNGIN_SUPPORT_EMAIL", "support@example.com")

LOG_SQL_APP_DB = truthy_env("LOG_SQL_APP_DB")
LOG_SQL_DWH = truthy_env("LOG_SQL_DWH")


def get_public_api_base_url():
    if serving_domain := os.environ.get("RAILWAY_PUBLIC_DOMAIN"):
        return f"https://{serving_domain}"
    if (hostname := XNGIN_PUBLIC_HOSTNAME) and (proto := XNGIN_PUBLIC_PROTOCOL):
        return f"{proto}://{hostname}"
    return ""


# The base URL for this API server. This is used only to populate fields in the generated OpenAPI spec.
XNGIN_PUBLIC_API_BASE_URL = get_public_api_base_url()

# Populates the description field of the generated OpenAPI spec.
XNGIN_PUBLIC_API_DESCRIPTION = os.environ.get("XNGIN_PUBLIC_API_DESCRIPTION", "")


class LogFormat(enum.StrEnum):
    FRIENDLY = "friendly"
    STRUCTURED_RAILWAY = "structured_railway"
    DEFAULT = "default"

    @classmethod
    def from_env(cls):
        if is_railway():
            return LogFormat.STRUCTURED_RAILWAY
        if is_dev_environment():
            return LogFormat.FRIENDLY
        return LogFormat.DEFAULT


LOG_FORMAT = LogFormat.from_env()
