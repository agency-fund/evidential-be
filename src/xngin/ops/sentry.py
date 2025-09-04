"""Common implementation of Sentry client configuration.

Entrypoints that want Sentry enabled when SENTRY_DSN is set can call setup().
"""

import os

from loguru import logger

from xngin.apiserver import constants


def setup():
    """Configures Sentry if SENTRY_DSN is present."""
    sentry_dsn = os.environ.get("SENTRY_DSN")
    if not sentry_dsn:
        return
    logger.info("Setting up Sentry: {sentry_dsn}", sentry_dsn=sentry_dsn)

    import sentry_sdk  # noqa: PLC0415
    from sentry_sdk import scrubber  # noqa: PLC0415

    denylist = [
        *scrubber.DEFAULT_DENYLIST,
        "dsn",  # command line flags
        "code",  # OIDC
        "code_verifier",  # OIDC
        "credentials_base64",  # sqlalchemy-bigquery
        "credentials_info",  # sqlalchemy-bigquery
        "database_url",  # common name of variable containing application database credentials
    ]
    pii_denylist = [
        *scrubber.DEFAULT_PII_DENYLIST,
        "webhook_token",
        # Header names are exposed to Sentry in a normalized form.
        constants.HEADER_WEBHOOK_TOKEN.lower().replace("-", "_"),
        "email",
    ]

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=os.environ.get("ENVIRONMENT", "local"),
        traces_sample_rate=1.0,
        _experiments={
            "continuous_profiling_auto_start": True,
        },
        send_default_pii=False,
        event_scrubber=sentry_sdk.scrubber.EventScrubber(denylist=denylist, pii_denylist=pii_denylist),
    )
