"""Tests of date and time columns through the admin API, against the stress test table in a new Redshift schema.

They are marked integration and need XNGIN_DEVDWH_AWS_REDSHIFT_DSN, XNGIN_DEVDWH_AWS_BUCKET and
XNGIN_DEVDWH_AWS_REDSHIFT_IAM_ROLE, and AWS credentials that may write to the bucket.
"""

import os
from collections.abc import Iterator

import pytest

from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.settings import Dsn
from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.dwh_time_types_helpers import check_time_fields
from xngin.apiserver.testing.stress_loaders import (
    REDSHIFT_BUCKET_VAR,
    REDSHIFT_DSN_VAR,
    REDSHIFT_IAM_ROLE_VAR,
    redshift_stress_schema,
)

_VARS = (REDSHIFT_DSN_VAR, REDSHIFT_BUCKET_VAR, REDSHIFT_IAM_ROLE_VAR)

pytestmark = [
    pytest.mark.integration,
    # botocore, which stages the CSV in S3 for Redshift's COPY, calls the deprecated datetime.utcnow().
    pytest.mark.filterwarnings("ignore:datetime.datetime.utcnow:DeprecationWarning"),
    pytest.mark.skipif(
        not all(os.environ.get(var) for var in _VARS),
        reason=f"{', '.join(_VARS)} are required for Redshift integration tests",
    ),
]


@pytest.fixture(name="stress_dsn", scope="module")
def fixture_stress_dsn() -> Iterator[Dsn]:
    with redshift_stress_schema("stress_time") as dsn:
        yield dsn


@pytest.fixture(name="real_dns")
def fixture_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves the Redshift host for real; the test suite otherwise sends every connection to 127.0.0.1."""
    monkeypatch.setattr(safe_resolve, "INTERCEPT_DNS_FOR_TESTING", False)


@pytest.mark.usefixtures("real_dns")
def test_redshift_date_and_time_fields(aclient: AdminAPIClient, stress_dsn: Dsn):
    # Redshift lists every name in lowercase.
    check_time_fields(aclient, stress_dsn, stress_dwh_def.REDSHIFT, name=str.lower)
