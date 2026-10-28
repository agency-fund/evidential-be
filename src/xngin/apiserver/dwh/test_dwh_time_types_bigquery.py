"""Tests of date and time columns through the admin API, against the stress test table in a new BigQuery dataset.

They are marked integration and need XNGIN_DEVDWH_GCP_CREDENTIALS.
"""

from collections.abc import Iterator

import pytest

from xngin.apiserver.settings import BqDsn
from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.dwh_time_types_helpers import check_time_fields
from xngin.apiserver.testing.helpers import skip_unless_gcp_credentials
from xngin.apiserver.testing.stress_loaders import bigquery_stress_dataset

pytestmark = [pytest.mark.integration, skip_unless_gcp_credentials]


@pytest.fixture(name="stress_dataset", scope="module")
def fixture_stress_dataset() -> Iterator[BqDsn]:
    with bigquery_stress_dataset() as dsn:
        yield dsn


# sqlalchemy-bigquery's reflection maps only some BigQuery types to SQLAlchemy types; it maps neither JSON nor, without
# GeoAlchemy2, GEOGRAPHY. It reflects those columns as NullType with this warning, and inspection skips them.
@pytest.mark.filterwarnings("ignore:Did not recognize type:sqlalchemy.exc.SAWarning")
def test_bigquery_date_and_time_fields(aclient: AdminAPIClient, stress_dataset: BqDsn):
    check_time_fields(aclient, stress_dataset, stress_dwh_def.BIGQUERY)
