"""Tests that the bqstress data source a dev environment's first user gets works, against a real BigQuery dataset that
the test loads the stress test table into. They are marked integration and need XNGIN_DEVDWH_GCP_CREDENTIALS.
"""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.storage.bootstrap import BQSTRESS_DATASOURCE_NAME
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.helpers import (
    GCP_CREDENTIALS_VAR,
    delete_seeded_users,
    skip_unless_gcp_credentials,
    stress_experiment_sample_size,
)
from xngin.apiserver.testing.stress_loaders import bigquery_stress_dataset


@pytest.fixture(name="stress_bigquery_dsn")
def fixture_stress_bigquery_dsn() -> Iterator[str]:
    """The DSN of a new BigQuery dataset holding the stress test table."""
    with bigquery_stress_dataset() as dsn:
        yield f"bigquery://{dsn.project_id}/{dsn.dataset_id}?credentials_path={os.environ[GCP_CREDENTIALS_VAR]}"


@pytest.mark.integration
@skip_unless_gcp_credentials
# sqlalchemy-bigquery's reflection maps only some BigQuery types to SQLAlchemy types; it maps neither JSON nor, without
# GeoAlchemy2, GEOGRAPHY. It reflects those columns as NullType with this warning, and inspection skips them.
@pytest.mark.filterwarnings("ignore:Did not recognize type:sqlalchemy.exc.SAWarning")
def test_the_first_user_gets_a_bqstress_data_source(
    xngin_session: Session, aclient: AdminAPIClient, monkeypatch: pytest.MonkeyPatch, stress_bigquery_dsn: str
):
    delete_seeded_users(xngin_session)
    monkeypatch.setattr(flags, "XNGIN_DEVDWH_GCP_BIGQUERY_DSN", stress_bigquery_dsn)
    monkeypatch.setattr(flags, "XNGIN_DEVDWH_AWS_REDSHIFT_DSN", "")
    # The experiment filters to participants in three cities, a RECORD sub-field, who logged in recently, and assigns
    # 40 of them.
    assert stress_experiment_sample_size(aclient, BQSTRESS_DATASOURCE_NAME) == 40
