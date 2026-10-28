"""Tests that the first user of a dev environment gets a working pgstress data source, through the API, against the
local Postgres testing warehouse, which task bootstrap-dwh-database loads the stress test table into.
"""

import pytest
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.storage.bootstrap import BQSTRESS_DATASOURCE_NAME, PGSTRESS_DATASOURCE_NAME
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.helpers import delete_seeded_users, stress_experiment_sample_size


def test_the_first_user_gets_a_pgstress_data_source(xngin_session: Session, aclient: AdminAPIClient):
    delete_seeded_users(xngin_session)
    # The experiment filters to active participants who logged in recently, and assigns 40 of them.
    assert stress_experiment_sample_size(aclient, PGSTRESS_DATASOURCE_NAME) == 40


def test_a_misconfigured_cloud_warehouse_does_not_block_the_first_sign_in(
    xngin_session: Session, aclient: AdminAPIClient, monkeypatch: pytest.MonkeyPatch
):
    delete_seeded_users(xngin_session)
    # Without ?credentials_path=, the bqstress data source cannot be created.
    monkeypatch.setattr(flags, "XNGIN_DEVDWH_GCP_BIGQUERY_DSN", "bigquery://some-project/some_dataset")
    assert stress_experiment_sample_size(aclient, PGSTRESS_DATASOURCE_NAME) == 40
    (organization,) = aclient.list_organizations().data.items
    datasources = aclient.list_organization_datasources(organization_id=organization.id).data.items
    assert BQSTRESS_DATASOURCE_NAME not in {d.name for d in datasources}
