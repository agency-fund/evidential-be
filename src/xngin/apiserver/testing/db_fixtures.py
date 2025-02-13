import dataclasses
import secrets

import pytest

from xngin.apiserver.apikeys import make_key, hash_key
from xngin.apiserver.conftest import get_settings_for_test
from xngin.apiserver.models.tables import Organization, Datasource, ApiKey


@pytest.fixture(scope="function")
def secured_datasource(db_session):
    """Creates a new test datasource with its associated organization."""
    run_id = secrets.token_hex(8)
    datasource_id = "ds" + run_id

    # We derive a new test datasource from the standard static "testing" datasource by
    # randomizing its unique ID and marking it as requiring an API key.
    # TODO: replace this with a non-sqliteconfig value.
    test_ds = get_settings_for_test().get_datasource("testing").config

    org = Organization(id="org" + run_id, name="test organization")

    datasource = Datasource(
        id=datasource_id, name="test ds", config=test_ds.model_dump()
    )
    datasource.organization = org

    key_id, key = make_key()
    kt = ApiKey(id=key_id, key=hash_key(key))
    kt.datasource = datasource

    db_session.add_all([org, datasource, kt])
    db_session.commit()
    assert db_session.get(Datasource, datasource_id) is not None
    return DatasourceMetadata(org=org, ds=datasource, key=key)


@dataclasses.dataclass
class DatasourceMetadata:
    """Describes an ephemeral datasource, organization, and API key."""

    org: Organization
    ds: Datasource
    key: str
