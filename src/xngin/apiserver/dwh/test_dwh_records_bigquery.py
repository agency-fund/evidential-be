"""Tests of RECORD sub-fields through the admin API, against the stress test table in a new BigQuery dataset.

They are marked integration and need XNGIN_DEVDWH_GCP_CREDENTIALS.
"""

from collections.abc import Iterator

import pytest

from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest, Filter, PowerRequest
from xngin.apiserver.routers.common_enums import DataType, Relation
from xngin.apiserver.settings import BqDsn
from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.dwh_identifiers_helpers import create_datasource
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
def test_bigquery_record_sub_fields(aclient: AdminAPIClient, stress_dataset: BqDsn):
    datasource_id = create_datasource(aclient, stress_dataset)
    table_name = stress_dwh_def.BIGQUERY.table

    inspected = aclient.inspect_table_in_datasource(datasource_id=datasource_id, table_name=table_name).data
    data_types = {f.field_name: f.data_type for f in inspected.fields}
    # The scalar sub-fields of address, at any depth, are offered. The RECORDs themselves, address and address.geo, are
    # not, nor are REPEATED fields: address.lines within the RECORD, and events and tags at the top level.
    assert {name: data_type for name, data_type in data_types.items() if name.startswith("address")} == {
        "address.city": DataType.CHARACTER_VARYING,
        "address.postal code": DataType.CHARACTER_VARYING,
        "address.group": DataType.CHARACTER_VARYING,
        "address.geo.lat": DataType.DOUBLE_PRECISION,
        "address.geo.lng": DataType.DOUBLE_PRECISION,
    }
    assert not {"events", "tags"} & data_types.keys()

    # A filter on a sub-field named by a reserved word.
    metric = DesignSpecMetricRequest(field_name=stress_dwh_def.BIGQUERY.column("visits").name, metric_pct_change=0.5)
    filter_ = Filter(field_name="address.group", relation=Relation.INCLUDES, value=["north"])
    power = aclient.power_check(
        datasource_id=datasource_id,
        body=PowerRequest(table_name=table_name, metrics=[metric], filters=[filter_], n_arms=2),
    ).data
    expected = sum(row["address_group"] == "north" for row in stress_dwh_def.generate_rows())
    assert power.analyses[0].metric_spec.available_n == expected
