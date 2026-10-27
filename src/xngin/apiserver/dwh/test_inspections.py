from sqlalchemy import JSON, Boolean, Column, MetaData, String, Table

from xngin.apiserver.dwh.inspections import create_inspect_table_response_from_table
from xngin.apiserver.routers.admin.admin_api_types import FieldMetadata, InspectDatasourceTableResponse
from xngin.apiserver.routers.common_enums import DataType


def test_create_inspect_table_response_from_table_success():
    my_table = Table(
        "flat_table",
        MetaData(),
        Column("participant_id", String, primary_key=True),
        Column("is_onboarded", Boolean),
        Column("payload", JSON),
    )

    assert create_inspect_table_response_from_table(my_table) == InspectDatasourceTableResponse(
        primary_key_fields=["participant_id"],
        detected_unique_id_fields=["participant_id"],
        # JSON columns are unsupported and left out.
        fields=[
            FieldMetadata(field_name="is_onboarded", data_type=DataType.BOOLEAN, description=""),
            FieldMetadata(field_name="participant_id", data_type=DataType.CHARACTER_VARYING, description=""),
        ],
    )
