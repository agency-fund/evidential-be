import pytest
from sqlalchemy import JSON, Boolean, Column, Integer, MetaData, String, Table
from sqlalchemy_bigquery import STRUCT

from xngin.apiserver.dwh.inspections import create_inspect_table_response_from_table
from xngin.apiserver.exceptions_common import LateValidationError
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


def test_create_inspect_table_response_from_table_skips_postgres_dotted_columns():
    # A dotted column whose parent is not a STRUCT is an ordinary (Postgres) column, not a nested
    # one. FieldName does not allow dots, so it is left out of the response rather than rejected.
    my_table = Table(
        "pg_table",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("score", Integer),
        Column("score.total", Integer),
        Column("score.total_id", Integer),
    )

    assert create_inspect_table_response_from_table(my_table) == InspectDatasourceTableResponse(
        primary_key_fields=["id"],
        detected_unique_id_fields=["id"],
        fields=[
            FieldMetadata(field_name="id", data_type=DataType.INTEGER, description=""),
            FieldMetadata(field_name="score", data_type=DataType.INTEGER, description=""),
        ],
    )


def test_create_inspect_table_response_from_table_rejects_bigquery_nested_columns():
    # Mirrors how the BigQuery dialect reflects a RECORD column: the parent, then one column per sub-field.
    my_table = Table(
        "replicated_table",
        MetaData(),
        Column("_id", String),
        Column("datastream_metadata", STRUCT(uuid=String, source_timestamp=Integer)),
        Column("datastream_metadata.uuid", String),
        Column("datastream_metadata.source_timestamp", Integer),
    )

    with pytest.raises(LateValidationError) as excinfo:
        create_inspect_table_response_from_table(my_table)

    message = str(excinfo.value)
    assert "Table 'replicated_table' has nested columns" in message
    assert "datastream_metadata.source_timestamp, datastream_metadata.uuid" in message
    assert "Create a view" in message
