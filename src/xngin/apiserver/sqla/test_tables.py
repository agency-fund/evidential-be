from sqlalchemy import select

from xngin.apiserver.sqla import tables


async def test_task_payload_none_is_sql_null(xngin_session):
    task = tables.Task(task_type="test", payload={})
    xngin_session.add(task)
    await xngin_session.flush()
    assert task.payload == {}
    payload_is_sql_null = await xngin_session.scalar(
        select(tables.Task.payload.is_(None)).where(tables.Task.id == task.id)
    )
    assert payload_is_sql_null is False

    task.payload = None
    await xngin_session.flush()
    assert task.payload is None
    payload_is_sql_null = await xngin_session.scalar(
        select(tables.Task.payload.is_(None)).where(tables.Task.id == task.id)
    )
    assert payload_is_sql_null is True


async def test_participant_type_inspection_response_none_is_sql_null(xngin_session, testing_datasource):
    inspection = tables.ParticipantTypesInspected(
        datasource_id=testing_datasource.datasource_id,
        participant_type="participant_type",
        response={},
    )
    xngin_session.add(inspection)
    await xngin_session.flush()
    response_is_sql_null = await xngin_session.scalar(
        select(tables.ParticipantTypesInspected.response.is_(None)).where(
            tables.ParticipantTypesInspected.datasource_id == inspection.datasource_id,
            tables.ParticipantTypesInspected.participant_type == inspection.participant_type,
        )
    )
    assert response_is_sql_null is False

    inspection.response = None
    await xngin_session.flush()
    assert inspection.response is None
    response_is_sql_null = await xngin_session.scalar(
        select(tables.ParticipantTypesInspected.response.is_(None)).where(
            tables.ParticipantTypesInspected.datasource_id == inspection.datasource_id,
            tables.ParticipantTypesInspected.participant_type == inspection.participant_type,
        )
    )
    assert response_is_sql_null is True


async def test_datasource_table_inspection_response_none_is_sql_null(xngin_session, testing_datasource):
    inspection = tables.DatasourceTablesInspected(
        datasource_id=testing_datasource.datasource_id,
        table_name="table_name",
        response={},
    )
    xngin_session.add(inspection)
    await xngin_session.flush()
    response_is_sql_null = await xngin_session.scalar(
        select(tables.DatasourceTablesInspected.response.is_(None)).where(
            tables.DatasourceTablesInspected.datasource_id == inspection.datasource_id,
            tables.DatasourceTablesInspected.table_name == inspection.table_name,
        )
    )
    assert response_is_sql_null is False

    inspection.response = None
    await xngin_session.flush()
    assert inspection.response is None
    response_is_sql_null = await xngin_session.scalar(
        select(tables.DatasourceTablesInspected.response.is_(None)).where(
            tables.DatasourceTablesInspected.datasource_id == inspection.datasource_id,
            tables.DatasourceTablesInspected.table_name == inspection.table_name,
        )
    )
    assert response_is_sql_null is True


async def test_turn_connection_journeys_dict_none_is_sql_null(xngin_session, testing_datasource):
    turn_connection = tables.TurnConnection(
        organization_id=testing_datasource.organization_id,
        encrypted_turn_api_token="encrypted",
        turn_api_token_preview="view",
        journeys_dict={},
    )
    xngin_session.add(turn_connection)
    await xngin_session.flush()
    assert turn_connection.journeys_dict == {}
    journeys_dict_is_sql_null = await xngin_session.scalar(
        select(tables.TurnConnection.journeys_dict.is_(None)).where(
            tables.TurnConnection.organization_id == turn_connection.organization_id
        )
    )
    assert journeys_dict_is_sql_null is False

    turn_connection.journeys_dict = None
    await xngin_session.flush()
    assert turn_connection.journeys_dict is None
    journeys_dict_is_sql_null = await xngin_session.scalar(
        select(tables.TurnConnection.journeys_dict.is_(None)).where(
            tables.TurnConnection.organization_id == turn_connection.organization_id
        )
    )
    assert journeys_dict_is_sql_null is True


async def test_datasource_table_list_none_is_sql_null(xngin_session, testing_datasource):
    datasource = testing_datasource.ds

    datasource.set_table_list([])
    await xngin_session.flush()
    assert datasource.table_list == []
    table_list_is_sql_null = await xngin_session.scalar(
        select(tables.Datasource.table_list.is_(None)).where(tables.Datasource.id == datasource.id)
    )
    assert table_list_is_sql_null is False

    datasource.set_table_list(None)
    await xngin_session.flush()
    assert datasource.table_list is None
    table_list_is_sql_null = await xngin_session.scalar(
        select(tables.Datasource.table_list.is_(None)).where(tables.Datasource.id == datasource.id)
    )
    assert table_list_is_sql_null is True
