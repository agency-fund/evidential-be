from sqlalchemy import select

from xngin.apiserver.sqla import tables


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
