from sqlalchemy import select

from xngin.apiserver.sqla import tables


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
