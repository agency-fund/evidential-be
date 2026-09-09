from sqlalchemy import select

from xngin.apiserver.sqla import tables


async def test_datasource_set_table_list(xngin_session, testing_datasource):
    datasource = testing_datasource.ds

    datasource.set_table_list([])
    await xngin_session.flush()
    assert datasource.table_list == []
    assert datasource.table_list_updated is not None
    assert (
        await xngin_session.scalar(
            select(tables.Datasource.table_list.is_(None)).where(tables.Datasource.id == datasource.id)
        )
    ) is False

    datasource.set_table_list(None)
    await xngin_session.flush()
    assert datasource.table_list is None
    assert datasource.table_list_updated is None
    assert (
        await xngin_session.scalar(
            select(tables.Datasource.table_list.is_(None)).where(tables.Datasource.id == datasource.id)
        )
    ) is True
