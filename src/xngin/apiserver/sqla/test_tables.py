import pytest
from pydantic import ValidationError
from sqlalchemy import select

from xngin.apiserver.settings import RemoteDatabaseConfig
from xngin.apiserver.sqla import tables


def test_datasource_set_table_list(xngin_session, testing_datasource):
    datasource = testing_datasource.ds

    datasource.set_table_list([])
    xngin_session.flush()
    assert datasource.table_list == []
    assert datasource.table_list_updated is not None
    assert (
        xngin_session.scalar(
            select(tables.Datasource.table_list.is_(None)).where(tables.Datasource.id == datasource.id)
        )
    ) is False

    datasource.set_table_list(None)
    xngin_session.flush()
    assert datasource.table_list is None
    assert datasource.table_list_updated is None
    assert (
        xngin_session.scalar(
            select(tables.Datasource.table_list.is_(None)).where(tables.Datasource.id == datasource.id)
        )
    ) is True


@pytest.mark.parametrize(
    "config",
    [
        # A config written before migration 20260901184333 stripped the removed "participants" key.
        {"type": "remote", "dwh": {"driver": "none"}, "participants": []},
        {"type": "remote", "dwh": {"driver": "none", "hostt": "typo"}},
    ],
)
def test_remote_database_config_rejects_unknown_keys(config):
    with pytest.raises(ValidationError) as excinfo:
        RemoteDatabaseConfig.model_validate(config)
    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())
