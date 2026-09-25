import pytest
from pydantic import ValidationError
from sqlalchemy import select

from xngin.apiserver.settings import RemoteDatabaseConfig
from xngin.apiserver.sqla import tables

#: A datasource config as written before migration 20260901184333 stripped the "participants" key.
LEGACY_CONFIG_WITH_PARTICIPANTS = {
    "type": "remote",
    "dwh": {"driver": "none"},
    "participants": [
        {
            "type": "schema",
            "participant_type": "users",
            "hidden": False,
            "table_name": "dwh",
            "fields": [{"field_name": "id", "data_type": "bigint", "is_unique_id": True}],
        }
    ],
}


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


def test_get_config_tolerates_legacy_participants_key(xngin_session, testing_datasource):
    """Datasources not yet reached by migration 20260901184333 must still load.

    get_config() runs on every authenticated request via edeps.datasource, so rejecting the
    removed "participants" key would take the public API down for any unmigrated datasource. Drop
    this test in the same change that restores extra="forbid" on RemoteDatabaseConfig.
    """
    datasource = testing_datasource.ds
    datasource.config = LEGACY_CONFIG_WITH_PARTICIPANTS
    xngin_session.flush()

    config = datasource.get_config()

    assert config.type == "remote"
    assert config.dwh.driver == "none"
    assert not hasattr(config, "participants")


def test_set_config_drops_legacy_participants_key(xngin_session, testing_datasource):
    """Rewriting a legacy config self-heals it, so the migration is not the only way to clean a row."""
    datasource = testing_datasource.ds
    datasource.config = LEGACY_CONFIG_WITH_PARTICIPANTS
    xngin_session.flush()

    datasource.set_config(datasource.get_config())
    xngin_session.flush()

    assert "participants" not in datasource.config
    assert (
        xngin_session.scalar(select(tables.Datasource.config).where(tables.Datasource.id == datasource.id))
    ).keys() == {"type", "dwh"}


def test_remote_database_config_still_rejects_unknown_dwh_keys():
    """Relaxing RemoteDatabaseConfig must not relax the nested dwh models."""
    with pytest.raises(ValidationError) as excinfo:
        RemoteDatabaseConfig.model_validate({"type": "remote", "dwh": {"driver": "none", "hostt": "typo"}})
    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())
