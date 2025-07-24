import pytest

from xngin.apiserver import conftest
from xngin.apiserver.settings import (
    CannotFindParticipantsError,
)


def test_get_settings_for_test():
    settings = conftest.get_settings_for_test()
    assert settings.get_datasource("testing").config.dwh.user == "postgres"


def test_root_get_api(client):
    response = client.get("/")
    assert response.status_code == 404


def test_find_participants_raises_exception_for_invalid_participant_type():
    settings = conftest.get_settings_for_test()
    config = settings.get_datasource("testing").config
    with pytest.raises(CannotFindParticipantsError):
        config.find_participants("bad_type")
