import pytest

from xngin.apiserver.settings import CannotFindParticipantsError


def test_root_get_api(client):
    response = client.get("/")
    assert response.status_code == 404


def test_static_settings_contains_testing_datasource(static_settings):
    assert static_settings.get_datasource("testing").config.dwh.user == "postgres"


def test_find_participants_raises_specific_exception_for_undefined_participants(
    static_settings,
):
    config = static_settings.get_datasource("testing").config
    with pytest.raises(CannotFindParticipantsError):
        config.find_participants("bad_type")
