from xngin.apiserver import constants

# CONFIG_ID_SECURED refers to a datasource defined in xngin.testing.settings.json
CONFIG_ID_SECURED = "testing-secured"


def test_secured_requires_apikey(client_v1):
    """Tests that a config marked as requiring an API key rejects requests w/o API keys."""
    response = client_v1.get(
        "/filters?participant_type=test_participant_type",
        headers={constants.HEADER_CONFIG_ID: CONFIG_ID_SECURED},
    )
    assert response.status_code == 403, response.content


def test_secured_wrong_apikey(client_v1):
    """Tests that a config marked as requiring an API key rejects requests when no API keys defined."""
    response = client_v1.get(
        "/filters?participant_type=test_participant_type",
        headers={
            constants.HEADER_CONFIG_ID: CONFIG_ID_SECURED,
            constants.HEADER_API_KEY: "nonexistent",
        },
    )
    assert response.status_code == 403, response.content


def test_secured_correct_apikey(testing_datasource, client_v1):
    """Tests that a config with API keys allows valid API keys and rejects invalid keys."""
    response = client_v1.get(
        "/filters?participant_type=test_participant_type",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200, response.content

    for bad_key in (
        "",
        testing_datasource.key + "suffix",
        testing_datasource.key.lower(),
        testing_datasource.key.upper(),
    ):
        response = client_v1.get(
            "/filters?participant_type=test_participant_type",
            headers={
                constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
                constants.HEADER_API_KEY: bad_key,
            },
        )
        assert response.status_code == 403, response.content
