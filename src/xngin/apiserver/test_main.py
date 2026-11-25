from xngin.apiserver import routes


def test_root_get_api(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == routes.SWAGGER_UI_PATH
