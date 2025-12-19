def test_root_get_api(client):
    response = client.get("/")
    assert response.status_code == 404
