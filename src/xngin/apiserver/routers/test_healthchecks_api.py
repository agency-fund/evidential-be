def test_db_healthcheck(client):
    response = client.get("/_healthchecks/db")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_deadlock_answers_409_and_leaves_the_database_usable(client):
    for _ in range(2):
        response = client.get("/_healthchecks/deadlock")
        assert response.status_code == 409
        assert response.json() == {
            "message": "The request conflicted with a concurrent transaction and was rolled back."
        }
    assert client.get("/_healthchecks/db").status_code == 200


def test_deadlock_does_not_exist_outside_dev_environments(client, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert client.get("/_healthchecks/deadlock").status_code == 404
