from fastapi.testclient import TestClient

from api.main import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "spatial_csv" in body


def test_states_endpoint():
    client = TestClient(app)
    r = client.get("/api/v1/states")
    assert r.status_code == 200
    body = r.json()
    assert "states" in body
    assert isinstance(body["states"], list)
