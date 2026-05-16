"""Tests de integracion livianos para la API FastAPI."""

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from app.api import server


@pytest.fixture()
def client():
    server.route_cache.clear()
    server.travel_graph.graph.clear()
    server.travel_graph.vertices.clear()
    server.reservation_manager.reservations.clear()
    return TestClient(server.app)


def test_health_endpoint_returns_system_stats(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "cache_stats" in payload
    assert "reservation_stats" in payload


def test_matrix_endpoint_validates_transport_and_metric(client):
    invalid = client.get("/routes/matrix", params={"transport": "barco"})
    assert invalid.status_code == 400

    response = client.get(
        "/routes/matrix",
        params={"transport": "tren", "optimize_by": "time"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "tren"
    assert payload["optimize_by"] == "time"
    assert len(payload["cities"]) == len(payload["matrix"])
    assert all(row[index] == 0.0 for index, row in enumerate(payload["matrix"]))


def test_shortest_route_endpoint_caches_second_request(client):
    request = {
        "origin": "Madrid",
        "destination": "Barcelona",
        "optimize_by": "cost",
    }

    first = client.post("/routes/shortest", json=request)
    second = client.post("/routes/shortest", json=request)

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["path"][0] == "Madrid"
    assert first_payload["path"][-1] == "Barcelona"
    assert first_payload["cached"] is False
    assert second_payload["cached"] is True
    assert second_payload["path"] == first_payload["path"]
